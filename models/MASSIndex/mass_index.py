"""Reference implementation of the first MASS-Index milestone.

MASS keeps HiLS's chunk-level retrieval abstraction but replaces its one
Taylor plane with an upper envelope of M planes.  This file is intentionally
PyTorch-only: it is a correctness/reference path and does not alter any
existing TileLang operator or HiLS model/configuration.

Tensor conventions:
    base_anchor: [B, N, Hq, D]
    key_chunks:  [B, N, S, Hkv, D]
    tangent_mu:  [B, N, M, Hq, D]
    tangent_b:   [B, N, M, Hq]
    queries:     [B, L, Hq, D]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MASSIndexConfig:
    """Hyperparameters owned by the standalone MASS module.

    ``num_anchors=1`` and zero offsets recover a single tangent plane.  The
    low-rank basis has shape ``[Hq, D, offset_rank]`` and is shared across
    chunks, while ``base_anchor`` supplies the block-conditioned component.
    """

    num_anchors: int = 4
    offset_rank: int = 8
    anchor_rms_norm: bool = True
    rms_norm_eps: float = 1e-6
    offset_init_std: float = 0.02
    score_reduction: str = "max"
    lse_temperature: float = 1.0
    normalize_lse: bool = False
    lse_anneal_start: Optional[float] = None
    lse_anneal_end: Optional[float] = None
    two_stage_backward: Optional[bool] = None
    # Shared-expert view (MoE shared expert): the HiLS landmark key joins the
    # upper envelope as fixed view 0 with a per-layer learnable logit bias.
    shared_view: bool = False
    shared_bias_init: float = 2.0


@dataclass
class MassTangentSummary:
    """The M tangent planes built for all chunks."""

    mu: torch.Tensor
    b: torch.Tensor
    anchors: torch.Tensor


@dataclass
class MassTopKResult:
    """Chunk candidates selected by the reference MASS scoring path."""

    indices: torch.Tensor
    selected_scores: torch.Tensor
    envelope_scores: Optional[torch.Tensor]
    winning_anchor: Optional[torch.Tensor]
    anchor_load: Optional[torch.Tensor] = None
    conditional_entropy: Optional[torch.Tensor] = None


def _rms_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps).to(x.dtype)


class MultiTangentAnchorGenerator(nn.Module):
    """Build ``a[c,m] = RMSNorm(a[c,0] + B_h e_m)``.

    ``base_anchor`` normally comes from HiLS's dedicated landmark-query
    projection at each chunk end.  The module itself is agnostic to that
    source, which makes it safe to develop separately from the existing
    attention implementation.
    """

    def __init__(self, num_query_heads: int, head_dim: int, config: MASSIndexConfig):
        super().__init__()
        if config.num_anchors < 1:
            raise ValueError("num_anchors must be >= 1")
        if config.offset_rank < 1:
            raise ValueError("offset_rank must be >= 1")

        self.num_query_heads = num_query_heads
        self.head_dim = head_dim
        self.config = config

        self.anchor_embedding = nn.Parameter(torch.empty(config.num_anchors, config.offset_rank))
        self.offset_basis = nn.Parameter(torch.empty(num_query_heads, head_dim, config.offset_rank))
        if config.shared_view:
            # 0-dim scalar params are rejected by FSDP2 fully_shard; keep numel-1 1D.
            self.shared_view_bias = nn.Parameter(torch.tensor([float(config.shared_bias_init)]))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize factorized offsets after meta tensors are materialized."""
        nn.init.normal_(self.anchor_embedding, std=self.config.offset_init_std)
        nn.init.normal_(self.offset_basis, std=self.config.offset_init_std)
        if hasattr(self, "shared_view_bias"):
            with torch.no_grad():
                self.shared_view_bias.fill_(float(self.config.shared_bias_init))

    def forward(self, base_anchor: torch.Tensor) -> torch.Tensor:
        if base_anchor.ndim != 4:
            raise ValueError(f"base_anchor must be [B, N, Hq, D], got {tuple(base_anchor.shape)}")
        B, N, Hq, D = base_anchor.shape
        if (Hq, D) != (self.num_query_heads, self.head_dim):
            raise ValueError(
                "base_anchor head shape does not match generator: "
                f"got (Hq={Hq}, D={D}), expected ({self.num_query_heads}, {self.head_dim})"
            )

        # [M, Hq, D], then broadcast over batch and chunks.
        offsets = torch.einsum("mr,hdr->mhd", self.anchor_embedding, self.offset_basis)
        anchors = base_anchor[:, :, None] + offsets[None, None]
        if self.config.anchor_rms_norm:
            anchors = _rms_normalize(anchors, self.config.rms_norm_eps)
        # Single-GPU training keeps the small trainable generator parameters in
        # FP32 while the frozen backbone and TileLang kernels run in BF16.
        # Preserve the FP32 gradient path, but match the backbone dtype at the
        # kernel boundary.
        return anchors.to(dtype=base_anchor.dtype)


def _expand_key_heads(key_chunks: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    """Expand GQA KV heads only for the reference tangent construction."""
    if key_chunks.ndim != 5:
        raise ValueError(f"key_chunks must be [B, N, S, Hkv, D], got {tuple(key_chunks.shape)}")
    h_kv = key_chunks.shape[3]
    if num_query_heads % h_kv != 0:
        raise ValueError(f"Hq ({num_query_heads}) must be divisible by Hkv ({h_kv})")
    groups = num_query_heads // h_kv
    return key_chunks if groups == 1 else key_chunks.repeat_interleave(groups, dim=3)


def build_mass_tangent_summary(
    base_anchor: torch.Tensor,
    key_chunks: torch.Tensor,
    anchor_generator: MultiTangentAnchorGenerator,
    sm_scale: Optional[float] = None,
) -> MassTangentSummary:
    """Pool a separate ``(mu[c,m], b[c,m])`` for every chunk and anchor.

    The final token is excluded, matching HiLS's existing Taylor pooling
    convention.  All probability and entropy calculations run in FP32;
    ``mu`` is returned in the input-key dtype and ``b`` in FP32.
    """
    if base_anchor.shape[:2] != key_chunks.shape[:2] or base_anchor.shape[-1] != key_chunks.shape[-1]:
        raise ValueError(
            "base_anchor and key_chunks must agree on (B, N, D): "
            f"got {tuple(base_anchor.shape)} and {tuple(key_chunks.shape)}"
        )
    if key_chunks.shape[2] < 2:
        raise ValueError("MASS chunk pooling needs chunk_size >= 2 because the last token is masked")

    anchors = anchor_generator(base_anchor)
    _, _, num_anchors, h_q, D = anchors.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    key_q = _expand_key_heads(key_chunks, h_q)
    logits = torch.einsum("bnmhd,bnshd->bnmsh", anchors.float(), key_q.float()) * sm_scale
    logits[:, :, :, -1, :] = float("-inf")
    probs = F.softmax(logits, dim=3)
    mu = torch.einsum("bnmsh,bnshd->bnmhd", probs, key_q.float()).to(key_chunks.dtype)

    log_probs = F.log_softmax(logits, dim=3)
    entropy = -(probs * torch.where(torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs))).sum(dim=3)
    return MassTangentSummary(mu=mu, b=entropy, anchors=anchors)


def upper_envelope_scores(
    queries: torch.Tensor,
    tangent_mu: torch.Tensor,
    tangent_b: torch.Tensor,
    sm_scale: Optional[float] = None,
    *,
    score_reduction: str = "max",
    lse_temperature: float = 1.0,
    normalize_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate ``max_m(q·mu[c,m]/sqrt(D) + b[c,m])``.

    Returns the envelope scores ``[B, L, N, Hq]`` and the winning anchor
    index with the same shape.  Keeping this separate makes expert usage
    available for a later load-balancing loss without changing selection.
    """
    if queries.ndim != 4 or tangent_mu.ndim != 5 or tangent_b.ndim != 4:
        raise ValueError("expected queries [B,L,H,D], mu [B,N,M,H,D], b [B,N,M,H]")
    B, L, Hq, D = queries.shape
    if tangent_mu.shape != (B, tangent_mu.shape[1], tangent_b.shape[2], Hq, D):
        raise ValueError(
            f"incompatible query/mu shapes: queries={tuple(queries.shape)}, mu={tuple(tangent_mu.shape)}, "
            f"b={tuple(tangent_b.shape)}"
        )
    if tangent_b.shape[:3] != tangent_mu.shape[:3] or tangent_b.shape[3] != Hq:
        raise ValueError(f"tangent_b must match mu's [B,N,M,H], got {tuple(tangent_b.shape)}")
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    if score_reduction not in {"max", "logsumexp"}:
        raise ValueError(f"unsupported MASS score reduction: {score_reduction!r}")
    if score_reduction == "logsumexp" and lse_temperature <= 0:
        raise ValueError("MASS LogSumExp temperature must be positive")

    planes = torch.einsum("blhd,bnmhd->blnmh", queries.float(), tangent_mu.float()) * sm_scale
    planes = planes + tangent_b.float()[:, None]
    winners = planes.argmax(dim=3)
    if score_reduction == "logsumexp":
        scores = lse_temperature * torch.logsumexp(planes / lse_temperature, dim=3)
        if normalize_lse:
            scores = scores - lse_temperature * math.log(tangent_mu.shape[2])
    else:
        scores = planes.max(dim=3).values
    return scores, winners


def mass_topk_reference(
    envelope_scores: torch.Tensor,
    winning_anchor: torch.Tensor,
    *,
    num_key_value_heads: int,
    topk: int,
    chunk_size: int,
    window_size: int,
    is_causal: bool = True,
    q_offset: int = 0,
) -> MassTopKResult:
    """Reference top-k selection from MASS envelope scores.

    One candidate list is produced per KV head by taking the max across its
    GQA query-head group.  This preserves the shape expected by the existing
    sparse-value attention call while leaving its operator untouched.
    """
    if envelope_scores.ndim != 4 or winning_anchor.shape != envelope_scores.shape:
        raise ValueError("envelope_scores and winning_anchor must both be [B, L, N, Hq]")
    if topk < 1 or chunk_size < 1:
        raise ValueError("topk and chunk_size must be positive")

    B, L, N, Hq = envelope_scores.shape
    if Hq % num_key_value_heads != 0:
        raise ValueError(f"Hq ({Hq}) must be divisible by Hkv ({num_key_value_heads})")
    groups = Hq // num_key_value_heads
    scores = envelope_scores.float()

    if is_causal:
        query_positions = torch.arange(L, device=scores.device) + q_offset
        chunk_positions = torch.arange(N, device=scores.device)
        if window_size > 0:
            first_retrievable = torch.div(query_positions - window_size + 1, chunk_size, rounding_mode="floor")
        else:
            first_retrievable = torch.div(query_positions, chunk_size, rounding_mode="floor")
        masked = chunk_positions[None, :] >= first_retrievable[:, None]
        scores = scores.masked_fill(masked[None, :, :, None], float("-inf"))

    grouped = scores.view(B, L, N, num_key_value_heads, groups).amax(dim=-1).permute(0, 1, 3, 2)
    actual_topk = min(topk, N)
    values, indices = torch.topk(grouped, k=actual_topk, dim=-1, sorted=False)
    indices = indices.masked_fill(values.eq(float("-inf")), -1)
    if actual_topk < topk:
        indices = torch.cat(
            [indices, torch.full((*indices.shape[:-1], topk - actual_topk), -1, device=indices.device, dtype=indices.dtype)],
            dim=-1,
        )

    # Existing HiLS attention consumes ascending chunk indices.  Per-query-head
    # scores are gathered from the MASS envelope for the selected KV candidates.
    sort_key = indices.masked_fill(indices.lt(0), N)
    indices = torch.sort(sort_key, dim=-1).values.masked_fill(sort_key.sort(dim=-1).values.eq(N), -1)
    safe_indices = indices.clamp_min(0)
    per_head = scores.permute(0, 1, 3, 2)
    head_indices = safe_indices.repeat_interleave(groups, dim=2)
    selected_scores = torch.gather(per_head, dim=-1, index=head_indices)
    selected_scores = selected_scores.masked_fill(head_indices.eq(0) & indices.repeat_interleave(groups, dim=2).eq(-1), float("-inf"))

    return MassTopKResult(
        indices=indices,
        selected_scores=selected_scores,
        envelope_scores=scores,
        winning_anchor=winning_anchor,
    )
