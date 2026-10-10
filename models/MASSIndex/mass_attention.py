"""A separate training-time attention layer for the MASS-Index MVP.

The class subclasses ``HiLSAttention`` only to reuse its projections, RoPE,
sliding-window attention and sparse value-attention call.  It does *not*
change that class or any kernel.  The replaced part is entirely in this file:
    single-tangent landmark pooling + HiLS Top-K becomes MASS tangent pooling +
    reference upper-envelope Top-K.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

from models.FlashHiLS.hils_attention import HiLSAttention
from ops.flex_attn_tilelang import flex_attn_tl
from ops.mass_retrieval_tilelang_v2 import mass_retrieval_tilelang_v2

from .mass_index import (
    MASSIndexConfig,
    MassTopKResult,
    MultiTangentAnchorGenerator,
    build_mass_tangent_summary,
)


class MASSAttention(HiLSAttention):
    """Drop-in HiLS-style attention with independent MASS retrieval logic.

    This first version supports full-sequence training/prefill only.  KV-cache
    continuation is intentionally deferred until the M tangent summaries have
    an explicit cache format.  ``mass_config`` is passed programmatically, so
    no existing JSON configuration needs new fields.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        norm_cls=None,
        mass_config: Optional[MASSIndexConfig] = None,
    ):
        super().__init__(config, layer_idx, norm_cls=norm_cls)
        self.mass_config = mass_config or MASSIndexConfig()
        self.mass_anchor_generator = MultiTangentAnchorGenerator(
            self.h_q, self.head_dim, self.mass_config
        )
        self.last_mass_balance_loss = None
        self.last_mass_anchor_load = None
        self.last_mass_conditional_entropy = None
        self.last_mass_marginal_entropy = None
        # MASS_SPEC_DIVERSE_MU: functional-diversity loss on tangent summaries
        # (replaces the routing-only MI objective; aggregated in
        # modeling_qwen_mass.MASSForCausalLM.forward).
        self.last_mass_diversity_loss = None
        self._mass_diverse_mu = None  # lazy env cache
        # MASS_NEEDLE_INFO_NCE: per-forward context is set/cleared by
        # MASSForCausalLM.forward; loss terms are collected there.
        self._mass_needle_ctx = None
        self._mass_nci_terms = []

    def build_mass_retrieval_plan(
        self,
        lmk_q_norm: torch.Tensor,
        lmk_k_source: torch.Tensor,
        *,
        q_offset: int = 0,
    ) -> MassTopKResult:
        """Create MASS Top-K candidates without an L x N x M workspace.

        Exposed separately to make the mathematical retrieval path testable
        without invoking either of the existing fused attention operators.
        """
        B, full_seq_len, _, _ = lmk_k_source.shape
        full_chunks = full_seq_len // self.chunk_size
        if full_chunks == 0:
            raise ValueError("MASS retrieval requires at least one complete chunk")

        key_chunks = lmk_k_source[:, : full_chunks * self.chunk_size]
        key_chunks = rearrange(key_chunks, "b (n s) h d -> b n s h d", s=self.chunk_size)
        base_anchor = lmk_q_norm[:, self.chunk_size - 1 :: self.chunk_size, :, :][:, :full_chunks]
        anchors = self.mass_anchor_generator(base_anchor).to(torch.bfloat16).contiguous()
        shared_key = None
        shared_bias = getattr(self.mass_anchor_generator, "shared_view_bias", None)
        if shared_bias is not None:
            # HiLS landmark key (paper's compressed chunk key) as shared view.
            shared_key = (
                lmk_k_source[:, self.chunk_size - 1 :: self.chunk_size, :, :][:, :full_chunks]
                .to(torch.bfloat16)
                .contiguous()
            )
        indices, selected_scores, anchor_load, conditional_entropy = mass_retrieval_tilelang_v2(
            lmk_q_norm.to(torch.bfloat16).contiguous(),
            key_chunks.to(torch.bfloat16).contiguous(),
            anchors,
            num_key_value_heads=self.h_kv,
            topk=self.topk,
            chunk_size=self.chunk_size,
            window_size=int(self.hils_sliding_window or 0),
            q_offset=q_offset,
            sm_scale=self.scaling,
            score_reduction=self.mass_config.score_reduction,
            lse_temperature=self.mass_config.lse_temperature,
            two_stage_backward=self.mass_config.two_stage_backward,
            shared_key=shared_key,
            shared_bias=shared_bias,
        )
        num_views = self.mass_config.num_anchors  # replace mode: count unchanged
        # Normalized LSE has the same router/anchor gradients as ordinary LSE
        # and leaves chunk Top-K unchanged.  Removing tau*log(M) only avoids a
        # fixed M-dependent bias when MASS scores are mixed with the SWA LSE.
        if self.mass_config.score_reduction == "logsumexp" and self.mass_config.normalize_lse:
            selected_scores = selected_scores - (
                self.mass_config.lse_temperature * math.log(num_views)
            )
        self._maybe_aux_losses(lmk_q_norm, base_anchor, key_chunks, full_chunks)
        return MassTopKResult(
            indices,
            selected_scores,
            None,
            None,
            anchor_load,
            conditional_entropy,
        )

    def _maybe_aux_losses(self, lmk_q_norm, base_anchor, key_chunks, full_chunks):
        """Env-gated auxiliary objectives computed on the reference scoring path.

        Both are no-ops unless their env flag is set and autograd is enabled
        (eval / no-grad forwards never pay the cost or touch state).

        * ``MASS_SPEC_DIVERSE_MU=1`` — functional diversity: push the two
          anchors' per-chunk pooled summaries towards orthogonal views
          (cos^2 -> 0) instead of only differentiating the router.
        * ``MASS_NEEDLE_INFO_NCE`` — context set by MASSForCausalLM.forward on
          RULER samples; multi-positive InfoNCE pulling the needle chunk(s)
          above the other window-eligible remote chunks at question rows.
        """
        if not torch.is_grad_enabled():
            return
        if self._mass_diverse_mu is None:
            self._mass_diverse_mu = (
                os.environ.get("MASS_SPEC_DIVERSE_MU", "0") == "1"
            )
        ctx = self._mass_needle_ctx
        if not self._mass_diverse_mu and ctx is None:
            return
        summary = build_mass_tangent_summary(
            base_anchor,
            key_chunks,
            self.mass_anchor_generator,
            sm_scale=self.scaling,
        )
        if self._mass_diverse_mu:
            mu = summary.mu.float()  # [B, N, M, Hq, D]
            if mu.shape[2] >= 2:
                cos = F.cosine_similarity(mu[:, :, 0], mu[:, :, 1], dim=-1)
                self.last_mass_diversity_loss = (cos * cos).mean()
        if ctx is not None:
            rows = ctx["rows_exp"]  # [R] expanded positions
            rows = rows[rows < lmk_q_norm.shape[1]]
            if rows.numel() > 0:
                shared_bias = getattr(self.mass_anchor_generator, "shared_view_bias", None)
                shared_mu = None
                if shared_bias is not None:
                    groups = self.h_q // self.h_kv
                    shared_mu = key_chunks[:, :, -1, :, :].repeat_interleave(groups, dim=2)
                self._mass_nci_terms.append(
                    self._needle_info_nce(
                        lmk_q_norm, summary, ctx, rows, full_chunks, shared_mu, shared_bias
                    )
                )

    def _needle_info_nce(
        self, lmk_q_norm, summary, ctx, rows, full_chunks, shared_mu=None, shared_bias=None
    ):
        """Multi-positive InfoNCE on the reference upper-envelope scores.

        Mirrors the kernel math exactly: planes = q.mu*sm_scale + b, envelope
        over anchors with the *current* training temperature and the same
        normalize_lse constant, eligibility n < (row - window + 1) / chunk_size.
        """
        nci_tau = float(os.environ.get("MASS_NCI_TAU", "1.0"))
        planes = torch.einsum(
            "brhd,bnmhd->brnmh", lmk_q_norm[:, rows, :, :].float(), summary.mu.float()
        ) * self.scaling
        planes = planes + summary.b.float()[:, None]
        if shared_mu is not None and shared_bias is not None:
            # Replace mode, mirroring the kernel-side swap (view 0 = shared).
            shared_plane = (
                torch.einsum("brhd,bnhd->brnh", lmk_q_norm[:, rows, :, :].float(), shared_mu.float())
                * self.scaling
                + shared_bias.float()
            )
            planes = torch.cat([shared_plane.unsqueeze(3), planes[..., 1:, :]], dim=3)
        num_views = planes.shape[3]
        if self.mass_config.score_reduction == "logsumexp":
            scores = self.mass_config.lse_temperature * torch.logsumexp(
                planes / self.mass_config.lse_temperature, dim=3
            )
            if self.mass_config.normalize_lse:
                scores = scores - self.mass_config.lse_temperature * math.log(num_views)
        else:
            scores = planes.max(dim=3).values
        # scores: [B, R, N, Hq]
        chunk_ids = torch.arange(full_chunks, device=scores.device)
        window = int(self.hils_sliding_window or 0)
        if window > 0:
            limits = (rows.to(torch.long) - window + 1) // self.chunk_size
        else:
            limits = rows.to(torch.long) // self.chunk_size
        eligible = chunk_ids[None, :] < limits.clamp_min(0)[:, None]  # [R, N]
        scores = scores.masked_fill(~eligible[None, :, :, None], float("-inf"))
        needle_ids = ctx["needle_chunks"]  # [B, K] long, -1 padded
        needle_mask = torch.zeros(
            scores.shape[0], full_chunks, dtype=torch.bool, device=scores.device
        )
        valid = (needle_ids >= 0) & (needle_ids < full_chunks)
        needle_mask.scatter_(1, needle_ids.clamp_min(0), valid)
        pos = scores.masked_fill(~needle_mask[:, None, :, None], float("-inf"))
        denom = torch.logsumexp(scores, dim=2)  # [B, R, Hq]
        numer = torch.logsumexp(pos, dim=2)
        # -log p(needle) (>= 0): minimizing the total loss MAXIMIZES the
        # needle's softmax mass.  (numer - denom would invert the gradient
        # and actively push the needle out of the top-k.)
        per_row = denom - numer  # [B, R, Hq]
        per_row = per_row.masked_fill(~torch.isfinite(per_row), 0.0)
        has = ctx["sample_has_needle"].float()  # [B]
        loss = (per_row.mean(dim=(1, 2)) * has).sum() / has.sum().clamp_min(1.0)
        return loss

    def _reference_sparse_value_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Small-sequence PyTorch reference for the unchanged sparse-V kernel.

        This is deliberately opt-in via ``MASS_REFERENCE_SPARSE=1``.  It is a
        correctness/smoke-test fallback when a locally installed TileLang build
        is not ABI-compatible with the GPU.  It materializes an L x L mask and
        must not be used for long-context training.
        """
        B, L, h_q, _ = q.shape
        max_seq_len = int(os.environ.get("MASS_REFERENCE_MAX_SEQ_LEN", "512"))
        if L > max_seq_len:
            raise RuntimeError(
                f"MASS reference sparse attention is limited to L <= {max_seq_len}; "
                "install a compatible TileLang sparse kernel for long-context training."
            )
        if h_q % self.h_kv:
            raise ValueError("MASS query-head count must be divisible by KV-head count")

        groups = h_q // self.h_kv
        selected_chunks = indices.to(torch.long).repeat_interleave(groups, dim=2)
        key_chunk = torch.arange(L, device=q.device) // self.chunk_size
        selected = (selected_chunks.unsqueeze(-1) == key_chunk.view(1, 1, 1, 1, L)).any(dim=3)
        query_pos = torch.arange(L, device=q.device)
        causal = torch.arange(L, device=q.device).view(1, L) <= query_pos.view(L, 1)
        valid = selected & causal.view(1, L, 1, L)
        has_selected = valid.any(dim=-1, keepdim=True)
        # Avoid a NaN softmax on early queries that have no eligible completed
        # chunk.  Their output is zeroed immediately after attention.
        valid[..., 0] |= ~has_selected.squeeze(-1)

        q_heads = q.transpose(1, 2).float()
        k_heads = k.repeat_interleave(groups, dim=2).transpose(1, 2).float()
        v_heads = v.repeat_interleave(groups, dim=2).transpose(1, 2).float()
        logits = torch.matmul(q_heads, k_heads.transpose(-1, -2)) * self.scaling
        logits = logits.masked_fill(~valid.transpose(1, 2), torch.finfo(logits.dtype).min)
        probs = F.softmax(logits, dim=-1)
        output = torch.matmul(probs, v_heads).transpose(1, 2)
        return (output * has_selected.to(output.dtype)).to(q.dtype)

    def _forward_with_cache(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        """KV-cache continuation, mirroring HiLSAttention's inference strategy.

        The cache keeps the full K/V history (landmark-layout K rows included
        whenever the caller inserted landmarks; raw streams pool by position).
        Per step: SWA runs over the full cache, the tangent summaries ``mu`` are
        recomputed from the full K (cheap for realistic chunk counts), the
        per-chunk anchor base rows (``lmk_q`` at chunk ends) come from
        ``past_key_values._mass_prior_cache`` because query projections of past
        positions cannot be recovered from K, and the sparse value attention
        consumes the full cache with the newly selected chunk indices.
        """
        B, L, _ = hidden_states.shape
        cos, sin = position_embeddings
        self.last_mass_balance_loss = None
        self.last_mass_anchor_load = None
        self.last_mass_conditional_entropy = None
        self.last_mass_marginal_entropy = None
        self.last_mass_diversity_loss = None

        hils_q_nope, hils_k_nope, hils_v, lmk_q = self._project_qkv(hidden_states)
        kernel_dtype = torch.bfloat16
        hils_q_nope = hils_q_nope.to(kernel_dtype)
        hils_k_nope = hils_k_nope.to(kernel_dtype)
        hils_v = hils_v.to(kernel_dtype)
        lmk_q = lmk_q.to(kernel_dtype)
        hils_q_rope, hils_k_rope, lmk_q = self._apply_hils_rope(
            hils_q_nope, hils_k_nope, lmk_q, cos, sin
        )

        hils_k_for_cache = hils_k_rope if self.apply_hils_rope else hils_k_nope
        hils_cache_idx = self.layer_idx + self.config.num_hidden_layers
        hils_k_full, hils_v_full = past_key_values.update(
            hils_k_for_cache, hils_v, hils_cache_idx, {"cache_position": cache_position}
        )
        if self.apply_hils_rope:
            hils_k_rope = hils_k_full
        else:
            hils_k_nope = hils_k_full
            hils_k_rope = hils_k_nope
        hils_v = hils_v_full
        full_seq_len = hils_k_full.shape[1]

        swa_o, lse_swa = flex_attn_tl(
            hils_q_rope,
            hils_k_rope,
            hils_v,
            window_size=self.hils_sliding_window,
            chunk_size=self.chunk_size,
            training=False,
            mask_lmk=True,
            expand_to_chunk=True,
        )
        lse_swa = lse_swa.contiguous().to(hidden_states.dtype)

        if full_seq_len < self.chunk_size:
            swa_weight = (
                torch.sigmoid(lse_swa).to(swa_o.dtype) if self.enable_softmax1 else 1.0
            )
            output = swa_o * swa_weight.unsqueeze(-1) if self.enable_softmax1 else swa_o
            output = rearrange(output, "b l h d -> b l (h d)")
            return self.o_proj(output.to(self.o_proj.weight.dtype)), None

        if L == full_seq_len:
            # Prompt prefill with cache enabled: identical math to the training
            # path, then stash the chunk-end anchor base rows for later steps.
            outputs = self._mass_prefill_core(
                hidden_states,
                hils_q_rope=hils_q_rope,
                hils_k_rope=hils_k_rope,
                hils_q_nope=hils_q_nope,
                hils_k_nope=hils_k_nope,
                hils_v=hils_v,
                lmk_q=lmk_q,
                swa_o=swa_o,
                lse_swa=lse_swa,
                is_training=False,
            )
            self._update_prior_anchor_cache(past_key_values, full_seq_len, lmk_q)
            return outputs

        # ---- decode continuation (L < full_seq_len) ----
        key_source = hils_k_rope if self.apply_hils_rope else hils_k_nope
        self._update_prior_anchor_cache(past_key_values, full_seq_len, lmk_q)
        prior_cache = past_key_values._mass_prior_cache[self.layer_idx]
        indices, scores = self._decode_retrieval(key_source, lmk_q, full_seq_len, L, prior_cache)

        if lse_swa.shape[2] != scores.shape[2]:
            if scores.shape[2] % lse_swa.shape[2] != 0:
                raise ValueError("SWA LSE head count is incompatible with MASS query heads")
            lse_swa = lse_swa.repeat_interleave(scores.shape[2] // lse_swa.shape[2], dim=2)
        if self.enable_softmax1:
            normalizer = torch.zeros(
                B, L, scores.shape[2], 1, device=scores.device, dtype=scores.dtype
            )
            mixture_logits = torch.cat(
                [scores, lse_swa.unsqueeze(-1).to(scores.dtype), normalizer], dim=-1
            )
            swa_weight_index = -2
        else:
            mixture_logits = torch.cat([scores, lse_swa.unsqueeze(-1).to(scores.dtype)], dim=-1)
            swa_weight_index = -1
        weights = F.softmax(mixture_logits, dim=-1).to(hidden_states.dtype)

        hils_q = hils_q_rope if self.apply_hils_rope else hils_q_nope
        # Mirror the parent HiLS decode call: default block_M/num_threads. The
        # training-tuned explicit block_M layout does not compile at L=1.
        hils_o = self.hils_func(
            hils_q,
            hils_k_full,
            hils_v,
            weights=weights,
            indices=indices.to(torch.int32).contiguous(),
            block_size=self.chunk_size,
            mask_last_token=True,
            is_training=False,
        )
        output = torch.addcmul(
            hils_o, swa_o, weights[:, :, :, swa_weight_index].unsqueeze(-1)
        )
        output = rearrange(output, "b l h d -> b l (h d)")
        return self.o_proj(output.to(self.o_proj.weight.dtype)), None

    def _update_prior_anchor_cache(self, past_key_values, full_seq_len, lmk_q):
        """Append the anchor base rows (chunk-end ``lmk_q``) of newly completed chunks.

        A chunk completes exactly when its end row is part of the current input,
        so the rows are always extractable from ``lmk_q`` of this forward.
        """
        if not hasattr(past_key_values, "_mass_prior_cache"):
            past_key_values._mass_prior_cache = {}
        cache = past_key_values._mass_prior_cache.get(self.layer_idx, None)
        cached_chunks = 0 if cache is None else cache["base_anchor"].shape[1]
        full_chunks = full_seq_len // self.chunk_size
        if full_chunks <= cached_chunks:
            return
        end_positions = torch.arange(
            cached_chunks * self.chunk_size + self.chunk_size - 1,
            full_chunks * self.chunk_size,
            self.chunk_size,
            device=lmk_q.device,
        )
        new_rows = lmk_q[:, end_positions - (full_seq_len - lmk_q.shape[1]), :, :]
        if cache is not None:
            new_rows = torch.cat([cache["base_anchor"], new_rows.to(cache["base_anchor"].dtype)], dim=1)
            cache["base_anchor"] = new_rows.to(torch.bfloat16).contiguous()
        else:
            past_key_values._mass_prior_cache[self.layer_idx] = {
                "base_anchor": new_rows.to(torch.bfloat16).contiguous(),
            }

    def _decode_retrieval(self, key_source, lmk_q, full_seq_len, L, prior_cache):
        """Reference (PyTorch) upper-envelope Top-K for cache continuation.

        Mirrors the kernel math exactly as documented in ``_needle_info_nce``:
        planes = q.mu*sm_scale + b, view-0 replaced by the shared landmark-key
        plane, envelope over views with the current temperature, window
        eligibility ``n < (row - window + 1) // chunk_size``. Selection is per
        KV-head group (mean over its query heads); scores are gathered per
        query head at its group's selected chunks.
        """
        full_chunks = full_seq_len // self.chunk_size
        base_anchor = prior_cache["base_anchor"]
        if base_anchor.shape[1] < full_chunks:
            raise RuntimeError(
                "MASS decode continuation requires the prior anchor cache; "
                "the prompt forward must run with use_cache=True."
            )
        base_anchor = base_anchor[:, :full_chunks]

        # mu/b of a completed chunk are immutable — recompute only the newly
        # completed chunks and reuse the cached rest (per decode step this
        # drops the N-chunk tangent pooling to an append every chunk_size
        # steps plus a small q.mu einsum).
        cached_summary = prior_cache.get("tangent_summary", None)
        if cached_summary is not None and cached_summary.mu.shape[1] == full_chunks:
            summary = cached_summary
        else:
            key_chunks = key_source[:, : full_chunks * self.chunk_size]
            key_chunks = rearrange(key_chunks, "b (n s) h d -> b n s h d", s=self.chunk_size)
            cached_n = 0
            if cached_summary is not None:
                cached_n = min(cached_summary.mu.shape[1], full_chunks)
            if cached_n < full_chunks:
                new_k = key_chunks[:, cached_n:]
                new_a = base_anchor[:, cached_n:]
                from .mass_index import MassTangentSummary
                new_s = build_mass_tangent_summary(
                    new_a, new_k, self.mass_anchor_generator, sm_scale=self.scaling
                )
                if cached_n:
                    summary = MassTangentSummary(
                        mu=torch.cat([cached_summary.mu, new_s.mu], dim=1),
                        b=torch.cat([cached_summary.b, new_s.b], dim=1),
                        anchors=torch.cat([cached_summary.anchors, new_s.anchors], dim=1),
                    )
                else:
                    summary = new_s
            else:
                summary = cached_summary
            prior_cache["tangent_summary"] = summary
        shared_bias = getattr(self.mass_anchor_generator, "shared_view_bias", None)
        groups = lmk_q.shape[2] // key_source.shape[2]
        planes = torch.einsum(
            "blhd,bnmhd->blnmh", lmk_q.float(), summary.mu.float()
        ) * self.scaling + summary.b.float()[:, None]
        if shared_bias is not None:
            shared_key = key_source[:, self.chunk_size - 1 :: self.chunk_size, :, :][
                :, :full_chunks
            ].float()
            shared_plane = (
                torch.einsum("blhd,bnhd->blnh", lmk_q.float(), shared_key.repeat_interleave(groups, dim=2))
                * self.scaling
                + shared_bias.float()
            )
            planes = torch.cat([shared_plane.unsqueeze(3), planes[..., 1:, :]], dim=3)
        num_views = planes.shape[3]
        if self.mass_config.score_reduction == "logsumexp":
            scores_all = self.mass_config.lse_temperature * torch.logsumexp(
                planes / self.mass_config.lse_temperature, dim=3
            )
            if self.mass_config.normalize_lse:
                scores_all = scores_all - self.mass_config.lse_temperature * math.log(
                    num_views
                )
        else:
            scores_all = planes.max(dim=3).values
        # scores_all: [B, L, N, Hq]

        q_offset = full_seq_len - L
        rows = q_offset + torch.arange(L, device=scores_all.device)
        window = int(self.hils_sliding_window or 0)
        if window > 0:
            limits = (rows - window + 1) // self.chunk_size
        else:
            limits = rows // self.chunk_size
        eligible = (
            torch.arange(full_chunks, device=scores_all.device)[None, None, :]
            < limits.clamp_min(0)[:, None]
        )  # [1, L, N]
        scores_all = scores_all.masked_fill(~eligible[..., None], float("-inf"))

        # The sparse kernels are only ever compiled for K = self.topk (the
        # prefill path always requests that many), so keep K fixed here too:
        # top-k over the available chunks, then pad to self.topk with -inf
        # scores (zero softmax mass) and dummy chunk-0 indices.
        k_eff = min(self.topk, full_chunks)
        # Per-KV-head selection aggregated exactly like the kernel: LSE over
        # the group's query-head scores at the retrieval temperature (or max
        # in non-LSE reduction mode). See mass_retrieval_tilelang_v2's
        # group_max/exp_sum loop.
        Bq, Lq, _, Hq = scores_all.shape
        scores_grouped = scores_all.view(Bq, Lq, full_chunks, key_source.shape[2], groups)
        if self.mass_config.score_reduction == "logsumexp":
            grouped = self.mass_config.lse_temperature * torch.logsumexp(
                scores_grouped / self.mass_config.lse_temperature, dim=-1
            )
        else:
            grouped = scores_grouped.max(dim=-1).values
        grouped = grouped.permute(0, 1, 3, 2)  # [B, L, Hkv, N]
        top_scores, top_idx = grouped.topk(k_eff, dim=-1)  # [B, L, Hkv, k_eff]
        if k_eff < self.topk:
            pad = self.topk - k_eff
            top_idx = F.pad(top_idx, (0, pad), value=0)
            top_scores = F.pad(top_scores, (0, pad), value=float("-inf"))
        indices_hq = top_idx.repeat_interleave(groups, dim=2).clamp_min(0)
        gathered = torch.gather(
            scores_all.permute(0, 1, 3, 2), 3, indices_hq
        )  # [B, L, Hq, K]: gather along the chunk axis (dim 3 after permute)
        if k_eff < self.topk:
            gathered = F.pad(gathered, (0, self.topk - k_eff), value=float("-inf"))
        # Early decode rows with no eligible chunk: kernel semantics yield a
        # SWA-only mixture (sparse mass zero). Gathered -inf scores produce a
        # zero softmax weight on every sparse branch, which matches.
        return top_idx, gathered.to(torch.bfloat16)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        if use_cache and past_key_values is not None:
            return self._forward_with_cache(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        if position_embeddings is None:
            raise ValueError("MASSAttention requires precomputed RoPE position_embeddings")

        B, L, _ = hidden_states.shape
        self.last_mass_balance_loss = None
        self.last_mass_anchor_load = None
        self.last_mass_conditional_entropy = None
        self.last_mass_marginal_entropy = None
        cos, sin = position_embeddings
        hils_q_nope, hils_k_nope, hils_v, lmk_q = self._project_qkv(hidden_states)
        # This separate layer owns the dtype boundary required by the existing
        # BF16 TileLang kernels.  The original HiLS implementation is left
        # untouched; FSDP/VeOmni may otherwise keep these activations in FP32.
        kernel_dtype = torch.bfloat16
        hils_q_nope = hils_q_nope.to(kernel_dtype)
        hils_k_nope = hils_k_nope.to(kernel_dtype)
        hils_v = hils_v.to(kernel_dtype)
        lmk_q = lmk_q.to(kernel_dtype)
        hils_q_rope, hils_k_rope, lmk_q = self._apply_hils_rope(hils_q_nope, hils_k_nope, lmk_q, cos, sin)

        swa_o, lse_swa = flex_attn_tl(
            hils_q_rope,
            hils_k_rope,
            hils_v,
            window_size=self.hils_sliding_window,
            chunk_size=self.chunk_size,
            training=True,
            mask_lmk=True,
            expand_to_chunk=True,
        )
        lse_swa = lse_swa.contiguous().to(hidden_states.dtype)

        return self._mass_prefill_core(
            hidden_states,
            hils_q_rope=hils_q_rope,
            hils_k_rope=hils_k_rope,
            hils_q_nope=hils_q_nope,
            hils_k_nope=hils_k_nope,
            hils_v=hils_v,
            lmk_q=lmk_q,
            swa_o=swa_o,
            lse_swa=lse_swa,
            is_training=self.training,
        )

    def _mass_prefill_core(
        self,
        hidden_states,
        *,
        hils_q_rope,
        hils_k_rope,
        hils_q_nope,
        hils_k_nope,
        hils_v,
        lmk_q,
        swa_o,
        lse_swa,
        is_training,
    ):
        """Shared prefill math for both the training path and cache prompts."""
        B, L, _ = hidden_states.shape
        if L < self.chunk_size:
            self.last_mass_anchor_load = torch.full(
                (self.mass_config.num_anchors,),
                1.0 / self.mass_config.num_anchors,
                device=hidden_states.device,
                dtype=torch.float32,
            )
            self.last_mass_balance_loss = hidden_states.new_zeros((), dtype=torch.float32)
            self.last_mass_conditional_entropy = hidden_states.new_zeros((), dtype=torch.float32)
            self.last_mass_marginal_entropy = hidden_states.new_zeros((), dtype=torch.float32)
            swa_weight = torch.sigmoid(lse_swa).to(swa_o.dtype) if self.enable_softmax1 else 1.0
            output = swa_o * swa_weight.unsqueeze(-1) if self.enable_softmax1 else swa_o
            output = rearrange(output, "b l h d -> b l (h d)")
            return self.o_proj(output.to(self.o_proj.weight.dtype)), None

        key_source = hils_k_rope if self.apply_hils_rope else hils_k_nope
        retrieval = self.build_mass_retrieval_plan(lmk_q, key_source)
        scores = retrieval.selected_scores
        indices = retrieval.indices
        self.last_mass_anchor_load = retrieval.anchor_load
        num_anchors = self.mass_config.num_anchors
        if num_anchors > 1 and retrieval.conditional_entropy is not None:
            marginal_probs = retrieval.anchor_load.float().clamp_min(
                torch.finfo(torch.float32).tiny
            )
            marginal_entropy = -(marginal_probs * marginal_probs.log()).sum()
            log_num_anchors = math.log(num_anchors)
            # Mutual-information specialization objective, normalized to [0, 1]
            # at its two important endpoints:
            #   uniform per-route collapse -> 1
            #   confident, globally balanced routing -> 0
            specialization_loss = (
                retrieval.conditional_entropy - marginal_entropy + log_num_anchors
            ) / log_num_anchors
            self.last_mass_conditional_entropy = retrieval.conditional_entropy
            self.last_mass_marginal_entropy = marginal_entropy
            self.last_mass_balance_loss = specialization_loss
        else:
            self.last_mass_conditional_entropy = scores.new_zeros((), dtype=torch.float32)
            self.last_mass_marginal_entropy = scores.new_zeros((), dtype=torch.float32)
            self.last_mass_balance_loss = scores.new_zeros((), dtype=torch.float32)

        # ``scores`` is per query head.  The existing sparse V-attention
        # operator consumes the same [B, L, Hq, K] mixture weights and the
        # original [B, L, Hkv, K] indices, so it remains unmodified.
        if lse_swa.shape[2] != scores.shape[2]:
            if scores.shape[2] % lse_swa.shape[2] != 0:
                raise ValueError("SWA LSE head count is incompatible with MASS query heads")
            lse_swa = lse_swa.repeat_interleave(scores.shape[2] // lse_swa.shape[2], dim=2)
        if self.enable_softmax1:
            normalizer = torch.zeros(B, L, scores.shape[2], 1, device=scores.device, dtype=scores.dtype)
            mixture_logits = torch.cat([scores, lse_swa.unsqueeze(-1).to(scores.dtype), normalizer], dim=-1)
            swa_weight_index = -2
        else:
            mixture_logits = torch.cat([scores, lse_swa.unsqueeze(-1).to(scores.dtype)], dim=-1)
            swa_weight_index = -1
        weights = F.softmax(mixture_logits, dim=-1).to(hidden_states.dtype)

        hils_q = hils_q_rope if self.apply_hils_rope else hils_q_nope
        hils_k = hils_k_rope if self.apply_hils_rope else hils_k_nope
        if os.environ.get("MASS_REFERENCE_SPARSE", "0") == "1":
            hils_o = self._reference_sparse_value_attention(hils_q, hils_k, hils_v, indices)
            # The reference path is exact for one selected chunk (the smoke
            # setup) and uses the total sparse mixture mass for a multi-chunk
            # diagnostic run.  Production runs must use the fused kernel.
            hils_o = hils_o * weights[:, :, :, : indices.shape[-1]].sum(dim=-1, keepdim=True).to(hils_o.dtype)
        else:
            # PyTorch ``topk`` returns int64, while the unchanged TileLang HiLS
            # kernel takes int32 block indices.
            indices = indices.to(torch.int32).contiguous()
            # TileLang's MMA layout requires the fused GEMM row count
            # (query-heads per KV-head times ``block_M``) to be a multiple of
            # 16.  Prefer the existing HiLS-tuned 4:1 grouping when present;
            # this is the 12-Q/3-KV-head ~204M MASS configuration.
            query_heads_per_kv_head = hils_q.shape[2] // hils_k.shape[2]
            if query_heads_per_kv_head == 4:
                block_m = 8
                sparse_num_threads_fwd = 64
                sparse_num_threads_bwd = 128
            else:
                block_m = 16 // math.gcd(query_heads_per_kv_head, 16)
                sparse_num_threads_fwd = math.lcm(
                    64, 2 * query_heads_per_kv_head * block_m
                )
                # bwd 线程 64→128:该 kernel 85KB smem 限制每 SM 1 block,64 线程
                # 严重欠载;每-query 索引(merged_len~480)实测 5071→3524ms(-30%),
                # chunk 共享索引(merged_len~32)-3%,数值与 64 线程逐位一致。
                sparse_num_threads_bwd = max(128, sparse_num_threads_fwd)
            hils_o = self.hils_func(
                hils_q,
                hils_k,
                hils_v,
                weights=weights,
                indices=indices,
                block_size=self.chunk_size,
                block_M=block_m,
                num_threads_fwd=sparse_num_threads_fwd,
                num_threads_bwd=sparse_num_threads_bwd,
                mask_last_token=True,
                is_training=is_training,
            )
        output = torch.addcmul(hils_o, swa_o, weights[:, :, :, swa_weight_index].unsqueeze(-1))
        output = rearrange(output, "b l h d -> b l (h d)")
        return self.o_proj(output.to(self.o_proj.weight.dtype)), None
