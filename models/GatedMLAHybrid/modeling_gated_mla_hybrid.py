from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_gated_mla_hybrid import GatedMLAHybridConfig


def _pack_dynamic_cache(layer_caches: list) -> DynamicCache:
    """Pack per-layer legacy caches into a ``DynamicCache`` for ``generate()``.

    Recompute-mode semantics: the model re-runs the full sequence every decode
    step and never reads these slots back, so the packing only needs
    ``get_seq_length`` on layer 0 to be correct, which holds for GatedMLA
    (``compressed_kv`` is ``[B, S, R]``) and SWA/HiLS (``k`` is ``[B, H, S, D]``)
    layers. KDA layers return an opaque fla ``Cache`` object; their slot stays
    empty (never read back). Slot assignment bypasses ``DynamicLayer.update``
    on purpose: its concat assumes 4-D ``[B, H, S, D]`` tensors.
    """
    cache = DynamicCache()
    for entry in layer_caches:
        if isinstance(entry, tuple) and len(entry) == 2 and torch.is_tensor(entry[0]) and torch.is_tensor(entry[1]):
            layer = DynamicLayer()
            layer.keys = entry[0]
            layer.values = entry[1]
            layer.dtype = entry[0].dtype
            layer.device = entry[0].device
            layer.is_initialized = True
            cache.layers.append(layer)
        else:
            cache.layers.append(DynamicLayer())
    return cache


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        return hidden_states * torch.rsqrt(variance + self.eps).to(hidden_states.dtype) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    position_ids: torch.Tensor,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = query.shape[-1]
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, device=query.device).float() / dim))
    freqs = torch.einsum("bt,d->btd", position_ids.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(2)
    cos, sin = emb.cos().to(query.dtype), emb.sin().to(query.dtype)
    return query * cos + _rotate_half(query) * sin, key * cos[:, :, :1] + _rotate_half(key) * sin[:, :, :1]


def _attention_mask(
    q_len: int,
    kv_len: int,
    device: torch.device,
    window_size: int,
    sink_size: int,
) -> torch.Tensor:
    query_positions = torch.arange(kv_len - q_len, kv_len, device=device)[:, None]
    key_positions = torch.arange(kv_len, device=device)[None, :]
    causal = key_positions <= query_positions
    local = key_positions > (query_positions - window_size)
    sink = key_positions < sink_size
    return causal & (local | sink)


def _attention_bias(
    q_len: int,
    kv_len: int,
    device: torch.device,
    dtype: torch.dtype,
    window_size: int,
    sink_size: int,
    sink_logits: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build an additive causal/local mask with optional learned sink logits."""
    valid = _attention_mask(q_len, kv_len, device, window_size, sink_size)
    if key_padding_mask is not None:
        if key_padding_mask.shape[-1] != kv_len:
            raise ValueError(
                f"attention_mask has length {key_padding_mask.shape[-1]}, expected {kv_len}"
            )
        valid = valid.unsqueeze(0) & key_padding_mask.to(torch.bool)[:, None, :]
        batch_size = key_padding_mask.shape[0]
    else:
        valid = valid.view(1, q_len, kv_len)
        batch_size = 1
    bias = torch.zeros((batch_size, 1, q_len, kv_len), device=device, dtype=dtype)
    bias.masked_fill_(~valid.unsqueeze(1), torch.finfo(dtype).min)
    if sink_logits is not None and sink_size:
        if sink_logits.ndim == 1:
            bias[..., :sink_size] += sink_logits.to(dtype).view(1, 1, 1, sink_size)
        else:
            bias = bias.expand(batch_size, sink_logits.shape[0], q_len, kv_len).clone()
            bias[..., :sink_size] += sink_logits.to(dtype).view(1, -1, 1, sink_size)
    return bias


class VectorOutputGate(nn.Module):
    def __init__(self, hidden_size: int, init_bias: float):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, init_bias)

    def forward(self, hidden_states: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        return branch * torch.sigmoid(self.proj(hidden_states))


class GatedMLA(nn.Module):
    """Reference MLA path that caches compressed KV and rotary K separately."""

    def __init__(self, config: GatedMLAHybridConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.qk_nope_dim = config.qk_nope_head_dim
        self.qk_rope_dim = 0 if config.use_nope_mla else config.qk_rope_head_dim
        self.qk_dim = self.qk_nope_dim + self.qk_rope_dim
        self.v_dim = config.v_head_dim
        self.kv_rank = config.kv_lora_rank
        self.window_size = config.window_size
        self.sink_size = config.attention_sink_size if config.attention_sink else 0
        self.use_nope_mla = config.use_nope_mla
        self.rope_theta = config.rope_theta

        if config.q_lora_rank:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(config.q_lora_rank, config.norm_eps)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_dim, bias=False)
        else:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_dim, bias=False)
        self.kv_a_proj = nn.Linear(config.hidden_size, self.kv_rank + self.qk_rope_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_rank, config.norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_rank,
            self.num_heads * (self.qk_nope_dim + self.v_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_dim, config.hidden_size, bias=False)
        self.output_gate = VectorOutputGate(config.hidden_size, config.output_gate_init)
        self.sink_logits = (
            nn.Parameter(torch.zeros(self.sink_size))
            if self.sink_size
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        batch, q_len, _ = hidden_states.shape
        if hasattr(self, "q_a_proj"):
            q = self.q_b_proj(self.q_norm(self.q_a_proj(hidden_states)))
        else:
            q = self.q_proj(hidden_states)
        q = q.view(batch, q_len, self.num_heads, self.qk_dim)
        kv = self.kv_a_proj(hidden_states)
        compressed_kv, k_rope = kv.split([self.kv_rank, self.qk_rope_dim], dim=-1)
        compressed_kv = self.kv_norm(compressed_kv)
        if self.qk_rope_dim:
            k_rope = k_rope.view(batch, q_len, 1, self.qk_rope_dim)
        else:
            k_rope = hidden_states.new_zeros(batch, q_len, 1, 0)

        past_len = 0 if past_key_value is None else past_key_value[0].shape[1]
        if past_key_value is not None:
            compressed_kv = torch.cat((past_key_value[0], compressed_kv), dim=1)
            k_rope = torch.cat((past_key_value[1], k_rope), dim=1)
        kv_len = compressed_kv.shape[1]

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + q_len, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(batch, -1)
        q_nope, q_rope = q.split([self.qk_nope_dim, self.qk_rope_dim], dim=-1)
        if self.qk_rope_dim:
            q_rope, _ = _apply_rope(
                q_rope,
                torch.zeros_like(q_rope),
                position_ids,
                self.rope_theta,
            )
        q = torch.cat((q_nope, q_rope), dim=-1)

        expanded = self.kv_b_proj(compressed_kv).view(batch, kv_len, self.num_heads, -1)
        k_nope, value = expanded.split([self.qk_nope_dim, self.v_dim], dim=-1)
        if self.qk_rope_dim:
            _, k_rope_rotated = _apply_rope(
                q_rope.new_zeros(batch, kv_len, self.num_heads, self.qk_rope_dim),
                k_rope,
                torch.arange(kv_len, device=hidden_states.device).unsqueeze(0).expand(batch, -1),
                self.rope_theta,
            )
            k_rope_for_attention = k_rope_rotated.expand(-1, -1, self.num_heads, -1)
        else:
            # nope path: k_rope has 1 head and an empty last dim; expand to num_heads so it
            # can be concatenated with k_nope (b, kv_len, num_heads, qk_nope_dim)
            k_rope_for_attention = k_rope.expand(-1, -1, self.num_heads, -1)
        key = torch.cat((k_nope, k_rope_for_attention), dim=-1).transpose(1, 2)
        query = q.transpose(1, 2)
        value = value.transpose(1, 2)

        mask = _attention_bias(
            q_len,
            kv_len,
            hidden_states.device,
            query.dtype,
            self.window_size,
            self.sink_size,
            self.sink_logits,
            attention_mask,
        )
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).reshape(batch, q_len, -1)
        output = self.o_proj(output)
        output = self.output_gate(hidden_states, output)
        return output, (compressed_kv, k_rope) if use_cache else None


class SlidingWindowAttention(nn.Module):
    def __init__(self, config: GatedMLAHybridConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.window_size = config.window_size
        self.sink_size = config.attention_sink_size if config.attention_sink else 0
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.output_gate = VectorOutputGate(config.hidden_size, config.output_gate_init)
        self.sink_logits = (
            nn.Parameter(torch.zeros(self.sink_size))
            if self.sink_size
            else None
        )

    def forward(
        self,
        hidden_states,
        use_cache=False,
        past_key_value=None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(hidden_states).view(batch, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(hidden_states).view(batch, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        if past_key_value is None:
            k, v = k_new, v_new
        else:
            if len(past_key_value) != 2:
                raise ValueError("SlidingWindowAttention cache must be a (key, value) tuple")
            k = torch.cat((past_key_value[0], k_new), dim=2)
            v = torch.cat((past_key_value[1], v_new), dim=2)
        mask = _attention_bias(
            q_len,
            k.shape[2],
            hidden_states.device,
            q.dtype,
            self.window_size,
            self.sink_size,
            self.sink_logits,
            attention_mask,
        )
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        output = output.transpose(1, 2).reshape(batch, q_len, -1)
        output = self.output_gate(hidden_states, self.o_proj(output))
        return output, (k, v) if use_cache else None


class HiLSAttention(SlidingWindowAttention):
    """Reference local/global path; optimized HiLS kernels can replace this module."""

    def forward(self, hidden_states, use_cache=False, past_key_value=None, **kwargs):
        output, cache = super().forward(hidden_states, use_cache, past_key_value, **kwargs)
        global_context = hidden_states.mean(dim=1, keepdim=True)
        return output + 0.1 * global_context, cache


class KDAAttention(nn.Module):
    def __init__(self, config: GatedMLAHybridConfig, layer_idx: int):
        super().__init__()
        try:
            from fla.layers.kda import KimiDeltaAttention
        except ImportError as exc:
            raise ImportError(
                "The KDA baseline requires flash-linear-attention with KDA kernels installed."
            ) from exc
        self.attn = KimiDeltaAttention(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_heads=config.num_attention_heads,
            mode="chunk",
            use_short_conv=True,
            conv_size=4,
            layer_idx=layer_idx,
            norm_eps=config.norm_eps,
        )
        self.output_gate = VectorOutputGate(config.hidden_size, config.output_gate_init)

    def forward(self, hidden_states, past_key_value=None, use_cache=False, **kwargs):
        if use_cache:
            from fla.models.utils import Cache

            if past_key_value is None:
                past_key_value = Cache()
            elif not isinstance(past_key_value, Cache):
                past_key_value = Cache.from_legacy_cache(past_key_value)
        else:
            past_key_value = None
        output, _, cache = self.attn(
            hidden_states=hidden_states,
            past_key_values=past_key_value,
            use_cache=use_cache,
            **kwargs,
        )
        return self.output_gate(hidden_states, output), cache


class SwiGLU(nn.Module):
    def __init__(self, config: GatedMLAHybridConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    @classmethod
    def _with_intermediate(cls, config: GatedMLAHybridConfig, intermediate_size: int) -> "SwiGLU":
        block = cls.__new__(cls)
        nn.Module.__init__(block)
        block.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        block.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        block.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        return block

    def forward(self, hidden_states):
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class MoEFFN(nn.Module):
    """Tier-1 eager MoE FFN (single-GPU reference, no EP/TP).

    Top-k softmax routing with per-expert token loops; expert width defaults to
    intermediate_size // top_k so ACTIVE parameters match the dense SwiGLU
    counterpart. Stores the switch-style load-balance loss of the last forward
    on ``self.last_balance_loss``; the CausalLM head adds the weighted sum to
    the CE loss during training (the multi-loss pipeline carries it through).
    """

    def __init__(self, config: GatedMLAHybridConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        d, e = config.hidden_size, config.moe_intermediate_size
        self.gate = nn.Linear(d, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU._with_intermediate(config, e) for _ in range(self.num_experts)]
        )
        self.shared_expert = (
            SwiGLU._with_intermediate(config, config.shared_expert_intermediate_size)
            if config.shared_expert_intermediate_size > 0
            else None
        )
        self.last_balance_loss = None

    def forward(self, hidden_states):
        bsz, seq, d = hidden_states.shape
        x = hidden_states.reshape(-1, d)
        router_logits = self.gate(x)                                # [T, E]
        probs = torch.softmax(router_logits.float(), dim=-1)
        topk_p, topk_i = probs.topk(self.top_k, dim=-1)
        topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True)          # renormalized
        output = torch.zeros_like(x)
        for expert_idx, expert in enumerate(self.experts):
            token_mask = (topk_i == expert_idx).any(dim=-1)         # [T]
            if not token_mask.any():
                continue
            xe = x[token_mask]
            we = (topk_p * (topk_i == expert_idx)).sum(dim=-1)[token_mask].unsqueeze(-1).to(x.dtype)
            output[token_mask] = output[token_mask] + we * expert(xe)
        if self.shared_expert is not None:
            output = output + self.shared_expert(x)
        # switch load-balance loss: E * sum_f (assign_frac_f * mean_prob_f)
        with torch.autocast(device_type="cuda", enabled=x.is_cuda):
            assign_frac = torch.zeros_like(probs[0]).scatter_add(
                0, topk_i.reshape(-1), torch.ones_like(topk_i.reshape(-1), dtype=probs.dtype)
            ) / topk_i.numel()
        self.last_balance_loss = self.num_experts * (assign_frac * probs.mean(dim=0)).sum()
        return output.reshape(bsz, seq, d)


class IdentityHyperConnection(nn.Module):
    """Per-sublayer iHC coefficient generator (DeepSeek-V4 style, B fixed to I).

    Given the stream state X [B, S, n, d], produces the dynamic read weights
    A [B, S, n] in (0, 1) and write weights C [B, S, n] in (0, 2) via
    X_hat = RMSNorm(flatten(X)) -> small linear -> sigmoid(alpha * . + bias).
    Biases initialize A to 1/n and C to 1 and alpha starts near zero, so the
    network is mathematically the standard-residual transformer at init.
    """

    def __init__(self, config: GatedMLAHybridConfig):
        super().__init__()
        n, d = config.ihc_num_streams, config.hidden_size
        self.n = n
        self.d = d
        self.pre_proj = nn.Linear(n * d, n, bias=False)
        self.post_proj = nn.Linear(n * d, n, bias=False)
        self.pre_bias = nn.Parameter(torch.full((n,), math.log((1.0 / n) / (1.0 - 1.0 / n))))
        self.post_bias = nn.Parameter(torch.zeros(n))
        self.alpha_pre = nn.Parameter(torch.tensor(float(config.ihc_alpha_init)))
        self.alpha_post = nn.Parameter(torch.tensor(float(config.ihc_alpha_init)))

    def forward(self, X: torch.Tensor):
        bsz, seq = X.shape[0], X.shape[1]
        x_hat = F.rms_norm(X.reshape(bsz, seq, -1), (self.n * self.d,))
        A = torch.sigmoid(self.alpha_pre * self.pre_proj(x_hat) + self.pre_bias)
        C = 2.0 * torch.sigmoid(self.alpha_post * self.post_proj(x_hat) + self.post_bias)
        return A, C


def hc_read(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Weighted stream read: [B,S,n] x [B,S,n,d] -> [B,S,d]."""
    return torch.einsum("bsn,bsnd->bsd", A, X)


def hc_write(X: torch.Tensor, C: torch.Tensor, out: torch.Tensor, B: torch.Tensor = None) -> torch.Tensor:
    """Residual update: B-mixed streams plus the broadcast sublayer output.

    With B=None (iHC) this is X + C (x) out; with B [B,S,n,n] (mHC) it is
    B @ X + C (x) out, B being the doubly-stochastic stream mixing matrix.
    """
    if B is not None:
        X = torch.einsum("bsij,bsjd->bsid", B, X)
    return X + C.unsqueeze(-1) * out.unsqueeze(2)


class ManifoldHyperConnection(nn.Module):
    """Per-sublayer mHC coefficient generator (DeepSeek-V4 2.2, B on the Birkhoff polytope).

    Extends the iHC generator with the stream-mixing matrix B_tilde =
    alpha_res * (X_hat W_res) + S_res; B = Sinkhorn(exp(B_tilde)) is doubly
    stochastic (non-expansive, stable through depth). S_res inits to
    -tau*(J-I) so exp(S_res) ~ I (a Sinkhorn fixed point): mHC starts as iHC,
    which starts as the standard residual.
    """

    def __init__(self, config: GatedMLAHybridConfig):
        super().__init__()
        n, d = config.mhc_num_streams, config.hidden_size
        self.n = n
        self.d = d
        self.sinkhorn_iters = config.mhc_sinkhorn_iters
        self.pre_proj = nn.Linear(n * d, n, bias=False)
        self.post_proj = nn.Linear(n * d, n, bias=False)
        self.res_proj = nn.Linear(n * d, n * n, bias=False)
        self.pre_bias = nn.Parameter(torch.full((n,), math.log((1.0 / n) / (1.0 - 1.0 / n))))
        self.post_bias = nn.Parameter(torch.zeros(n))
        # tau=0 -> exp(S_res) = all-ones -> Sinkhorn returns the uniform
        # matrix J/n exactly (the classic Sinkhorn seed, no exp saturation).
        tau = config.mhc_res_bias_init
        self.res_bias = nn.Parameter(-tau * (torch.ones(n, n) - torch.eye(n)).reshape(-1))
        self.alpha_pre = nn.Parameter(torch.tensor(float(config.mhc_alpha_init)))
        self.alpha_post = nn.Parameter(torch.tensor(float(config.mhc_alpha_init)))
        self.alpha_res = nn.Parameter(torch.tensor(float(config.mhc_alpha_init)))

    def forward(self, X: torch.Tensor):
        bsz, seq = X.shape[0], X.shape[1]
        x_hat = F.rms_norm(X.reshape(bsz, seq, -1), (self.n * self.d,))
        A = torch.sigmoid(self.alpha_pre * self.pre_proj(x_hat) + self.pre_bias)
        C = 2.0 * torch.sigmoid(self.alpha_post * self.post_proj(x_hat) + self.post_bias)
        B_tilde = self.alpha_res * self.res_proj(x_hat) + self.res_bias
        M = torch.exp(B_tilde).reshape(bsz, seq, self.n, self.n)
        # Sinkhorn-Knopp: alternate row/column normalization onto the Birkhoff
        # polytope. exp(tau*I) is already doubly stochastic up to a scalar, so
        # the identity init is a fixed point (the alpha term perturbs it).
        for _ in range(self.sinkhorn_iters):
            M = M / M.sum(dim=-1, keepdim=True)
            M = M / M.sum(dim=-2, keepdim=True)
        B = M
        return A, B, C


class HybridBlock(nn.Module):
    def __init__(self, config: GatedMLAHybridConfig, layer_idx: int):
        super().__init__()
        if config.use_engram:
            raise NotImplementedError(
                "Engram is reserved for the next integration step; "
                "keep it disabled while validating the baseline architecture."
            )
        self.attn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.mlp_norm = RMSNorm(config.hidden_size, config.norm_eps)
        if layer_idx in config.linear_layer_indices:
            if config.baseline == "kda_gated_mla":
                self.attn = KDAAttention(config, layer_idx)
            elif config.baseline == "hils_gated_mla":
                self.attn = HiLSAttention(config)
            else:
                self.attn = SlidingWindowAttention(config)
        else:
            self.attn = GatedMLA(config)
        self.mlp = MoEFFN(config) if config.num_experts > 1 else SwiGLU(config)
        self.layer_idx = layer_idx
        self.use_ihc = config.use_ihc
        self.use_mhc = config.use_mhc
        if self.use_ihc:
            self.attn_ihc = IdentityHyperConnection(config)
            self.mlp_ihc = IdentityHyperConnection(config)
        if self.use_mhc:
            self.attn_mhc = ManifoldHyperConnection(config)
            self.mlp_mhc = ManifoldHyperConnection(config)

    def forward(self, hidden_states, past_key_value=None, use_cache=False, **kwargs):
        residual = hidden_states
        mixed, next_cache = self.attn(
            self.attn_norm(hidden_states),
            past_key_value=past_key_value,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + mixed
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states, next_cache

    def forward_streams(self, X, past_key_value=None, use_cache=False, **kwargs):
        """Hyper-connection path: X [B,S,n,d] streams; A/C read/write, B=I (iHC)
        or doubly-stochastic (mHC)."""
        gen = self.attn_mhc if self.use_mhc else self.attn_ihc
        coeffs = gen(X)
        B = coeffs[1] if self.use_mhc else None
        A = coeffs[0]
        C = coeffs[-1]
        mixed, next_cache = self.attn(
            self.attn_norm(hc_read(A, X)),
            past_key_value=past_key_value,
            use_cache=use_cache,
            **kwargs,
        )
        X = hc_write(X, C, mixed, B)
        gen2 = self.mlp_mhc if self.use_mhc else self.mlp_ihc
        coeffs2 = gen2(X)
        B2 = coeffs2[1] if self.use_mhc else None
        X = hc_write(X, coeffs2[-1], self.mlp(self.mlp_norm(hc_read(coeffs2[0], X))), B2)
        return X, next_cache


class GatedMLAHybridPreTrainedModel(PreTrainedModel):
    config_class = GatedMLAHybridConfig
    base_model_prefix = "model"
    _no_split_modules = ["HybridBlock"]
    # generate() passes a DynamicCache; forward accepts it via recompute mode
    _supports_cache_class = True
    # Porting note (VeOmni): attention runs through
    # torch.nn.functional.scaled_dot_product_attention inside the layer forward,
    # so declare sdpa support or transformers refuses attn_implementation="sdpa"
    # at load time; the flag changes no computation.
    _supports_sdpa = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)


class GatedMLAHybridModel(GatedMLAHybridPreTrainedModel):
    def __init__(self, config: GatedMLAHybridConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [HybridBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.use_ihc = config.use_ihc
        self.use_mhc = config.use_mhc
        self.use_hc = config.use_ihc or config.use_mhc
        self.hc_num_streams = config.mhc_num_streams if config.use_mhc else config.ihc_num_streams
        if self.use_hc:
            # final readout: learned combination of the streams, init 1/n
            # (with identical streams this reduces to the plain hidden state)
            param_name = "mhc_readout" if config.use_mhc else "ihc_readout"
            setattr(
                self,
                param_name,
                nn.Parameter(torch.full((self.hc_num_streams,), 1.0 / self.hc_num_streams)),
            )
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _get_past_length(self, past_key_values) -> int:
        if isinstance(past_key_values, Cache):
            # Recompute mode: generate() passes a Cache object and
            # prepare_inputs_for_generation keeps the full input_ids, so every
            # step re-runs the whole sequence and positions start at 0.
            return 0
        if not past_key_values:
            return 0
        first = past_key_values[0]
        if first is None:
            return 0
        if isinstance(first, tuple):
            return first[0].shape[1] if first[0].ndim == 3 else first[0].shape[2]
        if hasattr(first, "get_seq_length"):
            return int(first.get_seq_length())
        raise TypeError(f"Unsupported cache type for {type(first).__name__}")

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        output_hidden_states=False,
        return_dict=None,
        position_ids=None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Exactly one of input_ids and inputs_embeds must be provided")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if self.training:
            use_cache = False
        if position_ids is None and input_ids is not None:
            start = self._get_past_length(past_key_values)
            position_ids = torch.arange(
                start,
                start + input_ids.shape[1],
                device=inputs_embeds.device,
            ).unsqueeze(0).expand(input_ids.shape[0], -1)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        next_cache = [] if use_cache else None
        cache_input = isinstance(past_key_values, Cache)
        if self.use_hc:
            # broadcast the embedding into n identical residual streams
            X = inputs_embeds.unsqueeze(2).expand(
                *inputs_embeds.shape[:2], self.hc_num_streams, inputs_embeds.shape[-1]
            )
        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            # Cache objects from generate() are not consumed (recompute mode);
            # legacy tuple caches keep the original incremental protocol.
            layer_past = None if (past_key_values is None or cache_input) else past_key_values[layer_idx]
            if self.use_hc:
                X, layer_cache = layer.forward_streams(
                    X,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                    position_ids=position_ids,
                    **kwargs,
                )
            else:
                hidden_states, layer_cache = layer(
                    hidden_states,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                    position_ids=position_ids,
                    **kwargs,
                )
            if use_cache:
                next_cache.append(layer_cache)
        if self.use_hc:
            # (output_hidden_states is not stream-aware in HC mode; entries
            # stay at the embedding level -- debug-only feature)
            readout = self.mhc_readout if self.use_mhc else self.ihc_readout
            hidden_states = torch.einsum("n,bsnd->bsd", readout, X)
        hidden_states = self.norm(hidden_states)
        aux_losses = None
        if any(getattr(layer.mlp, "last_balance_loss", None) is not None for layer in self.layers):
            aux_losses = sum(
                layer.mlp.last_balance_loss
                for layer in self.layers
                if getattr(layer.mlp, "last_balance_loss", None) is not None
            )
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if use_cache and cache_input:
            past_out = _pack_dynamic_cache(next_cache)
        else:
            past_out = tuple(next_cache) if next_cache is not None else None
        if not (self.config.use_return_dict if return_dict is None else return_dict):
            return tuple(x for x in (hidden_states, past_out, all_hidden_states) if x is not None)
        outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_out,
            hidden_states=all_hidden_states,
        )
        if aux_losses is not None:
            # ModelOutput is a fixed-field dataclass in transformers 4.57;
            # attach the MoE aux loss as a plain attribute instead.
            outputs.aux_losses = aux_losses
        return outputs


class GatedMLAHybridForCausalLM(GatedMLAHybridPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GatedMLAHybridConfig):
        super().__init__(config)
        self.model = GatedMLAHybridModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        self.criterion = None
        self.post_init()
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=True,
        **kwargs,
    ):
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            # legacy tuple cache: incremental decoding under the old protocol
            input_ids = input_ids[:, -1:]
            inputs_embeds = None
        # transformers Cache (generate()): keep the full input_ids -- the model
        # recomputes the whole sequence each step (recompute-mode cache support)
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }
        if past_key_values is None and inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds
            model_inputs.pop("input_ids")
        return model_inputs

    def forward(self, labels=None, logits_to_keep: int = 0, **kwargs):
        outputs = self.model(**kwargs)
        return_dict = kwargs.get("return_dict", self.config.use_return_dict)
        if return_dict:
            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values
            all_hidden_states = outputs.hidden_states
        else:
            hidden_states = outputs[0]
            past_key_values = outputs[1] if len(outputs) > 1 else None
            all_hidden_states = outputs[2] if len(outputs) > 2 else None
        logits_input = (
            hidden_states
            if labels is not None or not logits_to_keep
            else hidden_states[:, -logits_to_keep:]
        )
        logits = self.lm_head(logits_input)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            aux = getattr(outputs, "aux_losses", None)
            if aux is not None and self.training:
                loss = loss + self.config.moe_balance_loss_weight * aux
        if not return_dict:
            output = tuple(
                value
                for value in (logits, past_key_values, all_hidden_states)
                if value is not None
            )
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )

    def parameter_counts(self):
        return {"total_parameters": sum(p.numel() for p in self.parameters())}
