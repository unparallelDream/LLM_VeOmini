"""Separate Qwen model wrapper that installs :class:`MASSAttention`.

All shared decoder and LM-forward behaviour comes from the original HiLS
implementation.  No source file in ``FlashHiLS`` is modified.
"""

from __future__ import annotations

import os
from types import MethodType

import torch

from models.FlashHiLS.hils_attention import HiLSAttention
from models.FlashHiLS.modeling_qwen_hils import (
    HiLSForCausalLM,
    HiLSModel,
    Qwen3RMSNorm,
)

from .mass_attention import MASSAttention
from .mass_index import MASSIndexConfig, MultiTangentAnchorGenerator


def _mass_qwen_decoder_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
):
    """Qwen decoder forward returning checkpoint-safe MASS auxiliaries."""
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    balance_loss = self.self_attn.last_mass_balance_loss
    anchor_load = self.self_attn.last_mass_anchor_load
    if balance_loss is None or anchor_load is None:
        num_anchors = self.self_attn.mass_config.num_anchors
        balance_loss = hidden_states.new_zeros((), dtype=torch.float32)
        anchor_load = hidden_states.new_full(
            (num_anchors,), 1.0 / num_anchors, dtype=torch.float32
        )
    return hidden_states, balance_loss, anchor_load


def mass_index_config_from_model_config(config) -> MASSIndexConfig:
    return MASSIndexConfig(
        num_anchors=config.mass_num_anchors,
        offset_rank=config.mass_offset_rank,
        anchor_rms_norm=config.mass_anchor_rms_norm,
        offset_init_std=config.mass_offset_init_std,
        score_reduction=getattr(config, "mass_score_reduction", "max"),
        lse_temperature=getattr(config, "mass_lse_temperature", 1.0),
        normalize_lse=getattr(config, "mass_normalize_lse", False),
        lse_anneal_start=getattr(config, "mass_lse_anneal_start", None),
        lse_anneal_end=getattr(config, "mass_lse_anneal_end", None),
        two_stage_backward=getattr(config, "mass_two_stage_backward", None),
        shared_view=getattr(config, "mass_shared_view", False),
        shared_bias_init=getattr(config, "mass_shared_bias_init", 2.0),
    )


class MASSModel(HiLSModel):
    """HiLS transformer body with only landmark-attention layers replaced."""

    def __init__(self, config):
        super().__init__(config)
        self._collect_mass_aux = True
        mass_layer_indices = []
        mass_config = mass_index_config_from_model_config(config)
        for layer_idx, decoder_layer in enumerate(self.layers):
            old_attention = decoder_layer.self_attn
            if not isinstance(old_attention, HiLSAttention):
                continue
            mass_attention = MASSAttention(
                config,
                layer_idx,
                norm_cls=Qwen3RMSNorm,
                mass_config=mass_config,
            )
            # Existing Q/K/V/O and landmark-projection weights retain the
            # original names.  Only the new anchor parameters are fresh.
            missing, unexpected = mass_attention.load_state_dict(old_attention.state_dict(), strict=False)
            unexpected = [name for name in unexpected if not name.startswith("mass_anchor_generator")]
            if unexpected:
                raise RuntimeError(f"unexpected HiLS -> MASS state keys: {unexpected}")
            disallowed_missing = [name for name in missing if not name.startswith("mass_anchor_generator")]
            if disallowed_missing:
                raise RuntimeError(f"missing shared HiLS weights in MASS layer: {disallowed_missing}")
            decoder_layer.self_attn = mass_attention
            decoder_layer.forward = MethodType(_mass_qwen_decoder_forward, decoder_layer)
            mass_layer_indices.append(layer_idx)
        self._mass_layer_indices = tuple(mass_layer_indices)


class MASSForCausalLM(HiLSForCausalLM):
    """Causal LM wrapper selecting ``MASSModel`` instead of ``HiLSModel``."""

    def init_weights(self):
        super().init_weights()
        for module in self.modules():
            if isinstance(module, MultiTangentAnchorGenerator):
                module.reset_parameters()

    @staticmethod
    def get_optimizer_pre_hook(model, model_config, data_parallel_mode):
        """Fail fast if MASS factors or their first-step gradients are zero."""
        checked = False

        def _local(tensor):
            return tensor.to_local() if hasattr(tensor, "to_local") else tensor

        def _check_anchor_gradients(optimizer, args, kwargs):
            nonlocal checked
            if getattr(optimizer, "_mass_initializing_checkpoint_state", False):
                return None
            if checked:
                return None

            anchor_params = [
                (name, param)
                for name, param in model.named_parameters()
                if ".mass_anchor_generator." in name
            ]
            if not anchor_params:
                raise RuntimeError("MASS model has no trainable anchor parameters")

            sample = _local(anchor_params[0][1]).detach()
            device = sample.device
            param_absmax = torch.zeros((), device=device, dtype=torch.float32)
            grad_absmax = torch.zeros((), device=device, dtype=torch.float32)
            bad_values = torch.zeros((), device=device, dtype=torch.int32)
            grad_seen = torch.zeros((), device=device, dtype=torch.int32)
            for _, param in anchor_params:
                local_param = _local(param).detach()
                param_absmax = torch.maximum(param_absmax, local_param.float().abs().max())
                bad_values += (~torch.isfinite(local_param)).any().to(torch.int32)
                if param.grad is not None:
                    local_grad = _local(param.grad).detach()
                    grad_seen += 1
                    grad_absmax = torch.maximum(grad_absmax, local_grad.float().abs().max())
                    bad_values += (~torch.isfinite(local_grad)).any().to(torch.int32)

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(param_absmax, op=torch.distributed.ReduceOp.MAX)
                torch.distributed.all_reduce(grad_absmax, op=torch.distributed.ReduceOp.MAX)
                torch.distributed.all_reduce(bad_values, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(grad_seen, op=torch.distributed.ReduceOp.SUM)
                rank = torch.distributed.get_rank()
            else:
                rank = 0

            if bad_values.item() != 0:
                raise RuntimeError("MASS anchor parameters or gradients contain NaN/Inf")
            if param_absmax.item() == 0.0:
                raise RuntimeError("MASS anchor parameters are all zero after initialization/checkpoint load")
            if grad_seen.item() == 0 or grad_absmax.item() == 0.0:
                raise RuntimeError("MASS anchor gradients are missing or all zero on the first optimizer step")
            if rank == 0:
                print(
                    f"[mass-anchor-check] param_absmax={param_absmax.item():.6g} "
                    f"grad_absmax={grad_absmax.item():.6g}",
                    flush=True,
                )
            checked = True
            return None

        return _check_anchor_gradients

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = MASSModel(config)
        if config.tie_word_embeddings:
            self.tie_weights()

    def _build_needle_ctx(self, input_ids):
        """Locate RULER needles in RAW input ids and build the NCI context.

        The needle phrase never occurs in natural fineweb text, so a simple
        token-subsequence match is enough.  Positions are converted to
        expanded coordinates (r + r//63) and 63-token chunk ids here; the
        retrieval layers only see expanded tensors.
        """
        pattern = getattr(self.model, "_mass_needle_pattern", None)
        if pattern is None or input_ids is None or input_ids.dim() != 2:
            return None
        chunk_size = int(getattr(self.config, "chunk_size", 64) or 64)
        real = chunk_size - 1
        pattern = pattern.to(input_ids.device)
        B, L = input_ids.shape
        if pattern.numel() < 2 or L <= pattern.numel():
            return None
        n_query = int(os.environ.get("MASS_NCI_QUERY_ROWS", "64"))
        raw_rows = torch.arange(max(0, L - n_query), L, device=input_ids.device)
        rows_exp = raw_rows + torch.div(raw_rows, real, rounding_mode="floor")
        windows = input_ids.unfold(1, pattern.numel(), 1)
        matches = (windows == pattern[None, None, :]).all(dim=-1)  # [B, L-P+1]
        max_needles = int(os.environ.get("MASS_NCI_MAX_NEEDLES", "8"))
        needle_chunks = torch.full(
            (B, max_needles), -1, dtype=torch.long, device=input_ids.device
        )
        has = torch.zeros(B, dtype=torch.bool, device=input_ids.device)
        for b in range(B):
            pos = matches[b].nonzero(as_tuple=True)[0]
            if pos.numel() == 0:
                continue
            pos = pos[:max_needles]
            needle_chunks[b, : pos.numel()] = torch.div(
                pos, real, rounding_mode="floor"
            )
            has[b] = True
        if not bool(has.any()):
            return None
        return {
            "rows_exp": rows_exp,
            "needle_chunks": needle_chunks,
            "sample_has_needle": has,
        }

    def forward(self, *args, **kwargs):
        nci_enabled = os.environ.get("MASS_NEEDLE_INFO_NCE", "0") == "1"
        ctx = None
        if nci_enabled and torch.is_grad_enabled():
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            ctx = self._build_needle_ctx(input_ids)
        # Reset per-forward state BEFORE the pass and keep ctx alive until the
        # next forward: gradient-checkpointing recomputes decoder layers during
        # backward, and both passes must take the identical code path (a ctx
        # cleared mid-step makes recomputation save a different number of
        # tensors and trips CheckpointError).  Terms appended later by the
        # recompute pass are harmless; they are wiped here at the next entry.
        for layer_idx in self.model._mass_layer_indices:
            attn = self.model.layers[int(layer_idx)].self_attn
            attn._mass_nci_terms = []
            attn._mass_needle_ctx = ctx
        outputs = super().forward(*args, **kwargs)
        lm_loss_pure = outputs.loss  # pure LM loss; spec/nci are added below
        nci_terms = []
        if nci_enabled:
            for layer_idx in self.model._mass_layer_indices:
                attn = self.model.layers[int(layer_idx)].self_attn
                terms = getattr(attn, "_mass_nci_terms", [])
                nci_terms.extend(t for t in terms if t is not None)
        needle_loss = None
        nci_weight = float(os.environ.get("MASS_NCI_WEIGHT", "1.0"))
        if nci_terms:
            needle_loss = torch.stack(nci_terms).mean()
            if outputs.loss is not None:
                outputs.loss = outputs.loss + nci_weight * needle_loss
            outputs.mass_needle_loss = needle_loss
            outputs.mass_needle_loss_weight = nci_weight
        balance_losses = getattr(self.model, "_mass_balance_losses", ())
        anchor_loads = getattr(self.model, "_mass_anchor_loads", ())
        if balance_losses:
            lm_loss = lm_loss_pure
            balance_loss = torch.stack(balance_losses).mean()
            anchor_load = torch.stack(anchor_loads)
            balance_weight = float(
                os.environ.get(
                    "MASS_SPECIALIZATION_LOSS_WEIGHT",
                    os.environ.get(
                        "MASS_BALANCE_LOSS_WEIGHT",
                        getattr(
                            self.config,
                            "mass_specialization_loss_weight",
                            getattr(self.config, "mass_balance_loss_weight", 0.01),
                        ),
                    ),
                )
            )
            # MASS_SPEC_WARMUP_STEPS: linear ramp 0 -> balance_weight, driven
            # by the training loop via model._mass_spec_weight_step, so the
            # routing differentiation is not locked in before the LM (and any
            # retrieval supervision) has shaped the summaries.
            step_weight = getattr(self.model, "_mass_spec_weight_step", None)
            if step_weight is not None:
                balance_weight = float(step_weight)
            if os.environ.get("MASS_SPEC_DIVERSE_MU", "0") == "1":
                div_losses = [
                    self.model.layers[int(i)].self_attn.last_mass_diversity_loss
                    for i in self.model._mass_layer_indices
                ]
                div_losses = [d for d in div_losses if d is not None]
                if div_losses:
                    balance_loss = torch.stack(div_losses).mean()
            if lm_loss is not None:
                outputs.loss = lm_loss + balance_weight * balance_loss
            outputs.lm_loss = lm_loss
            outputs.mass_balance_loss = balance_loss
            outputs.mass_balance_loss_weight = balance_weight
            outputs.mass_specialization_loss = balance_loss
            outputs.mass_specialization_loss_weight = balance_weight
            outputs.mass_anchor_loads = anchor_load
            outputs.mass_layer_indices = self.model._mass_layer_indices
        return outputs
