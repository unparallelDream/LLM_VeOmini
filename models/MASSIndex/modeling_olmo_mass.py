"""OLMo3 wrapper that replaces HiLS retrieval layers with MASS retrieval."""

from __future__ import annotations

import os
from types import MethodType

import torch

from models.FlashHiLS.hils_attention import HiLSAttention
from models.FlashHiLS.modeling_olmo_hils import (
    HiLSForCausalLM,
    HiLSModel,
    Olmo3RMSNorm,
)

from .mass_attention import MASSAttention
from .mass_index import MultiTangentAnchorGenerator
from .modeling_qwen_mass import MASSForCausalLM as QwenMASSForCausalLM
from .modeling_qwen_mass import (
    MASSForCausalLM as QwenMASSForCausalLM,
    mass_index_config_from_model_config,
)


def _mass_decoder_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    use_cache=False,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
):
    """OLMo decoder forward that returns checkpoint-safe MASS auxiliaries."""
    residual = hidden_states
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.post_feedforward_layernorm(hidden_states)
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


class MASSModel(HiLSModel):
    """OLMo3 body with the same sparse-layer placement as OLMo3 HiLS."""

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
                norm_cls=Olmo3RMSNorm,
                mass_config=mass_config,
            )
            missing, unexpected = mass_attention.load_state_dict(
                old_attention.state_dict(), strict=False
            )
            unexpected = [
                name for name in unexpected
                if not name.startswith("mass_anchor_generator")
            ]
            if unexpected:
                raise RuntimeError(
                    f"unexpected OLMo HiLS -> MASS state keys: {unexpected}"
                )
            disallowed_missing = [
                name for name in missing
                if not name.startswith("mass_anchor_generator")
            ]
            if disallowed_missing:
                raise RuntimeError(
                    f"missing shared OLMo HiLS weights in MASS layer: {disallowed_missing}"
                )
            decoder_layer.self_attn = mass_attention
            decoder_layer.forward = MethodType(_mass_decoder_forward, decoder_layer)
            mass_layer_indices.append(layer_idx)
        self._mass_layer_indices = tuple(mass_layer_indices)


class MASSForCausalLM(HiLSForCausalLM):
    """OLMo3 causal LM initialized from an HF/ModelScope OLMo3 checkpoint."""

    def _init_weights(self, module):
        # VeOmni initializes checkpoint-missing parameters one leaf at a time.
        # Handle the direct anchor Parameter here so meta-device materialization
        # can never leave it zeroed or backed by uninitialized storage.
        if isinstance(module, MultiTangentAnchorGenerator):
            module.reset_parameters()
            return
        super()._init_weights(module)

    def init_weights(self):
        super().init_weights()
        for module in self.modules():
            if isinstance(module, MultiTangentAnchorGenerator):
                module.reset_parameters()

    @staticmethod
    def get_optimizer_pre_hook(model, model_config, data_parallel_mode):
        return QwenMASSForCausalLM.get_optimizer_pre_hook(
            model, model_config, data_parallel_mode
        )

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

