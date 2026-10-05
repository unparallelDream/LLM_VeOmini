#!/usr/bin/env python3
"""Deterministic 4K RULER evaluation for Qwen3-0.6B MASS and equal-budget BSA."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from types import MethodType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_mass_model(args, fixed_anchor_zero: bool):
    # Importing the registry module is intentionally explicit: models/__init__.py
    # does not import the opt-in MASS model registration.
    from models.MASSIndex import modeling_registry as _mass_registry  # noqa: F401
    from veomni.checkpoint import ckpt_to_state_dict
    from veomni.models import build_foundation_model

    model = build_foundation_model(
        config_path=args.mass_config,
        weights_path=args.mass_model,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
        init_device="cuda",
    )

    adapter_state = ckpt_to_state_dict(args.checkpoint)
    expected = {
        name
        for name, _ in model.named_parameters()
        if ".mass_anchor_generator." in name
    }
    provided = set(adapter_state)
    if provided != expected:
        raise RuntimeError(
            "Step checkpoint is not the expected anchor-only adapter: "
            f"missing={sorted(expected - provided)}, unexpected={sorted(provided - expected)}"
        )
    incompat = model.load_state_dict(adapter_state, strict=False)
    if incompat.unexpected_keys:
        raise RuntimeError(f"Unexpected DCP keys: {incompat.unexpected_keys}")

    loaded_norms = {
        name: float(param.detach().float().norm().cpu())
        for name, param in model.named_parameters()
        if name in provided
    }
    if any(value == 0.0 or not np.isfinite(value) for value in loaded_norms.values()):
        raise RuntimeError(f"Invalid loaded anchor norm(s): {loaded_norms}")

    fixed_modules = 0
    if fixed_anchor_zero:
        with torch.no_grad():
            for module in model.modules():
                generator = getattr(module, "mass_anchor_generator", None)
                if generator is None:
                    continue
                anchor_embedding = generator.anchor_embedding
                anchor_embedding.copy_(anchor_embedding[0:1].expand_as(anchor_embedding))
                if not torch.equal(
                    anchor_embedding,
                    anchor_embedding[0:1].expand_as(anchor_embedding),
                ):
                    raise RuntimeError("Failed to replicate fixed anchor 0")
                fixed_modules += 1
        expected_fixed_modules = len(model.model._mass_layer_indices)
        if fixed_modules != expected_fixed_modules:
            raise RuntimeError(
                f"Expected {expected_fixed_modules} MASS layers, fixed {fixed_modules}"
            )

    model.config.use_cache = False
    model.eval()
    return model, {
        "adapter_key_count": len(provided),
        "adapter_norm_min": min(loaded_norms.values()),
        "adapter_norm_max": max(loaded_norms.values()),
        "fixed_anchor_modules": fixed_modules,
    }


def load_bsa_model(args, score_mode: str = "exact"):
    """Build a block-sparse-attention baseline from the identical MASS base weights.

    Only the eight retrieval attention modules are replaced.  The Q/K/V/O and
    QK-normalization tensors are copied from the matching MASS layers, while
    MLPs, embeddings, norms, and the remaining 24 layers stay bit-identical.

    ``score_mode`` selects the chunk-ranking scorer while keeping everything
    else (budget, layout, masks, mixture structure) identical:
      * ``"exact"`` — per-token block mass ``log sum exp(q·k)`` (NSA/DSA-style)
      * ``"mean"``  — block-mean pooled score ``q · mean(k_chunk)``
    """
    from models.MASSIndex import modeling_registry as _mass_registry  # noqa: F401
    from models.FlashHiLS.modeling_qwen_hils import (
        Qwen3DecoderLayer,
        Qwen3RMSNorm,
    )
    from models.NaiveBSA import NaiveBSA
    from veomni.models import build_foundation_model

    model = build_foundation_model(
        config_path=args.mass_config,
        weights_path=args.mass_model,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
        init_device="cuda",
    )
    # Chunk-ranking scorer for the replaced layers; see load_bsa_model docstring.
    model.config.bsa_score_mode = score_mode

    replaced_layers = []
    copied_keys = {}
    allowed_source_only_prefixes = (
        "lmk_q_proj.",
        "lmk_q_norm.",
        "mass_anchor_generator.",
    )
    for layer_idx in model.model._mass_layer_indices:
        decoder_layer = model.model.layers[layer_idx]
        source_attention = decoder_layer.self_attn
        source_state = source_attention.state_dict()
        source_device = source_attention.q_proj.weight.device
        source_dtype = source_attention.q_proj.weight.dtype

        bsa_attention = NaiveBSA(
            model.config,
            layer_idx,
            norm_cls=Qwen3RMSNorm,
        ).to(device=source_device, dtype=source_dtype)
        incompat = bsa_attention.load_state_dict(source_state, strict=False)
        if incompat.missing_keys:
            raise RuntimeError(
                f"BSA layer {layer_idx} did not receive all shared weights: "
                f"{incompat.missing_keys}"
            )
        disallowed_source_only = [
            key
            for key in incompat.unexpected_keys
            if not key.startswith(allowed_source_only_prefixes)
        ]
        if disallowed_source_only:
            raise RuntimeError(
                f"Unexpected non-MASS source keys at layer {layer_idx}: "
                f"{disallowed_source_only}"
            )

        # Verify the load itself instead of trusting strict=False bookkeeping.
        for key, tensor in bsa_attention.state_dict().items():
            if key not in source_state or not torch.equal(tensor, source_state[key]):
                raise RuntimeError(
                    f"Shared BSA tensor mismatch at layer {layer_idx}: {key}"
                )

        decoder_layer.self_attn = bsa_attention
        decoder_layer.forward = MethodType(Qwen3DecoderLayer.forward, decoder_layer)
        replaced_layers.append(int(layer_idx))
        copied_keys[str(layer_idx)] = sorted(bsa_attention.state_dict())

    if replaced_layers != [3, 7, 11, 15, 19, 23, 27]:
        raise RuntimeError(f"Unexpected BSA layer placement: {replaced_layers}")
    model.model._collect_mass_aux = False
    model.config.use_cache = False
    model.eval()
    return model, {
        "adapter_key_count": 0,
        "adapter_norm_min": None,
        "adapter_norm_max": None,
        "fixed_anchor_modules": 0,
        "bsa_replaced_layers": replaced_layers,
        "bsa_shared_state_keys": copied_keys,
        "bsa_uses_mass_base_weights": True,
        "bsa_uses_qwen_rope_on_replaced_layers": True,
        "bsa_masks_chunk_end_landmarks": bool(model.config.mask_lmk_token),
        "bsa_score_mode": score_mode,
        "baseline_definition": (
            "Qwen3-0.6B original shared weights with only the seven MASS/HiLS "
            "retrieval layers replaced by equal-budget block-sparse attention; "
            + (
                "chunks ranked by exact per-token block mass logsumexp(q·k) "
                "(NSA/DSA-style, O(L^2) scoring)"
                if score_mode == "exact"
                else "chunks ranked by block-mean pooled score q·mean(k_chunk) "
                "(O(L*N) scoring)"
            )
        ),
    }


class DenseRetrievalAttention(torch.nn.Module):
    """Dense causal attention replacement for the retrieval layers.

    Quality ceiling at budget=100%: identical shared Q/K/V/O + QK-norm +
    RoPE as every other variant, but every key is attended (causally).
    Chunk-end landmark keys stay masked so the value-KV accounting matches
    the sparse variants; everything else about the input layout is shared.
    Eval-only, no KV cache.
    """

    def __init__(self, config, layer_idx, norm_cls=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self.h_q = config.num_attention_heads
        self.h_kv = config.num_key_value_heads
        self.d_query = self.head_dim * self.h_q

        self.q_proj = torch.nn.Linear(self.hidden_size, self.d_query, bias=False)
        self.k_proj = torch.nn.Linear(
            self.hidden_size, self.h_kv * self.head_dim, bias=False
        )
        self.v_proj = torch.nn.Linear(
            self.hidden_size, self.h_kv * self.head_dim, bias=False
        )
        self.o_proj = torch.nn.Linear(self.d_query, self.hidden_size, bias=False)

        self.layerwise_qk_norm = getattr(config, "layerwise_qk_norm", False)
        if norm_cls is not None and not self.layerwise_qk_norm:
            self.q_norm = norm_cls(self.head_dim)
            self.k_norm = norm_cls(self.head_dim)
        self.apply_hils_rope = getattr(config, "apply_hils_rope", True)
        self.mask_lmk_token = bool(getattr(config, "mask_lmk_token", False))
        self.chunk_size = config.chunk_size
        self.scaling = self.head_dim ** -0.5

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
    ):
        if use_cache or past_key_values is not None:
            raise NotImplementedError("DenseRetrievalAttention is eval-only.")
        if position_embeddings is None:
            raise ValueError("DenseRetrievalAttention needs (cos, sin).")
        B, L, _ = hidden_states.shape
        cos, sin = position_embeddings

        q = self.q_proj(hidden_states).view(B, L, self.h_q, self.head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.h_kv, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.h_kv, self.head_dim)
        if not self.layerwise_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2)  # (B, h_q, L, d)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.apply_hils_rope:
            from models.NaiveBSA.naive_bsa_layer import apply_rotary_pos_emb

            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        G = self.h_q // self.h_kv
        if G > 1:
            k = k.repeat_interleave(G, dim=1)
            v = v.repeat_interleave(G, dim=1)

        j = torch.arange(L, device=hidden_states.device).view(1, L)
        i = torch.arange(L, device=hidden_states.device).view(L, 1)
        allowed = j <= i
        if self.mask_lmk_token:
            allowed = allowed & (((j + 1) % self.chunk_size) != 0)
        o = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, scale=self.scaling
        )
        o = o.transpose(1, 2).reshape(B, L, self.d_query)
        return self.o_proj(o), None


def load_mass_lora_model(args):
    """Trained anchor + full-token-mixer LoRA adapter (launch_anchor_lora_kv_ft).

    Injects the same LoRA configuration used in training (rank 16, alpha 32,
    q/k/v/o on every layer) and loads the trainable-only checkpoint that
    contains the 14 anchor tensors plus 224 LoRA tensors."""
    from models.MASSIndex import modeling_registry as _mass_registry  # noqa: F401
    from utils.lora_adapt import inject_lora
    from veomni.checkpoint import ckpt_to_state_dict
    from veomni.models import build_foundation_model

    model = build_foundation_model(
        config_path=args.mass_config,
        weights_path=args.mass_model,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
        init_device="cuda",
    )
    wrapped, lora_param_count = inject_lora(
        model,
        rank=16,
        alpha=32.0,
        targets="self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj",
        only_layers=None,
    )
    adapter_state = ckpt_to_state_dict(args.checkpoint)
    # Tied snapshots carry an lm_head duplicate of embed_tokens; it is not an
    # adapter tensor, drop it before the strict inventory check.
    adapter_state.pop("lm_head.weight", None)
    expected = {
        name
        for name, _ in model.named_parameters()
        if ".mass_anchor_generator." in name or ".lora_" in name
        or ".lmk_q_proj." in name or ".lmk_q_norm." in name
    }
    provided = set(adapter_state)
    if provided != expected:
        raise RuntimeError(
            f"anchor+lora checkpoint mismatch: missing={sorted(expected - provided)[:6]}, "
            f"unexpected={sorted(provided - expected)[:6]}"
        )
    incompat = model.load_state_dict(adapter_state, strict=False)
    if incompat.unexpected_keys:
        raise RuntimeError(f"unexpected adapter keys: {incompat.unexpected_keys}")
    missing_adapters = sorted(expected.intersection(incompat.missing_keys))
    if missing_adapters:
        raise RuntimeError(f"adapter keys not loaded: {missing_adapters}")
    model.config.use_cache = False
    model.eval()
    return model, {
        "adapter_key_count": len(provided),
        "lora_linears": wrapped,
        "lora_param_count": lora_param_count,
        "baseline_definition": (
            "trained 2-anchor MASS scoring + rank-16 qkvo LoRA on all layers "
            "(full-token-mixer adaptation, frozen backbone weights)"
        ),
    }


def load_hils_lora_model(args):
    """Trained lmk_embed + full-token-mixer LoRA (hils_lora control arm)."""
    model, metadata = load_hils_model(args)
    from utils.lora_adapt import inject_lora
    from veomni.checkpoint import ckpt_to_state_dict

    wrapped, lora_param_count = inject_lora(
        model,
        rank=16,
        alpha=32.0,
        targets="self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj",
        only_layers=None,
    )
    adapter_state = ckpt_to_state_dict(args.checkpoint)
    adapter_state.pop("lm_head.weight", None)
    expected = {
        name
        for name, _ in model.named_parameters()
        if name == "model.lmk_embed" or ".lora_" in name
    }
    provided = set(adapter_state)
    if provided != expected:
        raise RuntimeError(
            f"lmk+lora checkpoint mismatch: missing={sorted(expected - provided)[:6]}, "
            f"unexpected={sorted(provided - expected)[:6]}"
        )
    incompat = model.load_state_dict(adapter_state, strict=False)
    if incompat.unexpected_keys:
        raise RuntimeError(f"unexpected adapter keys: {incompat.unexpected_keys}")
    missing_adapters = sorted(expected.intersection(incompat.missing_keys))
    if missing_adapters:
        raise RuntimeError(f"adapter keys not loaded: {missing_adapters}")
    metadata = dict(metadata)
    metadata["hils_zero_shot"] = False
    metadata["adapter_key_count"] = len(provided)
    metadata["lora_param_count"] = lora_param_count
    metadata["baseline_definition"] = (
        "HiLS landmark scoring (trained lmk_embed) + rank-16 qkvo LoRA on all "
        "layers — the LoRA-matched control for mass_lora"
    )
    return model, metadata


def load_bsa_lora_model(args):
    """Trained pure-LoRA BSA control arm (launch_lora_controls_20260915).

    Block-mean scoring with zero retrieval parameters, plus the same rank-16
    qkvo LoRA on every layer as mass_lora/hils_lora — the scoring-free control
    for the three-way attribution."""
    model, metadata = load_bsa_model(args, score_mode="mean")
    from utils.lora_adapt import inject_lora
    from veomni.checkpoint import ckpt_to_state_dict

    wrapped, lora_param_count = inject_lora(
        model,
        rank=16,
        alpha=32.0,
        targets="self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj",
        only_layers=None,
    )
    adapter_state = ckpt_to_state_dict(args.checkpoint)
    adapter_state.pop("lm_head.weight", None)
    expected = {
        name
        for name, _ in model.named_parameters()
        if ".lora_" in name
    }
    provided = set(adapter_state)
    if provided != expected:
        raise RuntimeError(
            f"lora-only checkpoint mismatch: missing={sorted(expected - provided)[:6]}, "
            f"unexpected={sorted(provided - expected)[:6]}"
        )
    incompat = model.load_state_dict(adapter_state, strict=False)
    if incompat.unexpected_keys:
        raise RuntimeError(f"unexpected adapter keys: {incompat.unexpected_keys}")
    missing_adapters = sorted(expected.intersection(incompat.missing_keys))
    if missing_adapters:
        raise RuntimeError(f"adapter keys not loaded: {missing_adapters}")
    metadata = dict(metadata)
    metadata["adapter_key_count"] = len(provided)
    metadata["lora_linears"] = wrapped
    metadata["lora_param_count"] = lora_param_count
    metadata["baseline_definition"] = (
        "block-mean chunk scoring (zero retrieval parameters, no lmk_embed "
        "training) + rank-16 qkvo LoRA on all layers — the scoring-free "
        "control for the mass_lora/hils_lora/bsa_lora three-way attribution"
    )
    return model, metadata


def load_full_model(args):
    """Budget=100% ceiling: identical base weights, retrieval layers swapped
    to dense causal attention (landmark value keys masked, layout shared)."""
    from models.MASSIndex import modeling_registry as _mass_registry  # noqa: F401
    from models.FlashHiLS.modeling_qwen_hils import (
        Qwen3DecoderLayer,
        Qwen3RMSNorm,
    )
    from veomni.models import build_foundation_model

    model = build_foundation_model(
        config_path=args.mass_config,
        weights_path=args.mass_model,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
        init_device="cuda",
    )
    # 让 attention_budget 上报 ~100%(512 + 57*64 = 4160 ≈ 4161);稠密层不读 topk。
    model.config.hils_topk = 57

    replaced_layers = []
    for layer_idx in model.model._mass_layer_indices:
        decoder_layer = model.model.layers[layer_idx]
        source_state = decoder_layer.self_attn.state_dict()
        dense_attention = DenseRetrievalAttention(
            model.config, layer_idx, norm_cls=Qwen3RMSNorm
        ).to(
            device=decoder_layer.self_attn.q_proj.weight.device,
            dtype=decoder_layer.self_attn.q_proj.weight.dtype,
        )
        incompat = dense_attention.load_state_dict(source_state, strict=False)
        if incompat.missing_keys:
            raise RuntimeError(
                f"Dense layer {layer_idx} missing shared weights: {incompat.missing_keys}"
            )
        disallowed = [
            key
            for key in incompat.unexpected_keys
            if not key.startswith(
                ("lmk_q_proj.", "lmk_q_norm.", "mass_anchor_generator.")
            )
        ]
        if disallowed:
            raise RuntimeError(
                f"Unexpected non-shared source keys at layer {layer_idx}: {disallowed}"
            )
        decoder_layer.self_attn = dense_attention
        decoder_layer.forward = MethodType(Qwen3DecoderLayer.forward, decoder_layer)
        replaced_layers.append(int(layer_idx))

    if replaced_layers != [3, 7, 11, 15, 19, 23, 27]:
        raise RuntimeError(f"Unexpected dense layer placement: {replaced_layers}")
    model.model._collect_mass_aux = False
    model.config.use_cache = False
    model.eval()
    return model, {
        "adapter_key_count": 0,
        "adapter_norm_min": None,
        "adapter_norm_max": None,
        "fixed_anchor_modules": 0,
        "dense_replaced_layers": replaced_layers,
        "baseline_definition": (
            "Qwen3-0.6B original shared weights, the seven retrieval layers "
            "replaced by dense causal attention (chunk-end landmark value "
            "keys masked; budget=100% quality ceiling)"
        ),
    }


def load_hils_model(args):
    """Zero-shot HiLS control: identical MASS base weights, retrieval layers
    swapped to HiLSAttention (landmark-key scoring, no anchors)."""
    from models.MASSIndex import modeling_registry as _mass_registry  # noqa: F401
    from models.FlashHiLS.hils_attention import HiLSAttention
    from models.FlashHiLS.modeling_qwen_hils import (
        Qwen3DecoderLayer,
        Qwen3RMSNorm,
    )
    from veomni.models import build_foundation_model

    model = build_foundation_model(
        config_path=args.mass_config,
        weights_path=args.mass_model,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
        init_device="cuda",
    )

    replaced_layers = []
    for layer_idx in model.model._mass_layer_indices:
        decoder_layer = model.model.layers[layer_idx]
        source_attention = decoder_layer.self_attn
        source_state = source_attention.state_dict()

        hils_attention = HiLSAttention(
            model.config,
            layer_idx,
            norm_cls=Qwen3RMSNorm,
        ).to(
            device=source_attention.q_proj.weight.device,
            dtype=source_attention.q_proj.weight.dtype,
        )
        incompat = hils_attention.load_state_dict(source_state, strict=False)
        if incompat.missing_keys:
            raise RuntimeError(
                f"HiLS layer {layer_idx} did not receive all shared weights: "
                f"{incompat.missing_keys}"
            )
        disallowed = [
            key
            for key in incompat.unexpected_keys
            if not key.startswith(
                ("lmk_q_proj.", "lmk_q_norm.", "mass_anchor_generator.")
            )
        ]
        if disallowed:
            raise RuntimeError(
                f"Unexpected non-shared source keys at layer {layer_idx}: {disallowed}"
            )
        for key, tensor in hils_attention.state_dict().items():
            if key not in source_state or not torch.equal(tensor, source_state[key]):
                raise RuntimeError(
                    f"Shared HiLS tensor mismatch at layer {layer_idx}: {key}"
                )

        decoder_layer.self_attn = hils_attention
        decoder_layer.forward = MethodType(Qwen3DecoderLayer.forward, decoder_layer)
        replaced_layers.append(int(layer_idx))

    if replaced_layers != [3, 7, 11, 15, 19, 23, 27]:
        raise RuntimeError(f"Unexpected HiLS layer placement: {replaced_layers}")
    model.model._collect_mass_aux = False
    model.config.use_cache = False
    model.eval()
    return model, {
        "adapter_key_count": 0,
        "adapter_norm_min": None,
        "adapter_norm_max": None,
        "fixed_anchor_modules": 0,
        "hils_replaced_layers": replaced_layers,
        "hils_zero_shot": True,
        "baseline_definition": (
            "Qwen3-0.6B original shared weights with the seven MASS/HiLS "
            "retrieval layers scored by HiLS landmark keys (no anchors, "
            "no additional training)"
        ),
    }


def prepare_answer_positions(input_ids, labels, insert_landmarks, chunk_size, device, lmk_fill_id):
    from utils.landmark_utils import (
        create_position_ids_with_landmarks,
        insert_special_tokens,
    )

    input_ids = input_ids.to(device)
    labels = labels.to(device)
    answer_len = labels.shape[1]
    original_len = input_ids.shape[1]
    answer_start = original_len - answer_len
    original_answer_positions = torch.arange(
        answer_start,
        answer_start + answer_len,
        device=device,
    )
    original_logit_positions = original_answer_positions - 1
    if torch.any(original_logit_positions < 0):
        raise RuntimeError(f"Invalid answer start {answer_start}")

    position_ids = None
    if insert_landmarks:
        # Must equal the model's lmk_id (= config.vocab_size) so the frozen
        # standalone lmk_embed is injected at chunk-end slots, exactly like
        # training (hils_forward inserts self.lmk_id).  The legacy hardcoded
        # 100278 (an ordinary token, '拼') silently disabled that injection.
        input_ids = insert_special_tokens(
            input_ids,
            fill_id=lmk_fill_id,
            chunk_size=chunk_size,
        )
        position_ids = create_position_ids_with_landmarks(
            None,
            original_len,
            chunk_size=chunk_size,
            device=device,
        )
        logit_positions = original_logit_positions + (
            original_logit_positions // (chunk_size - 1)
        )
    else:
        logit_positions = original_logit_positions

    logits_start = int(logit_positions.min().item())
    offsets = logit_positions - logits_start
    return input_ids, labels, position_ids, logits_start, offsets, original_len


def evaluate_task(model, tokenizer, dataset, args, task_id: int, insert_landmarks: bool):
    from data import RulerSynthesizer

    synthesizer = RulerSynthesizer(tokenizer, task_id=task_id)
    device = torch.device("cuda")
    task_names = {
        0: "single_niah",
        1: "multi_query",
        2: "variable_tracking",
        3: "frequent_words",
    }
    task_name = task_names[task_id]
    correct_tokens = 0
    total_tokens = 0
    exact_matches = 0
    elapsed_ms = []
    sample_hashes = []
    expanded_lengths = []

    for sample_idx in range(args.max_samples):
        raw_sample = dataset[sample_idx]
        batch = synthesizer.single_token_eval_collate_fn([raw_sample])
        raw_input = batch["input_ids"]
        if args.max_seq_len > raw_input.shape[1]:
            # 长度外推评测: 数据池按 8192 打包, 更长预算用相邻样本拼接
            j = sample_idx + 1
            while raw_input.shape[1] < args.max_seq_len and j < len(dataset):
                nxt = torch.from_numpy(dataset[j].astype("int64")).unsqueeze(0)
                raw_input = torch.cat([raw_input, nxt], dim=1)
                j += 1
            raw_input = raw_input[:, :args.max_seq_len]
        digest = hashlib.sha256(raw_input.numpy().tobytes()).hexdigest()[:16]
        sample_hashes.append(digest)

        input_ids, labels, position_ids, logits_start, offsets, original_len = (
            prepare_answer_positions(
                raw_input,
                batch["labels"],
                insert_landmarks,
                args.chunk_size,
                device,
                int(getattr(model, "lmk_id", model.config.vocab_size)),
            )
        )
        expanded_lengths.append(int(input_ids.shape[1]))
        cache_position = torch.arange(input_ids.shape[1], device=device)
        logits_to_keep = input_ids.shape[1] - logits_start

        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.cuda.synchronize()
            start.record()
            output = model(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=False,
                logits_to_keep=logits_to_keep,
            )
            end.record()
            torch.cuda.synchronize()
        sample_elapsed = float(start.elapsed_time(end))
        elapsed_ms.append(sample_elapsed)

        answer_logits = output.logits[:, offsets, :]
        predictions = answer_logits.argmax(dim=-1)
        valid_predictions = predictions.flatten()[:-1]
        valid_labels = labels.flatten()[:-1]
        if valid_predictions.shape != valid_labels.shape:
            raise RuntimeError(
                f"Prediction/label shape mismatch: {valid_predictions.shape} vs {valid_labels.shape}"
            )
        matches = valid_predictions.eq(valid_labels)
        sample_correct = int(matches.sum().item())
        sample_total = int(matches.numel())
        correct_tokens += sample_correct
        total_tokens += sample_total
        exact_matches += int(sample_correct == sample_total)

        print(
            json.dumps(
                {
                    "event": "sample",
                    "variant": args.variant,
                    "task_id": task_id,
                    "task": task_name,
                    "sample": sample_idx,
                    "hash": digest,
                    "original_len": original_len,
                    "model_input_len": int(input_ids.shape[1]),
                    "answer_tokens": sample_total,
                    "correct_tokens": sample_correct,
                    "exact": sample_correct == sample_total,
                    "elapsed_ms": sample_elapsed,
                    "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del output, answer_logits, predictions, input_ids, labels, position_ids

    # The first sample includes lazy operator/JIT initialization. Keep it in
    # raw timings, but use the remaining samples for steady-state throughput.
    steady = elapsed_ms[1:] if len(elapsed_ms) > 1 else elapsed_ms
    effective_tokens_per_s = [
        expanded_lengths[i] * 1000.0 / elapsed_ms[i]
        for i in range(1 if len(elapsed_ms) > 1 else 0, len(elapsed_ms))
    ]
    return {
        "task_id": task_id,
        "task": task_name,
        "samples": args.max_samples,
        "token_accuracy": correct_tokens / total_tokens if total_tokens else 0.0,
        "exact_match": exact_matches / args.max_samples,
        "correct_tokens": correct_tokens,
        "total_tokens": total_tokens,
        "sample_hashes": sample_hashes,
        "model_input_len": expanded_lengths[0],
        "first_sample_ms": elapsed_ms[0],
        "steady_median_ms": statistics.median(steady),
        "steady_mean_ms": statistics.mean(steady),
        "steady_median_effective_tokens_per_s": statistics.median(effective_tokens_per_s),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("mass", "fixed0", "bsa", "dsa", "hils", "hils_t", "full", "mass_lora", "hils_lora", "bsa_lora"),
        required=True,
    )
    parser.add_argument("--mass-config", required=True)
    parser.add_argument("--mass-model", required=True)
    parser.add_argument("--original-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_device(0)
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.original_model,
        trust_remote_code=True,
    )

    if args.variant == "hils_t":
        # Trained-HiLS control: zero-shot HiLS swap + trained lmk_embed adapter.
        model, load_metadata = load_hils_model(args)
        from veomni.checkpoint import ckpt_to_state_dict

        adapter_state = ckpt_to_state_dict(args.checkpoint)
        if set(adapter_state) != {"model.lmk_embed"}:
            raise RuntimeError(
                f"hils_t checkpoint must contain exactly model.lmk_embed, "
                f"got {sorted(adapter_state)}"
            )
        lmk = model.model.lmk_embed
        loaded = adapter_state["model.lmk_embed"].to(device=lmk.device, dtype=lmk.dtype)
        with torch.no_grad():
            lmk.copy_(loaded)
        if not torch.equal(model.model.lmk_embed.detach(), loaded):
            raise RuntimeError("Failed to apply trained lmk_embed")
        load_metadata = dict(load_metadata)
        load_metadata["hils_zero_shot"] = False
        load_metadata["adapter_key_count"] = 1
        load_metadata["adapter_norm_min"] = float(
            model.model.lmk_embed.detach().float().norm().cpu()
        )
        load_metadata["adapter_norm_max"] = load_metadata["adapter_norm_min"]
        load_metadata["baseline_definition"] = (
            "HiLS landmark scoring with the lmk_embed adapter trained under the "
            "same frozen-backbone recipe as the MASS anchors"
        )
        insert_landmarks = True
    elif args.variant == "mass_lora":
        model, load_metadata = load_mass_lora_model(args)
        insert_landmarks = True
    elif args.variant == "hils_lora":
        model, load_metadata = load_hils_lora_model(args)
        insert_landmarks = True
    elif args.variant == "bsa_lora":
        model, load_metadata = load_bsa_lora_model(args)
        insert_landmarks = True
    elif args.variant == "full":
        model, load_metadata = load_full_model(args)
        insert_landmarks = True
    elif args.variant == "hils":
        model, load_metadata = load_hils_model(args)
        insert_landmarks = True
    elif args.variant in ("bsa", "dsa"):
        # dsa = exact per-token block-mass scoring (NSA/DSA-style);
        # bsa = block-mean pooled scoring.  Same weights/budget/layout.
        model, load_metadata = load_bsa_model(
            args, score_mode="mean" if args.variant == "bsa" else "exact"
        )
        # Use the same expanded sequence, position ids, and masked chunk-end
        # landmark layout as MASS.  This makes the value-KV budget exact, not
        # merely equal in the nominal W/K/S hyperparameters.
        insert_landmarks = True
    else:
        model, load_metadata = load_mass_model(
            args,
            fixed_anchor_zero=args.variant == "fixed0",
        )
        insert_landmarks = True

    from data import build_numpy_dataset

    dataset = build_numpy_dataset(
        args.corpus,
        args.max_seq_len,
        namespace="test",
        sort_files=True,
    )
    results = []
    for task_id in args.tasks:
        results.append(
            evaluate_task(model, tokenizer, dataset, args, task_id, insert_landmarks)
        )

    all_correct = sum(result["correct_tokens"] for result in results)
    all_tokens = sum(result["total_tokens"] for result in results)
    summary = {
        "variant": args.variant,
        "seed": args.seed,
        "raw_max_seq_len": args.max_seq_len,
        "max_samples_per_task": args.max_samples,
        "tasks": results,
        "macro_token_accuracy": statistics.mean(r["token_accuracy"] for r in results),
        "micro_token_accuracy": all_correct / all_tokens if all_tokens else 0.0,
        "macro_exact_match": statistics.mean(r["exact_match"] for r in results),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "wall_seconds": time.time() - started,
        "load_metadata": load_metadata,
        "attention_budget": {
            "local_window": int(model.config.hils_sliding_window),
            "remote_topk_chunks": int(model.config.hils_topk),
            "chunk_size": int(model.config.chunk_size),
            "nominal_max_kv_tokens_per_query": int(
                model.config.hils_sliding_window
                + model.config.hils_topk * model.config.chunk_size
            ),
            "kv_budget_note": (
                "MASS, fixed0, and BSA use identical landmark-expanded inputs, "
                "position ids, local window, remote top-k chunks, and masked "
                "chunk-end landmark value keys. Only retrieval scoring differs."
            ),
        },
        "mass_config": args.mass_config,
        "model_path": args.mass_model,
        "checkpoint": args.checkpoint if args.variant != "bsa" else None,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "summary", **summary}, sort_keys=True), flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
