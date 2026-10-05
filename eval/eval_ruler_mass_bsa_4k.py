#!/usr/bin/env python3
"""Deterministic 4K RULER evaluation for MASS and equal-budget BSA."""

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
        if fixed_modules != 8:
            raise RuntimeError(f"Expected 8 MASS layers, fixed {fixed_modules}")

    model.config.use_cache = False
    model.eval()
    return model, {
        "adapter_key_count": len(provided),
        "adapter_norm_min": min(loaded_norms.values()),
        "adapter_norm_max": max(loaded_norms.values()),
        "fixed_anchor_modules": fixed_modules,
    }


def load_bsa_model(args):
    """Build an exact-score BSA baseline from the identical MASS base weights.

    Only the eight retrieval attention modules are replaced.  The Q/K/V/O and
    QK-normalization tensors are copied from the matching MASS layers, while
    MLPs, embeddings, norms, and the remaining 24 layers stay bit-identical.
    """
    from models.MASSIndex import modeling_registry as _mass_registry  # noqa: F401
    from models.FlashHiLS.modeling_olmo_hils import (
        Olmo3DecoderLayer,
        Olmo3RMSNorm,
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
            norm_cls=Olmo3RMSNorm,
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
        decoder_layer.forward = MethodType(Olmo3DecoderLayer.forward, decoder_layer)
        # MASS retrieval layers use HoPE; keep that position encoding constant.
        decoder_layer.use_hils_rotary_embedding = True
        replaced_layers.append(int(layer_idx))
        copied_keys[str(layer_idx)] = sorted(bsa_attention.state_dict())

    if replaced_layers != [3, 7, 11, 15, 19, 23, 27, 31]:
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
        "bsa_uses_hope_on_replaced_layers": True,
        "bsa_masks_chunk_end_landmarks": bool(model.config.mask_lmk_token),
    }


def prepare_answer_positions(input_ids, labels, insert_landmarks, chunk_size, device):
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
        input_ids = insert_special_tokens(
            input_ids,
            fill_id=100278,
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
    task_names = {0: "single_niah", 1: "multi_query", 2: "variable_tracking"}
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
        digest = hashlib.sha256(raw_input.numpy().tobytes()).hexdigest()[:16]
        sample_hashes.append(digest)

        input_ids, labels, position_ids, logits_start, offsets, original_len = (
            prepare_answer_positions(
                raw_input,
                batch["labels"],
                insert_landmarks,
                args.chunk_size,
                device,
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
    parser.add_argument("--variant", choices=("mass", "fixed0", "bsa"), required=True)
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

    if args.variant == "bsa":
        model, load_metadata = load_bsa_model(args)
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
