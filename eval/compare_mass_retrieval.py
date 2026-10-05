#!/usr/bin/env python3
"""Compare MASS retrieval/output signatures for two anchor checkpoints."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(config: str, weights: str, checkpoint: str):
    from models.MASSIndex import modeling_registry as _registry  # noqa: F401
    from veomni.checkpoint import ckpt_to_state_dict
    from veomni.models import build_foundation_model

    model = build_foundation_model(
        config_path=config,
        weights_path=weights,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
        init_device="cuda",
    )
    state = ckpt_to_state_dict(checkpoint)
    expected = {
        name for name, _ in model.named_parameters()
        if ".mass_anchor_generator." in name
    }
    if set(state) != expected:
        raise RuntimeError(
            f"adapter key mismatch: missing={sorted(expected-set(state))}, "
            f"unexpected={sorted(set(state)-expected)}"
        )
    incompat = model.load_state_dict(state, strict=False)
    if incompat.unexpected_keys:
        raise RuntimeError(f"unexpected adapter keys: {incompat.unexpected_keys}")
    model.config.use_cache = False
    model.eval()
    return model


def prepare_sample(tokenizer, corpus: str, max_seq_len: int, task_id: int):
    from data import RulerSynthesizer, build_numpy_dataset
    from utils.landmark_utils import create_position_ids_with_landmarks, insert_special_tokens

    dataset = build_numpy_dataset(
        corpus, max_seq_len, namespace="test", sort_files=True
    )
    synth = RulerSynthesizer(tokenizer, task_id=task_id)
    batch = synth.single_token_eval_collate_fn([dataset[0]])
    raw = batch["input_ids"].cuda()
    labels = batch["labels"].cuda()
    original_len = raw.shape[1]
    answer_len = labels.shape[1]
    raw_answer_logits = torch.arange(
        original_len - answer_len - 1,
        original_len - 1,
        device="cuda",
    )
    model_input = insert_special_tokens(raw, fill_id=100278, chunk_size=64)
    position_ids = create_position_ids_with_landmarks(
        None, original_len, chunk_size=64, device=torch.device("cuda")
    )
    answer_logits = raw_answer_logits + raw_answer_logits // 63
    return model_input, position_ids, labels, answer_logits


def run_one(model, model_input, position_ids, answer_positions):
    captures: dict[int, dict[str, object]] = {}
    handles = []
    sketch_positions = torch.unique(
        torch.cat(
            [
                torch.arange(0, model_input.shape[1], 128, device="cuda"),
                answer_positions,
            ]
        )
    )

    for layer_idx in model.model._mass_layer_indices:
        module = model.model.layers[layer_idx].self_attn
        original = module.build_mass_retrieval_plan

        def wrapped(self, *args, _original=original, _layer=int(layer_idx), **kwargs):
            result = _original(*args, **kwargs)
            finite = torch.isfinite(result.selected_scores)
            captures.setdefault(_layer, {})["indices"] = (
                result.indices.detach().cpu().to(torch.int16)
            )
            captures[_layer]["scores"] = (
                result.selected_scores.detach().cpu().to(torch.float32)
            )
            captures[_layer]["score_finite"] = finite.detach().cpu()
            captures[_layer]["anchor_load"] = (
                result.anchor_load.detach().cpu().float()
            )
            # Quantify the learned perturbation relative to the actual dynamic
            # base anchor used for this RULER sample.
            base_anchor = args[0][
                :, self.chunk_size - 1 :: self.chunk_size, :, :
            ]
            full_chunks = args[1].shape[1] // self.chunk_size
            base_anchor = base_anchor[:, :full_chunks]
            anchors = self.mass_anchor_generator(base_anchor)
            offsets = torch.einsum(
                "mr,hdr->mhd",
                self.mass_anchor_generator.anchor_embedding,
                self.mass_anchor_generator.offset_basis,
            )
            base_norm = base_anchor.float().norm()
            captures[_layer]["effective_scale"] = {
                "base_anchor_norm": float(base_norm),
                "offset_norm": float(offsets.float().norm()),
                "offset_to_base_ratio": float(offsets.float().norm() / base_norm),
                "normalized_anchor_pairwise_mean": float(
                    torch.pdist(anchors[0, -1].float().flatten(1)).mean()
                ),
                "normalized_anchor_norm_mean": float(
                    anchors[0, -1].float().flatten(1).norm(dim=1).mean()
                ),
            }
            return result

        module.build_mass_retrieval_plan = MethodType(wrapped, module)

        def output_hook(_module, _inputs, output, _layer=int(layer_idx)):
            captures.setdefault(_layer, {})["output_sketch"] = (
                output[0][:, sketch_positions].detach().cpu().float()
            )

        handles.append(module.register_forward_hook(output_hook))

    cache_position = torch.arange(model_input.shape[1], device="cuda")
    logits_start = int(answer_positions.min())
    offsets = answer_positions - logits_start
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            input_ids=model_input,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=False,
            logits_to_keep=model_input.shape[1] - logits_start,
        )
    answer_logits = output.logits[:, offsets].detach().cpu().float()
    for handle in handles:
        handle.remove()
    return captures, answer_logits


def compare_indices(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    active = a.ge(0).any(dim=-1) | b.ge(0).any(dim=-1)
    row_equal = a.eq(b).all(dim=-1)
    slot_valid = a.ge(0) | b.ge(0)
    slot_equal = a.eq(b) & slot_valid
    intersections = torch.zeros_like(row_equal, dtype=torch.int32)
    for k in range(a.shape[-1]):
        valid = a[..., k].ge(0)
        intersections += (
            a[..., k, None].eq(b).any(dim=-1) & valid
        ).to(torch.int32)
    size_a = a.ge(0).sum(dim=-1)
    size_b = b.ge(0).sum(dim=-1)
    unions = size_a + size_b - intersections
    jaccard = intersections.float() / unions.clamp_min(1).float()
    return {
        "active_rows": int(active.sum()),
        "active_row_exact_match_fraction": float(row_equal[active].float().mean()),
        "valid_slot_exact_match_fraction": float(
            slot_equal.sum().float() / slot_valid.sum().clamp_min(1)
        ),
        "mean_set_jaccard": float(jaccard[active].mean()),
    }


def tensor_delta(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    x, y = a.float().flatten(), b.float().flatten()
    d = y - x
    return {
        "relative_l2": float(d.norm() / x.norm().clamp_min(1e-30)),
        "cosine": float(F.cosine_similarity(x, y, dim=0)),
        "abs_max": float(d.abs().max()),
        "abs_mean": float(d.abs().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--step-a", required=True)
    parser.add_argument("--step-b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--task-id", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    set_seed(42)
    torch.cuda.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model_input, position_ids, labels, answer_positions = prepare_sample(
        tokenizer, args.corpus, args.max_seq_len, args.task_id
    )

    model_a = load_model(args.config, args.weights, args.step_a)
    cap_a, logits_a = run_one(model_a, model_input, position_ids, answer_positions)
    del model_a
    torch.cuda.empty_cache()

    model_b = load_model(args.config, args.weights, args.step_b)
    cap_b, logits_b = run_one(model_b, model_input, position_ids, answer_positions)
    del model_b
    torch.cuda.empty_cache()

    layers = []
    for layer in sorted(cap_a):
        a, b = cap_a[layer], cap_b[layer]
        same = a["indices"].eq(b["indices"])
        aligned = same.repeat_interleave(1, dim=2) & a["score_finite"] & b["score_finite"]
        score_delta = (
            tensor_delta(a["scores"][aligned], b["scores"][aligned])
            if aligned.any()
            else None
        )
        layers.append(
            {
                "layer": layer,
                "retrieval": compare_indices(a["indices"], b["indices"]),
                "aligned_selected_score_delta": score_delta,
                "anchor_load_a": [float(x) for x in a["anchor_load"]],
                "anchor_load_b": [float(x) for x in b["anchor_load"]],
                "anchor_load_l1": float((a["anchor_load"] - b["anchor_load"]).abs().sum()),
                "attention_output_sketch_delta": tensor_delta(
                    a["output_sketch"], b["output_sketch"]
                ),
                "effective_scale_a": a["effective_scale"],
                "effective_scale_b": b["effective_scale"],
            }
        )

    labels_valid = labels.flatten()[:-1].cpu()
    pred_a = logits_a.argmax(dim=-1).flatten()[:-1]
    pred_b = logits_b.argmax(dim=-1).flatten()[:-1]
    result = {
        "step_a": args.step_a,
        "step_b": args.step_b,
        "task_id": args.task_id,
        "raw_length": args.max_seq_len,
        "model_input_length": int(model_input.shape[1]),
        "layers": layers,
        "answer_logits_delta": tensor_delta(logits_a, logits_b),
        "prediction_ids_a": [int(x) for x in pred_a],
        "prediction_ids_b": [int(x) for x in pred_b],
        "predictions_equal": bool(torch.equal(pred_a, pred_b)),
        "correct_a": int(pred_a.eq(labels_valid).sum()),
        "correct_b": int(pred_b.eq(labels_valid).sum()),
        "answer_token_count": int(labels_valid.numel()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
