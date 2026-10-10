"""RULER eval for FULL-PARAM OLMo checkpoints (olmo_hils & olmo_mass).

Unlike eval_ruler_mass_bsa_4k.py (anchor-only adapter checkpoints), this loads
DCP step checkpoints that contain every trainable parameter of the full CPT
runs (e.g. olmo200m_hils_cpt_ft15000 / olmo200m_mass_cpt_ft15000), and extends
the task set to the full four-task RULER suite (incl. task 3 frequent_words).

The scoring loop is task-agnostic (token accuracy + exact match), mirroring
eval_ruler_mass_bsa_4k.evaluate_task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time

import torch
from transformers import AutoTokenizer

from data import build_numpy_dataset
from data.data_transform import RulerSynthesizer
from eval.eval_ruler_mass_bsa_4k import prepare_answer_positions, set_seed

TASK_NAMES = {
    0: "single_niah",
    1: "multi_query",
    2: "variable_tracking",
    3: "frequent_words",
}


def load_full_param_model(args):
    if args.flavor == "hils":
        import models.FlashHiLS  # noqa: F401  (registers olmo_hils)
    else:
        from models.MASSIndex import modeling_registry  # noqa: F401  (registers olmo_mass)
    from veomni.checkpoint import ckpt_to_state_dict
    from veomni.models import build_foundation_model

    model = build_foundation_model(
        config_path=args.config,
        weights_path=args.base_weights,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
        init_device="cuda",
    )

    if args.checkpoint is None:
        print("[load] zero-shot: base weights + freshly initialized MASS modules", flush=True)
        if hasattr(model.model, "_collect_mass_aux"):
            model.model._collect_mass_aux = False
        model.config.use_cache = False
        model.eval()
        return model, {"flavor": args.flavor, "checkpoint": None}
    state = ckpt_to_state_dict(args.checkpoint)
    params = dict(model.named_parameters())
    # 200M runs started before the FSDP2 scalar-param fix: their DCP ckpts
    # store shared_view_bias as a 0-dim tensor while this build uses (1,).
    adapted = 0
    for key in list(state.keys()):
        param = params.get(key)
        if param is not None and state[key].shape != param.shape and state[key].numel() == param.numel():
            state[key] = state[key].view(param.shape)
            adapted += 1
    if adapted:
        print(f"[load] shape-adapted {adapted} numel-matched tensor(s) (scalar<->(1,))", flush=True)
    param_names = set(params)
    missing = sorted(param_names - set(state))
    unexpected = sorted(set(state) - param_names)
    if missing:
        raise RuntimeError(f"checkpoint missing model parameters: {missing[:8]}...")
    incompat = model.load_state_dict(state, strict=False)
    if incompat.unexpected_keys:
        raise RuntimeError(f"unexpected DCP keys: {incompat.unexpected_keys[:8]}...")
    loaded = sum(1 for n in param_names if n in state)
    print(f"[load] {loaded}/{len(param_names)} parameter tensors loaded from {args.checkpoint}", flush=True)

    if hasattr(model.model, "_collect_mass_aux"):
        model.model._collect_mass_aux = False
    model.config.use_cache = False
    model.eval()
    return model, {
        "flavor": args.flavor,
        "config": args.config,
        "base_weights": args.base_weights,
        "checkpoint": args.checkpoint,
        "param_tensors_loaded": loaded,
        "param_tensors_total": len(param_names),
        "unexpected_keys": unexpected,
    }


def evaluate_task(model, tokenizer, dataset, args, task_id: int):
    synthesizer = RulerSynthesizer(tokenizer, task_id=task_id)
    device = torch.device("cuda")
    task_name = TASK_NAMES[task_id]
    correct_tokens = 0
    total_tokens = 0
    exact_matches = 0
    elapsed_ms = []

    filler_pool = None
    for sample_idx in range(args.max_samples):
        raw_sample = dataset[sample_idx]
        batch = synthesizer.single_token_eval_collate_fn([raw_sample])
        raw_input = batch["input_ids"]
        if args.pad_fixed_len and raw_input.shape[1] < args.pad_fixed_len:
            # prepend corpus filler so every sample hits the same kernel shape;
            # all positions shift equally, keeping question/answer/needle relative
            # structure intact (filler sits before the context).
            need = args.pad_fixed_len - raw_input.shape[1]
            j = (sample_idx + 1) % len(dataset)
            filler = torch.from_numpy(dataset[j].astype("int64"))[:need]
            while filler.shape[0] < need:
                j += 1
                filler = torch.cat([filler, torch.from_numpy(dataset[j % len(dataset)].astype("int64"))])
            raw_input = torch.cat([filler.unsqueeze(0), raw_input], dim=1)

        input_ids, labels, position_ids, logits_start, offsets, original_len = (
            prepare_answer_positions(
                raw_input,
                batch["labels"],
                True,  # insert_landmarks: same masked chunk-end layout as training
                args.chunk_size,
                device,
            )
        )
        cache_position = torch.arange(input_ids.shape[1], device=device)
        logits_to_keep = input_ids.shape[1] - logits_start

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
        elapsed_ms.append(float(start.elapsed_time(end)))

        answer_logits = output.logits[:, offsets, :]
        predictions = answer_logits.argmax(dim=-1)
        valid_predictions = predictions.flatten()[:-1]
        valid_labels = labels.flatten()[:-1]
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
                    "task": task_name,
                    "sample": sample_idx,
                    "answer_tokens": sample_total,
                    "correct_tokens": sample_correct,
                    "exact": sample_correct == sample_total,
                }
            ),
            flush=True,
        )
        del output, answer_logits, predictions, input_ids, labels, position_ids

    return {
        "task_id": task_id,
        "task": task_name,
        "correct_tokens": correct_tokens,
        "total_tokens": total_tokens,
        "token_accuracy": correct_tokens / total_tokens if total_tokens else 0.0,
        "exact_match": exact_matches / args.max_samples,
        "mean_ms": statistics.mean(elapsed_ms) if elapsed_ms else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flavor", choices=("hils", "mass"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-weights", required=True)
    parser.add_argument("--checkpoint", required=False, default=None, help="omit for zero-shot base+random MASS modules (kernel debugging)")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pad-fixed-len", type=int, default=0,
                        help="Pad every sample to this exact raw length with corpus filler "
                             "prepended at the front (constant kernel shape: tilelang "
                             "compiles once instead of per-sample-length).")
    parser.add_argument("--lse-temperature", type=float, default=None,
                        help="Override mass_config.lse_temperature after load "
                             "(mid-training probes must use the annealed tau at that step).")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_device(0)
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        "/9950backfile/zhangem/.cache/modelscope/models/allenai--Olmo-3-1025-7B/snapshots/master",
        trust_remote_code=True,
    )
    model, load_metadata = load_full_param_model(args)
    if args.lse_temperature is not None:
        n = 0
        for _m in model.modules():
            _mc = getattr(_m, "mass_config", None)
            if _mc is not None:
                object.__setattr__(_mc, "lse_temperature", args.lse_temperature)
                n += 1
        print(f"[tau] overrode lse_temperature={args.lse_temperature} on {n} modules", flush=True)

    dataset = build_numpy_dataset(args.corpus, args.max_seq_len, namespace="test", sort_files=True)
    results = [evaluate_task(model, tokenizer, dataset, args, t) for t in args.tasks]

    all_correct = sum(r["correct_tokens"] for r in results)
    all_tokens = sum(r["total_tokens"] for r in results)
    summary = {
        "variant": f"{args.flavor}_fullcpt",
        "checkpoint": args.checkpoint,
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
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps({k: summary[k] for k in
                      ("variant", "checkpoint", "macro_token_accuracy", "macro_exact_match")}), flush=True)


if __name__ == "__main__":
    main()
