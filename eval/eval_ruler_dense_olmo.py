#!/usr/bin/env python3
"""RULER eval for the dense recovered OLMo-199M (no landmarks, plain causal LM).

Serves as the `full`-variant ceiling reference for the OLMo line, mirroring
what `full` does for the Qwen line.  Reuses the eval script's sample
generation (same synthesizer/seed/hashes) but loads a plain Olmo3ForCausalLM
via MODELING_BACKEND=hf.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF dir with model.safetensors (pruned base)")
    ap.add_argument("--checkpoint", required=False, default=None, help="DCP checkpoint dir of the recovery run; omit for zero-shot HF weights")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    model.eval()

    from veomni.checkpoint import ckpt_to_state_dict
    if args.checkpoint:
        sd = ckpt_to_state_dict(args.checkpoint)
        keep = {k: v for k, v in sd.items() if not k.startswith(("optimizer", "dataloader", "lr_scheduler", "extra"))}
        incompat = model.load_state_dict(keep, strict=False)
        loaded = len(keep) - len(incompat.missing_keys)
        print(f"[load] {loaded} tensors loaded, missing={len(incompat.missing_keys)}, unexpected={len(incompat.unexpected_keys)}")
        if incompat.unexpected_keys:
            raise RuntimeError(f"unexpected keys: {incompat.unexpected_keys[:5]}")
    else:
        print("[load] no --checkpoint given: zero-shot HF weights as-is")

    from data import RulerSynthesizer, build_numpy_dataset

    dataset = build_numpy_dataset(args.corpus, args.max_seq_len, namespace="test", sort_files=True)
    task_names = {0: "single_niah", 1: "multi_query", 2: "variable_tracking", 3: "frequent_words"}
    results = []
    for task_id in args.tasks:
        synth = RulerSynthesizer(tokenizer, task_id=task_id)
        correct_tokens = total_tokens = exact = 0
        hashes = []
        t0 = time.time()
        for idx in range(args.max_samples):
            batch = synth.single_token_eval_collate_fn([dataset[idx]])
            raw = batch["input_ids"]
            if raw.shape[1] < args.max_seq_len:
                j = idx + 1
                while raw.shape[1] < args.max_seq_len and j < len(dataset):
                    nxt = torch.from_numpy(dataset[j].astype("int64")).unsqueeze(0)
                    raw = torch.cat([raw, nxt], dim=1)
                    j += 1
                raw = raw[:, : args.max_seq_len]
            hashes.append(hashlib.sha256(raw.numpy().tobytes()).hexdigest()[:16])
            input_ids = raw.cuda()
            labels = batch["labels"].cuda()
            answer_len = labels.shape[1]
            answer_logits = torch.arange(
                raw.shape[1] - answer_len - 1, raw.shape[1] - 1, device="cuda"
            )
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids).logits[:, answer_logits, :]
                preds = logits.argmax(dim=-1)
            matches = preds.eq(labels)
            correct_tokens += int(matches.sum())
            total_tokens += labels.numel()
            exact += int(matches.all())
        results.append(
            {
                "task": task_names[task_id],
                "task_id": task_id,
                "token_accuracy": correct_tokens / total_tokens,
                "exact_match": exact / args.max_samples,
                "correct_tokens": correct_tokens,
                "total_tokens": total_tokens,
                "sample_hashes": hashes,
                "samples": args.max_samples,
                "wall_seconds": time.time() - t0,
            }
        )
        print(f"[task {task_id}] {task_names[task_id]}: acc={correct_tokens/total_tokens:.4f} em={exact/args.max_samples:.3f}", flush=True)

    out = {
        "variant": "dense_recovered_olmo200m",
        "model": args.model,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "raw_max_seq_len": args.max_seq_len,
        "max_samples_per_task": args.max_samples,
        "tasks": results,
        "macro_token_accuracy": statistics.mean(r["token_accuracy"] for r in results),
        "macro_exact_match": statistics.mean(r["exact_match"] for r in results),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=1)
    print("WROTE", args.output)


if __name__ == "__main__":
    main()
