#!/usr/bin/env python3
"""Three-way controlled RULER probe on IDENTICAL generated samples.

mass fixkernel@70000 vs hils cont@latest vs dense cont@latest, same corpus,
same seed, same per-sample token sequence (sha256 recorded). One generation
pass; every raw sample is evaluated on all three models:

  * mass / hils: prepare_answer_positions (landmark insertion) forward,
    identical to eval_ruler_olmo_full_ckpt.py
  * dense: plain causal forward scored at the answer tail,
    identical to eval_ruler_dense_olmo.py

Filler convention unified to the mass script's PREPEND rule (corpus filler
before the context, answer stays at the tail) for all models — the dense
script's post-append variant would shift answers off the scored tail on
short samples.

Usage:
  CUDA_VISIBLE_DEVICES=3 python eval/eval_ruler_threeway_same_data.py \
    --mass-ckpt .../global_step_70000 --hils-ckpt .../global_step_80000 \
    --dense-ckpt .../global_step_95000 \
    --corpus data/dolma3_1025_3B_packed --max-seq-len 8192 \
    --output out.json [--tasks 0 1 2] [--max-samples 50]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

BASE = "/9950backfile/zhangem/.cache/modelscope/models/allenai--Olmo-3-1025-200M-HiLS-CPT/snapshots/master"
DENSE_MODEL = "/9950backfile/zhangem/.cache/modelscope/models/allenai--Olmo-3-1025-200M-pruned"
TOK = "/9950backfile/zhangem/.cache/modelscope/models/allenai--Olmo-3-1025-7B/snapshots/master"
MASS_CFG = str(PROJECT_ROOT / "configs/mass_attention/config_mass_olmo3_200M_cpt_sharedview.json")
HILS_CFG = str(PROJECT_ROOT / "configs/olmo3_200M_cpt/olmo3_200M_hils_yarn.json")
TASK_NAMES = {0: "single_niah", 1: "multi_query", 2: "variable_tracking", 3: "frequent_words"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass-ckpt", required=True)
    ap.add_argument("--hils-ckpt", required=True)
    ap.add_argument("--dense-ckpt", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from eval.eval_ruler_mass_bsa_4k import prepare_answer_positions
    from eval.eval_ruler_olmo_full_ckpt import load_full_param_model

    class _NS:
        pass

    def _load(flavor, cfg, ckpt):
        ns = _NS()
        ns.flavor, ns.config, ns.base_weights, ns.checkpoint = flavor, cfg, BASE, ckpt
        return load_full_param_model(ns)

    print("[load] mass ...", flush=True)
    mass_model, _ = _load("mass", MASS_CFG, args.mass_checkpoint if False else args.mass_ckpt)
    print("[load] hils ...", flush=True)
    hils_model, _ = _load("hils", HILS_CFG, args.hils_ckpt)

    print("[load] dense ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dense_model = AutoModelForCausalLM.from_pretrained(
        DENSE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    dense_model.eval()
    from veomni.checkpoint import ckpt_to_state_dict
    sd = ckpt_to_state_dict(args.dense_ckpt)
    keep = {k: v for k, v in sd.items() if not k.startswith(("optimizer", "dataloader", "lr_scheduler", "extra"))}
    incompat = dense_model.load_state_dict(keep, strict=False)
    print(f"[load] dense {len(keep) - len(incompat.missing_keys)} tensors", flush=True)

    from data import RulerSynthesizer, build_numpy_dataset

    tokenizer = AutoTokenizer.from_pretrained(TOK)
    dataset = build_numpy_dataset(args.corpus, args.max_seq_len, namespace="test", sort_files=True)

    models = {"mass": mass_model, "hils": hils_model, "dense": dense_model}
    results = {m: {} for m in models}
    hashes_all = {}

    for task_id in args.tasks:
        synth = RulerSynthesizer(tokenizer, task_id=task_id)
        stats = {m: [0, 0, 0] for m in models}  # correct, total, exact
        hashes = []
        t0 = time.time()
        for idx in range(args.max_samples):
            batch = synth.single_token_eval_collate_fn([dataset[idx]])
            raw = batch["input_ids"]
            labels_cpu = batch["labels"]
            if raw.shape[1] < args.max_seq_len:
                need = args.max_seq_len - raw.shape[1]
                j = (idx + 1) % len(dataset)
                filler = torch.from_numpy(dataset[j].astype("int64"))[:need]
                while filler.shape[0] < need:
                    j += 1
                    filler = torch.cat([filler, torch.from_numpy(dataset[j % len(dataset)].astype("int64"))])
                raw = torch.cat([filler.unsqueeze(0), raw], dim=1)
            if raw.shape[1] > args.max_seq_len:
                raw = raw[:, -args.max_seq_len:]  # keep the tail (question+answer)
            hashes.append(hashlib.sha256(raw.numpy().tobytes()).hexdigest()[:16])
            answer_len = labels_cpu.shape[1]

            for name, model in models.items():
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    if name == "dense":
                        input_ids = raw.cuda()
                        pos = torch.arange(input_ids.shape[1] - answer_len - 1, input_ids.shape[1] - 1, device="cuda")
                        logits = model(input_ids=input_ids).logits[:, pos, :]
                        preds = logits.argmax(dim=-1)
                        matches = preds.eq(labels_cpu.cuda())
                    else:
                        input_ids, labels_dev, position_ids, logits_start, offsets, _ = (
                            prepare_answer_positions(
                                raw, labels_cpu, True, args.chunk_size, torch.device("cuda")
                            )
                        )
                        cache_position = torch.arange(input_ids.shape[1], device="cuda")
                        logits_to_keep = input_ids.shape[1] - logits_start
                        out = model(
                            input_ids=input_ids,
                            position_ids=position_ids,
                            cache_position=cache_position,
                            use_cache=False,
                            logits_to_keep=logits_to_keep,
                        )
                        preds = out.logits[:, offsets, :].argmax(dim=-1)
                        vp = preds.flatten()[:-1]
                        vl = labels_dev.flatten()[:-1]
                        matches = vp.eq(vl)
                c = int(matches.sum())
                t = int(matches.numel())
                stats[name][0] += c
                stats[name][1] += t
                stats[name][2] += int(c == t)
        hashes_all[TASK_NAMES[task_id]] = hashes
        for name in models:
            c, t, e = stats[name]
            results[name][TASK_NAMES[task_id]] = {
                "token_accuracy": c / max(t, 1), "exact_match": e / args.max_samples,
                "correct_tokens": c, "total_tokens": t,
            }
            print(f"[task {task_id}] {name}: TA={c/max(t,1):.4f} EM={e/args.max_samples:.3f}", flush=True)
        print(f"[task {task_id}] wall={time.time()-t0:.1f}s hashes[0:3]={hashes[:3]}", flush=True)

    out = {
        "protocol": "identical-samples threeway (single generation pass, sha256 per sample)",
        "seed": args.seed,
        "max_seq_len": args.max_seq_len,
        "max_samples": args.max_samples,
        "checkpoints": {"mass": args.mass_ckpt, "hils": args.hils_ckpt, "dense": args.dense_ckpt},
        "results": results,
        "sample_hashes": hashes_all,
    }
    Path(args.output).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"[done] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
