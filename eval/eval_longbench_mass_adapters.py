"""LongBench v1 evaluation for the 90M adapter models (mass_lora / hils_lora).

Reuses prompt templates, metrics and middle-truncation from
``eval_longbench_v1``; model loading reuses the adapter loaders from
``eval_ruler_qwen_mass_bsa_4k``.  Generation relies on the model's
generate-time chunk-end landmark insertion (gated by ``insert_landmarks``;
the HiLS prepare_inputs_for_generation path inserts on prefill exactly like
the training-time layout).

English F1/classification subset only (no jieba/rouge deps in env):
narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique triviaqa trec.

Usage:
  HF_ENDPOINT=https://hf-mirror.com python eval/eval_longbench_mass_adapters.py \
    --variant mass_lora --checkpoint <ckpt> --output out.json \
    --datasets narrativeqa,... --max-length 8192 --max-samples 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from eval.eval_longbench_v1 import (
    DATASET2MAXGEN,
    DATASET2PROMPT,
    classification_score,
    qa_f1_score,
    truncate_middle,
)
from eval.eval_ruler_qwen_mass_bsa_4k import load_hils_lora_model, load_mass_lora_model

METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "triviaqa": qa_f1_score,
    "trec": classification_score,
}


class _Args:
    """Minimal namespace understood by the shared adapter loaders."""

    def __init__(self, variant, config, model_path, checkpoint):
        self.variant = variant
        self.mass_config = config
        self.mass_model = model_path
        self.original_model = (
            "/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master"
        )
        self.checkpoint = checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("mass_lora", "hils_lora"), required=True)
    parser.add_argument(
        "--mass-config",
        default="configs/mass_attention/config_mass_qwen3_90M_8KA2K_RoPE_m2_r8_asym0p5_anneal.json",
    )
    parser.add_argument(
        "--mass-model",
        default="/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-90M-MASS-m2-recovered/snapshots/master",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--original-model",
        default="/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master",
    )
    parser.add_argument("--datasets", default=",".join(METRIC))
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument(
        "--max-new-cap",
        type=int,
        default=0,
        help="cap generation length (0 = official per-task max_gen)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    loader = load_mass_lora_model if args.variant == "mass_lora" else load_hils_lora_model
    model, load_metadata = loader(_Args(args.variant, args.mass_config, args.mass_model, args.checkpoint))
    tokenizer = AutoTokenizer.from_pretrained(args.original_model, trust_remote_code=True)

    results = {}
    raw_predictions = {}
    local_dir = Path(
        os.environ.get(
            "LONG BENCH_LOCAL_DIR", "/9950backfile/zhangem/data/longbench/data"
        ).replace(" ", "")
    )
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        t0 = time.time()
        local_file = local_dir / f"{name}.jsonl"
        if local_file.is_file():
            rows = [
                json.loads(line)
                for line in local_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            rows = list(load_dataset("THUDM/LongBench", name, split="test"))
        n = min(args.max_samples, len(rows))
        scores, preds = [], []
        max_gen = DATASET2MAXGEN[name]
        if args.max_new_cap > 0:
            max_gen = min(max_gen, args.max_new_cap)
        for i in range(n):
            sample = rows[i]
            prompt = DATASET2PROMPT[name].format(
                context=sample["context"], input=sample["input"]
            )
            if sample.get("length", 0) > 0:
                pass  # keep official template; truncation below is token-based
            prompt = truncate_middle(tokenizer, prompt, args.max_length)
            inputs = tokenizer(
                prompt, truncation=False, return_tensors="pt"
            ).input_ids.cuda()
            with torch.inference_mode():
                out = model.generate(
                    inputs,
                    max_new_tokens=max_gen,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            pred = tokenizer.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)
            gold = sample["answers"]
            if isinstance(sample["answers"], str):
                gold = [sample["answers"]]
            score = max(METRIC[name](pred, g) for g in gold)
            scores.append(score)
            preds.append({"pred": pred, "gold": gold, "score": score})
        results[name] = {
            "n": n,
            "score": sum(scores) / len(scores),
            "max_length": args.max_length,
        }
        raw_predictions[name] = preds
        print(
            f"[lb] {name}: score={results[name]['score']:.4f} (n={n}, "
            f"{time.time()-t0:.0f}s)",
            flush=True,
        )

    summary = {
        "variant": args.variant,
        "checkpoint": args.checkpoint,
        "max_length": args.max_length,
        "macro": sum(r["score"] for r in results.values()) / max(1, len(results)),
        "datasets": results,
        "load_metadata": {
            k: load_metadata[k]
            for k in ("adapter_key_count", "lora_param_count", "baseline_definition")
            if k in load_metadata
        },
        "predictions": raw_predictions,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v["score"] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
