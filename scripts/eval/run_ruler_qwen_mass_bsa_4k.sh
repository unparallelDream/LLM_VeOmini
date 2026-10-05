#!/usr/bin/env bash
set -euo pipefail

cd /9950backfile/zhangem/HiLS-Attention-main

PYTHON=/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin/python
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

MASS_CONFIG=/9950backfile/zhangem/HiLS-Attention-main/configs/mass_attention/config_mass_qwen3_0p6B_8KA2K_RoPE_m2_r8_lse_norm_tau1.json
MASS_MODEL=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B-MASS-m2/snapshots/master
ORIGINAL_MODEL=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master
CHECKPOINT=${CHECKPOINT:-/9950backfile/zhangem/outputs/mass_qwen3_0p6b_anchor_only_m2_lse_norm_tau1_balance_mb4_gpu07_v2/checkpoints/global_step_5000}
CORPUS=/9950backfile/zhangem/data/fineweb_qwen3_data
OUTPUT_DIR=${1:?output directory is required}
MAX_SAMPLES=${MAX_SAMPLES:-50}
EVAL_GPU=${EVAL_GPU:-0}

test -f "$CHECKPOINT/COMPLETED.json"
mkdir -p "$OUTPUT_DIR"

for variant in ${VARIANTS:-mass fixed0 bsa}; do
  echo "[$(date '+%F %T')] starting ${variant} on physical GPU ${EVAL_GPU}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
  "$PYTHON" eval/eval_ruler_qwen_mass_bsa_4k.py \
    --variant "$variant" \
    --mass-config "$MASS_CONFIG" \
    --mass-model "$MASS_MODEL" \
    --original-model "$ORIGINAL_MODEL" \
    --checkpoint "$CHECKPOINT" \
    --corpus "$CORPUS" \
    --output "$OUTPUT_DIR/${variant}.json" \
    --tasks 0 1 2 \
    --max-samples "$MAX_SAMPLES" \
    --max-seq-len 4096 \
    --seed 42 \
    2>&1 | tee "$OUTPUT_DIR/${variant}.log"
  echo "[$(date '+%F %T')] completed ${variant}"
done

"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
names = ("mass", "fixed0", "bsa")
rows = {name: json.loads((root / f"{name}.json").read_text()) for name in names}
reference = rows["mass"]
for name, row in rows.items():
    if row["attention_budget"] != reference["attention_budget"]:
        raise RuntimeError(f"Attention budget mismatch for {name}")
    for task_ref, task_row in zip(reference["tasks"], row["tasks"]):
        if task_ref["sample_hashes"] != task_row["sample_hashes"]:
            raise RuntimeError(f"Sample mismatch for {name}/{task_row['task']}")
        if task_ref["model_input_len"] != task_row["model_input_len"]:
            raise RuntimeError(f"Model-input layout mismatch for {name}/{task_row['task']}")

comparison = {
    "checkpoint": reference["checkpoint"],
    "sample_hashes_verified_equal": True,
    "attention_budget_verified_equal": True,
    "model_input_layout_verified_equal": True,
    "attention_budget": reference["attention_budget"],
    "raw_context_length": reference["raw_max_seq_len"],
    "tasks": [task["task"] for task in reference["tasks"]],
    "variants": {
        name: {
            "label": (
                "Qwen3-0.6B original weights + equal-KV exact block-sparse attention"
                if name == "bsa" else name
            ),
            "macro_token_accuracy": row["macro_token_accuracy"],
            "micro_token_accuracy": row["micro_token_accuracy"],
            "macro_exact_match": row["macro_exact_match"],
            "peak_memory_gib": row["peak_memory_gib"],
            "wall_seconds": row["wall_seconds"],
            "load_metadata": row["load_metadata"],
            "tasks": [
                {
                    "task": task["task"],
                    "model_input_len": task["model_input_len"],
                    "token_accuracy": task["token_accuracy"],
                    "exact_match": task["exact_match"],
                    "steady_median_ms": task["steady_median_ms"],
                    "steady_median_effective_tokens_per_s": task["steady_median_effective_tokens_per_s"],
                }
                for task in row["tasks"]
            ],
        }
        for name, row in rows.items()
    },
}
(root / "comparison.json").write_text(
    json.dumps(comparison, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(comparison, indent=2, sort_keys=True))
PY
