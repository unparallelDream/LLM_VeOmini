#!/usr/bin/env bash
set -euo pipefail

cd /9950backfile/zhangem/HiLS-Attention-main

PYTHON=/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin/python
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
MASS_CONFIG=/9950backfile/zhangem/HiLS-Attention-main/configs/mass_attention/config_mass_olmo3_7B_8KA2K_HoPE_m8_r8.json
MASS_MODEL=/9950backfile/zhangem/.cache/modelscope/models/allenai--Olmo-3-7B-Think-MASS-m8/snapshots/master
ORIGINAL_MODEL=/9950backfile/zhangem/.cache/modelscope/models/allenai--Olmo-3-7B-Think/snapshots/master
CHECKPOINT=/9950backfile/zhangem/outputs/mass_olmo3_7b_think_anchor_only_m8_balance_v1_gpu36/checkpoints/global_step_3000
CORPUS=/9950backfile/zhangem/data/fineweb_olmo3_data
OUTPUT_DIR=${1:?output directory is required}
MAX_SAMPLES=${MAX_SAMPLES:-20}

mkdir -p "$OUTPUT_DIR"
for variant in mass fixed0 original; do
  echo "[$(date '+%F %T')] starting ${variant}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON" eval/eval_ruler_mass_ablation.py \
    --variant "$variant" \
    --mass-config "$MASS_CONFIG" \
    --mass-model "$MASS_MODEL" \
    --original-model "$ORIGINAL_MODEL" \
    --checkpoint "$CHECKPOINT" \
    --corpus "$CORPUS" \
    --output "$OUTPUT_DIR/${variant}.json" \
    --tasks 0 1 2 \
    --max-samples "$MAX_SAMPLES" \
    --max-seq-len 8192 \
    --seed 42 \
    2>&1 | tee "$OUTPUT_DIR/${variant}.log"
  echo "[$(date '+%F %T')] completed ${variant}"
done

"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = {name: json.loads((root / f"{name}.json").read_text()) for name in ("mass", "fixed0", "original")}
reference = rows["mass"]
for name, row in rows.items():
    for task_ref, task_row in zip(reference["tasks"], row["tasks"]):
        if task_ref["sample_hashes"] != task_row["sample_hashes"]:
            raise RuntimeError(f"Sample mismatch for {name}/{task_row['task']}")

comparison = {
    "sample_hashes_verified_equal": True,
    "variants": {
        name: {
            "macro_token_accuracy": row["macro_token_accuracy"],
            "micro_token_accuracy": row["micro_token_accuracy"],
            "macro_exact_match": row["macro_exact_match"],
            "peak_memory_gib": row["peak_memory_gib"],
            "tasks": [
                {
                    "task": task["task"],
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
(root / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
print(json.dumps(comparison, indent=2, sort_keys=True))
PY
