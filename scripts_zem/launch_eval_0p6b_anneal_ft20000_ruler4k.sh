#!/usr/bin/env bash
# 0.6B anchor 退火模型(ft20000, asym0p5_anneal 配方)的 4K RULER 四变体评测:
#   mass   = 冻结基座 + 训练后的双 anchor adapter
#   fixed0 = 消融: 所有 anchor 强制复制为 anchor0(退化为单 anchor)
#   dsa    = 等预算精确块打分: 逐 token logsumexp 块质量(NSA/DSA 式, O(L^2) 打分)
#   bsa    = 等预算均值池化块打分: q·mean(k_chunk)(O(L*N) 打分, 块大小与 mass 一致)
# 四者共用同一 chunk_size=64 / window / top-k / KV 预算与样本,仅打分函数不同。
# 基于 scripts/eval/run_ruler_qwen_mass_bsa_4k.sh 独立成脚本(原脚本硬编码旧
# checkpoint 且要求 COMPLETED.json, 本 checkpointer 不写)。
set -euo pipefail

cd /9950backfile/zhangem/HiLS-Attention-main

PYTHON=/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin/python
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TMPDIR=/dev/shm/zhangem_tmp
export PYTHONPYCACHEPREFIX=/9950backfile/zhangem/tmp/pycache
mkdir -p "$TMPDIR"
unset CODEBUDDY_SAFE_DELETE_BULK_STATE_DIR CODEBUDDY_TOOL_CALL_ID \
      CODEBUDDY_SAFE_DELETE_BULK_GUARD CODEBUDDY_NODE_BIN GENIE_TRASH_DIR 2>/dev/null

MASS_CONFIG=/9950backfile/zhangem/HiLS-Attention-main/configs/mass_attention/config_mass_qwen3_0p6B_8KA2K_RoPE_m2_r8_asym0p5_anneal.json
MASS_MODEL=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B-MASS-m2/snapshots/master
ORIGINAL_MODEL=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master
CHECKPOINT=/9950backfile/zhangem/outputs/mass_qwen3_0p6b_anchor_only_m2_r8_asym0p5_anneal_ft20000/checkpoints/global_step_20000_anchor_only
CORPUS=/9950backfile/zhangem/data/fineweb_qwen3_data
OUTPUT_DIR=/9950backfile/zhangem/outputs/eval_qwen3_0p6b_mass_anneal_ft20000_ruler4k_5variants_fillfix_n50_20260905
MAX_SAMPLES=${MAX_SAMPLES:-50}
# GPU0 被 90M 训练占用; GPU3 当前空闲
EVAL_GPU=${EVAL_GPU:-3}

# anchor-only DCP(由 extract_anchor_only_ckpt.py 从 global_step_20000 提取)
ls "$CHECKPOINT"/__0_0.distcp > /dev/null
mkdir -p "$OUTPUT_DIR"

for variant in mass fixed0 dsa bsa hils; do
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
names = ("mass", "fixed0", "dsa", "bsa", "hils")
labels = {
    "mass": "trained 2-anchor MASS scoring",
    "fixed0": "collapsed single-anchor (anchor0) ablation",
    "dsa": "exact per-token block-mass scoring logsumexp(q·k) (NSA/DSA-style)",
    "bsa": "block-mean pooled scoring q·mean(k_chunk)",
    "hils": "zero-shot HiLS landmark-key scoring (no anchors)",
}
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
            "label": labels[name],
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
