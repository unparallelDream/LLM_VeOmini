#!/usr/bin/env bash
# 90M 七变体评测(等预算 2560,4K RULER,n=50):
#   mass / fixed0(训练 anchor) dsa / bsa(基线) hils(零样本) hils_t(训练 lmk_embed) full(稠密上限)
# mass/fixed0 用 anchor-only checkpoint;hils_t 用 lmk_embed trainable-only checkpoint。
set -uo pipefail

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

MASS_CONFIG=configs/mass_attention/config_mass_qwen3_90M_8KA2K_RoPE_m2_r8_asym0p5_anneal.json
MASS_MODEL=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-90M-MASS-m2-recovered/snapshots/master
ORIGINAL_MODEL=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master
CKPT_ANCHOR=/9950backfile/zhangem/outputs/mass_qwen3_90m_anchor_only_m2_r8_asym0p5_anneal_ft20000/checkpoints/global_step_20000_anchor_only
CKPT_LMK=/9950backfile/zhangem/outputs/hils_qwen3_90m_lmkembed_only_ft20000/checkpoints/global_step_20000
CORPUS=/9950backfile/zhangem/data/fineweb_qwen3_data
OUTPUT_DIR=/9950backfile/zhangem/outputs/eval_qwen3_90m_mass_vs_hils_trained_ruler4k_7variants_n50_20260906
MAX_SAMPLES=${MAX_SAMPLES:-50}
EVAL_GPU=${EVAL_GPU:-0}

mkdir -p "$OUTPUT_DIR"

for variant in mass fixed0 hils_t hils dsa bsa full; do
  CKPT=$CKPT_ANCHOR
  [ "$variant" = "hils_t" ] && CKPT=$CKPT_LMK
  echo "[$(date '+%F %T')] starting ${variant} on GPU ${EVAL_GPU}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
  "$PYTHON" eval/eval_ruler_qwen_mass_bsa_4k.py \
    --variant "$variant" \
    --mass-config "$MASS_CONFIG" \
    --mass-model "$MASS_MODEL" \
    --original-model "$ORIGINAL_MODEL" \
    --checkpoint "$CKPT" \
    --corpus "$CORPUS" \
    --output "$OUTPUT_DIR/${variant}.json" \
    --tasks 0 1 2 \
    --max-samples "$MAX_SAMPLES" \
    --max-seq-len 4096 \
    --seed 42 \
    2>&1 | tee "$OUTPUT_DIR/${variant}.log" > /dev/null
  echo "[$(date '+%F %T')] completed ${variant}"
done

"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
names = ("mass", "fixed0", "hils_t", "hils", "dsa", "bsa", "full")
labels = {
    "mass": "trained 2-anchor MASS scoring",
    "fixed0": "collapsed single-anchor ablation",
    "hils_t": "HiLS landmark scoring + trained lmk_embed (equal-recipe control)",
    "hils": "HiLS landmark scoring, zero-shot",
    "dsa": "exact per-token block-mass scoring (NSA/DSA-style)",
    "bsa": "block-mean pooled scoring",
    "full": "dense causal attention (budget=100% ceiling)",
}
rows = {n: json.loads((root / f"{n}.json").read_text()) for n in names}
reference = rows["mass"]
for name, row in rows.items():
    if name in ("full",):
        continue  # full 的预算上报不同(4160),其余必须相等
    if row["attention_budget"] != reference["attention_budget"]:
        raise RuntimeError(f"Attention budget mismatch for {name}")

comparison = {
    "checkpoint_anchor": reference["checkpoint"],
    "checkpoint_lmk": rows["hils_t"]["checkpoint"],
    "tasks": [t["task"] for t in reference["tasks"]],
    "variants": {
        name: {
            "label": labels[name],
            "macro_token_accuracy": row["macro_token_accuracy"],
            "macro_exact_match": row["macro_exact_match"],
            "peak_memory_gib": row["peak_memory_gib"],
            "attention_budget": row["attention_budget"]["nominal_max_kv_tokens_per_query"],
            "tasks": [
                {
                    "task": t["task"],
                    "token_accuracy": t["token_accuracy"],
                    "exact_match": t["exact_match"],
                }
                for t in row["tasks"]
            ],
        }
        for name, row in rows.items()
    },
}
(root / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
print(json.dumps(comparison["variants"], indent=2, sort_keys=True))
PY
