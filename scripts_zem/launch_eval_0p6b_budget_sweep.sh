#!/usr/bin/env bash
# 0.6B 退火模型的 KV budget 扫描评测:
#   6 档预算(b1/b2/b5/b10/b15/b20) × 4 变体(mass/fixed0/dsa/bsa),hils 跳过(未训练)。
#   预算 = 检索层每 query KV(window + topk*64),% 按 4161 计;非检索层窗口固定 512。
#   topk 受 MASS kernel 约束取 2 的幂,故实际预算为最接近目标的可达值(见配置注释)。
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

MASS_MODEL=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B-MASS-m2/snapshots/master
ORIGINAL_MODEL=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master
CHECKPOINT=/9950backfile/zhangem/outputs/mass_qwen3_0p6b_anchor_only_m2_r8_asym0p5_anneal_ft20000/checkpoints/global_step_20000_anchor_only
CORPUS=/9950backfile/zhangem/data/fineweb_qwen3_data
OUTROOT=/9950backfile/zhangem/outputs/eval_qwen3_0p6b_mass_anneal_ft20000_ruler4k_budget_sweep_n50_20260905
MAX_SAMPLES=${MAX_SAMPLES:-50}
EVAL_GPU=${EVAL_GPU:-3}

mkdir -p "$OUTROOT"

for B in b20 b15 b10 b5 b2 b1; do
  CFG=configs/mass_attention/budget_sweep_0p6b/config_${B}.json
  OUTDIR=$OUTROOT/$B
  mkdir -p "$OUTDIR"
  for variant in mass fixed0 dsa bsa; do
    if [ -f "$OUTDIR/${variant}.json" ]; then
      echo "[$B/${variant}] already done, skip"
      continue
    fi
    echo "[$(date '+%F %T')] $B/${variant} starting on GPU ${EVAL_GPU}"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
    "$PYTHON" eval/eval_ruler_qwen_mass_bsa_4k.py \
      --variant "$variant" \
      --mass-config "$CFG" \
      --mass-model "$MASS_MODEL" \
      --original-model "$ORIGINAL_MODEL" \
      --checkpoint "$CHECKPOINT" \
      --corpus "$CORPUS" \
      --output "$OUTDIR/${variant}.json" \
      --tasks 0 1 2 \
      --max-samples "$MAX_SAMPLES" \
      --max-seq-len 4096 \
      --seed 42 \
      > "$OUTDIR/${variant}.log" 2>&1 || echo "[$B/${variant}] FAILED (see log)"
    echo "[$(date '+%F %T')] $B/${variant} done"
  done
done

"$PYTHON" - "$OUTROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
budgets = ["b20", "b15", "b10", "b5", "b2", "b1"]
variants = ["mass", "fixed0", "dsa", "bsa"]
summary = {"reference_full_budget": 2560, "context_len": 4161, "budgets": {}}
for b in budgets:
    rows = {}
    for v in variants:
        p = root / b / f"{v}.json"
        if not p.is_file():
            rows[v] = None
            continue
        d = json.loads(p.read_text())
        t = {tt["task"]: tt for tt in d["tasks"]}
        rows[v] = {
            "macro_token_accuracy": d["macro_token_accuracy"],
            "macro_exact_match": d["macro_exact_match"],
            "niah": t["single_niah"]["token_accuracy"],
            "multi_query": t["multi_query"]["token_accuracy"],
            "variable_tracking": t["variable_tracking"]["token_accuracy"],
            "budget": d["attention_budget"]["nominal_max_kv_tokens_per_query"],
        }
    ref = next((r for r in rows.values() if r), None)
    summary["budgets"][b] = {
        "kv_per_query": ref["budget"] if ref else None,
        "pct_of_context": round((ref["budget"] / 4161 * 100), 1) if ref else None,
        "variants": rows,
    }
(root / "sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
