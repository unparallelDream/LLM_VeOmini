#!/usr/bin/env bash
# mass vs hils 训练轨迹评测:在匹配步数的 checkpoint 上各评一次,回答
# "随 step 增加 mass 是否有反超 hils 的趋势"。
#   90M : steps 2000 5000 8000 11000 14000 17000 20000(mass 需先提取 anchor-only)
#   0.6B: steps 1000 3000 5000 6500(hils 对照尚未完成, 只评到当前)
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
EVAL_GPU=${EVAL_GPU:-0}

run_eval () {  # scale variant ckpt outjson
  local SCALE=$1 VAR=$2 CKPT=$3 OUT=$4
  local CFG SNAP
  if [ "$SIZE" = 90m ]; then
    CFG=configs/mass_attention/config_mass_qwen3_90M_8KA2K_RoPE_m2_r8_asym0p5_anneal.json
    SNAP=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-90M-MASS-m2-recovered/snapshots/master
  else
    CFG=configs/mass_attention/config_mass_qwen3_0p6B_8KA2K_RoPE_m2_r8_asym0p5_anneal.json
    SNAP=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B-MASS-m2/snapshots/master
  fi
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
  "$PYTHON" eval/eval_ruler_qwen_mass_bsa_4k.py \
    --variant "$VAR" --mass-config "$CFG" --mass-model "$SNAP" \
    --original-model /9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master \
    --checkpoint "$CKPT" \
    --corpus /9950backfile/zhangem/data/fineweb_qwen3_data \
    --output "$OUT" --tasks 0 1 2 --max-samples 50 --max-seq-len 4096 --seed 42 \
    > "${OUT%.json}.log" 2>&1
}

for SIZE in 90m 0p6b; do
  case "$SIZE" in
    90m)  STEPS="2000 5000 8000 11000 14000 17000 20000"
          MASSDIR=/9950backfile/zhangem/outputs/mass_qwen3_90m_anchor_only_m2_r8_asym0p5_anneal_ft20000/checkpoints
          HILSDIR=/9950backfile/zhangem/outputs/hils_qwen3_90m_lmkembed_only_ft20000/checkpoints ;;
    0p6b) STEPS="1000 3000 5000 6500"
          MASSDIR=/9950backfile/zhangem/outputs/mass_qwen3_0p6b_anchor_only_m2_r8_asym0p5_anneal_ft20000/checkpoints
          HILSDIR=/9950backfile/zhangem/outputs/hils_qwen3_0p6b_lmkembed_only_ft20000/checkpoints ;;
  esac
  OUTROOT=/9950backfile/zhangem/outputs/eval_trajectory_${SIZE}_mass_vs_hils_n50_20260907
  mkdir -p "$OUTROOT"

  for S in $STEPS; do
    # mass: 提取 anchor-only(若未提取)
    if [ ! -d "$MASSDIR/global_step_${S}_anchor_only" ]; then
      "$PYTHON" /9950backfile/zhangem/extract_anchor_only_ckpt.py \
        "$MASSDIR/global_step_${S}" "$MASSDIR/global_step_${S}_anchor_only" \
        > /dev/null 2>&1 || { echo "[$SIZE/$S] anchor 提取失败"; continue; }
    fi
    for V in mass hils_t; do
      [ "$V" = mass ] && CKPT=$MASSDIR/global_step_${S}_anchor_only || CKPT=$HILSDIR/global_step_${S}
      [ -f "$OUTROOT/${V}_${S}.json" ] && continue
      echo "[$(date '+%T')] $SIZE $V step=$S"
      run_eval "$SIZE" "$V" "$CKPT" "$OUTROOT/${V}_${S}.json" \
        || echo "[$SIZE/$V/$S] FAILED"
    done
  done
done

"$PYTHON" - <<'PY'
import json, pathlib
root90 = pathlib.Path("/9950backfile/zhangem/outputs/eval_trajectory_90m_mass_vs_hils_n50_20260907")
root06 = pathlib.Path("/9950backfile/zhangem/outputs/eval_trajectory_0p6b_mass_vs_hils_n50_20260907")
def load(root, steps):
    out = {}
    for s in steps:
        row = {}
        for v in ("mass", "hils_t"):
            p = root / f"{v}_{s}.json"
            if p.is_file():
                d = json.loads(p.read_text())
                t = {tt["task"]: tt for tt in d["tasks"]}
                row[v] = (round(d["macro_token_accuracy"], 4),
                          round(t["single_niah"]["token_accuracy"], 3),
                          round(t["multi_query"]["token_accuracy"], 3),
                          round(t["variable_tracking"]["token_accuracy"], 3))
        if len(row) == 2:
            out[s] = row
    return out
summary = {"90m": load(root90, [2000,5000,8000,11000,14000,17000,20000]),
           "0p6b": load(root06, [1000,3000,5000,6500])}
for scale, data in summary.items():
    print(f"=== {scale} (macro_tok, niah, multiq, vartrk) ===")
    print(f"{'step':>6s} {'mass':>26s} {'hils_t':>26s} {'gap(m-h)':>8s}")
    for s in sorted(data):
        m, h = data[s]["mass"], data[s]["hils_t"]
        print(f"{s:6d} {str(m):>26s} {str(h):>26s} {m[0]-h[0]:+8.4f}")
(pathlib.Path("/9950backfile/zhangem/outputs/eval_trajectory_summary.json")).write_text(
    json.dumps(summary, indent=2) + "\n")
PY
