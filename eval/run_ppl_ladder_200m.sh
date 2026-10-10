#!/bin/bash
# 用法: run_ppl_ladder_200m.sh <hf_ckpt> <model_type> <gpu> <out_jsonl> [max_samples] [config_json]
set -u
HF=$1; MT=$2; GPU=$3; OUT=$4; NS=${5:-64}; CFG=${6:-}
export PYTHONPATH=/9950backfile/zhangem/HiLS-Attention-main MODELING_BACKEND=veomni
export TILELANG_CACHE_DIR=/dev/shm/tilelang_cache TMPDIR=/dev/shm/tmp CUDA_HOME=/usr/local/cuda
export PYTHONPYCACHEPREFIX=/9950backfile/zhangem/tmp/pycache CUDA_VISIBLE_DEVICES=$GPU
PY=/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin/python
cd /9950backfile/zhangem/HiLS-Attention-main
: > $OUT
for L in 64 128 512 8192 32768 131072; do
  $PY - "$HF" "$MT" "$L" "$NS" "$CFG" <<'PYEOF' >> $OUT 2>>${OUT%.jsonl}.err
import sys, runpy
hf, mt, L, ns, cfg = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
if mt in ("olmo_hils", "olmo_mass"):
    from eval.run_ruler_lmeval_200m import register_custom
    register_custom(mt)
sys.argv = ["eval_ppl.py", "--use_hf_model", "--checkpoint_path", hf,
            "--data_path", "/9950backfile/zhangem/data/dolma3_1025_3B_packed",
            "--max_seq_len", str(L), "--max_samples", str(ns),
            "--attn_implementation", "sdpa"]
if cfg:
    sys.argv += ["--config_path", cfg]
runpy.run_path("eval/eval_ppl.py", run_name="__main__")
PYEOF
done
echo "LADDER DONE $(date +%H:%M)" >> $OUT
