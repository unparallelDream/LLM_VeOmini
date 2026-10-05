#!/bin/bash
# 0.6B 三向归因评测(2026-09-26):补齐 hils_lora / bsa_lora 两臂,与 9/15 的 mass_lora 对齐
# 用法: bash eval_0p6b_threeway_lora_20260926.sh smoke|full [GPU]
set -eu
MODE=${1:?smoke|full}
GPU=${2:-0}

REPO=/9950backfile/zhangem/HiLS-Attention-main
PY=/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin/python
OUTROOT=/9950backfile/zhangem/outputs/eval_qwen3_0p6b_threeway_lora_ruler4k_n50_20260926

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$REPO
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TMPDIR=/dev/shm/zhangem_tmp
export PYTHONPYCACHEPREFIX=/9950backfile/zhangem/tmp/pycache
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
mkdir -p "$TMPDIR"
cd "$REPO"

MASSCFG=configs/mass_attention/config_mass_qwen3_0p6B_8KA2K_RoPE_m2_r8_asym0p5_anneal.json
MASSMODEL=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B-MASS-m2/snapshots/master
ORIGMODEL=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master
CORPUS=/9950backfile/zhangem/data/fineweb_qwen3_data
HILS_CKPT=/9950backfile/zhangem/outputs/hils_lora_qwen3_0p6b_lmk_qkvo_all_r16_ft20000/checkpoints/global_step_20000
BSA_CKPT=/9950backfile/zhangem/outputs/bsa_lora_qwen3_0p6b_mean_qkvo_all_r16_ft20000/checkpoints/global_step_20000

if [ "$MODE" = smoke ]; then
  OUTDIR=/tmp/threeway_smoke
  NS=1; TASKS="0"
else
  OUTDIR=$OUTROOT
  NS=50; TASKS="0 1 2"
fi
mkdir -p "$OUTDIR"

run() { # variant ckpt taskset outfile
  local V=$1 CKPT=$2 T=$3 F=$4
  echo "=== [$V] tasks=[$T] -> $F"
  $PY eval/eval_ruler_qwen_mass_bsa_4k.py \
    --variant "$V" \
    --mass-config "$MASSCFG" \
    --mass-model "$MASSMODEL" \
    --original-model "$ORIGMODEL" \
    --checkpoint "$CKPT" \
    --corpus "$CORPUS" \
    --output "$F" \
    --tasks $T \
    --max-samples "$NS" \
    --max-seq-len 4096 \
    --seed 42
}

run hils_lora "$HILS_CKPT" "$TASKS" "$OUTDIR/hils_lora_tasks012.json"
run bsa_lora  "$BSA_CKPT"  "$TASKS" "$OUTDIR/bsa_lora_tasks012.json"

if [ "$MODE" = full ]; then
  run hils_lora "$HILS_CKPT" "3" "$OUTDIR/hils_lora_task3.json"
  run bsa_lora  "$BSA_CKPT"  "3" "$OUTDIR/bsa_lora_task3.json"
fi
echo ALL_DONE
