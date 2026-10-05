#!/bin/bash
# 0.6B 三向归因扩量评测(2026-09-26): n=50 -> n=200, tasks 0/1/2
# 变体: mass_lora / bsa_lora / hils_lora (step 20000) + mass / dsa / full 参照
# 前 50 个样本与 eval_qwen3_0p6b_threeway_lora_ruler4k_n50_20260926 完全连续(同数据顺序)
set -eu
GPU=${1:-0}

REPO=/9950backfile/zhangem/HiLS-Attention-main
PY=/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin/python
OUTROOT=/9950backfile/zhangem/outputs/eval_qwen3_0p6b_threeway_lora_ruler4k_n200_20260926

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$REPO
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TMPDIR=/dev/shm/zhangem_tmp
export PYTHONPYCACHEPREFIX=/9950backfile/zhangem/tmp/pycache
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
mkdir -p "$TMPDIR" "$OUTROOT"
cd "$REPO"

MASSCFG=configs/mass_attention/config_mass_qwen3_0p6B_8KA2K_RoPE_m2_r8_asym0p5_anneal.json
MASSMODEL=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B-MASS-m2/snapshots/master
ORIGMODEL=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master
CORPUS=/9950backfile/zhangem/data/fineweb_qwen3_data
LORA_MASS=/9950backfile/zhangem/outputs/mass_lora_qwen3_0p6b_anchor_qkvo_all_r16_ft20000/checkpoints/global_step_20000
LORA_BSA=/9950backfile/zhangem/outputs/bsa_lora_qwen3_0p6b_mean_qkvo_all_r16_ft20000/checkpoints/global_step_20000
LORA_HILS=/9950backfile/zhangem/outputs/hils_lora_qwen3_0p6b_lmk_qkvo_all_r16_ft20000/checkpoints/global_step_20000
ANCHOR=/9950backfile/zhangem/outputs/mass_qwen3_0p6b_anchor_only_m2_r8_asym0p5_anneal_ft20000/checkpoints/global_step_20000_anchor_only

run() { # variant ckpt outfile
  local V=$1 CKPT=$2 F=$3
  echo "=== [$V] n=200 tasks 0 1 2 -> $F"
  $PY eval/eval_ruler_qwen_mass_bsa_4k.py \
    --variant "$V" \
    --mass-config "$MASSCFG" \
    --mass-model "$MASSMODEL" \
    --original-model "$ORIGMODEL" \
    --checkpoint "$CKPT" \
    --corpus "$CORPUS" \
    --output "$F" \
    --tasks 0 1 2 \
    --max-samples 200 \
    --max-seq-len 4096 \
    --seed 42
}

run mass_lora "$LORA_MASS" "$OUTROOT/mass_lora_tasks012.json"
run bsa_lora  "$LORA_BSA"  "$OUTROOT/bsa_lora_tasks012.json"
run hils_lora "$LORA_HILS" "$OUTROOT/hils_lora_tasks012.json"
run mass      "$ANCHOR"    "$OUTROOT/mass_tasks012.json"
run dsa       "$ANCHOR"    "$OUTROOT/dsa_tasks012.json"
run full      "$ANCHOR"    "$OUTROOT/full_tasks012.json"
echo ALL_DONE
