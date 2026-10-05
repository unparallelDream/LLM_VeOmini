#!/usr/bin/env bash
set -euo pipefail

# Production TileLang launcher for the standalone MASS model.  The optional
# trailing arguments make short smoke runs possible without altering this file
# or any original HiLS recipe/configuration.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,7}"
export NPROC_PER_NODE=2
export WANDB_MODE=disabled
export FSDP_ACCUM_NO_SYNC="${FSDP_ACCUM_NO_SYNC:-1}"
export MASS_TWO_STAGE_BACKWARD="${MASS_TWO_STAGE_BACKWARD:-1}"
unset MASS_REFERENCE_SPARSE
unset MASS_REFERENCE_MAX_SEQ_LEN

CORPUS_PATH="${CORPUS_PATH:-/9950backfile/zhangem/data/fineweb_olmo3_data}"
OUTPUT_DIR="${OUTPUT_DIR:-/9950backfile/zhangem/outputs/mass_200m_fineweb_4B_tilelang}"

# 4B tokens = 200M parameters * 20.  With a global batch of 8 and an 8,192
# token context, ceil(4B / (8 * 8192)) = 61,036 optimizer updates.
bash train.sh tasks/pretrain_mass_with_ruler.py configs/training_recipes/pretrain_345M_8K_30B.yaml \
  --model.config_path configs/mass_attention/config_mass_attn_200M_m4_r8.json \
  --model.attn_implementation sdpa \
  --data.train_path "$CORPUS_PATH" \
  --data.data_type ruler_0.05 \
  --data.datasets_type olmo3 \
  --data.max_seq_len 8192 \
  --data.train_size 4000000000 \
  --data.num_workers 8 \
  --data.sort_files true \
  --train.max_steps 61036 \
  --train.micro_batch_size "${MICRO_BATCH_SIZE:-4}" \
  --train.global_batch_size 8 \
  --train.use_wandb false \
  --train.output_dir "$OUTPUT_DIR" \
  --train.save_steps 5000 \
  --train.data_parallel_mode fsdp2 \
  --train.init_device meta \
  --train.rmpad false \
  --train.rmpad_with_pos_ids false \
  --train.lr 3e-4 \
  --train.lr_min 3e-5 \
  --train.no_decay_params norm bias embed \
  "$@"
