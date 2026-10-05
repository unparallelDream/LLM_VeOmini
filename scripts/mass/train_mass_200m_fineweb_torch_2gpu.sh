#!/usr/bin/env bash
set -euo pipefail

# PyTorch diagnostic backend: it is correct for short contexts but not a
# replacement for the TileLang kernel in a production long-context run.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,6}"
export NPROC_PER_NODE=2
export WANDB_MODE=disabled
export MASS_REFERENCE_SPARSE=1
export MASS_REFERENCE_MAX_SEQ_LEN=1024

CORPUS_PATH="${CORPUS_PATH:-/9950backfile/zhangem/data/fineweb_olmo3_data}"
OUTPUT_DIR="${OUTPUT_DIR:-/9950backfile/zhangem/outputs/mass_200m_fineweb_4B_torch}"

bash train.sh tasks/pretrain_mass_with_ruler.py configs/training_recipes/pretrain_345M_8K_30B.yaml \
  --model.config_path configs/mass_attention/config_mass_attn_200M_m4_r8.json \
  --model.attn_implementation sdpa \
  --data.train_path "$CORPUS_PATH" \
  --data.data_type ruler_0.05 \
  --data.datasets_type olmo3 \
  --data.max_seq_len 512 \
  --data.train_size 4000000000 \
  --data.num_workers 8 \
  --data.sort_files true \
  --train.max_steps 122071 \
  --train.micro_batch_size 4 \
  --train.global_batch_size 64 \
  --train.use_wandb false \
  --train.output_dir "$OUTPUT_DIR" \
  --train.save_steps 5000 \
  --train.data_parallel_mode fsdp2 \
  --train.init_device meta \
  --train.rmpad false \
  --train.rmpad_with_pos_ids false \
  --train.lr 3e-4 \
  --train.lr_min 3e-5 \
  --train.no_decay_params norm bias embed
