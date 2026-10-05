#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_HOME="/usr/local/cuda-12.4"
export PATH="$CUDA_HOME/bin:$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export WANDB_MODE=disabled
export FSDP_ACCUM_NO_SYNC=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASS_TWO_STAGE_BACKWARD=1
export MASS_TRAINABLE_ONLY_CHECKPOINT=1
export MASS_DCP_THREAD_COUNT=1
export MASS_SPECIALIZATION_LOSS_WEIGHT="${MASS_SPECIALIZATION_LOSS_WEIGHT:-0.01}"
export MASS_SPECIALIZATION_LOG_INTERVAL="${MASS_SPECIALIZATION_LOG_INTERVAL:-10}"
unset MASS_SMOKE_SKIP_CLIP
unset MASS_REFERENCE_SPARSE
unset MASS_REFERENCE_MAX_SEQ_LEN

MODEL_PATH="${MODEL_PATH:-/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B-MASS-m2/snapshots/master}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master}"
CORPUS_PATH="${CORPUS_PATH:-/9950backfile/zhangem/data/fineweb_qwen3_data}"
OUTPUT_DIR="${OUTPUT_DIR:-/9950backfile/zhangem/outputs/mass_qwen3_0p6b_anchor_only_m2_lse_norm_tau1_mi_spec_mb4_v1}"
FREEZE_PATTERN='^(?!.*mass_anchor_generator\.(?:anchor_embedding|offset_basis)$).*'

cd "$REPO_ROOT"
bash train.sh tasks/pretrain_mass_with_ruler.py configs/training_recipes/pretrain_345M_8K_30B.yaml \
  --model.config_path configs/mass_attention/config_mass_qwen3_0p6B_8KA2K_RoPE_m2_r8_lse_norm_tau1_mi.json \
  --model.model_path "$MODEL_PATH" \
  --model.tokenizer_path "$TOKENIZER_PATH" \
  --model.attn_implementation sdpa \
  --data.train_path "$CORPUS_PATH" \
  --data.data_type ruler_0.05 \
  --data.datasets_type olmo3 \
  --data.max_seq_len 8192 \
  --data.train_size "${TRAIN_SIZE:-327680000}" \
  --data.num_workers "${DATA_WORKERS:-8}" \
  --data.sort_files true \
  --train.max_steps "${MAX_STEPS:-5000}" \
  --train.micro_batch_size "${MICRO_BATCH_SIZE:-4}" \
  --train.global_batch_size 8 \
  --train.enable_gradient_checkpointing true \
  --train.broadcast_model_weights_from_rank0 true \
  --train.use_wandb false \
  --train.output_dir "$OUTPUT_DIR" \
  --train.save_steps "${SAVE_STEPS:-1000}" \
  --train.save_epochs 0 \
  --train.save_hf_weights false \
  --train.freeze_pattern "$FREEZE_PATTERN" \
  --train.include_frozen_params_in_optimizer false \
  --train.data_parallel_mode fsdp2 \
  --train.init_device meta \
  --train.rmpad false \
  --train.rmpad_with_pos_ids false \
  --train.lr "${LR:-2e-5}" \
  --train.lr_min "${LR_MIN:-2e-6}" \
  --train.lr_warmup_ratio "${LR_WARMUP_RATIO:-0.03}" \
  --train.lr_decay_style cosine \
  --train.lr_decay_ratio 1.0 \
  --train.max_grad_norm 1.0 \
  --train.no_decay_params norm bias embed \
  "$@"
