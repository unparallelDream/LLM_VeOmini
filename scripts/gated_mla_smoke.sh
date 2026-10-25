#!/bin/bash
# Smoke test for the gated_mla_hybrid family (ported from LLM_Standardization).
# Runs 2 training steps on 1 GPU and exports an HF checkpoint at the last step.
#
# Usage:
#   bash scripts/gated_mla_smoke.sh                          # gated_mla_50m
#   CONFIG=configs/gated_mla_hybrid/kda_gated_mla_50m.json GPU=6 bash scripts/gated_mla_smoke.sh
set -x
set -o pipefail
cd "$(dirname "$0")/.."

# torchrun children don't inherit the repo root on sys.path; the task imports
# top-level packages (models/utils/data) relative to it
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
# fla triton backend: the env's tilelang pin serves the HiLS kernels, not fla's
export FLA_TILELANG=${FLA_TILELANG:-0}
# KDA needs pure-bf16 params: fla's fp32 chunk_bwd variant fails to compile on
# this env's triton 3.2 (bf16 compiles fine). Harmless for the other baselines.
export GATED_PURE_BF16=${GATED_PURE_BF16:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PYTHON=${PYTHON:-/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin/python}
GPU=${GPU:-6}
STEPS=${STEPS:-2}
CONFIG=${CONFIG:-configs/gated_mla_hybrid/gated_mla_50m.json}
OUT=${OUT:-tmp/gated_smoke_$(basename "$CONFIG" .json)}
DATA=${DATA:-/9950backfile/zhangem/LLM_Standardization/data/smoke.jsonl}
# DP_MODE=fsdp2 gives bf16 compute + fp32 master via fully_shard mixed precision
# (valid on a single GPU); ddp applies the GATED_PURE_BF16 path instead.
DP_MODE=${DP_MODE:-ddp}
INIT_DEVICE=${INIT_DEVICE:-cuda}
ENABLE_FULL_SHARD=${ENABLE_FULL_SHARD:-false}

rm -rf "$OUT"
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" -m torch.distributed.run \
  --nnodes=1 --nproc-per-node="${NPROC:-1}" --node-rank=0 --standalone \
  tasks/pretrain_gated.py \
    --model.config_path "$CONFIG" \
    --model.tokenizer_path "${TOKENIZER:-configs/gated_mla_hybrid/llama32k_tokenizer}" \
    --model.attn_implementation sdpa \
    --data.train_path "$DATA" \
    --data.data_type plaintext \
    --data.datasets_type mapping \
    --data.dataloader_type native \
    --data.text_keys text \
    --data.max_seq_len 512 \
    --data.num_workers 0 \
    --train.output_dir "$OUT" \
    --train.data_parallel_mode "$DP_MODE" \
    --train.init_device "$INIT_DEVICE" \
    --train.enable_full_shard "$ENABLE_FULL_SHARD" \
    --train.broadcast_model_weights_from_rank0 true \
    --train.micro_batch_size "${MICRO:-4}" \
    --train.global_batch_size "${GLOBAL:-${MICRO:-4}}" \
    --train.max_steps "$STEPS" \
    --train.seed "${SEED:-42}" \
    --train.num_train_epochs 1 \
    --train.enable_mixed_precision true \
    --train.enable_gradient_checkpointing false \
    --train.rmpad false \
    --train.rmpad_with_pos_ids false \
    --train.lr 3e-4 \
    --train.lr_min 3e-5 \
    --train.lr_warmup_ratio "${WARMUP:-0.5}" \
    --train.lr_decay_style cosine \
    --train.lr_decay_ratio 1.0 \
    --train.max_grad_norm 1.0 \
    --train.weight_decay 0.0 \
    --train.no_decay_params norm bias embed alpha ihc_readout mhc_readout \
    --train.ckpt_manager dcp \
    --train.save_steps "${SAVE_STEPS:-$STEPS}" \
    --train.save_epochs 0 \
    --train.save_hf_weights "${SAVE_HF:-true}" \
    --train.use_wandb false \
    2>&1 | tee "$OUT.log"

rc=$?
echo "SMOKE-EXIT: $rc (config=$CONFIG)"
exit $rc
