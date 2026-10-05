#!/bin/bash
# Olmo-3-1025-200M-pruned dense recovery(2026-10-03)
# 数据:dolma3_3B_packed(OLMo 词表);配方参照 90M recovery:全参 lr 3e-4→3e-5, 15000 步(0.98B tokens)
# 用法: bash launch_olmo3_200m_recovery_20261003.sh [GPU]
set -u
GPU=${1:-4}

REPO=/9950backfile/zhangem/HiLS-Attention-main
PY=/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin/python
OUTDIR=/9950backfile/zhangem/outputs/olmo3_200m_recovery_ft15000

export CUDA_VISIBLE_DEVICES=$GPU
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TMPDIR=/dev/shm/zhangem_tmp
export PYTHONPYCACHEPREFIX=/9950backfile/zhangem/tmp/pycache
export MODELING_BACKEND=hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
mkdir -p "$TMPDIR"

cd "$REPO"
export PYTHONPATH="$REPO"
unset CODEBUDDY_SAFE_DELETE_BULK_STATE_DIR CODEBUDDY_TOOL_CALL_ID \
      CODEBUDDY_SAFE_DELETE_BULK_GUARD CODEBUDDY_NODE_BIN GENIE_TRASH_DIR 2>/dev/null

LOG=/9950backfile/zhangem/logs/olmo3_200m_recovery_ft15000.log

nohup $PY -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=1 --node-rank=0 --standalone \
  tasks/pretrain_with_ruler.py configs/training_recipes/pretrain_345M_8K_30B.yaml \
  --model.config_path /9950backfile/zhangem/.cache/modelscope/models/allenai--Olmo-3-1025-200M-pruned \
  --model.model_path /9950backfile/zhangem/.cache/modelscope/models/allenai--Olmo-3-1025-200M-pruned \
  --model.tokenizer_path /9950backfile/zhangem/.cache/modelscope/models/allenai--Olmo-3-1025-7B/snapshots/master \
  --model.attn_implementation sdpa \
  --data.train_path /9950backfile/zhangem/data/dolma3_1025_3B_packed \
  --data.data_type ruler_0.05 --data.datasets_type olmo3 --data.max_seq_len 8192 \
  --data.train_size 327680000 --data.num_workers 8 --data.sort_files true \
  --train.max_steps 15000 \
  --train.micro_batch_size 4 --train.global_batch_size 8 \
  --train.enable_gradient_checkpointing true \
  --train.broadcast_model_weights_from_rank0 true \
  --train.use_wandb false \
  --train.output_dir "$OUTDIR" \
  --train.save_steps 500 --train.save_epochs 0 --train.save_hf_weights false \
  --train.include_frozen_params_in_optimizer false \
  --train.data_parallel_mode ddp --train.init_device cuda \
  --train.rmpad false --train.rmpad_with_pos_ids false \
  --train.lr 3e-4 --train.lr_min 3e-5 --train.lr_warmup_ratio 0.03 \
  --train.lr_decay_style cosine --train.lr_decay_ratio 1.0 \
  --train.max_grad_norm 1.0 --train.no_decay_params norm bias embed \
  > "$LOG" 2>&1 &

PID=$!
echo $PID > /9950backfile/zhangem/pids/olmo3_200m_recovery.pid
echo "launched olmo3-200m-recovery pid=$PID gpu=$GPU log=$LOG outdir=$OUTDIR"
