#!/bin/bash
# 90M 共享专家臂(2026-09-27):HiLS landmark 键作为共享视图 0 + m2 专用锚 + LoRA + NCI
# 对照:mass_lora_qwen3_90m(run#7, macro4 0.3436)与 hils_lora_90m
# 用法: bash launch_mass_shared_90m_20260927.sh [GPU]
set -u
GPU=${1:-4}

REPO=/9950backfile/zhangem/HiLS-Attention-main
PY=/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin/python
OUTDIR=/9950backfile/zhangem/outputs/mass_lora_qwen3_90m_shared_m2_ft20000

export CUDA_VISIBLE_DEVICES=$GPU
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TMPDIR=/dev/shm/zhangem_tmp
export PYTHONPYCACHEPREFIX=/9950backfile/zhangem/tmp/pycache
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"

# 共享专家臂配置:MI spec 关闭(负载统计保留为诊断,视图0=共享份额)
export MASS_ANCHOR_REINIT=1
export MASS_SPECIALIZATION_LOSS_WEIGHT=0.0
export MASS_LSE_ANNEAL_START=4.0
export MASS_LSE_ANNEAL_END=0.5
export MASS_TWO_STAGE_BACKWARD=1
export MASS_TRAINABLE_ONLY_CHECKPOINT=1
# 检索监督(共享视图参与包络,NCI 同时训练 k_proj/lmk 键、λ 与专用锚)
export MASS_NEEDLE_INFO_NCE=1
export MASS_NCI_WEIGHT=8.0
export MASS_NCI_TAU=1.0
export MASS_NCI_QUERY_ROWS=64
# LoRA(与 run#7 同款)
export MASS_LORA_ADAPT=1
export MASS_LORA_RANK=16
export MASS_LORA_ALPHA=32
export MASS_LORA_TARGETS=self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj
export MASS_LORA_SCOPE=all
mkdir -p "$TMPDIR"

cd "$REPO"
export PYTHONPATH="$REPO"
unset CODEBUDDY_SAFE_DELETE_BULK_STATE_DIR CODEBUDDY_TOOL_CALL_ID \
      CODEBUDDY_SAFE_DELETE_BULK_GUARD CODEBUDDY_NODE_BIN GENIE_TRASH_DIR 2>/dev/null

LOG=/9950backfile/zhangem/logs/mass_lora_90m_shared_ft20000.log
FREEZE='^(?!.*(?:mass_anchor_generator\.(?:anchor_embedding|offset_basis|shared_view_bias)|\.lora_[AB]\.weight)$).*'

nohup $PY -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=1 --node-rank=0 --standalone \
  tasks/pretrain_mass_with_ruler.py configs/training_recipes/pretrain_345M_8K_30B.yaml \
  --model.config_path configs/mass_attention/config_mass_qwen3_90M_8KA2K_RoPE_m2_r8_sharedview.json \
  --model.model_path /9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-90M-MASS-m2-recovered/snapshots/master \
  --model.tokenizer_path /9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master \
  --model.attn_implementation sdpa \
  --data.train_path /9950backfile/zhangem/data/fineweb_qwen3_data \
  --data.data_type ruler_0.05 --data.datasets_type olmo3 --data.max_seq_len 8192 \
  --data.train_size 327680000 --data.num_workers 8 --data.sort_files true \
  --train.max_steps 20000 \
  --train.micro_batch_size 4 --train.global_batch_size 8 \
  --train.enable_gradient_checkpointing true \
  --train.broadcast_model_weights_from_rank0 true \
  --train.use_wandb false \
  --train.output_dir "$OUTDIR" \
  --train.save_steps 500 --train.save_epochs 0 --train.save_hf_weights false \
  --train.freeze_pattern "$FREEZE" \
  --train.include_frozen_params_in_optimizer false \
  --train.data_parallel_mode ddp --train.init_device cuda \
  --train.rmpad false --train.rmpad_with_pos_ids false \
  --train.lr 2e-5 --train.lr_min 2e-6 --train.lr_warmup_ratio 0.03 \
  --train.lr_decay_style cosine --train.lr_decay_ratio 1.0 \
  --train.max_grad_norm 1.0 --train.no_decay_params norm bias embed \
  > "$LOG" 2>&1 &

PID=$!
echo $PID > /9950backfile/zhangem/pids/mass_lora_90m_shared.pid
echo "launched 90m-shared pid=$PID gpu=$GPU log=$LOG outdir=$OUTDIR"
