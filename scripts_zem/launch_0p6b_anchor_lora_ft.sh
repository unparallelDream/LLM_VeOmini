#!/bin/bash
# 0.6B anchor + LoRA 适配臂(中间方案):
#   冻结 backbone,联合训练 28,784 个 anchor 参数 + 检索层 k/v 投影上的 LoRA
#   (默认 rank=16 alpha=32, 7 个检索层 k_proj+v_proj ≈ 45.9 万 LoRA 参数)。
#   设计依据: anchor 已承担 query 侧低秩适配,冻结的是 K/V 表征侧——
#   LoRA 放 k/v 恰好补上"让块摘要变得可选"的缺口(与论文 Q-Cal 互补)。
#   LoRA 零初始化,step-0 输出与冻结基座位级一致(已单元验证)。
#   产物为 trainable-only checkpoint(anchors + lora),评测端可直接加载。
# 用法: bash launch_0p6b_anchor_lora_ft.sh [GPU]
set -u

GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export MASS_ANCHOR_REINIT=1
export MASS_SPECIALIZATION_LOSS_WEIGHT=1.0
export MASS_LSE_ANNEAL_START=4.0
export MASS_LSE_ANNEAL_END=0.1
export MASS_TWO_STAGE_BACKWARD=1
export MASS_TRAINABLE_ONLY_CHECKPOINT=1
export MASS_LORA_ADAPT=1
export MASS_LORA_RANK=${MASS_LORA_RANK:-16}
export MASS_LORA_ALPHA=${MASS_LORA_ALPHA:-32}
export MASS_LORA_TARGETS=${MASS_LORA_TARGETS:-self_attn.k_proj,self_attn.v_proj}
export MASS_LORA_SCOPE=${MASS_LORA_SCOPE:-retrieval}

export TMPDIR=/dev/shm/zhangem_tmp
export PYTHONPYCACHEPREFIX=/9950backfile/zhangem/tmp/pycache
mkdir -p "$TMPDIR"

cd /9950backfile/zhangem/HiLS-Attention-main
export PYTHONPATH=/9950backfile/zhangem/HiLS-Attention-main
unset CODEBUDDY_SAFE_DELETE_BULK_STATE_DIR CODEBUDDY_TOOL_CALL_ID \
      CODEBUDDY_SAFE_DELETE_BULK_GUARD CODEBUDDY_NODE_BIN GENIE_TRASH_DIR 2>/dev/null
export PATH=/usr/local/cuda-12.4/bin:$PATH

LOG=/9950backfile/zhangem/mass_lora_qwen3_0p6b_anchor_kv_r${MASS_LORA_RANK}_ft20000.log

nohup /9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin/python -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=1 --node-rank=0 --standalone \
  tasks/pretrain_mass_with_ruler.py configs/training_recipes/pretrain_345M_8K_30B.yaml \
  --model.config_path configs/mass_attention/config_mass_qwen3_0p6B_8KA2K_RoPE_m2_r8_asym0p5_anneal.json \
  --model.model_path /9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B-MASS-m2/snapshots/master \
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
  --train.output_dir /9950backfile/zhangem/outputs/mass_lora_qwen3_0p6b_anchor_kv_r${MASS_LORA_RANK}_ft20000 \
  --train.save_steps 500 --train.save_epochs 0 --train.save_hf_weights false \
  --train.freeze_pattern '^(?!.*(?:mass_anchor_generator\.(?:anchor_embedding|offset_basis)|\.lora_[AB]\.weight)$).*' \
  --train.include_frozen_params_in_optimizer false \
  --train.data_parallel_mode ddp --train.init_device cuda \
  --train.rmpad false --train.rmpad_with_pos_ids false \
  --train.lr 2e-5 --train.lr_min 2e-6 --train.lr_warmup_ratio 0.03 \
  --train.lr_decay_style cosine --train.lr_decay_ratio 1.0 \
  --train.max_grad_norm 1.0 --train.no_decay_params norm bias embed \
  > "$LOG" 2>&1 &

PID=$!
echo $PID > /9950backfile/zhangem/mass_lora_0p6b.pid
echo "launched anchor+lora pid=$PID gpu=$GPU rank=$MASS_LORA_RANK log=$LOG"
echo "可变体: MASS_LORA_SCOPE=all(全层) / MASS_LORA_TARGETS=self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj"
