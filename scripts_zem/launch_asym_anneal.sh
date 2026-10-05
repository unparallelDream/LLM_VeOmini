#!/bin/bash
# 对照实验：锚点不对称初始化 (offset_init_std=0.5) + lse 温度退火 (4.0 -> 0.1)
set -o pipefail
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_VISIBLE_DEVICES=4
export MASS_ANCHOR_REINIT=1
export MASS_SPECIALIZATION_LOSS_WEIGHT=1.0
export MASS_LSE_ANNEAL_START=4.0
export MASS_LSE_ANNEAL_END=0.1

cd /9950backfile/zhangem/HiLS-Attention-main
export PYTHONPATH=/9950backfile/zhangem/HiLS-Attention-main:$PYTHONPATH
export PATH=/usr/local/cuda-12.4/bin:$PATH

nohup /9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin/python -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=1 --node-rank=0 --standalone \
  tasks/pretrain_mass_with_ruler.py configs/training_recipes/pretrain_345M_8K_30B.yaml \
  --model.config_path configs/mass_attention/config_mass_qwen3_0p6B_8KA2K_RoPE_m2_r8_asym0p5_anneal.json \
  --model.model_path /9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B-MASS-m2/snapshots/master \
  --model.tokenizer_path /9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master \
  --model.attn_implementation sdpa \
  --data.train_path /9950backfile/zhangem/data/fineweb_qwen3_data \
  --data.data_type ruler_0.05 \
  --data.datasets_type olmo3 \
  --data.max_seq_len 8192 \
  --data.train_size 327680000 \
  --data.num_workers 8 \
  --data.sort_files true \
  --train.max_steps 300 \
  --train.micro_batch_size 4 \
  --train.global_batch_size 8 \
  --train.enable_gradient_checkpointing true \
  --train.broadcast_model_weights_from_rank0 true \
  --train.use_wandb false \
  --train.output_dir /9950backfile/zhangem/outputs/mass_qwen3_0p6b_anchor_only_m2_r8_asym0p5_anneal_smoke300 \
  --train.save_steps 150 \
  --train.save_epochs 0 \
  --train.save_hf_weights false \
  --train.freeze_pattern '^(?!.*mass_anchor_generator\.(?:anchor_embedding|offset_basis)$).*' \
  --train.include_frozen_params_in_optimizer false \
  --train.data_parallel_mode ddp \
  --train.init_device cuda \
  --train.rmpad false \
  --train.rmpad_with_pos_ids false \
  --train.lr 2e-5 \
  --train.lr_min 2e-6 \
  --train.lr_warmup_ratio 0.03 \
  --train.lr_decay_style cosine \
  --train.lr_decay_ratio 1.0 \
  --train.max_grad_norm 1.0 \
  --train.no_decay_params norm bias embed \
  > /9950backfile/zhangem/mass_qwen3_0p6b_anchor_only_m2_r8_asym2p0_anneal_smoke300_v3.log 2>&1 &

echo $! > /9950backfile/zhangem/asym_anneal.pid

echo "launched PID $!"
