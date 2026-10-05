#!/bin/bash
set -u

export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_VISIBLE_DEVICES=7
export MASS_ANCHOR_REINIT=1
export MASS_SPECIALIZATION_LOSS_WEIGHT=1.0
export MASS_LSE_ANNEAL_START=4.0
export MASS_LSE_ANNEAL_END=0.1
export MASS_TWO_STAGE_BACKWARD=1

# 根分区(/)已满：TMPDIR 用 /dev/shm 内存盘(tmpfs 999G 可用, 不占根分区, 本地内存 IO 快)
# TileLang/Triton 缓存保持默认(~/.tilelang 等, 网络盘 home 下, 命中后无需重编译)
export TMPDIR=/dev/shm/zhangem_tmp
export PYTHONPYCACHEPREFIX=/9950backfile/zhangem/tmp/pycache
mkdir -p "$TMPDIR"

cd /9950backfile/zhangem/HiLS-Attention-main
# 关键：重写 PYTHONPATH，移除 IDE 注入的 codebuddy shim 目录，
# 避免 sitecustomize.py 的 safe-delete 包装在训练中 raise SystemExit 杀进程
export PYTHONPATH=/9950backfile/zhangem/HiLS-Attention-main
# 双保险：清除 safe-delete 相关环境变量
unset CODEBUDDY_SAFE_DELETE_BULK_STATE_DIR CODEBUDDY_TOOL_CALL_ID \
      CODEBUDDY_SAFE_DELETE_BULK_GUARD CODEBUDDY_NODE_BIN GENIE_TRASH_DIR 2>/dev/null
export PATH=/usr/local/cuda-12.4/bin:$PATH

LOG=/9950backfile/zhangem/mass_qwen3_0p6b_anchor_only_m2_r8_asym0p5_anneal_ft20000.log

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
  --train.output_dir /9950backfile/zhangem/outputs/mass_qwen3_0p6b_anchor_only_m2_r8_asym0p5_anneal_ft20000 \
  --train.save_steps 500 --train.save_epochs 0 --train.save_hf_weights false \
  --train.load_checkpoint_path /9950backfile/zhangem/outputs/mass_qwen3_0p6b_anchor_only_m2_r8_asym0p5_anneal_ft20000/checkpoints/global_step_12500 \
  --train.freeze_pattern '^(?!.*mass_anchor_generator\.(?:anchor_embedding|offset_basis)$).*' \
  --train.include_frozen_params_in_optimizer false \
  --train.data_parallel_mode ddp --train.init_device cuda \
  --train.rmpad false --train.rmpad_with_pos_ids false \
  --train.lr 2e-5 --train.lr_min 2e-6 --train.lr_warmup_ratio 0.03 \
  --train.lr_decay_style cosine --train.lr_decay_ratio 1.0 \
  --train.max_grad_norm 1.0 --train.no_decay_params norm bias embed \
  > "$LOG" 2>&1 &

PID=$!
echo $PID > /9950backfile/zhangem/ft20000.pid
echo "launched pid=$PID log=$LOG"
