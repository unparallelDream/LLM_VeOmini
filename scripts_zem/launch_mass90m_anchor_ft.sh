#!/bin/bash
# 90M 压缩模型(anchor-only)冻结微调：
#   基座 = Qwen3-0.6B 宽度裁剪到 90M(utils/prune_qwen3_width.py --preset 90m)
#   经 qwen3_90m_pruned_recovery(15000 步, 2026-09-03 完成, 末段 loss≈3.24)恢复预训练后,
#   由 convert_dcp_to_hf_ckpt.py + convert_qwen3_to_mass.py 移植 MASS(锚点全新初始化)
#   除 mass_anchor_generator.{anchor_embedding,offset_basis} 外全部冻结
# 上游产物路径见 PRUNED_BASE / MASS_SNAPSHOT 两个变量。
set -u

# 空闲卡按需调整(GPU2 被 0.6B ft20000 占用, 勿动)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export MASS_ANCHOR_REINIT=1
export MASS_SPECIALIZATION_LOSS_WEIGHT=1.0
export MASS_LSE_ANNEAL_START=4.0
export MASS_LSE_ANNEAL_END=0.1
export MASS_TWO_STAGE_BACKWARD=1

# 根分区(/)已满：TMPDIR 用 /dev/shm 内存盘(同 launch_ft20000.sh)
export TMPDIR=/dev/shm/zhangem_tmp
export PYTHONPYCACHEPREFIX=/9950backfile/zhangem/tmp/pycache
mkdir -p "$TMPDIR"

cd /9950backfile/zhangem/HiLS-Attention-main
export PYTHONPATH=/9950backfile/zhangem/HiLS-Attention-main
unset CODEBUDDY_SAFE_DELETE_BULK_STATE_DIR CODEBUDDY_TOOL_CALL_ID \
      CODEBUDDY_SAFE_DELETE_BULK_GUARD CODEBUDDY_NODE_BIN GENIE_TRASH_DIR 2>/dev/null
export PATH=/usr/local/cuda-12.4/bin:$PATH

MASS_SNAPSHOT=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-90M-MASS-m2-recovered/snapshots/master
LOG=/9950backfile/zhangem/mass_qwen3_90m_anchor_only_m2_r8_asym0p5_anneal_ft20000.log

nohup /9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin/python -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=1 --node-rank=0 --standalone \
  tasks/pretrain_mass_with_ruler.py configs/training_recipes/pretrain_345M_8K_30B.yaml \
  --model.config_path configs/mass_attention/config_mass_qwen3_90M_8KA2K_RoPE_m2_r8_asym0p5_anneal.json \
  --model.model_path "$MASS_SNAPSHOT" \
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
  --train.output_dir /9950backfile/zhangem/outputs/mass_qwen3_90m_anchor_only_m2_r8_asym0p5_anneal_ft20000 \
  --train.save_steps 500 --train.save_epochs 0 --train.save_hf_weights false \
  --train.freeze_pattern '^(?!.*mass_anchor_generator\.(?:anchor_embedding|offset_basis)$).*' \
  --train.include_frozen_params_in_optimizer false \
  --train.data_parallel_mode ddp --train.init_device cuda \
  --train.rmpad false --train.rmpad_with_pos_ids false \
  --train.lr 2e-5 --train.lr_min 2e-6 --train.lr_warmup_ratio 0.03 \
  --train.lr_decay_style cosine --train.lr_decay_ratio 1.0 \
  --train.max_grad_norm 1.0 --train.no_decay_params norm bias embed \
  > "$LOG" 2>&1 &

PID=$!
echo $PID > /9950backfile/zhangem/mass90m_anchor_ft.pid
echo "launched pid=$PID gpu=$CUDA_VISIBLE_DEVICES log=$LOG"
