#!/bin/bash
# 【可选】裁剪后基座的恢复预训练(recovery):
#   90M 裁剪掉了 ~83% 参数, 若不做恢复训练, 基座 LM 能力损失较大且 anchor-FT 无法弥补。
#   本脚本用 veomni 原生 qwen3 路径在 fineweb 上继续预训练【未裁剪前】的 pruned 基座
#   (不移植 MASS、不冻结任何参数, lr 3e-4 cosine)。
#   产物仅含 Transformer 权重, 完成后需重新执行 convert_qwen3_to_mass.py 移植 MASS
#   再跑 launch_mass90m_anchor_ft.sh。
#   200M 档削减 63%, 可先用 smoke 对比损失决定是否需要本步。
set -u

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

export TMPDIR=/dev/shm/zhangem_tmp
export PYTHONPYCACHEPREFIX=/9950backfile/zhangem/tmp/pycache
mkdir -p "$TMPDIR"

cd /9950backfile/zhangem/HiLS-Attention-main
export PYTHONPATH=/9950backfile/zhangem/HiLS-Attention-main
unset CODEBUDDY_SAFE_DELETE_BULK_STATE_DIR CODEBUDDY_TOOL_CALL_ID \
      CODEBUDDY_SAFE_DELETE_BULK_GUARD CODEBUDDY_NODE_BIN GENIE_TRASH_DIR 2>/dev/null
export PATH=/usr/local/cuda-12.4/bin:$PATH

PRUNED_BASE=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B-pruned-90M/snapshots/master
LOG=/9950backfile/zhangem/qwen3_90m_pruned_recovery.log

# 1B tokens ≈ 15000 步(global batch 8×8192=65536 tok/step); 先跑 500 步观察 loss 回落斜率亦可
MAX_STEPS=${MAX_STEPS:-15000}

nohup /9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin/python -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=1 --node-rank=0 --standalone \
  tasks/pretrain_with_ruler.py configs/training_recipes/pretrain_345M_8K_30B.yaml \
  --model.config_path "$PRUNED_BASE/config.json" \
  --model.model_path "$PRUNED_BASE" \
  --model.tokenizer_path /9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master \
  --model.attn_implementation sdpa \
  --data.train_path /9950backfile/zhangem/data/fineweb_qwen3_data \
  --data.data_type ruler_0.05 --data.datasets_type olmo3 --data.max_seq_len 8192 \
  --data.train_size 327680000 --data.num_workers 8 --data.sort_files true \
  --train.max_steps "$MAX_STEPS" \
  --train.micro_batch_size 4 --train.global_batch_size 8 \
  --train.enable_gradient_checkpointing true \
  --train.broadcast_model_weights_from_rank0 true \
  --train.use_wandb false \
  --train.output_dir /9950backfile/zhangem/outputs/qwen3_90m_pruned_recovery \
  --train.save_steps 500 --train.save_epochs 0 --train.save_hf_weights false \
  --train.data_parallel_mode ddp --train.init_device cuda \
  --train.rmpad false --train.rmpad_with_pos_ids false \
  --train.lr 3e-4 --train.lr_min 3e-5 --train.lr_warmup_ratio 0.03 \
  --train.lr_decay_style cosine --train.lr_decay_ratio 1.0 \
  --train.max_grad_norm 1.0 --train.no_decay_params norm bias embed \
  > "$LOG" 2>&1 &

PID=$!
echo $PID > /9950backfile/zhangem/mass90m_recovery.pid
echo "launched pid=$PID gpu=$CUDA_VISIBLE_DEVICES log=$LOG"
