#!/bin/bash
set -u
cd /9950backfile/zhangem/HiLS-Attention-main
export PYTHONPATH=/9950backfile/zhangem/HiLS-Attention-main
PYTHON=/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin/python
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export TOKENIZERS_PARALLELISM=false TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TMPDIR=/dev/shm/zhangem_tmp PYTHONPYCACHEPREFIX=/9950backfile/zhangem/tmp/pycache
export MASS_PURE_BF16=1
mkdir -p "$TMPDIR"

CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -m torch.distributed.run --nnodes=1 --nproc-per-node=1 --standalone \
  tasks/pretrain_mass_with_ruler.py configs/training_recipes/pretrain_345M_8K_30B.yaml \
  --model.config_path configs/mass_attention/config_mass_qwen3_100M_18L_8KA2K_RoPE_m2_r8.json \
  --model.model_path /9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-100M-MASS-m2-init/snapshots/master \
  --model.tokenizer_path /9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master \
  --model.attn_implementation sdpa \
  --data.train_path /9950backfile/zhangem/data/fineweb_qwen3_data \
  --data.data_type ruler_0.05 --data.datasets_type olmo3 --data.max_seq_len 8192 \
  --data.train_size 327680000 --data.num_workers 8 --data.sort_files true \
  --train.max_steps 60 --train.micro_batch_size 8 --train.global_batch_size 8 \
  --train.enable_gradient_checkpointing true --train.enable_mixed_precision true \
  --train.broadcast_model_weights_from_rank0 true --train.use_wandb false \
  --train.output_dir /9950backfile/zhangem/tmp/bench_final_mb8 \
  --train.save_steps 50000 --train.save_epochs 0 --train.save_hf_weights false \
  --train.data_parallel_mode ddp --train.init_device cuda \
  --train.rmpad false --train.rmpad_with_pos_ids false \
  --train.lr 3e-4 --train.lr_min 3e-5 --train.lr_warmup_ratio 0.03 \
  --train.lr_decay_style cosine --train.lr_decay_ratio 1.0 \
  --train.max_grad_norm 1.0 --train.no_decay_params norm bias embed \
  > /9950backfile/zhangem/bench_final_mb8.log 2>&1 &
LAUNCHER=$!
CHILD=""
for i in $(seq 1 36); do sleep 5; CHILD=$(pgrep -P $LAUNCHER | head -1); [ -n "$CHILD" ] && grep -q "Set gradient accumulation" /9950backfile/zhangem/bench_final_mb8.log 2>/dev/null && break; done
echo "child=$CHILD"
[ -n "$CHILD" ] && while kill -0 $LAUNCHER 2>/dev/null; do nvidia-smi pmon -s u -c 1 2>/dev/null | awk -v pid="$CHILD" '$2==pid && $4!="-"' >> /9950backfile/zhangem/bench_final_mb8.pmon; sleep 2; done
wait $LAUNCHER
rm -rf /9950backfile/zhangem/tmp/bench_final_mb8
echo "done"
