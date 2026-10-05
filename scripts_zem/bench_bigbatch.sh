#!/bin/bash
# 100M/18L 单卡大 batch 测试(GPU4): mb ∈ {8,16,24,32}, 各 30 步
# 目标: 吞吐/SM 是否随 batch 提升、显存是否守住 24GB
set -u
cd /9950backfile/zhangem/HiLS-Attention-main
export PYTHONPATH=/9950backfile/zhangem/HiLS-Attention-main
PYTHON=/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin/python
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TMPDIR=/dev/shm/zhangem_tmp
export PYTHONPYCACHEPREFIX=/9950backfile/zhangem/tmp/pycache
export MASS_PURE_BF16=1
mkdir -p "$TMPDIR"

for MB in 8 16 24 32; do
  LOG=/9950backfile/zhangem/benchbig_mb${MB}.log
  rm -rf /9950backfile/zhangem/tmp/benchbig_mb${MB}
  echo "[big] mb=$MB start $(date '+%T')"
  CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" -m torch.distributed.run --nnodes=1 --nproc-per-node=1 --standalone \
    tasks/pretrain_mass_with_ruler.py configs/training_recipes/pretrain_345M_8K_30B.yaml \
    --model.config_path configs/mass_attention/config_mass_qwen3_100M_18L_8KA2K_RoPE_m2_r8.json \
    --model.model_path /9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-100M-MASS-m2-init/snapshots/master \
    --model.tokenizer_path /9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master \
    --model.attn_implementation sdpa \
    --data.train_path /9950backfile/zhangem/data/fineweb_qwen3_data \
    --data.data_type ruler_0.05 --data.datasets_type olmo3 --data.max_seq_len 8192 \
    --data.train_size 327680000 --data.num_workers 8 --data.sort_files true \
    --train.max_steps 30 --train.micro_batch_size $MB --train.global_batch_size $MB \
    --train.enable_gradient_checkpointing true --train.enable_mixed_precision true \
    --train.broadcast_model_weights_from_rank0 true --train.use_wandb false \
    --train.output_dir /9950backfile/zhangem/tmp/benchbig_mb${MB} \
    --train.save_steps 50000 --train.save_epochs 0 --train.save_hf_weights false \
    --train.data_parallel_mode ddp --train.init_device cuda \
    --train.rmpad false --train.rmpad_with_pos_ids false \
    --train.lr 3e-4 --train.lr_min 3e-5 --train.lr_warmup_ratio 0.03 \
    --train.lr_decay_style cosine --train.lr_decay_ratio 1.0 \
    --train.max_grad_norm 1.0 --train.no_decay_params norm bias embed \
    > "$LOG" 2>&1
  RC=$?
  rm -rf /9950backfile/zhangem/tmp/benchbig_mb${MB}
  MEM=$(grep "VRAM usage" "$LOG" | tail -1 | grep -oE "max [0-9.]+GB" || echo "?")
  echo "[big] mb=$MB rc=$RC $MEM done $(date '+%T')"
done
echo "BIG DONE"
