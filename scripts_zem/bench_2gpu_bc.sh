#!/bin/bash
# 100M/18L 双卡加速测试(GPU4+GPU7, 共享卡, 背靠背三相位):
#   A) 1×GPU4  mb8(global 8)        —— 强扩展基线
#   B) 2×GPU   mb4/卡(global 8)     —— 强扩展: 同总batch, 纯加速比
#   C) 2×GPU   mb8/卡(global 16)    —— 弱扩展: 吞吐
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

run_phase () {  # name nproc gpus mb global
  local NAME=$1 NPROC=$2 GPUS=$3 MB=$4 GB=$5
  local LOG=/9950backfile/zhangem/bench2_${NAME}.log
  rm -rf /9950backfile/zhangem/tmp/bench2_${NAME}
  CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" -m torch.distributed.run --nnodes=1 --nproc-per-node=$NPROC --node-rank=0 --standalone \
    tasks/pretrain_mass_with_ruler.py configs/training_recipes/pretrain_345M_8K_30B.yaml \
    --model.config_path configs/mass_attention/config_mass_qwen3_100M_18L_8KA2K_RoPE_m2_r8.json \
    --model.model_path /9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-100M-MASS-m2-init/snapshots/master \
    --model.tokenizer_path /9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master \
    --model.attn_implementation sdpa \
    --data.train_path /9950backfile/zhangem/data/fineweb_qwen3_data \
    --data.data_type ruler_0.05 --data.datasets_type olmo3 --data.max_seq_len 8192 \
    --data.train_size 327680000 --data.num_workers 8 --data.sort_files true \
    --train.max_steps 30 --train.micro_batch_size $MB --train.global_batch_size $GB \
    --train.enable_gradient_checkpointing true --train.enable_mixed_precision true \
    --train.broadcast_model_weights_from_rank0 true --train.use_wandb false \
    --train.output_dir /9950backfile/zhangem/tmp/bench2_${NAME} \
    --train.save_steps 50000 --train.save_epochs 0 --train.save_hf_weights false \
    --train.data_parallel_mode ddp --train.init_device cuda \
    --train.rmpad false --train.rmpad_with_pos_ids false \
    --train.lr 3e-4 --train.lr_min 3e-5 --train.lr_warmup_ratio 0.03 \
    --train.lr_decay_style cosine --train.lr_decay_ratio 1.0 \
    --train.max_grad_norm 1.0 --train.no_decay_params norm bias embed \
    > "$LOG" 2>&1
  rm -rf /9950backfile/zhangem/tmp/bench2_${NAME}
  echo "[$NAME] done: $(grep -c 'mass-specialization' "$LOG" 2>/dev/null || echo 0)/3 spec行"
}

echo "[A] 1xGPU4 mb8 ..."
echo "[B] 2xGPU mb4/卡 ..."
run_phase B2gpu_strong 2 4,7 4 8
echo "[C] 2xGPU mb8/卡 ..."
run_phase C2gpu_weak 2 4,7 8 16
echo "ALL DONE"
