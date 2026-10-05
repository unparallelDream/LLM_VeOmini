#!/bin/bash
# 三臂补齐对照(2026-09-15):
#   hils_lora_0p6b : 0.6B lmk_embed + LoRA(2×2 矩阵 0.6B 侧最后一格)
#   bsa_lora_0p6b  : 0.6B 纯 LoRA + 块均值打分(零检索参数, 三-way 归因)
#   bsa_lora_90m   : 90M 同款(先出结果的快速版)
# 用法: bash launch_lora_controls_20260915.sh hils_lora_0p6b|bsa_lora_0p6b|bsa_lora_90m [GPU]
set -u

ARM=${1:?用法见脚本头}
case "$ARM" in
  hils_lora_0p6b)
    DEFAULT_GPU=0
    CONFIG=configs/mass_attention/config_hils_qwen3_0p6B_8KA2K_RoPE.json
    SNAPSHOT=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B-HiLS/snapshots/master
    OUTDIR=/9950backfile/zhangem/outputs/hils_lora_qwen3_0p6b_lmk_qkvo_all_r16_ft20000
    FREEZE='^(?!.*(?:model\.lmk_embed$|\.lora_[AB]\.weight)$).*' ;;
  bsa_lora_0p6b)
    DEFAULT_GPU=6
    CONFIG=configs/mass_attention/config_bsamean_qwen3_0p6B_8KA2K_RoPE.json
    SNAPSHOT=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B-HiLS/snapshots/master
    OUTDIR=/9950backfile/zhangem/outputs/bsa_lora_qwen3_0p6b_mean_qkvo_all_r16_ft20000
    FREEZE='^(?!.*\.lora_[AB]\.weight$).*' ;;
  bsa_lora_90m)
    DEFAULT_GPU=7
    CONFIG=configs/mass_attention/config_bsamean_qwen3_90M_8KA2K_RoPE.json
    SNAPSHOT=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-90M-HiLS-recovered/snapshots/master
    OUTDIR=/9950backfile/zhangem/outputs/bsa_lora_qwen3_90m_mean_qkvo_all_r16_ft20000
    FREEZE='^(?!.*\.lora_[AB]\.weight$).*' ;;
  *) echo "未知臂: $ARM"; exit 1 ;;
esac
GPU=${2:-$DEFAULT_GPU}

export CUDA_VISIBLE_DEVICES=$GPU
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export MASS_TRAINABLE_ONLY_CHECKPOINT=1
export MASS_LORA_ADAPT=1
export MASS_LORA_RANK=16
export MASS_LORA_ALPHA=32
export MASS_LORA_TARGETS=self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj
export MASS_LORA_SCOPE=all

export TMPDIR=/dev/shm/zhangem_tmp
export PYTHONPYCACHEPREFIX=/9950backfile/zhangem/tmp/pycache
mkdir -p "$TMPDIR"

cd /9950backfile/zhangem/HiLS-Attention-main
export PYTHONPATH=/9950backfile/zhangem/HiLS-Attention-main
unset CODEBUDDY_SAFE_DELETE_BULK_STATE_DIR CODEBUDDY_TOOL_CALL_ID \
      CODEBUDDY_SAFE_DELETE_BULK_GUARD CODEBUDDY_NODE_BIN GENIE_TRASH_DIR 2>/dev/null
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"

LOG=/9950backfile/zhangem/${ARM}_ft20000.log

nohup /9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin/python -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=1 --node-rank=0 --standalone \
  tasks/pretrain_mass_with_ruler.py configs/training_recipes/pretrain_345M_8K_30B.yaml \
  --model.config_path "$CONFIG" \
  --model.model_path "$SNAPSHOT" \
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
echo $PID > /9950backfile/zhangem/${ARM}.pid
echo "launched arm=$ARM pid=$PID gpu=$GPU log=$LOG"
