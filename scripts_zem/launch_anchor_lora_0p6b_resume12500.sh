#!/bin/bash
# anchor + 全 token mixer LoRA 适配臂(0.6B / 90M 通用):
#   冻结 backbone,联合训练 anchor(28,784)+ 全部 28 层 q/k/v/o 投影 LoRA(rank16 α32)
#   0.6B: ~462 万可训练参数(0.78%)  90M: ~129 万(1.4%)
#   其余配方与对应规模的 anchor-only 臂完全一致,唯一变量 = 是否加 LoRA。
# 用法: bash launch_anchor_lora_kv_ft.sh 0p6b|90m [GPU]
set -u

SIZE=${1:?用法: bash launch_anchor_lora_kv_ft.sh 0p6b|90m [GPU]}
case "$SIZE" in
  0p6b)
    DEFAULT_GPU=0
    CONFIG=configs/mass_attention/config_mass_qwen3_0p6B_8KA2K_RoPE_m2_r8_asym0p5_anneal.json
    SNAPSHOT=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B-MASS-m2/snapshots/master
    OUTDIR=/9950backfile/zhangem/outputs/mass_lora_qwen3_0p6b_anchor_qkvo_all_r16_ft20000
    RESUME_CKPT=/9950backfile/zhangem/outputs/mass_lora_qwen3_0p6b_anchor_qkvo_all_r16_ft20000/checkpoints/global_step_12500 ;;
  90m)
    DEFAULT_GPU=6
    CONFIG=configs/mass_attention/config_mass_qwen3_90M_8KA2K_RoPE_m2_r8_asym0p5_anneal.json
    SNAPSHOT=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-90M-MASS-m2-recovered/snapshots/master
    OUTDIR=/9950backfile/zhangem/outputs/mass_lora_qwen3_90m_anchor_qkvo_all_r16_ft20000 ;;
  *) echo "未知规模: $SIZE"; exit 1 ;;
esac
GPU=${2:-$DEFAULT_GPU}

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
export PATH=/usr/local/cuda-12.4/bin:$PATH

LOG=/9950backfile/zhangem/mass_lora_qwen3_${SIZE}_anchor_qkvo_all_r16_ft20000_resume_from12500.log

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
  --train.load_checkpoint_path "$RESUME_CKPT" \
  --train.freeze_pattern '^(?!.*(?:mass_anchor_generator\.(?:anchor_embedding|offset_basis)|\.lora_[AB]\.weight)$).*' \
  --train.include_frozen_params_in_optimizer false \
  --train.data_parallel_mode ddp --train.init_device cuda \
  --train.rmpad false --train.rmpad_with_pos_ids false \
  --train.lr 2e-5 --train.lr_min 2e-6 --train.lr_warmup_ratio 0.03 \
  --train.lr_decay_style cosine --train.lr_decay_ratio 1.0 \
  --train.max_grad_norm 1.0 --train.no_decay_params norm bias embed \
  > "$LOG" 2>&1 &

PID=$!
echo $PID > /9950backfile/zhangem/mass_lora_${SIZE}.pid
echo "launched size=$SIZE pid=$PID gpu=$GPU log=$LOG"
