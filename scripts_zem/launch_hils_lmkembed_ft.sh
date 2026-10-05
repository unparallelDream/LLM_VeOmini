#!/bin/bash
# HiLS 对照组训练(与 MASS anchor-FT 同范式):
#   基座 = 与对应 MASS run 完全相同的冻结 backbone(仅剥离 anchor 张量)
#   唯一可训练参数 = model.lmk_embed(专用 landmark 嵌入, EOS 初始化)
#   MASS 侧训练的是 28,784 个 anchor 参数;HiLS 侧结构上唯一对应的
#   检索可学习参数就是 lmk_embed —— 训练配方(步数/lr/数据)与 anchor FT 一致。
# 训练时模型自动以 lmk_id(=vocab_size)插入 landmark 并注入 lmk_embed
# (hils_forward.py:283-285),因此该参数真实参与前向与梯度。
# 用法: bash launch_hils_lmkembed_ft.sh 0p6b|90m
set -u

SIZE=${1:?用法: bash launch_hils_lmkembed_ft.sh 0p6b|90m}
case "$SIZE" in
  0p6b)
    DEFAULT_GPU=7
    CONFIG=configs/mass_attention/config_hils_qwen3_0p6B_8KA2K_RoPE.json
    SNAPSHOT=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B-HiLS/snapshots/master
    OUTDIR=/9950backfile/zhangem/outputs/hils_qwen3_0p6b_lmkembed_only_ft20000
    TAG=0p6b ;;
  90m)
    DEFAULT_GPU=6
    CONFIG=configs/mass_attention/config_hils_qwen3_90M_8KA2K_RoPE.json
    SNAPSHOT=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-90M-HiLS-recovered/snapshots/master
    OUTDIR=/9950backfile/zhangem/outputs/hils_qwen3_90m_lmkembed_only_ft20000
    TAG=90m ;;
  *) echo "未知规模: $SIZE"; exit 1 ;;
esac

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$DEFAULT_GPU}
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
# trainable-only checkpoint: 只存 lmk_embed, 评测端可直接 ckpt_to_state_dict
export MASS_TRAINABLE_ONLY_CHECKPOINT=1

export TMPDIR=/dev/shm/zhangem_tmp
export PYTHONPYCACHEPREFIX=/9950backfile/zhangem/tmp/pycache
mkdir -p "$TMPDIR"

cd /9950backfile/zhangem/HiLS-Attention-main
export PYTHONPATH=/9950backfile/zhangem/HiLS-Attention-main
unset CODEBUDDY_SAFE_DELETE_BULK_STATE_DIR CODEBUDDY_TOOL_CALL_ID \
      CODEBUDDY_SAFE_DELETE_BULK_GUARD CODEBUDDY_NODE_BIN GENIE_TRASH_DIR 2>/dev/null
export PATH=/usr/local/cuda-12.4/bin:$PATH

LOG=/9950backfile/zhangem/hils_qwen3_${TAG}_lmkembed_only_ft20000.log

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
  --train.freeze_pattern '^(?!.*model\.lmk_embed$).*' \
  --train.include_frozen_params_in_optimizer false \
  --train.data_parallel_mode ddp --train.init_device cuda \
  --train.rmpad false --train.rmpad_with_pos_ids false \
  --train.lr 2e-5 --train.lr_min 2e-6 --train.lr_warmup_ratio 0.03 \
  --train.lr_decay_style cosine --train.lr_decay_ratio 1.0 \
  --train.max_grad_norm 1.0 --train.no_decay_params norm bias embed \
  > "$LOG" 2>&1 &

PID=$!
echo $PID > /9950backfile/zhangem/hils_${TAG}_lmkembed.pid
echo "launched size=$TAG pid=$PID gpu=$CUDA_VISIBLE_DEVICES log=$LOG"
