#!/bin/bash
# MASS 检索改进训练(2026-09-26):多 anchor 分化效果改进的两个对照臂
#   nci    : anchor-only + needle-chunk InfoNCE(MASS_NEEDLE_INFO_NCE)+ spec warmup
#   divmu  : anchor-only + spec 换成 μ 功能分化(MASS_SPEC_DIVERSE_MU, cos^2 惩罚)
# 基线 = 已有 run#2(mass_qwen3_0p6b_anchor_only_m2_r8_asym0p5_anneal_ft20000)
# 用法: bash launch_mass_retrieval_improve_20260926.sh nci|divmu [GPU]
set -u
ARM=${1:?用法见脚本头}
case "$ARM" in
  nci)
    DEFAULT_GPU=0
    OUTDIR=/9950backfile/zhangem/outputs/mass_qwen3_0p6b_anchor_nci_specwu_ft20000
    EXTRA_ENV=(
      MASS_NEEDLE_INFO_NCE=1
      MASS_NCI_WEIGHT=8.0
      MASS_NCI_TAU=1.0
      MASS_NCI_QUERY_ROWS=64
      MASS_SPEC_WARMUP_STEPS=4000
    ) ;;
  divmu)
    DEFAULT_GPU=5
    OUTDIR=/9950backfile/zhangem/outputs/mass_qwen3_0p6b_anchor_divmu_ft20000
    EXTRA_ENV=(MASS_SPEC_DIVERSE_MU=1) ;;
  *) echo "未知臂: $ARM"; exit 1 ;;
esac
GPU=${2:-$DEFAULT_GPU}

export CUDA_VISIBLE_DEVICES=$GPU
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TMPDIR=/dev/shm/zhangem_tmp
export PYTHONPYCACHEPREFIX=/9950backfile/zhangem/tmp/pycache
export MASS_ANCHOR_REINIT=1
export MASS_SPECIALIZATION_LOSS_WEIGHT=1.0
export MASS_LSE_ANNEAL_START=4.0
export MASS_LSE_ANNEAL_END=0.1
export MASS_TWO_STAGE_BACKWARD=1
export MASS_TRAINABLE_ONLY_CHECKPOINT=1
for kv in "${EXTRA_ENV[@]}"; do export "$kv"; done
mkdir -p "$TMPDIR"

cd /9950backfile/zhangem/HiLS-Attention-main
export PYTHONPATH=/9950backfile/zhangem/HiLS-Attention-main
unset CODEBUDDY_SAFE_DELETE_BULK_STATE_DIR CODEBUDDY_TOOL_CALL_ID \
      CODEBUDDY_SAFE_DELETE_BULK_GUARD CODEBUDDY_NODE_BIN GENIE_TRASH_DIR 2>/dev/null
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"

LOG=/9950backfile/zhangem/logs/mass_anchor_${ARM}_0p6b_ft20000.log
FREEZE='^(?!.*(?:mass_anchor_generator\.(?:anchor_embedding|offset_basis)$)).*'

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
  --train.output_dir "$OUTDIR" \
  --train.save_steps 500 --train.save_epochs 0 --train.save_hf_weights false \
  --train.freeze_pattern "$FREEZE" \
  --train.include_frozen_params_in_optimizer false \
  --train.data_parallel_mode ddp --train.init_device cuda \
  --train.rmpad false --train.rmpad_with_pos_ids false \
  --train.lr 2e-5 --train.lr_min 2e-6 --train.lr_warmup_ratio 0.03 \
  --train.lr_decay_style cosine --train.lr_decay_ratio 1.0 \
  --train.max_grad_norm 1.0 --train.no_decay_params norm bias embed \
  ${RESUME_CKPT:+--train.load_checkpoint_path $RESUME_CKPT} \
  > "$LOG" 2>&1 &

PID=$!
echo $PID > /9950backfile/zhangem/pids/mass_anchor_${ARM}_0p6b.pid
echo "launched arm=$ARM pid=$PID gpu=$GPU log=$LOG outdir=$OUTDIR"
