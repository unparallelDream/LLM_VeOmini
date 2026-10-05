#!/bin/bash
# 0.6B 全参(解冻)A/B 对照:与既有冻结臂构成 2×2 矩阵(冻结×{mass,hils})
#   fullparam_mass : MASS 双 anchor 模型,全参数训练(不再冻结 backbone)
#   fullparam_hils : HiLS 模型,全参数训练
# 其余配方(数据/步数/lr/batch)与冻结臂完全一致,唯一变量 = 是否冻结。
# lr 沿用 2e-5(保守,防灾难遗忘;如要更贴 CPT 惯例可改 5e-5 起步)。
# 用法: bash launch_0p6b_fullparam_ab.sh mass|hils [GPU]
set -u

ARM=${1:?用法: bash launch_0p6b_fullparam_ab.sh mass|hils [GPU]}
case "$ARM" in
  mass)
    DEFAULT_GPU=0
    CONFIG=configs/mass_attention/config_mass_qwen3_0p6B_8KA2K_RoPE_m2_r8_asym0p5_anneal.json
    SNAPSHOT=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B-MASS-m2/snapshots/master
    OUTDIR=/9950backfile/zhangem/outputs/fullparam_qwen3_0p6b_mass_ft20000
    TAG=mass
    EXTRA_ENV="export MASS_ANCHOR_REINIT=1 MASS_SPECIALIZATION_LOSS_WEIGHT=1.0 MASS_LSE_ANNEAL_START=4.0 MASS_LSE_ANNEAL_END=0.1 MASS_TWO_STAGE_BACKWARD=1" ;;
  hils)
    DEFAULT_GPU=6
    CONFIG=configs/mass_attention/config_hils_qwen3_0p6B_8KA2K_RoPE.json
    SNAPSHOT=/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B-HiLS/snapshots/master
    OUTDIR=/9950backfile/zhangem/outputs/fullparam_qwen3_0p6b_hils_ft20000
    TAG=hils
    EXTRA_ENV="true" ;;
  *) echo "未知臂: $ARM"; exit 1 ;;
esac
GPU=${2:-$DEFAULT_GPU}

export CUDA_VISIBLE_DEVICES=$GPU
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
eval "$EXTRA_ENV"

export TMPDIR=/dev/shm/zhangem_tmp
export PYTHONPYCACHEPREFIX=/9950backfile/zhangem/tmp/pycache
mkdir -p "$TMPDIR"

cd /9950backfile/zhangem/HiLS-Attention-main
export PYTHONPATH=/9950backfile/zhangem/HiLS-Attention-main
unset CODEBUDDY_SAFE_DELETE_BULK_STATE_DIR CODEBUDDY_TOOL_CALL_ID \
      CODEBUDDY_SAFE_DELETE_BULK_GUARD CODEBUDDY_NODE_BIN GENIE_TRASH_DIR 2>/dev/null
export PATH=/usr/local/cuda-12.4/bin:$PATH

LOG=/9950backfile/zhangem/fullparam_qwen3_0p6b_${TAG}_ft20000.log

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
  --train.data_parallel_mode ddp --train.init_device cuda \
  --train.rmpad false --train.rmpad_with_pos_ids false \
  --train.lr 2e-5 --train.lr_min 2e-6 --train.lr_warmup_ratio 0.03 \
  --train.lr_decay_style cosine --train.lr_decay_ratio 1.0 \
  --train.max_grad_norm 1.0 --train.no_decay_params norm bias embed \
  > "$LOG" 2>&1 &

PID=$!
echo $PID > /9950backfile/zhangem/fullparam_0p6b_${TAG}.pid
echo "launched arm=$TAG pid=$PID gpu=$GPU log=$LOG"
echo "注意: 无 --train.freeze_pattern,全参数训练"
