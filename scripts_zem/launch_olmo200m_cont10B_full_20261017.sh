#!/bin/bash
# 200M 三臂跑满 10B-minus-3B:从 30000 步检查点续到 121850 步(吃掉剩余 6.02B token)
# 数据: dolma3_1025_10Bminus3B_packed(10B 采样去掉与 3B pack 逐字节重合的前缀,前缀已验证)
# 配方: 延续原配方,lr 3e-5→3e-6 cosine(衔接上段终点),优化器/调度器/数据状态全新;mass τ 钉 0.5
# 用法: bash launch_olmo200m_cont10B_full_20261017.sh <mass|hils|recover> [GPU]
set -u
ARM=${1:?mass|hils|recover}
GPU=${2:-2}
REPO=/9950backfile/zhangem/HiLS-Attention-main
PY=/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin/python
DATA=/9950backfile/zhangem/data/dolma3_1025_10Bminus3B_packed
BASE_HILS=/9950backfile/zhangem/.cache/modelscope/models/allenai--Olmo-3-1025-200M-HiLS-CPT/snapshots/master
BASE_PRUNED=/9950backfile/zhangem/.cache/modelscope/models/allenai--Olmo-3-1025-200M-pruned
TOK7B=/9950backfile/zhangem/.cache/modelscope/models/allenai--Olmo-3-1025-7B/snapshots/master

export CUDA_VISIBLE_DEVICES=$GPU TOKENIZERS_PARALLELISM=false TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TMPDIR=/dev/shm/zhangem_tmp PYTHONPYCACHEPREFIX=/9950backfile/zhangem/tmp/pycache
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
mkdir -p "$TMPDIR"

case $ARM in
  mass)
    export MODELING_BACKEND=veomni TILELANG_CACHE_DIR=/dev/shm/zhangem_tmp/tilelang_cache
    export MASS_PURE_BF16=1 MASS_LSE_ANNEAL_START=0.5 MASS_LSE_ANNEAL_END=0.5
    CFG=configs/mass_attention/config_mass_olmo3_200M_cpt_sharedview.json
    MPATH=$BASE_HILS
    CKPT=/9950backfile/zhangem/outputs/olmo200m_mass_cont10B_ft30000/checkpoints/global_step_30000
    OUTDIR=/9950backfile/zhangem/outputs/olmo200m_mass_cont10B_full
    ;;
  hils)
    export MODELING_BACKEND=veomni TILELANG_CACHE_DIR=/dev/shm/zhangem_tmp/tilelang_cache
    export MASS_PURE_BF16=1  # 名字带 MASS,实为全局跳过 fp32 主权重转换;tilelang kernel 需 bf16
    CFG=configs/olmo3_200M_cpt/olmo3_200M_hils_yarn.json
    MPATH=$BASE_HILS
    CKPT=/9950backfile/zhangem/outputs/olmo200m_hils_cont10B_ft30000/checkpoints/global_step_30000
    OUTDIR=/9950backfile/zhangem/outputs/olmo200m_hils_cont10B_full
    ;;
  recover)
    export MODELING_BACKEND=hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    CFG=$BASE_PRUNED
    MPATH=$BASE_PRUNED
    CKPT=/9950backfile/zhangem/outputs/olmo3_200m_recovery_cont10B_ft30000/checkpoints/global_step_30000
    OUTDIR=/9950backfile/zhangem/outputs/olmo3_200m_recovery_cont10B_full
    ;;
  *) echo "unknown arm $ARM"; exit 1;;
esac

LOG=/9950backfile/zhangem/logs/$(basename $OUTDIR).log
cd $REPO
export PYTHONPATH=$REPO
TASK=tasks/pretrain_with_ruler.py
if [ "$ARM" = mass ]; then TASK=tasks/pretrain_mass_with_ruler.py; fi
nohup $PY -m torch.distributed.run --nnodes=1 --nproc-per-node=1 --node-rank=0 --standalone \
  $TASK configs/training_recipes/pretrain_345M_8K_30B.yaml \
  --model.config_path $CFG --model.model_path $MPATH --model.tokenizer_path $TOK7B \
  --model.attn_implementation sdpa \
  --data.train_path $DATA --data.data_type ruler_0.05 --data.datasets_type olmo3 \
  --data.max_seq_len 8192 --data.train_size 327680000 --data.num_workers 8 --data.sort_files true \
  --train.load_checkpoint_path $CKPT \
  --train.load_optimizer_state true --train.load_lr_scheduler_state false \
  --train.load_dataloader_state true --train.load_rng_state true \
  --train.max_steps 121850 --train.micro_batch_size 4 --train.global_batch_size 8 \
  --train.enable_gradient_checkpointing true --train.broadcast_model_weights_from_rank0 true \
  --train.use_wandb false --train.output_dir $OUTDIR \
  --train.save_steps 2500 --train.save_epochs 0 --train.save_hf_weights false \
  --train.include_frozen_params_in_optimizer false \
  --train.data_parallel_mode ddp --train.init_device cuda \
  --train.rmpad false --train.rmpad_with_pos_ids false \
  --train.lr 3e-5 --train.lr_min 3e-6 --train.lr_warmup_ratio 0.01 \
  --train.lr_decay_style cosine --train.lr_decay_ratio 1.0 \
  --train.max_grad_norm 1.0 --train.no_decay_params norm bias embed \
  > $LOG 2>&1 &
echo $! > /9950backfile/zhangem/pids/$(basename $OUTDIR).pid
echo "launched $ARM cont10B pid=$! gpu=$GPU log=$LOG"
