#!/bin/bash
# OLMo-199M 最终配方:MASS+shared+lmkq+NCI w2 + LoRA + 新协议(dolma3 数据)
set -u
GPU=${1:-4}
cd /9950backfile/zhangem/HiLS-Attention-main
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$GPU TOKENIZERS_PARALLELISM=false TORCH_NCCL_AVOID_RECORD_STREAMS=1 TMPDIR=/dev/shm/zhangem_tmp PYTHONPYCACHEPREFIX=/9950backfile/zhangem/tmp/pycache CUDA_HOME=/usr/local/cuda-12.4 PATH=/usr/local/cuda-12.4/bin:$PATH
export TILELANG_CACHE_DIR=/dev/shm/zhangem_tmp/tilelang_cache MODELING_BACKEND=veomni
export MASS_ANCHOR_REINIT=1 MASS_SPECIALIZATION_LOSS_WEIGHT=0.0 MASS_LSE_ANNEAL_START=4.0 MASS_LSE_ANNEAL_END=0.5 MASS_TWO_STAGE_BACKWARD=1 MASS_TRAINABLE_ONLY_CHECKPOINT=1
export MASS_NEEDLE_INFO_NCE=1 MASS_NCI_WEIGHT=2.0 MASS_NCI_TAU=1.0 MASS_NCI_QUERY_ROWS=64 MASS_LMKQ_WARMSTART=1
export MASS_LORA_ADAPT=1 MASS_LORA_RANK=16 MASS_LORA_ALPHA=32 MASS_LORA_TARGETS=self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj MASS_LORA_SCOPE=all
mkdir -p $TMPDIR
LOG=/9950backfile/zhangem/logs/olmo200m_final_recipe.log
nohup /9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin/python -m torch.distributed.run --nnodes=1 --nproc-per-node=1 --node-rank=0 --standalone \
  tasks/pretrain_mass_with_ruler.py configs/training_recipes/pretrain_345M_8K_30B.yaml \
  --model.config_path configs/mass_attention/config_mass_olmo3_200M_8KA2K_RoPE_m2_sharedview_lmkq.json \
  --model.model_path /9950backfile/zhangem/.cache/modelscope/models/allenai--Olmo-3-1025-200M-HiLS/snapshots/master \
  --model.tokenizer_path /9950backfile/zhangem/.cache/modelscope/models/allenai--Olmo-3-1025-200M-HiLS/snapshots/master \
  --model.attn_implementation sdpa \
  --data.train_path /9950backfile/zhangem/data/dolma3_1025_3B_packed \
  --data.data_type ruler_0.05 --data.datasets_type olmo3 --data.max_seq_len 8192 \
  --data.train_size 327680000 --data.num_workers 8 --data.sort_files true \
  --train.max_steps 20000 --train.micro_batch_size 4 --train.global_batch_size 8 \
  --train.enable_gradient_checkpointing true \
  --train.broadcast_model_weights_from_rank0 true \
  --train.use_wandb false --train.output_dir /9950backfile/zhangem/outputs/olmo200m_shared_lmkq_nciw2_ft20000 \
  --train.save_steps 500 --train.save_epochs 0 --train.save_hf_weights false \
  --train.freeze_pattern '^(?!.*(?:mass_anchor_generator\.(?:anchor_embedding|offset_basis|shared_view_bias)|self_attn\.lmk_q_(?:proj|norm)\.weight|\.lora_[AB]\.weight)$).*' \
  --train.include_frozen_params_in_optimizer false --train.data_parallel_mode ddp --train.init_device cuda \
  --train.rmpad false --train.rmpad_with_pos_ids false \
  --train.lr 2e-5 --train.lr_min 2e-6 --train.lr_warmup_ratio 0.03 \
  --train.lr_decay_style cosine --train.lr_decay_ratio 1.0 \
  --train.max_grad_norm 1.0 --train.no_decay_params norm bias embed ${RESUME_CKPT:+--train.load_checkpoint_path $RESUME_CKPT} \
  > $LOG 2>&1 &
echo $! > /9950backfile/zhangem/pids/olmo200m_final.pid
echo "launched olmo200m-final pid=$! gpu=$GPU"
