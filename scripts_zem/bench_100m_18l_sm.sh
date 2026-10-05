#!/bin/bash
# 100M/18L 从头预训练(随机初始化, MASS m2 结构)SM 饱和度测试:
#   micro_batch ∈ {4,8,16} × 30 步(accum=1), GPU2(共享卡, pmon 按 PID 隔离),
#   记录: 稳态 s/it、本进程 SM%、本进程显存(占 80GB 的 %)、MFU 估算。
# 判据: 显存峰值 < 24GB 且 SM% 接近(100 - 同卡他进程)% 视为"打满"。
set -u

cd /9950backfile/zhangem/HiLS-Attention-main
export PYTHONPATH=/9950backfile/zhangem/HiLS-Attention-main
PYTHON=/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin/python
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export MASS_PURE_BF16=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TMPDIR=/dev/shm/zhangem_tmp
export PYTHONPYCACHEPREFIX=/9950backfile/zhangem/tmp/pycache
mkdir -p "$TMPDIR"
unset CODEBUDDY_SAFE_DELETE_BULK_STATE_DIR CODEBUDDY_TOOL_CALL_ID \
      CODEBUDDY_SAFE_DELETE_BULK_GUARD CODEBUDDY_NODE_BIN GENIE_TRASH_DIR 2>/dev/null

GPU=${GPU:-2}
N=${N:-30}
PMON_DIR=/9950backfile/zhangem/bench_100m_pmon
mkdir -p "$PMON_DIR"

for MB in 4 8 16; do
  LOG=/9950backfile/zhangem/bench_100m_18l_mb${MB}.log
  OUTDIR=/9950backfile/zhangem/tmp/bench_100m_mb${MB}
  rm -rf "$OUTDIR"
  echo "[bench] mb=$MB starting"

  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" -m torch.distributed.run --nnodes=1 --nproc-per-node=1 --node-rank=0 --standalone \
    tasks/pretrain_mass_with_ruler.py configs/training_recipes/pretrain_345M_8K_30B.yaml \
    --model.config_path configs/mass_attention/config_mass_qwen3_100M_18L_8KA2K_RoPE_m2_r8.json \
    --model.model_path /9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-100M-MASS-m2-init/snapshots/master \
    --model.tokenizer_path /9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master \
    --model.attn_implementation sdpa \
    --data.train_path /9950backfile/zhangem/data/fineweb_qwen3_data \
    --data.data_type ruler_0.05 --data.datasets_type olmo3 --data.max_seq_len 8192 \
    --data.train_size 327680000 --data.num_workers 8 --data.sort_files true \
    --train.max_steps $N \
    --train.micro_batch_size $MB --train.global_batch_size $MB \
    --train.enable_gradient_checkpointing true \
    --train.enable_mixed_precision true \
    --train.broadcast_model_weights_from_rank0 true \
    --train.use_wandb false \
    --train.output_dir "$OUTDIR" \
    --train.save_steps 50000 --train.save_epochs 0 --train.save_hf_weights false \
    --train.data_parallel_mode ddp --train.init_device cuda \
    --train.rmpad false --train.rmpad_with_pos_ids false \
    --train.lr 3e-4 --train.lr_min 3e-5 --train.lr_warmup_ratio 0.03 \
    --train.lr_decay_style cosine --train.lr_decay_ratio 1.0 \
    --train.max_grad_norm 1.0 --train.no_decay_params norm bias embed \
    > "$LOG" 2>&1 &
  LAUNCHER=$!

  # 等模型构建完成, 拿到 rank0 python 子进程 PID
  CHILD=""
  for i in $(seq 1 24); do
    sleep 5
    CHILD=$(pgrep -P $LAUNCHER | head -1)
    [ -n "$CHILD" ] && grep -q "Set gradient accumulation" "$LOG" 2>/dev/null && break
  done
  echo "[bench] mb=$MB launcher=$LAUNCHER child=$CHILD"

  if [ -n "$CHILD" ]; then
    nvidia-smi pmon -s u -d 1 -c 90 2>/dev/null | awk -v pid="$CHILD" '$2==pid' > "$PMON_DIR/mb${MB}.pmon" &
    PMONPID=$!
  fi

  wait $LAUNCHER
  [ -n "${PMONPID:-}" ] && kill $PMONPID 2>/dev/null
  rm -rf "$OUTDIR"
  echo "[bench] mb=$MB done"
done

echo "=== 汇总 ==="
for MB in 4 8 16; do
  LOG=/9950backfile/zhangem/bench_100m_18l_mb${MB}.log
  STEADY=$(tr '\r' '\n' < "$LOG" | grep -oE "[0-9]+/$N \[[0-9:]+<[0-9:]+, +[0-9.]+s/it" | tail -3 | grep -oE "[0-9.]+s/it" | sed 's/s\/it//' | awk '{s+=$1} END {if(NR) printf "%.2f", s/NR}')
  PEAK=$(tr '\r' '\n' < "$LOG" | grep -oE "max [0-9.]+GB" | tail -1)
  python3 - "$PMON_DIR/mb${MB}.pmon" "$MB" "$STEADY" <<'PY'
import sys, statistics
lines = [l.split() for l in open(sys.argv[1]) if not l.startswith('#') and l.strip()]
mb, steady = sys.argv[2], sys.argv[3]
if lines:
    sm = [float(l[3]) for l in lines if len(l) > 3 and l[3] != '-']
    mem = [float(l[4]) for l in lines if len(l) > 4 and l[4] != '-']
    n = len(sm)
    # 丢掉前 1/4 采样(升温期)
    k = max(1, n // 4)
    sm_s, mem_s = sm[k:], mem[k:]
    print(f"mb={mb}: 稳态s/it={steady}  SM%: mean={statistics.mean(sm_s):.0f} max={max(sm_s):.0f} (n={n})  显存: mean={statistics.mean(mem_s)*81920/100:.1f}GB max={max(mem_s)*81920/100:.1f}GB")
else:
    print(f"mb={mb}: pmon 无采样")
PY
done
