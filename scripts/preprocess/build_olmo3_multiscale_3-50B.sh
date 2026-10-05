#!/bin/bash
# 分块下载 allenai/dolma3_mix-6T-1025-7B 并采样出 3B/10B/20B/50B 嵌套 token 子集。
#
# 带宽实测(2026-09-10): hf-mirror 对本机限流(~1MB/s, 并发压测后曾临时封 IP);
# NFS 写 ~64MB/s。瓶颈在下载 -> 并发 1 顺序处理, 不打满带宽。
# --download_budget_gb 40: 只下载覆盖 50B tokens 的分片子集(50B/5.93T × 4.05TB
# ≈ 34GB 压缩分片, 全量是 4TB), 配额在选中分片上按大小比例重新分配。
#
# 流程: 先跑一次小规模冒烟验证整条流水线, 通过后自动进入正式运行。
# hf-mirror 间歇不可达时每 5 分钟整体重试(进度按分片路径保留, 断点续跑),
# 默认重试 60 次(约 9 小时窗口), 可用 ATTEMPTS= 环境变量覆盖。
# 所有数据(下载缓存/staging/输出)均在 BASE 下; 单分片下载失败自动退避重试 5 次。
#
# 用法: bash scripts/preprocess/build_olmo3_multiscale_3-50B.sh
# 监控: tail -f <BASE>/build.log    停止: kill $(cat <BASE>/build.pid)
set -uo pipefail
cd "$(dirname "$0")/../.."

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export TOKENIZERS_PARALLELISM=false

PYTHON=/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin/python
SCRIPT=preprocess/download_sample_dolma3_multiscale.py
BASE=/9950backfile/zhangem/data/dolma3_multiscale
SMOKE=$BASE/smoke
ATTEMPTS=${ATTEMPTS:-60}

mkdir -p "$BASE"

for i in $(seq 1 "$ATTEMPTS"); do
  echo "=== attempt $i/$ATTEMPTS: smoke ==="
  rm -rf "$SMOKE"
  if nice -n 19 ionice -c3 $PYTHON $SCRIPT \
      --vocab_dir configs/olmo3_vocab \
      --scales 3000000 10000000 \
      --download_budget_gb 0.09 --limit_files 30 \
      --shm_dir "$BASE/hf_cache" \
      --staging_dir "$SMOKE/staging" \
      --output_dir "$SMOKE/out" \
      --list_cache "$BASE/shard_list.json" \
      --num_workers 1 --seed 1234; then
    echo "=== smoke ok, starting full run ==="
    rm -rf "$SMOKE"
    if nice -n 19 ionice -c3 $PYTHON $SCRIPT \
        --vocab_dir configs/olmo3_vocab \
        --scales 3000000000 10000000000 20000000000 50000000000 \
        --download_budget_gb 40 \
        --shm_dir "$BASE/hf_cache" \
        --staging_dir "$BASE/staging" \
        --output_dir "$BASE" \
        --list_cache "$BASE/shard_list.json" \
        --num_workers 1 --seed 1234; then
      echo "=== ALL DONE ==="
      ls -lh "$BASE"/*.numpy
      exit 0
    fi
    echo "=== full run failed, will retry ==="
  fi
  echo "=== retry in 300s ==="
  sleep 300
done

echo "=== giving up after $ATTEMPTS attempts ==="
exit 1
