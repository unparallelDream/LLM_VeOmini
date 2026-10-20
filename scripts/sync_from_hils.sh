#!/bin/bash
# 把 HiLS-Attention-main 的最新修改同步进 LLM_VeOmni（2026-10-20 首次人工同步的固化脚本）。
# 规则：
#   1) 内容有差异的文件 → 用 HiLS 版本覆盖，唯 models/__init__.py 与 .gitignore 保留本仓库版本
#      （前者本仓库额外导入了 GatedMLAHybrid；后者含本机软链 outputs/logs/.cache 的忽略项；
#       若哪天 HiLS 侧也改了这两个文件，需人工三路合并后更新豁免清单）
#   2) 仅存在于 HiLS 的新文件/目录 → 拷入
#   3) HiLS 侧 git 删除的文件 → 本仓库同步删除
#   4) 本仓库独有文件（GatedMLAHybrid/、configs/gated_mla_hybrid/、tasks/pretrain_gated.py、
#      scripts/gated_mla_smoke.sh、tmp/、outputs|logs|.cache 软链）→ 不受影响
# 用法: bash scripts/sync_from_hils.sh [--dry-run]
set -e
SRC=${SRC:-/9950backfile/zhangem/HiLS-Attention-main}
DST=$(cd "$(dirname "$0")/.." && pwd)
DRY=${1:-}
cd "$(dirname "$SRC")"
SRC_NAME=$(basename "$SRC")

diff -rq "$SRC_NAME" "$(basename "$DST")" \
  --exclude=.git --exclude=__pycache__ --exclude="*.pyc" --exclude=.pytest_cache \
  --exclude=tmp --exclude=outputs --exclude=logs --exclude=.cache 2>/dev/null | python3 -c "
import sys, os, shutil
dry = '$DRY' == '--dry-run'
for line in sys.stdin:
    line = line.strip()
    if line.startswith('Files '):
        a = line.split(' and ')[0][len('Files $SRC_NAME/'.replace('\$','')):]
        a = line.split(' and ')[0]
        a = a[len('Files '):].split(' and ')[0]
        a = os.path.relpath(a, '$SRC_NAME')
        if a in ('models/__init__.py', '.gitignore'):  # 本仓库两侧各自维护的文件
            print('SKIP(ours)', a); continue
        print('UPD', a)
        if not dry: shutil.copy(os.path.join('$SRC_NAME', a), os.path.join('$(basename "$DST")', a))
    elif line.startswith('Only in $SRC_NAME'):
        rest = line[len('Only in $SRC_NAME/'):]
        d, n = rest.rsplit(': ', 1)
        dst_path = os.path.join('$(basename "$DST")', d, n)
        if os.path.exists(dst_path):
            print('SKIP(exists)', d + '/' + n); continue
        print('NEW', d + '/' + n)
        if not dry:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            s = os.path.join('$SRC_NAME', d, n)
            shutil.copytree(s, dst_path, dirs_exist_ok=True) if os.path.isdir(s) else shutil.copy(s, dst_path)
"
cd "$SRC"
git status --short | awk '$1=="D" {print $2}' | while read f; do
  if [ -f "$DST/$f" ]; then echo "DEL $f"; [ "$DRY" != "--dry-run" ] && rm "$DST/$f"; fi
done
echo "=== SYNC DONE (${DRY:-applied}) ==="
