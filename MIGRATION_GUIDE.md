# 7B 双锚(mass)迁移指南 — 换 hard-NFS 节点续跑

**日期**:2026-10-10
**用途**:把 56 节点(soft NFS,需 shm 暂存+mover)上的 7B mass 训练迁到 hard-NFS 节点(如 47),直写检查点,简化架构。**56 节点当前 run 继续挂着不动**。

---

## 一、前置检查(目标节点执行)

```bash
# 1. 确认 NFS 挂载为 hard
mount | grep 9950-backfile | grep -o "hard\|soft"
# 2. 确认能看到共享存储
ls /9950backfile/zhangem/HiLS-Attention-main/tasks/pretrain_mass_with_ruler.py
# 3. 检查有空闲 GPU(需 2 张,各 ≥52GB 空闲)
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
# 4. 创建节点本地 tmpfs 目录(tilelang 缓存必须本地,NFS 会 mmap 崩溃)
mkdir -p /dev/shm/tilelang_cache /dev/shm/zhangem_tmp
```

## 二、断点信息

56 节点当前进度:step ~2500/13000(以 `logs/olmo7b_mass_cpt.log` 尾部为准)。
NFS 侧最新完整检查点:查看 `outputs/olmo7b_mass_cpt13000/checkpoints/` 中**最新含 `.metadata` 的 `global_step_*`**:

```bash
for d in /9950backfile/zhangem/outputs/olmo7b_mass_cpt13000/checkpoints/global_step_*; do
  [ -f "$d/.metadata" ] && echo "$d"
done | sort -t_ -k3 -n | tail -1
```

## 三、启动命令(hard 节点变体,无 shm/mover)

固化脚本:`outputs/olmo_7b_dual_anchor_experiment/launch_7bB_fsdp_hardnode.sh`

```bash
# 冒烟(12 步,验证续跑正常):
bash /9950backfile/zhangem/outputs/olmo_7b_dual_anchor_experiment/launch_7bB_fsdp_hardnode.sh "<两卡号>" smoke nooff \
  && echo "冒烟 OK"

# full(13000 步,直写 NFS):
bash /9950backfile/zhangem/outputs/olmo_7b_dual_anchor_experiment/launch_7bB_fsdp_hardnode.sh "<两卡号>" full nooff
```

脚本与原版差异:无 `TRAIN_SAVE_CKPT_DIR`、无 mover 后台进程、DCP 写线程=8(hard 下无需限速)、检查点直写 `output_dir/checkpoints/`。

## 四、断点续跑(可选)

如果需要从 56 节点的最新检查点续跑而非从头,在启动命令中追加:

```bash
--train.load_checkpoint_path /9950backfile/zhangem/outputs/olmo7b_mass_cpt13000/checkpoints/global_step_<N>
```

**注意**:当前 run 的 shared_view_bias 是 `(1,)` 形状(10-07 修复后),新代码构建也是 `(1,)`,无需标量技巧。

## 五、关键 env(脚本已内置,备忘)

```bash
MASS_PURE_BF16=1                    # 纯 bf16 存储(参数+梯度+AdamW)
MASS_ANCHOR_REINIT=1               # 锚生成器用 config 初始化
MASS_LSE_ANNEAL_START=4.0           # 退火起点
MASS_LSE_ANNEAL_END=0.5             # 退火终点
MASS_TWO_STAGE_BACKWARD=1           # 两段式反向
MASS_TRAINABLE_ONLY_CHECKPOINT=1    # 全参下等效全量
MASS_NEEDLE_INFO_NCE=1              # NCI w=2
MASS_NCI_WEIGHT=2.0 MASS_NCI_TAU=1.0 MASS_NCI_QUERY_ROWS=64
MASS_LMKQ_WARMSTART=1               # lmk_q warm-start(低秩上投影零初始化)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TILELANG_CACHE_DIR=/dev/shm/tilelang_cache   # 必须本地!
```

## 六、与 56 节点并行运行的冲突说明

- **可以并行**:两节点写同一 NFS 目录的检查点不冲突(DCP 按 rank 分片文件名,但步数不同);
- **建议不并行**:如果两边同时跑 full,检查点会交错,推理/评测读哪个取决于步数排序。**建议迁移时二选一**:
  - 方案 A:停 56 的 run → NFS 上最新 ckpt 续跑(零损失,推荐)
  - 方案 B:56 继续跑,目标节点从头跑(浪费算力但零风险)
- 本文档按方案 A 编写:先停 56 的 run,再在目标节点启动。

## 七、停 56 节点 run(迁移前执行,56 上)

```bash
# 找到 7B mass 的 torchrun 主进程
TP=$(ps -ef | grep "[o]lmo7b_mass_cpt13000" | grep torch.distributed.run | awk '{print $2}' | head -1)
kill -TERM $TP && echo "stopped $TP"
sleep 20
# 确认 GPU 释放
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 4,3
# 等 mover 把 shm 里的最新 ckpt 全部搬到 NFS(shm 清空即完成)
while [ -n "$(ls /dev/shm/ckpt_stage/olmo7b_mass_cpt13000/ 2>/dev/null)" ]; do
  echo "shm 尚有 $(ls /dev/shm/ckpt_stage/olmo7b_mass_cpt13000/ | wc -l) 个待搬"
  sleep 30
done
echo "shm 已清空,可以安全迁移"
```

## 八、启动后核查清单

1. step 1-3 正常出 loss(~2.x,非 10+)
2. 日志出现 `[mass-anchor-check] param_absmax>0 grad_absmax>0`(锚有梯度)
3. 日志出现 `[mass-anneal] step 1/13000 lse_temperature≈4.0`(退火激活)
4. step 200 时检查点落 NFS(无 `.incomplete` 残留)
5. GPU 显存稳定(每卡 ~50-55GB,纯 bf16 方案)
6. 速度:独占卡 ~27s/it;共享卡会更慢

## 九、环境备忘

- Python: `/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin/python`(共享 NFS,两节点同一份)
- 代码仓: `/9950backfile/zhangem/HiLS-Attention-main`(共享 NFS,含全部修复)
- 模型基座: `.cache/modelscope/models/allenai--Olmo-3-1025-7B-HiLS/snapshots/master`(共享 NFS)
- 数据: `data/dolma3_1025_3B_packed/`(共享 NFS,3B tokens)
- 42B 备用包: `data/dolma3_1025_42B_packed/`(共享 NFS,41.88B tokens)

—— 完 ——
