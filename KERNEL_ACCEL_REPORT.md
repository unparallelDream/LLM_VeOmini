# 7B MASS 训练 kernel 剖析与加速分析报告

**日期**:2026-10-10(同日完成优化落地,见文末"实施结果")
**运行**:hy-127-59,GPU 0/1(A800 80G),产线 2-rank FSDP2 拓扑(mbs=1,gbs=8,每 rank 4 micro-batch/step),PURE_BF16 + 双锚 Arm B 全套 env(MASS_ANCHOR_REINIT/TWO_STAGE_BACKWARD/NCI/LMKQ_WARMSTART),14 步,profiler 录第 8–13 步(24 个 forward/backward pass)。
**Trace(优化前)**:`outputs/olmo7b_mass_kernel_profile/trace/veomni_rank0_1791617678.pt.trace.json.gz`
**Trace(优化后)**:`outputs/olmo7b_mass_kernel_profile/trace/veomni_rank0_1791621487.pt.trace.json.gz`
**分析脚本**:`scripts_zem/analyze_kernel_trace.py`(分类聚合)+ External-id 关联(把 tilelang 匿名 kernel `kernel_kernel` 归属到调用点)

> 说明:原计划的 fp32 主权重方案在本机不可行——veomni 的 fsdp2 分支没接 `enable_fsdp_offload`(仅 fsdp1 接线),fp32 全参状态(~112GB)双卡也放不下;且 tilelang 检索 kernel 入口硬性只收 BF16(fp32 需在边界转换)。本剖析按产线 PURE_BF16 方案执行,结论对 fp32 变体同样成立(dq kernel 的问题与精度无关)。

---

## 一、总览(6 个 profiled step,24 个 pass,kernel 时间合计 278.0s)

| 类别 | 时间 | 占比 | 说明 |
|---|---|---|---|
| **MASS 检索(tilelang,合计)** | **112.2s** | **40.4%** | 8 个检索层/趟(另 8 层共享 indices) |
| dense GEMM(cuBLAS 族) | 70.4s | 25.3% | 主干,接近不可约 |
| elementwise 族 | 20.2s | 7.3% | 大头来自检索 autograd 的 fp32 glue |
| cast(fp32↔bf16) | 17.7s | 6.4% | 同上 |
| HiLS 稀疏注意力(有名 kernel) | 38.9s | 14.0% | fwd 10.2s + **bwd 28.7s** |
| NCCL | 12.0s | 4.3% | AllGather 8.9s + ReduceScatter 3.1s |
| flash 窗口注意力 fwd/bwd | 4.0s | 1.4% | |
| optimizer(fused adamw) | 0.34s | 0.1% | |

GPU 并行占用 89.2%(297s 窗口);每步尾部有 ~4.5s 空泡(共 27.7s/9.3%),空泡内 CPU 无大 op——**主要是 profiler record_shapes 对 130 万 cpu_op 的放大开销**,生产(无 profiler)需另行确认,暂不计入优化目标。

## 二、MASS 检索 kernel 逐个拆解(grid 签名已关联到源码)

| kernel(源码) | 阶段 | 次数 | 均耗 | 合计 | 占总量 |
|---|---|---|---|---|---|
| **`mass_selected_dq_kernel`**(grid 8322×32) | **bwd** | 192 | **422ms** | **81.1s** | **29.2%** |
| `mass_selected_partial_bwd_kernel`(grid 261×32,67KB smem,两 launch/tangent) | bwd | 384 | 19.2ms | 7.4s | 2.6% |
| `mass_selected_partial_reduce_kernel`(grid 260×32) | bwd | 192 | 15.2ms | 2.9s | 1.0% |
| `mass_tangent_summary_bwd_kernel` | bwd | 192 | 23.7ms | 4.6s | 1.6% |
| `mass_envelope_candidates_kernel`(grid 261×32,occ 30%) | fwd(+重算) | 384 | 18.6ms | 7.1s | 2.6% |
| `mass_bitonic_topk_kernel`(grid 1041×32) | fwd(+重算) | 384 | 11.3ms | 4.3s | 1.6% |
| `recompute_topk_max_pooling_scores`(tensor-core 重算 GEMM) | fwd(+重算) | 384 | 12.0ms | 4.6s | 1.7% |
| `mass_sort_selected_indices_kernel` | fwd(+重算) | 384 | 0.6ms | 0.2s | 0.1% |
| `mass_tangent_summary_fwd_kernel` | fwd | 384 | 0.06ms | 0.02s | ~0% |

结构事实:梯度检查点使每个检索层的 envelope/topk/recompute 在 backward 阶段**再各跑一次**;`_MassEnvelopeTopKV2FnBackward` 每层 launch 4 个 kernel(dq + 2×partial + reduce),平均 476ms/层。

## 三、加速机会(按预期收益排序)

### 1. `mass_selected_dq_kernel` 是最大单点,占全部 GPU 时间 29.2% ★★★
- 现状:grid=(L,H,B)=8322×32,**每 block 只处理 1 个 query 行**,每线程标量循环 k=0..31 做 `grad_planes[k,m]·Mu[indices[k],m,d]` 的 gather-FMA,无 tensor core、纯延迟 bound。roofline 估算(≈10.6GB gather 读 @1.5TB/s)约 **3–6ms/次,实测 422ms,慢 ~100×**。
- **方案:改写成 gather-GEMM(tensor core)**。前向的 `recompute_topk_max_pooling_scores_kernel` 已经是同一问题的转置版本([block_l,D]@[D,K·M] 选行 GEMM);dq 只是把左操作数换成权重 `W[l,h,k,m]=grad_planes·sm_scale`,即 `[block_l, K·M] @ gather(Mu)[K·M, D]`,复用同一 gather-GEMM 骨架即可。产线 L=8322,H=32,K·M=64,D=128。
- 预期:422ms → ~2–5ms/层,每 pass 省 ~3.3s,**整体 kernel 时间 -28% 左右**,是所有优化里投入产出比最高的一项。
- 顺带:`GradQ` 可直接乘 `sm_scale` 融入 GEMM epilogue。

### 2. 两段式 backward 的 partial→reduce 合并(用户点名的问题)★★
- 现状(stage 结构,见 `mass_retrieval_tilelang_v2.py`):
  - **Stage 1** partial:按 (query_tile,h) 分块在 65KB shared 表里累加,再写 **dense 全局部分表** `[B,H,261 tiles,260 nm,128 D] fp32 ≈ 1.1GB`;M=2 个 tangent **各 launch 一次**(各写一半行),67KB smem 导致 occupancy≈0。
  - **Stage 2** reduce:把 261 个 tile 的部分表再读回来归约。
  - 拆两段的原因:direct 全局原子 kernel 在 tilelang free-mode 下每 256 线程块重复累加(代码里 /256 补偿的历史问题),跨 query-tile 原子冲突高。
- 合并路径(由易到难):
  - **(a) M 个 tangent 合一**:把 tangent 循环放进 kernel(shared `[N,D]` 每 tangent 复用两次全局写),或 `alloc_shared[N*M,D]`=133KB 走 dynamic smem(A800 164KB 上限内)。launch 数减半,预计省 ~3–4ms/层。
  - **(b) 消除 1.1GB 往返**:转置并行布局——block 拥有 (h, n-tile, d-tile),扫描全部 query tile 的 indices 归属于自己的 n(scatter→gather 反转)。N=130 小,每 block 的 membership 探测用 shared 位图/直接比较即可。目标 partial(19.2×2)+reduce(15.2) → 10ms 内,省 ~2% 总量。
  - **(c) 终态:与 dq 融合成单一 backward kernel**——dq 走 gather-GEMM 路径,partial/reduce 走转置块路径,合一个 launch,顺带把 `grad_planes` 的 fp32 链(masked_fill/logsumexp 导数)收进 epilogue。与机会 1 合计,`_MassEnvelopeTopKV2FnBackward` 的 91.4s(32.9%)有望压到 ~15s(≈5%)。
- 注意:**单独做 (a)/(b) 只动 10.3s(3.7%),真正的量在 dq(81.1s)**——合并工程应与机会 1 同批做。

### 3. 检索 autograd 的 fp32 glue(cast 6.4% + elementwise 大头)★★
- forward 里 `gather→masked_fill→logsumexp→softmax→routing_probs` 全在 PyTorch fp32 走,~50 个小 op/层,还把 routing_probs 存 bf16 再回 fp32。
- envelope kernel **本来就在算每个 (q,n) 的 logsumexp**(代码 111-117 行);让它在选出 top-k 的同时直接写出 `routing_probs[l,h,k,m]`(或 scores 与 winner 平面),forward 的整条 glue 与 cast 大部分可删。预计 3–5% 总量。
- backward 的 `grad_planes` 组装(grad_scores·probs + load/entropy Jacobian)同理可并入 dq/partial 的输入准备。

### 4. `hils_bwd_dqkv_block_M_inverse_reuseIndices`(10.3%)★★
- 150ms/次×192,稀疏注意力 backward 的 dqkv 累加。独立优化方向(分块/向量化/原子策略),量级与两段式合并相当,建议单独立项。

### 5. envelope candidates 前向(2.6%)与 summary bwd(1.6%)★
- envelope occ 30%、block_n=16 串行 9 轮小 GEMM,可调 block 尺寸/双缓冲;summary bwd 23.7ms 有改善空间但收益 ≤2%。

### 不建议动的
- dense GEMM 25.3%(主干)、NCCL 4.3%(FSDP 通信,拓扑决定)、optimizer 0.1%。
- 若未来按原计划换 fp32 主权重:每层 all-gather 后多一次 fp32→bf16 cast、AdamW 状态翻倍(优化器时间 ~0.3s→1.2s)、NCCL reduce 流量×2;且需先给 veomni fsdp2 接上 offload 或换 ≥4 卡,当前 2×A800 放不下。

## 四、建议实施顺序

1. **dq gather-GEMM 重写**(单 kernel,可独立验证:数值等价 + 单测基准,预期 -28%)
2. **partial/reduce 合并(a→b)**(与 1 同一文件,预期再 -2~3%)
3. **routing_probs 由 envelope 直写,删 forward fp32 glue**(-3~5%)
4. `hils_bwd_dqkv` 专项(潜在 -5~8%)
5. 生产环境(无 profiler)复测 step 边界空泡是否真实存在

合计潜力:kernel 时间约 **-35~40%**(27s/it → ~17-19s/it 量级,以无 profiler 复测为准)。

## 附:复现

```bash
bash outputs/olmo_7b_dual_anchor_experiment/launch_7bB_kernel_profile_1gpu.sh 0,1
python scripts_zem/analyze_kernel_trace.py outputs/olmo7b_mass_kernel_profile/trace/<trace>.json.gz --top 35
```

---

# 实施结果(2026-10-10 当日落地)

改动文件:`ops/mass_retrieval_tilelang_v2.py`(原版备份 `ops/mass_retrieval_tilelang_v2.py.bak_20261010`),新增测试 `unittests/test_mass_dq_gemm.py`。

## ⚠ 先说重大附带发现:产线两段式 partial kernel 数值错误

优化过程中的三路对照(dense einsum 参考 vs direct kernel vs 两段式 partial+reduce)证实:
- **direct kernel 与参考完全一致**(偏差 1e-3,舍入级);
- **产线在用的两段式 partial+reduce 路径(MASS_TWO_STAGE_BACKWARD=1)数值错误**:35% 的 grad_mu 单元、73% 的 grad_bias 单元偏差 >0.05,最大偏差 31.8(随机数据,L=512/N=130 生产形状)。

**根因**:partial kernel 的清零/写回按 `row_tile=8` 分轮(`for rb in T.serial(cdiv(N,8))` 包着 `T.Parallel(8, D)`),与累加段对同一 shared 表的动态下标原子访问在 tilelang free-mode 布局推导下产生冲突的布局假设,丢/重共享原子更新(纯计数测试可差 10+)。旧源码注释里"cannot infer a legal free-mode layout … process eight selected chunks per round"描述了这个风险但从未实现缓解。改成整块 `T.Parallel(N, D)` 清零/写回后,所有形状精确归零误差。

**影响**:56 节点正在跑的 13000 步 run 自 step 0 起,dmu/dbias(流向 mu/bias → 锚生成器与 key 路径的梯度)一直是错的(噪声级污染,dq 是好的)。**建议从最新检查点用修复后的 kernel 重启**。direct 路径虽正确但慢 455×(18.5s vs 40.7ms,free-mode 256 线程复制),不可用。

## 实施的两项优化(默认启用,env 可回退)

1. **`mass_selected_dq_gemm_kernel`(新)**:dQ 从"每 block 一个 query 行"的标量 gather 循环改为块对角 gather-GEMM(与前向 recompute kernel 同族,bf16 tensor core + fp32 累加)。数值:与 fp32 参考相对偏差 0.2%(bf16 权重量化,与 forward 对 q 的处理同量级)。回退:`MASS_DQ_LEGACY=1`。
2. **`mass_selected_partial_bwd_mt_kernel`(新)**:tangent 折进 grid 头维(by = h*M+t),M 次 launch 合一;累加段重构为 Parallel(q,d)+serial k;清零/写回整块化(修复上述数值 bug)。抽样 24 个 (h,tile) 块与 dense 参考精确一致。回退:`MASS_PARTIAL_LEGACY=1`(注意:回退即回到错误数值)。

## 基准与端到端

单 kernel 微基准(产线形状 L=8322/N=130/K=32/M=2/D=128,A800):

| kernel | 优化前 | 优化后 | 加速 |
|---|---|---|---|
| dQ | 422.0ms | **7.2ms** | **58×** |
| partial(M=2) | 30.0ms(2 launch) | 17.8ms(1 launch) | 1.7× |

端到端(同 14 步冒烟、同 profile 窗口 8–13 步、rank0):

| 指标 | 优化前 | 优化后 | 变化 |
|---|---|---|---|
| 剖析窗口 | 297.2s | 158.9s | **-46.5%** |
| kernel 时间合计 | 278.0s | 166.0s | -40.3% |
| `kernel_kernel`(MASS tilelang 族) | 107.6s(38.7%) | 20.0s(12.1%) | -81% |
| `_MassEnvelopeTopKV2FnBackward` 全部 | 91.4s(32.9%) | 6.3s(3.8%) | -93% |
| GPU 占用率 | 89.2% | **97.5%** | |
| 最大空泡 | 4.59s | 0.21s | -95% |
| 14 步总时长(含启动/JIT) | 13:41 | 8:47 | -36% |
| profiled step 墙钟 | 40–75s/it | 21–28s/it | ~-50% |

产线预估:独占卡 27s/it → **~14–16s/it**(13000 步约省 40 小时;以无 profiler 复测为准)。loss/退火/grad_norm 冒烟核查全部正常。

## 剩余热点(下一批)

1. `hils_bwd_dqkv_block_M_inverse_reuseIndices`(现为 24.4s/14.7%,150ms/次×192)——下一个最大单点;
2. dense GEMM 36.4%(主干,不动);
3. fp32 glue(cast 9.2% + elementwise 10.1%):envelope 直写 routing_probs 方案未实施,预期再 -3~5%;
4. envelope fwd(15.6ms/次)与 summary bwd(20.5ms/次)有中等优化空间。

---

# 补充:hils_bwd_dqkv 热点处理(2026-10-10 第三批)

## 单锚问题的回答

`hils_bwd_dqkv` / `hils_block_M_reuseIndices`(ops/hils_fwd_bwd_head.py)是**稀疏注意力核心**,本身没有"锚"的概念:它只消费检索层产出的 top-k 块索引 + 每块权重,锚数(M=1 单锚 / M=2 双锚)只存在于上游的 MASS 检索(mass_envelope/partial 等 kernel)。单锚(Arm A / HiLS 基线)与双锚(Arm B / MASS)用的是**同一个** hils 注意力 kernel——所以这个 kernel 的优化对两臂同时生效。

## 慢因(trace + 源码)

- bwd:grid=(521,32),**block 只有 64 线程**,85KB shared,**255 寄存器/线程(溢出边缘)**→ 每 SM 仅驻 1 个 block ≈ 3% 占用率,K/V 加载延迟完全暴露(`num_stages=0` 无流水)。
- 每个 (16-query, head) 块串行跑 merged chunk 列表,每块 4 个小 GEMM + dk/dv 全局原子。

## 做了基准的旋钮(产线形状 B1/L8322/H32/G1/D128/S32)

| 配置 | 每-query 随机索引(merged_len≈483,重负载) | chunk 共享索引(merged_len=32,轻负载) |
|---|---|---|
| t=64(产线现状) | 5071ms | 332ms |
| t=128 | **3524ms(-30%)** | 322ms(-3%) |
| t=256 / stages=2/3 | ~3462ms(不再提升) | ~308ms(-7%) |
| block_M=32 | 编译 187KB smem > A800 164KB,非法 | 同 |

数值:三档线程配置的 dq/dk/dv 与纯 torch 真值全部一致(rel≤0.008),线程数不改变语义。

## 已接线(待端到端复测)

`mass_attention.py`:GQA else 分支的 `sparse_num_threads_bwd` 64→`max(128, …)`(fwd 不变)。kernel 侧把 `hils_sparse_attention_bwd…(num_stages=0)` 的硬编码暴露为参数(默认 0,行为不变)。**端到端冒烟因 hanjs 新作业占满 8 卡暂未复测**,GPU 空出后重跑 `launch_7bB_kernel_profile_1gpu.sh 0,1` 即可验证。

## 剩余结构性优化空间(真实存在,需要重写)

线程/流水旋钮天花板 ~30%(重负载情形);要再往下打必须动结构:
1. **smem 减脂**(85KB→<54KB 可 3 block/SM):`merged_s_indices` shared 占 32KB(S*M×M int32)可移 global 或按需加载;P_shared/dO_weighted 可复用;
2. **merged 列表切分并行**:把每块的串行 chunk 循环切到 Z 个 block(dk/dv 已是原子,dq 需原子化或 partial+reduce);
3. flash 式重写(寄存器驻留 + TMA)。估算合计可再 2–3×(占端到端 ~5–8%)。

---

# 补充:200M 形状同类问题核查(2026-10-10 第四批)

200M 产线(config_mass_olmo3_200M_8KA2K_RoPE_m2_sharedview_lmkq.json,launch_olmo200m_final_recipe_20261006.sh)形状 B1/L8322/N130/**H4/Hkv4(G=1)**/M2/D128/K32,`MASS_TWO_STAGE_BACKWARD=1` 且 config `mass_two_stage_backward: True` → **走的同一条 buggy partial 路径**。

实测(unittests/test_mass_200m_kernels.py):

| kernel | 结果 |
|---|---|
| legacy 两段式 partial | **同样数值错误**:36.4% grad_mu、73.9% grad_bias 单元偏差>0.05,max dbias 90.8 |
| 修复版合并 partial | 0.00% 偏差(精确) |
| dq 旧标量 | 正确,88.6ms |
| dq gather-GEMM | 正确(rel 0.0019),**1.84ms,48×** |
| partial 合并(2 launch→1) | 8.8ms → 4.5ms |

三项修复(dq GEMM、partial 修复+合并、hils bwd 线程)对 200M **自动生效**(共享同一份 ops 代码,无需改配置)。

**影响面**:已完成的 200M CPT(15000 步,final recipe)自始至终带着被污染的 dmu/dbias 梯度训练(锚生成器/landmark key 路径);dq 路径与 LM 主干梯度正常。当前在跑的 lmeval 200M 评测是纯推理(is_training=False,不走 backward),不受影响。是否用修复后 kernel 重训 200M 由实验决定。
