# MASS / HiLS 实验状态报告

**日期**:2026-09-05 15:30 前后
**机器**:hy-a800-127-56,A800-SXM4-80GB ×8(多用户共享,全部满载)
**环境**:`/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311`
**代码仓**:`/9950backfile/zhangem/HiLS-Attention-main`(HiLS-Attention,MASS 为其扩展)

---

## 一、实验总览:流水线与当前状态

```
Qwen3-0.6B(官方,~36T tokens 预训练)
├─[宽度裁剪 90M]→ pruned-90M → [recovery 预训练 ✅] → recovered-90M
│      └─[移植 MASS m2]→ 90M anchor-FT 🔄(65%,GPU0)
│      └─[移植 HiLS]  → 90M HiLS 对照 🔄(1%,GPU6)
└─[直接移植 MASS m2]→ 0.6B anchor-FT(ft20000)✅ 完成
       └─[剥离 anchor]→ 0.6B HiLS 对照 🔄(1%,GPU7)
```

| # | 任务 | 状态 | GPU | 速度 | 预计完成 |
|---|---|---|---|---|---|
| 1 | 90M 裁剪基座 recovery(15000 步) | ✅ 9/3 09:42 | (3) | 28.9s/步 | — |
| 2 | 0.6B anchor 退火 ft20000 | ✅ 9/5 08:04 | (2) | 36.1s/步 | — |
| 3 | 90M anchor-FT | ✅ 9/6 09:08(lm 3.514,spec 0.118) | (0) | 12.4s/步 | — |
| 4 | 90M HiLS 对照(lmk_embed-only) | ✅ 9/6 08:14(loss 3.490) | (6) | 3.3s/步 | — |
| 5 | 0.6B HiLS 对照(lmk_embed-only) | ✅ 9/7 13:41(loss 5.449;终局评测 hils_t=0.1626/EM 0.0133) | (7) | 8.7s/步 | — |
| 6 | 0.6B anchor + 全 token mixer LoRA(qkvo, r16, 462 万参) | ✅ 9/13 22:06(lm 2.809,spec 0.006;9/11 崩溃一次已恢复) | (0) | 15.9s/步 | — |
| 7 | 90M anchor + 全 token mixer LoRA(qkvo, r16, 129 万参) | ✅ 9/9 11:48(lm 3.374;**评测 0.2596 超 full 5pt**) | (6) | 9.3s/步 | — |
| 8 | 90M hils+LoRA 对照(lmk_embed + 同款 LoRA) | ✅ 9/10 08:06(loss 3.366;**评测 0.3425 与 mass_lora 打平——归因完成**) | (6) | 2.6–4.1s/步 | — |
| 9 | **0.6B hils+LoRA**(lmk 1024 + LoRA 462 万) | 🔄 起步正常(trainable=225) | 0 | 5.6s/步 | ~9/16 晚 |
| 10 | **0.6B bsa+LoRA**(均值池化,零检索参数,纯 LoRA) | 🔄 起步正常(trainable=224) | 6 | 16.3s/步 | ~9/19 |
| 11 | **90M bsa+LoRA**(同款,三-way 归因快速版) | 🔄 起步正常(trainable=224) | 7 | 4.6s/步 | ~9/16 午 |

注:#9–11 由 `launch_lora_controls_20260915.sh` 启动(2026-09-15);bsa 臂走 HiLSModel 的
`use_naive_bsa` 原生路径 + `bsa_score_mode=mean`(自实现均值打分,可微)。
三-way 归因设计:LoRA+anchor(mass_lora)/ LoRA+lmk(hils_lora)/ 纯 LoRA+均值(bsa_lora)——
分离"表征适配"与"检索侧训练"各自贡献;90M 侧三臂已齐待 bsa_lora,0.6B 侧三臂在跑。

注:#6/#7 由 `launch_anchor_lora_kv_ft.sh` 启动(2026-09-06 16:45),LoRA 注入经零初始化位级
等价与设备/属性代理修复后验证通过(`utils/lora_adapt.py` + `pretrain_mass_with_ruler.py` 的
MASS_LORA_ADAPT 门控);trainable=238(14 anchor + 224 lora),trainable-only checkpoint。
起步信号:90M 首步 lm 3.51(冻结臂同期 3.62);0.6B 首步 lm 5.76(冻结臂 5.87)——LoRA 从第一步
就在收窄 loss,grad_norm 偏高(8.1,clip 1.0 主导早期更新),warmup 后观察。

注:#3 于 9/5 06:07 因 dataloader 共享内存瞬时耗尽崩溃于 12609 步,从 12500 存档恢复,损失 109 步;#4 启动时因 90M snapshot 词表尺寸不匹配失败两次(embed 须 151968 行/lm_head 151936 行),已重建。

---

## 二、训练配置(所有 run 共用骨架)

### 2.1 数据与优化

| 项 | 值 |
|---|---|
| 数据 | fineweb_qwen3_data,`ruler_0.05` 混合(≈5% RULER 式长程合成 + 95% fineweb),olmo3 打包 |
| 数据池大小 | 327,680,000 tokens(合 5000 步/遍) |
| **训练序列长度** | **8192**(打包;landmark 在模型 forward 内自动插入,63+1/块) |
| batch | micro 4 × grad-accum 2 = global 8 → **65,536 tokens/步** |
| 优化器 | AdamW,lr 2e-5 → 2e-6 cosine,warmup 3%,grad clip 1.0 |
| 步数 | 20,000 步(anchor-FT 与 HiLS 对照)/ 15,000 步(recovery,lr 3e-4→3e-5) |
| 冻结策略 | anchor-FT:仅 `mass_anchor_generator.{anchor_embedding,offset_basis}`(14 张量,28,784 参数);HiLS 对照:仅 `model.lmk_embed`(0.6B:1024 / 90M:320 参数);其余 311 张量全冻结(FP32 优化器态,backbone BF16) |
| checkpoint | 每 500 步;HiLS 对照启用 trainable-only(只存 lmk_embed) |

### 2.2 模型结构(两规模同构)

| 项 | 0.6B | 90M |
|---|---|---|
| 来源 | Qwen3-0.6B 原始权重 | 0.6B 宽度裁剪(hidden 1024→320 等)+ recovery 0.98B tokens |
| 层数/隐藏维 | 28 层 / 1024 | 28 层 / 320 |
| 注意力头 | 16 q / 8 kv,head_dim 128,GQA=2 | 4 q / 2 kv,head_dim 128,GQA=2 |
| 参数量 | ~596M(+lmk_embed 1024) | **89,932,608**(tie 修正后 untied 存储) |
| MASS/HiLS 检索层 | **[3,7,11,15,19,23,27] 共 7 层**(每 4 层 1 个),其余 21 层普通 SWA | 同左 |
| 词表 | 151,936(输入侧嵌入补齐 151,968 行给 lmk_id 留位) | 同左 |
| 位置编码 | RoPE,theta=1e6,max_position_embeddings=40,960 | 同左 |

### 2.3 稀疏注意力预算(训练与评测一致)

| 项 | 值 |
|---|---|
| chunk_size | 64(63 真实 token + 1 块尾 landmark) |
| 局部滑窗 | 512 |
| 远程 top-k | 32 块 × 64 = 2048 |
| **KV budget/query** | **2560 tokens**(512 局部 + 2048 远程),landmark 值键掩码 |
| MASS anchor | M=2/层,offset rank 8;LSE 温度 4.0→0.1 全程退火;spec_loss 权重 1.0 |
| HiLS 打分 | landmark 键 + prior-query 路径(结构打分,无可学习检索参数,除 lmk_embed) |

### 2.4 五种检索打分(评测变体定义)

| 变体 | 块打分 | 打分开销 | 可学习参数 |
|---|---|---|---|
| mass | 2-anchor 低秩摘要(训练后) | O(L·d) 级 | 28,784(已训 20K 步) |
| fixed0 | anchor0 复制消融 | 同上 | 0(评测时塌缩) |
| hils | landmark 键打分(原生) | O(L·N·d) | 0(零样本)/ lmk_embed(对照组训练中) |
| bsa | 块均值池化 q·k̄ | O(L·N·d) | 0 |
| dsa | 逐 token logsumexp 块质量 | O(L²·d) | 0(质量上界参照) |

---

## 三、Token 消耗与 Chinchilla 分析

| run | 步数 | tokens | 状态 |
|---|---|---|---|
| 90M recovery | 15,000 | 9.83 亿 | ✅ |
| 0.6B anchor-FT | 20,000 | 13.11 亿 | ✅ |
| 90M anchor-FT | (13,081) | (8.58 亿)/ 13.11 亿 | 🔄 |
| 0.6B HiLS 对照 | (130) | (0.09 亿)/ 13.11 亿 | 🔄 |
| 90M HiLS 对照 | (218) | (0.14 亿)/ 13.11 亿 | 🔄 |
| **已完成合计** | | **≈ 22.9 亿** | |
| **全部计划合计** | | **≈ 62.2 亿** | |

- **90M 流水线 Chinchilla 比**:(0.98 + 1.31)B ÷ 89.9M ≈ **25.5×**,落在 20–30× 最优区间(名义口径);
  但数据池去重仅 3.28 亿(unique ÷ params ≈ 3.6×),整个流水线约重复 7 遍,等效 fresh tokens 明显低于名义值。
- **0.6B 侧不适用 Chinchilla**(backbone 继承官方 ~36T 预训练,增量微调)。
- 吞吐:0.6B anchor-FT ≈ 1,810 tok/s;90M anchor-FT 3.7(GPU 空闲时)~12.4(抢占)s/步;HiLS 对照因无 spec loss 与双段反传,单步更快(90M 仅 3.4s)。

---

## 四、Loss 轨迹与 Gap

### 4.1 各 run 起点/当前/终点

| run | 起点 lm_loss | 当前/终点 lm_loss | spec_loss 轨迹 |
|---|---|---|---|
| 90M recovery(全参,dense) | 9.95 | **3.24**(终) | — |
| 90M anchor-FT | 3.62(step10,冻结+新anchor) | **3.49** @13081(当前) | 0.966 → 0.780,负载 0.419/0.581 |
| 90M HiLS 对照 | ≈3.54 | **3.39** @218(早期) | —(无spec) |
| 0.6B anchor-FT | 5.87(step1) | **5.12**(终,step20000) | 0.99 → **0.152**,负载 0.430/0.570,无 dead anchor |
| 0.6B HiLS 对照 | ≈5.9 | 5.71 @130(早期) | — |

### 4.2 Loss Gap 读法

- **0.6B dense 参照(新测,2026-09-05)**:原始 Qwen3-0.6B(全注意力)在 fineweb@8192 上
  **CE = 3.089**(48 条序列,~0.4M tokens,test 命名空间)。按 ruler_0.05 混合的合成噪声
  (~5% 高熵 RULER 文本)修正后,该混合上的 dense 参照约 3.3。
- **0.6B 冻结代价 ≈ 1.5–2.0 nats**:冻结 backbone 的 mass 训练 lm 从 5.87 → 5.12,
  距 dense 参照(~3.3)仍差 ~1.8 nats,anchor 训练仅收回了 0.75。
  **backbone 不动,这个 gap 无法闭合**——选择器是在"未适配稀疏检索的表征"上打分。
- **90M 检索开销 gap**:recovery(dense 全注意力)= 3.24 为该 backbone 在此数据上的最优参照;
  MASS anchor-FT 当前 3.49 → gap ≈ +0.25 nats(冻结 backbone + 稀疏检索的代价,且 anchor 仍在收敛);
  HiLS 对照早期 3.39 → gap ≈ +0.15 nats。90M 侧 gap 远小于 0.6B,因其 backbone 已在同数据上
  recovery 适应过、且小模型对长程检索依赖更低。
- **教师参照**:裁剪时校准集(12×8192)上教师 Qwen3-0.6B CE=3.06,恢复后 90M 为 3.24(差 0.18)。
- **0.6B**:anchor 训练使 lm 从 5.87→5.12(−0.75),全部来自检索质量改善(backbone 冻结)。
  零样本 HiLS 损失 ≈5.7–5.9,与 anchor 训练前相当。
- spec_loss 是 anchor 分化指标:0.6B 完整退火后 0.152(强分化);90M 当前 0.78(温度退火约 55% 处)。

### 4.3 冻结 vs 全参的实验设计决策(2026-09-05)

**结论:0.6B 侧有必要补全参臂。** 论文的中心主张(学习型选块可超精确选块,VT +49pt、
≥128K 外推)依赖端到端协同适应——K/V 表征需要变得"可被选择";我们冻结范式测得的
dsa−mass=1.57pt 只反映"固定表征下选择器的差距",不构成学习型方法的上界。
数据可行性:fineweb 30GB ≈ 10B tokens,对 0.6B 的 1.31B 等预算 A/B 与 2–5B CPT 均充足
(本地已 tokenize 池:test 命名空间 14 文件 9.9 亿 tokens)。
已备好 `launch_0p6b_fullparam_ab.sh mass|hils`(与冻结臂同配方、唯一变量=是否冻结;
lr 沿用 2e-5 保守),待 GPU 释放即可启动,构成 2×2(冻结×方法)矩阵。

---

## 五、评测结果(4K RULER,n=50/任务,seed 42)

任务:single_niah / multi_query / variable_tracking;输入 4096→4161(含 65 landmark);等预算 2560 KV/query。

### 5.1 六变体(landmark id 已修复:100278'拼' → lmk_id 151936,lmk_embed 正确注入)

| 变体 | macro_tok | macro_EM | niah | multiq | vartrk | 吞吐* |
|---|---|---|---|---|---|---|
| **full(稠密因果,100% 预算上限)** | **0.1815** | 0.0400 | **0.2275** | **0.1862** | **0.1309** | 26.7K |
| dsa(精确,61.5%) | 0.1785 | **0.0467** | 0.2250 | 0.1831 | 0.1274 | 10.4K |
| fixed0 | 0.1636 | 0.0200 | 0.2225 | 0.1585 | 0.1099 | 19.0K |
| mass(训练双anchor) | 0.1628 | 0.0400 | 0.2050 | 0.1646 | 0.1187 | 19.1K |
| hils(零样本) | 0.1586 | 0.0200 | 0.2050 | 0.1677 | 0.1030 | **21.1K** |
| bsa(块均值) | 0.1489 | 0.0067 | 0.2100 | 0.1615 | 0.0750 | 10.4K |

\* 同期 GPU3 被外部任务抢占,绝对值约为无抢占时一半;无抢占参考:mass 39.2K、dsa 22.4K。

full 变体定义:同 backbone,7 个检索层换成稠密因果 SDPA(块尾 landmark 值键按统一口径掩掉,
其余 21 层 SWA-512 与所有变体相同)——隔离出的恰是"检索层稀疏化"的代价;预算上报 4160/4161 ≈ 100%。
实现:`eval_ruler_qwen_mass_bsa_4k.py` 的 `DenseRetrievalAttention`/`load_full_model`。

**关键读数:dsa(61.5% 预算)距 full 仅 0.30pt**——top-32 精确选块几乎捕获了稠密注意力的全部检索收益;
mass 距 full 1.87pt,其中 1.57pt 是打分近似(dsa−mass),仅 0.30pt 是稀疏化本身。
另注:4K 解码下 full(单融合 SDPA kernel)反而比 dsa/bsa 的非融合选择流水线快(26.7K vs 10.4K),
与"打分非瓶颈、融合度才是"的结论一致;更长短上下文下 O(L²) 才会反超。

### 5.2 关键 Gap(下游性能)

| 对比 | macro_tok 差 | 说明 |
|---|---|---|
| **full(100%)− dsa(61.5%)** | **+0.30pt** | 稀疏化本身几乎无损,预算上界主要被 dsa 代表 |
| **full − mass** | **+1.87pt** | 其中 1.57pt 为打分近似(dsa−mass),0.30pt 为稀疏化 |
| dsa(上界)− mass | **+1.57pt** | 学习型 anchor 打分离最优块选择的剩余差距 |
| mass − hils(零样本) | **+0.42pt**(EM 2×:0.04 vs 0.02) | 暂列优势;**公平比较待 HiLS 对照训完** |
| mass − bsa(均值) | +1.39pt | 均值池化稀释尖峰信号,崩在 vartrk(0.075) |
| mass − fixed0 | −0.08pt(tok)/ +2pt(EM) | 修复 fill bug 后 tok 差距落入 n=50 噪声;EM 仍 2× 支持双 anchor;n 需加大 |
| 速度:mass/dsa | **+83%**(19.1K vs 10.4K;无抢占 39.2 vs 22.4) | 主因是融合检索路径而非打分算法(换均值打分不提速已证) |
| 速度:hils/mass | +10% | 原生 HiLS 路径最快 |

### 5.3 配方迭代收益(0.6B)

| 配方 | macro_tok | macro_EM |
|---|---|---|
| 旧 lse_norm_tau1_mi(step10000,8/24 测) | 0.1598 | 0.0133 |
| asym0p5_anneal ft20000(本次,修复前) | 0.1672 | 0.0400 |
| 同上(修复后) | 0.1628 | 0.0400 |

退火配方:token acc +0.3~0.7pt,**EM 约 3×**。

### 5.4 KV Budget 扫描(4 变体 × 6 档预算,4K RULER,n=50)

预算定义:检索层每 query 实际 KV = `window + topk×64`,% 按 4161 计。topk 受 MASS kernel
约束取 2 的幂,故为最接近目标的可达值;窗口仅在低档收缩(非检索层 `sliding_window=512` 全程不变)。
hils 跳过(lmk_embed 对照尚未训完)。配置:`configs/mass_attention/budget_sweep_0p6b/`,
结果:`outputs/eval_..._budget_sweep_n50_20260905/sweep_summary.json`。

| 档位 | window | topk | KV/query | 占比 | mass tok/EM | fixed0 tok/EM | dsa tok/EM | bsa tok/EM |
|---|---|---|---|---|---|---|---|---|
| full(参照) | 512 | 32 | 2560 | 61.5% | 0.1628 / 0.0400 | 0.1636 / 0.0200 | 0.1785 / 0.0467 | 0.1489 / 0.0067 |
| b20 | 512 | 4 | 768 | 18.5% | 0.1627 / 0.0267 | 0.1697 / 0.0200 | 0.1769 / 0.0467 | 0.1485 / 0.0067 |
| b15 | 512 | 2 | 640 | 15.4% | **0.1647** / 0.0267 | 0.1657 / 0.0200 | 0.1748 / 0.0467 | 0.1469 / 0.0067 |
| b10 | 256 | 2 | 384 | 9.2% | 0.1618 / 0.0133 | 0.1630 / 0.0133 | 0.1748 / 0.0467 | 0.1419 / 0.0000 |
| b5 | 128 | 1 | 192 | 4.6% | 0.1613 / 0.0200 | 0.1569 / 0.0067 | 0.1705 / 0.0467 | 0.1425 / 0.0067 |
| b2 | 64 | 1 | 128 | 3.1% | 0.1578 / 0.0133 | 0.1561 / 0.0067 | 0.1724 / 0.0467 | 0.1299 / 0.0067 |
| b1 | 64 | 0 | 64 | 1.5% | N/A¹ | N/A¹ | 0.1288 / 0.0067 | 0.1288 / 0.0067 |

¹ topk=0 触发 MASS TileLang kernel TVM 编译错误(`no available layout found`),mass/fixed0 无法跑纯局部档。

**扫描发现:**

0. **全局上限参照(full,稠密,预算无关)**:macro_tok 0.1815 / EM 0.0400。dsa 在 61.5% 预算即达其
   99%(差 0.30pt);各预算档的所有变体均以 0.1815 为绝对上限。
1. **15–20% 预算几乎零损失**:四变体在 b15/b20 与 full(61.5%)的 token acc 差均在噪声内
   (mass 甚至 0.1647 vs 0.1628)——当前 32 块/2048 远程 token 的预算在 4K 下过于富裕,
   **topk 4–8(≈18–25% 预算)即可保住全部检索质量**。
2. **精确打分(dsa)对预算极端鲁棒**:从 61.5% 压到 3.1%(topk=1)只掉 0.6pt
   (0.1785→0.1724),EM 全程保持 0.0467,niah 甚至维持 0.228——单块精确质量的排序在
   极小预算下仍能选中正确块。
3. **mass 的压缩韧性弱于 dsa 但明显强于 bsa**:mass 从 full→3.1% 掉 3.0pt(0.1628→0.1578),
   bsa 同区间掉 1.9pt 但基线低(0.1489→0.1299,vartrk 崩至 0.073);dsa−mass 差距在
   各预算档稳定在 ~1.5pt,说明学习型打分的相对劣势不随预算变化。
4. **fixed0 ≈ mass 在所有预算档成立**(token acc 噪声内),但 **EM 在多数档位 mass 翻倍**
   (b20: 0.0267 vs 0.0200;b5: 0.0200 vs 0.0067;b2: 0.0133 vs 0.0067)——双 anchor 的
   收益集中在精确匹配维度。
5. **b1 纯局部档是有效性校验**:dsa 与 bsa 数字逐位相同(0.1288/0.0067)——两者在
   无远程检索时退化为同一条纯 SWA 路径,证明预算管线上报与实现一致。

### 5.6 与 HiLS 原论文(arXiv 2607.02980)的对照

论文 345M 实验的超参与我们完全一致(chunk 64 / top-K 32 / window 512,即 2048+512=2560 激活预算,
5% RULER 混合、8K 训练),其 Naive-BSA 基线即我们的 dsa 变体。

**论文核心主张:学习型选块可以超过精确逐 token 选块(naive BSA 不是上界)。**

论文 Tab.2(345M,8K 训练,RULER):

| 模型 | 8K S-N/MK-MQ/VT | 32K | 128K | 512K |
|---|---|---|---|---|
| Full-Attn HoPE | 100/98/**36** | 72/11/11 | 0/0/0 | 0/0/0 |
| Naive-BSA(=dsa) | 100/97/**23** | 98/77/13 | 46/0/0 | 崩(OOM) |
| HiLS(完整版) | 100/95/**72** | 100/95/68 | 99/95/61 | 99/91/66 |

- **VT 任务:HiLS 72 vs Naive-BSA 23(+49pt),甚至超过 full attention 36(+36pt)**;
- PPL:训练长度内两者打平(4.94 vs 4.94),32K 插值区 BSA 反而最好(3.94 vs 4.34),
  ≥128K BSA 崩(8.67→OOM),HiLS 稳定(4.71→5.95);
- 机制解释:full-attention 诱导的块质量是**含噪信号**(无关 token 拿到非零质量,随长度累积),
  学习型压缩摘要能让噪声对消、保留语义——这就是"精确 ≠ 上界"的原因。

**我们的复现差距(为什么我们的 hils 落后 dsa 1.99pt 而论文里反超):**
论文消融(Tab.7,8K)显示我们对照组缺失的三个组件恰好是大头:

| 消融 | S-N/MK-MQ/VT @8K |
|---|---|
| HiLS 完整版(HoPE + Q-Cal) | 100/95/72 |
| w/o Q-Cal(我们的对照即此,**且无训练**) | 99/77/49(VT −23pt) |
| RoPE 版(我们用 RoPE) | 99/81/52(VT −20pt,>8K 全崩) |
| LMK-Attn(原始 landmark 键,无 Taylor 校正) | 96/79/38(VT −34pt) |

我们的 hils 对照 = RoPE + 无 Q-Cal + 冻结 backbone + 仅训 lmk_embed,处于消融表中最弱组合附近;
论文完整版 HiLS 还含 Taylor 校正摘要(Prop 3.1)、Q-Cal 低秩查询校准、HoPE 位置编码和原生端到端训练
(345M 从零 30B tokens;7B CPT 50B tokens)。

**对本实验的启示:**
1. 我们测得的 dsa−mass = 1.57pt 不构成 mass 的上界约束——论文证明学习型选块在 VT/长上下文可反超精确选块;
2. 我们的评测在 4K(**低于** 8K 训练长度),恰是论文中 BSA 最强的插值区间;HiLS 类方法的优势区间
   (≥训练长度、VT 类任务)尚未测——长度外推实验(8K/16K/32K)应列为高优先级;
3. 若要让 mass/hils 对照触及论文水平,需补 Q-Cal 类查询校准模块与 HoPE(当前 MASS anchor 打分
   部分承担了 Q-Cal 的角色:anchor 的 query 侧门控即学习型查询变换)。

### 5.8 90M 七变体评测(2026-09-06,mass vs hils-trained 首个公平对比)

90M anchor-FT 与 90M HiLS 对照(lmk_embed-only)均完成 20000 步,同配方同数据,
等预算评测(n=50,seed 42,fill 已修复)。结果:
`outputs/eval_qwen3_90m_mass_vs_hils_trained_ruler4k_7variants_n50_20260906/comparison.json`

| 变体 | macro_tok | macro_EM | niah | multiq | vartrk |
|---|---|---|---|---|---|
| full(稠密上限) | **0.2099** | **0.0533** | 0.2125 | 0.2062 | 0.2112 |
| dsa(精确) | 0.2090 | **0.0533** | 0.2100 | 0.2077 | 0.2094 |
| **hils_t(训练 lmk_embed,等配方对照)** | 0.2073 | 0.0467 | 0.2000 | **0.2108** | **0.2112** |
| hils(零样本) | 0.2058 | 0.0467 | 0.2000 | 0.2062 | 0.2112 |
| mass(训练双 anchor) | 0.2029 | 0.0467 | 0.2100 | 0.2015 | 0.1972 |
| fixed0 | 0.1981 | 0.0467 | 0.2000 | 0.1954 | 0.1990 |
| bsa(均值) | 0.1971 | 0.0467 | 0.2100 | 0.1877 | 0.1937 |

**核心发现:**

1. **90M 上 HiLS 家族反超 MASS**:训练后的 hils_t(0.2073)> 零样本 hils(0.2058)> mass(0.2029),
   且训练 loss 同向(hils 3.490 < mass 3.514)——**同配方下,320 参数的结构打分对照 ≥ 28,784 参数的
   双 anchor 机制**。0.6B 上的 "mass > hils(零样本)" 在 90M 不成立;mass 的优势可能是 0.6B 强骨干
   特有,或本就处于噪声内。
2. **训练 lmk_embed 的下游收益很小**(+0.15pt,噪声级),但 LM loss 收益可见(3.54→3.49)——
   该对照的天花板受"单向量检索参数"限制,离论文完整版(Q-Cal+HoPE+Taylor+全参)仍远。
3. **mass > fixed0 在 90M 上更清晰**(+0.48pt,multiq/vartrk 均领先)——双 anchor 在弱骨干上
   边际收益反而更大。
4. full ≈ dsa(0.2099 vs 0.2090)再次确认 4K 下稀疏化本身近乎无损。
5. 90M 绝对分数普遍高于 0.6B(~0.20 vs ~0.16),因 90M backbone 经 recovery 在同分布数据上
   适应过,而 0.6B 是原始 Qwen3 权重(未适配该数据格式);vartrk 差距最大(0.21 vs 0.12)。
6. 局限:n=50(±1.5pt 噪声),hils_t/hils/mass 三者差距 0.15–0.44pt 部分落入噪声;
   0.6B 侧 hils_t 对照仍在训练(35%,预计 9/8),届时完成 2×2 矩阵的另一格。

### 5.10 训练轨迹评测:随 step 增加 mass 是否反超 hils(2026-09-07)

方法:在匹配步数的 checkpoint 上各跑同预算评测(n=50,seed 42);loss 轨迹取自训练日志。
产物:`outputs/eval_trajectory_{90m,0p6b}_mass_vs_hils_n50_20260907/`、`eval_trajectory_summary.json`。

**90M(两 run 已完成,7 个匹配点):**

| step | mass | hils_t | gap(m−h) |
|---|---|---|---|
| 2000 | 0.2003 | 0.2024 | −0.0021 |
| 5000 | 0.2012 | 0.2012 | 0.0000 |
| 8000 | 0.2010 | 0.2044 | −0.0034 |
| 11000 | 0.2016 | 0.2065 | −0.0049 |
| 14000 | 0.2022 | 0.2052 | −0.0030 |
| 17000 | 0.2027 | 0.2056 | −0.0029 |
| 20000 | 0.2029 | 0.2078 | −0.0049 |

**0.6B(hils 已完成,终局点补齐 2026-09-08):**

| step | mass | hils_t | gap(m−h) |
|---|---|---|---|
| 1000 | 0.1525 | 0.1588 | −0.0063 |
| 3000 | 0.1550 | 0.1556 | −0.0006 |
| 5000 | 0.1657 | 0.1570 | **+0.0087** |
| 6500 | 0.1659 | 0.1631 | +0.0028 |
| **20000(终局)** | 0.1628(EM 0.0400) | 0.1626(EM 0.0133) | **+0.0002(打平)** |

终局判定:token acc 三方平手(mass 0.1628 ≈ hils_t 0.1626 ≈ fixed0 0.1636,均在 n=50 噪声内);
**EM 上 mass 3 倍于 hils_t(0.0400 vs 0.0133)**——0.6B 侧 mass 的优势集中在精确匹配维度。
且两个 0.6B 训练臂的下游分数都在 5000-6500 步见顶后小幅回落(mass 0.1659→0.1628,
hils_t 0.1631→0.1626),后 1.35 万步对下游无增益(与 4 遍重复数据的过拟合一致)。
2×2 矩阵(冻结×方法)至此补齐:90M 上 hils 家族 token acc 占优,0.6B 上 EM 占优的是 mass。

**结论:**

1. **90M 无反超趋势**:gap 全程稳定在 −0.2~−0.5pt(hils 领先),7 个点无系统性收窄;
   loss gap 亦全程恒定(+0.024~0.036 nats)。两者都在 ~2000-5000 步早早进入平台期。
2. **0.6B 在 3000→5000 步之间发生反超**(−0.0006 → +0.0087),6500 步维持 +0.0028;
   但 hils_t 在 5000→6500 段以 +0.6pt/1500步 的斜率快速追赶(mass 同期走平),
   且 hils 还剩 13500 步与末段 lr 退火(90M 侧 hils_t 末段即因此 +0.2pt)——**0.6B 终局未定**。
3. **稳定的任务偏好差异(两规模、所有步数一致)**:mass 在 single_niah 持续领先(+1~2pt,
   anchor 打分利于"单针尖峰信号");hils 在 multi_query/variable_tracking 持续领先
   (landmark 打分利于多跳/跟踪)。这是打分函数性质,与训练进度无关。
4. 注记:0.6B mass 的下游分数在 5000 步后走平甚至略降(0.1657@5000 → 0.1628@20000),
   末段 spec 退火(温度→0.1)存在过度特化可能;n=50 噪声内,n≥200 待确认。

### 5.11 90M anchor+LoRA 臂终局:超越稠密上限(2026-09-09)

90M anchor+全 token mixer LoRA(qkvo r16,129 万可训练参数)20000 步完成(9/9 11:48,
中途 12619 步因宿主机内存压力崩溃一次、从 12500 无损恢复)。终局 lm 3.374(spec 0.054)。
评测新增 `mass_lora` 变体(注入同款 LoRA + 加载 anchors/LoRA adapter checkpoint):

| 变体 | macro_tok | macro_EM | niah | multiq | vartrk |
|---|---|---|---|---|---|
| **mass_lora(训练 anchor+LoRA)** | **0.2596** | **0.0533** | 0.2225 | **0.2631** | **0.2932** |
| full(稠密上限) | 0.2099 | 0.0533 | 0.2125 | 0.2062 | 0.2112 |
| dsa(精确选块) | 0.2090 | 0.0533 | 0.2100 | 0.2077 | 0.2094 |
| hils_t(训练 lmk) | 0.2073 | 0.0467 | 0.2000 | 0.2108 | 0.2112 |
| mass(冻结+训练 anchor) | 0.2029 | 0.0467 | 0.2100 | 0.2015 | 0.1972 |

**核心发现:**

1. **稀疏+LoRA 超过稠密注意力 +5.0pt(+24% 相对)**:multiq +5.7pt、vartrk +8.2pt——
   检索类任务上,经适配的训练型稀疏模型优于同 backbone 的 full attention。
   这正是 HiLS 论文"学习型分配优于精确质量分配"主张在我们设置下的首次实证。
2. **精确选块(dsa)被大幅甩开**(0.2596 vs 0.2090,+5.1pt)——dsa 不是上界,
   训练型选块(配表征适配)显著超越它。此前"dsa−mass=1.57pt 是打分近似差距"的
   解读被修正:那只是"冻结表征下"的差距。
3. **LoRA 的作用是双向的**:它让表征变得"可被选择"(K/V 侧),也让 anchor 的
   query 侧门控有更好的输入——28,784 anchor 参数在适配后的表征上发挥出真实能力。
4. 吞吐 31.7K tok/s(LoRA 推理开销可忽略,仍是检索层的稀疏注意力)。
5. 待办:0.6B LoRA 臂(47%,9/12 前后)完成后同款评测;给 hils 侧也挂同款 LoRA 对照
   (hils+LoRA vs mass+LoRA)以分离"LoRA 表征适配"与"anchor 打分"各自的贡献。

### 5.12 90M 完整 RULER 四任务对比(2026-09-09)

仓库 RULER 合成器共 4 任务:0 single_niah / 1 multi_query / 2 variable_tracking /
**3 frequent_words(FWE,本次新增,修复了任务映射缺失与 scipy 缺依赖两处问题)**。
FWE 探测(n=10)显示 token acc 有区分度(full 0.49 / mass_lora 0.55),EM 全 0 无区分度——
予以保留,以 token acc 计。n=50,seed 42,等预算 2560。

| 变体 | niah | multiq | vartrk | FWE | **macro4** |
|---|---|---|---|---|---|
| **mass_lora(anchor+LoRA)** | **0.2225** | **0.2631** | **0.2932** | **0.5958** | **0.3436** |
| full(稠密上限) | 0.2125 | 0.2062 | 0.2112 | 0.4813 | 0.2778 |
| dsa(精确) | 0.2100 | 0.2077 | 0.2094 | 0.4836 | 0.2777 |
| hils_t | 0.2000 | 0.2108 | 0.2112 | 0.4883 | 0.2776 |
| hils(零样本) | 0.2000 | 0.2062 | 0.2112 | 0.4883 | 0.2764 |
| mass(冻结) | 0.2100 | 0.2015 | 0.1972 | 0.4907 | 0.2749 |
| fixed0 | 0.2000 | 0.1954 | 0.1990 | 0.4930 | 0.2718 |
| bsa(均值) | 0.2100 | 0.1877 | 0.1937 | 0.4907 | 0.2705 |
| **hils_lora(lmk+LoRA 归因对照,9/11 补)** | 0.2175 | **0.2677** | 0.2914 | 0.5935 | **0.3425** |

**结论:**

1. **mass_lora 全部四任务第一,macro4 超稠密上限 +6.6pt(0.3436 vs 0.2778)**;
   FWE 上优势同样显著(+11.5pt)——FWE 需要全上下文的词频聚合,稀疏检索+适配表征
   在聚合类任务上同样超过稠密注意力。
2. **七个非 LoRA 变体在 macro4 上聚成 0.271–0.278 的窄带**(0.7pt,基本噪声),
   其中 FWE 完全无区分(0.481–0.493)——打分函数的差异只体现在检索类任务,
   LoRA 表征适配才带来全面跃升。
3. **归因结论(2026-09-11,hils_lora 补齐)**:hils_lora macro4 **0.3425 ≈ mass_lora 0.3437**
   (差 0.1pt,噪声)——**+6.6pt 的跃升完全来自 LoRA 表征适配,与打分函数无关**:
   配上 LoRA 后,320 参数的 HiLS landmark 打分与 28,784 参数的双 anchor 机制打平
   (LM loss 同向:hils_lora 3.366 ≈ mass_lora 3.374)。任务偏好签名保留
   (mass_lora 的 niah 略高 / hils_lora 的 multiq 略高),但聚合无差。
   2×2 归因矩阵:冻结(0.2749 vs 0.2776)与 LoRA(0.3437 vs 0.3425)两列均平手。
   **MASS anchor 的价值定位 = 参数高效的检索适配器,而非能力上不可替代的组件;
   能力瓶颈与释放都在表征侧(K/V 投影适配)。**

### 5.13 详细评测:RULER 长度扫描 + LongBench(2026-09-11)

**RULER 长度扫描**(mass_lora vs hils_lora,四任务,n=50,等预算 2560——注意预算固定,
长度加倍即相对压缩率减半:4K=61.5% / 8K=31.2% / 16K=15.5%;>8K 用相邻样本拼接实现,
`eval_ruler` 已支持):

| 模型 | 4K | 8K | 16K(外推) |
|---|---|---|---|
| mass_lora | 0.3437 | 0.1431 | 0.0733 |
| hils_lora | 0.3425 | 0.1464 | **0.0884** |

分任务(8K→16K):mass niah 0.1200→0.0725、multiq 0.2108→0.0738、vartrk 0.2417→0.1468;
hils niah 0.1275→**0.1100**、multiq 0.2077→0.0969、vartrk 0.2504→0.1468;FWE 两模型 8K 起均 0。

**发现:**

1. **长度维度的分离出现了**:4K/8K 两者打平,但 **16K 外推区 hils_lora 反超 mass_lora
   +1.5pt macro4(niah +3.75pt,+52% 相对)**——HiLS 的 landmark 结构打分在外推区间更稳,
   与原论文"HiLS 强外推"的主张方向一致;MASS anchor 是在训练长度内学出的低秩摘要,
   外推时失配更快。
2. 分解看退化来源:固定绝对预算下,长度 ×4(4K→16K)等于相对预算 ÷4,退化是
   "任务变难 + 相对预算收缩"的叠加,不能全归因外推。
3. FWE 在 8K+ 全零(4K 为 0.59):聚合类能力对长度最脆弱。

**LongBench(v1 英文子集,缩减协议)**:适配驱动 `eval_longbench_mass_adapters.py`
(本地 data.zip 数据、auto_insert_lmk 生成、官方 F1/分类指标);**解码速度是硬约束**——
MASS 检索 TileLang kernel 仅支持全序列训练模式,逐 token 解码回退慢路径
(mass_lora 14.6s/tok、hils_lora 1.67s/tok),且 90M 模型在 few-shot 提示下输出退化为续写。
缩减为 3 任务(hotpotqa/2wikimqa/trec)× n=10 × max_new≤24 × 截断 8192:
hils_lora hotpotqa F1=0.070(近地板,LongBench 面向 7B+ 模型,90M 区分度有限);
mass_lora 全量在跑。**结论:LongBench 对该规模模型不是有效区分基准,
长度扫描(teacher-forcing)才是长度维度的可靠读数。**

### 5.14 0.6B mass_lora 终局:达到稠密平手,未复现 90M 的超越(2026-09-15)

0.6B anchor+LoRA 完成(9/13,lm 2.809)。四任务评测(4K,等预算,n=50):
`outputs/eval_qwen3_0p6b_mass_lora_20260915/`

| 0.6B @4K | niah | multiq | vartrk | **macro3** | FWE |
|---|---|---|---|---|---|
| **mass_lora** | 0.1950 | **0.2077** | **0.2304** | **0.2110** | 0.0000 |
| full(稠密) | **0.2125** | 0.2062 | 0.2112 | 0.2100 | 0.0023 |
| dsa | 0.2250 | 0.1831 | 0.1274 | 0.1785 | — |
| mass(冻结) | 0.2050 | 0.1646 | 0.1187 | 0.1628 | — |
| hils_t(冻结) | 0.2050 | 0.1815 | 0.1012 | 0.1626 | — |

**读数:**

1. **LoRA 增益复现**:冻结 mass 0.1628 → mass_lora 0.2110(+4.8pt),vartrk 甚至超稠密 +1.9pt;
   EM 0.0467 亦为全场最高。
2. **但 90M 的"超稠密 +6.6pt"未复现**:0.6B 只到平手(+0.1pt)。且 0.6B 的 FWE 对所有
   变体均为地板(稠密上限 0.0023)——90M 的 FWE 优势(+11.5pt)很大程度来自其 backbone
   在 ruler 混合上 recovery 过;0.6B backbone 是通用 Qwen3。**修正此前推断:决定
   "训练型稀疏超越稠密"幅度的更可能是 backbone 与评测分布的匹配度(90M recovery 过),
   而非骨干规模本身**(§"MASS 优势区间"第 1 点按此修正)。
3. 2×2 矩阵 0.6B 侧的 LoRA 行仍缺 hils_lora 臂(GPU 已空,脚本可复用,~3 天)。

### 5.13 已修复的评测 bug 与其影响(汇总)

- 训练插入 landmark 用 `lmk_id=151936`(嵌入替换为 lmk_embed);评测曾硬编码 100278(普通 token '拼',olmo 时代遗留)→ lmk_embed 从未注入。
- 修复后:dsa/bsa **逐位不变**(其打分与 V 不读 landmark 键);mass/fixed0 小幅移动(噪声级)。
- 该修复是 HiLS 对照训练(lmk_embed)有意义的前提。

---

## 六、训练/推理长度与外推状态

| 项 | 值 | 备注 |
|---|---|---|
| 训练长度 | 8192(ruler_0.05 混合,打包) | landmark 插入后模型内序列 ~8321 |
| 评测长度 | 4096 原始 → 4161 含 landmark | **训 8K 测 4K,未测试长度外推** |
| 位置编码上限 | 40,960(RoPE theta 1e6,Qwen3 原生) | 8K→32K/4K→8K 外推实验待做 |
| KV budget | 训推一致:2560/query | 4K 时压缩率 4161/2560 ≈ 1.63×;8K 训练时 ≈ 3.25× |

---

## 七、100M/18L 从头预训练 SM 饱和度测试(2026-09-06)

问题:~100M 参数、17–19 层的模型从头预训练,能否在显存 <24GB 下打满 SM?

**配置**:18 层 / hidden 320 / inter 2432 / 4+2 头 / head_dim 128 / 词表 151,936,
MASS m2 结构,**99.54M 参数**(非嵌入 50.9M + 嵌入/head 48.6M);随机初始化 bf16
snapshot(seed 42);8K 序列,ruler_0.05 混合,梯度检查点,纯 bf16 训练
(为绕开全参 fp32 主权重 vs bf16 TileLang kernel 冲突,新增 `MASS_PURE_BF16=1` 旁路)。
GPU2(共享卡),mb ∈ {4,8,16} × 30 步,accum=1。

| micro-batch | tokens/步 | 稳态 s/it | torch 峰值显存 | 有效 SM 样本 | 达成 FLOPs |
|---|---|---|---|---|---|
| 4 | 32,768 | 4.8 | **3.6 GiB** | 中位 24–38%,瞬时 82% | ~4.9 TFLOPS |
| 8 | 65,536 | 13.8 | **6.4 GiB** | 27–42% | ~4.9 TFLOPS |
| 16 | 131,072 | 19.8 | **12.0 GiB** | ≤33%(采样稀疏) | ~5.3 TFLOPS |

**结论:**

1. **显存远不是约束**:mb=16 峰值仅 12 GiB(torch 分配器口径),mb 24–32 仍在 24GB 内。
2. **单卡单进程打不满**:步时随 batch 线性放大(4× tokens → 4.4× 时间,每 token 耗时不变),
   没有出现"大批量→高利用率"转折;MFU ≈ 1.7%(A800 312 TFLOPS 峰值),按抢占损失折算
   也就个位数百分比;SM 中位 24–42%,仅瞬时冲到 82%。
3. **结构性原因**(与抢占无关):hidden=320 的小 GEMM 计算密度低;15 万词表的
   lm_head+交叉熵主导显存带宽(liger 分块 CE 仍是带宽受限);18 层 × 多小 kernel
   + 7 个 MASS 检索层的 tilelang kernel 与 spec 归约,launch 开销大;梯度检查点 +33% 计算。
4. **要打满 SM 的路径**:数据并行多卡(每卡 mb4–8,8 卡即论文 345M 的做法)、
   增大序列或词表裁剪、减少小 kernel(融合)。单卡 <24GB 下,该工作负载本质接近带宽受限,
   追求 SM 饱和不如追求 tokens/s(受带宽上限)。
5. 附带产出:从头训练全参 MASS 的完整链路已打通(随机初始化 snapshot 生成脚本、
   `MASS_PURE_BF16` 旁路、bench 脚本 `bench_100m_18l_sm.sh`),可复用于真正的
   100M 从头预训练实验。

测试期间失败原因记录:fp32 随机初始化路径撞 bf16 kernel(dtype)、fsdp2 要求 meta
初始化(与 ddp 断言冲突)、GPU6 被外部任务 OOM、DDP 包装后 `model.config` 属性访问
崩溃(已修:`raw_model = getattr(model, "module", model)` 统一解包,多卡路径从此可用)——
均已绕开或修复。

### 7.2 单卡大 batch 测试(2026-09-07,GPU4,mb 8/16/24/32 × 30 步)

| mb | tok/步 | 稳态 s/it | 吞吐 tok/s | µs/token | VRAM 峰值 | 步时线性比 |
|---|---|---|---|---|---|---|
| 8 | 65,536 | 4.1 | 15,984 | 62.6 | 6.4 GiB | — |
| **16** | 131,072 | **7.7** | **17,022** | **58.7** | 12.0 GiB | 1.88×(亚线性,好) |
| 24 | 196,608 | 15.6 | 12,644 | 79.1 | 17.6 GiB | 2.02×(超线性,坏) |
| 32 | 262,144 | 19.0 | 13,797 | 72.5 | **23.3 GiB** | 1.22× |

**结论:**

1. **吞吐甜点在 mb16**(17.0K tok/s,12GB):mb8→16 提升 +6%,再往上不升反降
   (mb24 掉 25%;受同卡抢占波动影响,方向可信、幅度存噪)。
2. **mb32 = 23.3 GiB 触到 24GB 预算的天花板**,且无吞吐收益——增大 batch 的
   实用上限就是 mb16–24。
3. mb16 时 MFU ≈ 4.3%(13.5 TFLOPS,抢占下)——即使按甜点配置,该模型仍是
   带宽/launch 受限,与 §七 结论一致:**单卡堆 batch 到 mb16 即榨干,再想快只能
   接受多卡的负载不均风险,或改 kernel 融合/词表**。
4. 建议 100M/18L 从头训练配置:**mb16(12GB)单卡**,或 mb8×accum2 起步求稳。

### 7.1 双卡加速测试(2026-09-07,GPU4+GPU7)

三相位背靠背实测(100M/18L,同数据同配置,每相位 30 步,取每 10 步时间戳的稳态):

| 相位 | 配置 | tokens/步 | 稳态 s/it | 吞吐 |
|---|---|---|---|---|
| A | 1×GPU4,mb8 | 65,536 | **4.0** | 16.4K tok/s |
| B | 2×GPU(GPU4+GPU7),mb4/卡,同全局 batch | 65,536 | **4.05** | 16.2K tok/s |
| C | 2×GPU,mb8/卡 | 131,072 | **7.1** | 18.5K tok/s |

**强扩展加速比 0.99×(零加速),弱扩展吞吐 1.13×(每卡效率 56%)。**

原因分解:
1. **卡间负载不均主导**:GPU4 同驻 ~58%,GPU7 同驻 ~98%(含我们的 hils 训练)——
   DDP 每步等最慢的 rank,GPU7 上的 rank 定步调,2 卡时间 ≈ 慢卡 mb8 时间;
2. **模型本身 launch/带宽受限**(§七 单卡结论):每步存在与 batch 无关的固定开销,
   摊到多卡不能消除;
3. **NCCL 通信本身可忽略**:99.5M bf16 梯度 allreduce ≈ 200MB,NVLink 上 ~1ms 量级。

**结论与建议**:这台共享机上 2 卡对该负载实测无净收益(0.99×/1.13×);干净双卡的
理论上限约 1.7–1.9×(compute-bound 部分近线性,通信开销小)。对 100M/18L 小模型,
**单卡加 mb(12GB@mb16,24GB 内可到 mb24–32)比多卡更划算**;多卡价值在需要更大
global batch 时。优先级:增大每卡 batch > 多卡。

---

## 八、产物与脚本索引(更新)

| 类别 | 路径 |
|---|---|
| 0.6B 最终模型 | `outputs/mass_qwen3_0p6b_..._ft20000/checkpoints/global_step_20000`(+`_anchor_only` 评测版) |
| 评测结果 | `outputs/eval_qwen3_0p6b_mass_anneal_ft20000_ruler4k_5variants_fillfix_n50_20260905/comparison.json`(及 4variants、equal_kv 两版) |
| HiLS 基座 | `.cache/modelscope/models/Qwen--Qwen3-0.6B-HiLS/`、`Qwen--Qwen3-90M-HiLS-recovered/` |
| HiLS config | `configs/mass_attention/config_hils_qwen3_{0p6B,90M}_8KA2K_RoPE.json` |
| 训练脚本 | `launch_mass90m_anchor_ft.sh`(及 `_resume12500.sh`)、`launch_hils_lmkembed_ft.sh 0p6b|90m` |
| 评测脚本 | `launch_eval_0p6b_anneal_ft20000_ruler4k.sh`(五变体) |
| 工具 | `extract_anchor_only_ckpt.py`(DCP→anchor-only) |
| 日志 | `mass_qwen3_{0p6b,90m}_*_ft20000*.log`、`hils_qwen3_{0p6b,90m}_lmkembed_only_ft20000.log` |
| PID | `mass90m_anchor_ft.pid`、`hils_{0p6b,90m}_lmkembed.pid` |

代码改动(均在仓库内):`eval_ruler_qwen_mass_bsa_4k.py`(dsa/bsa/hils 变体 + fill_id 修复)、`models/NaiveBSA/{naive_bsa_kernel,naive_bsa_layer,naive_bsa_ref}.py`(score_mode=mean 均值池化打分)。

---

## 九、待办与风险

1. **9/6 上午**:90M HiLS 对照完成 → 提取 lmk_embed,跑 90M 五变体评测(90M 的 dsa/bsa/hils/mass 全套);
2. **9/6 下午**:90M anchor-FT 完成 → `extract_anchor_only_ckpt.py` 提取后评测,得 90M mass vs hils-trained 的**公平对比**;
3. **9/8**:0.6B HiLS 对照完成 → 同上,补齐 0.6B 侧公平对比 + hils 的预算扫描;
4. mass vs fixed0 在 n=50 下不可区分 → 需要 n≥200 复测(EM 的 2× 差异待确认);
5. 待补:原始 Qwen3-0.6B(dense)在 ruler_0.05 数据上的 CE(0.6B 侧 loss gap 的 dense 参照);
6. 长度外推(8K→32K)未测;预算扫描目前仅 4K,8K 下的预算-质量曲线待补;
7. 风险:GPU 全线被外部任务抢占,ETA 可能漂移;如再遇 dataloader shm 崩溃,用对应 resume 脚本从最近 500 步存档恢复;
8. 已知限制:b1(纯局部)档 mass/fixed0 因 kernel 不支持 topk=0 而缺失。
