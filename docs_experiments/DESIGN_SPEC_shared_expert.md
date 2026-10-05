# 共享专家重构方案:MASS-anchor MoE 化(OLMo3-1025-7B + Dolma3)

日期:2026-09-27。前置结论:anchor 旁路(独立打分)在 0.6B 上被证明不可训——
表达力不足(NCI 权重 8、梯度 5-10 倍仍零效应)+ 目标错位(spec=MI 路由目标,分化无功能)。
本方案按 MoE 共享专家路线重构,并把实验平台迁移到 OLMo3-1025-7B + Dolma3。

## 一、方法:共享专家重构(anchor 基于 HiLS 论文)

### 1.1 HiLS 论文原式(共享专家 = 论文机制本身,不动)

HiLS(arXiv 2607.02980):每个 chunk 末尾的 landmark token 提供**压缩 chunk 键** k_lmk(c),
检索分数 = q·k_lmk(c)(单泰勒平面),注意力分解为 inter-chunk(选择)与 intra-chunk(聚合)
双重 softmax,端到端由 next-token loss 学习。

### 1.2 重构式(共享专家 + 专用残差专家,上包络形式)

把 HiLS 的 landmark 键作为**第 0 个固定视图**放进 MASS 上包络:

```
score(q,c) = τ·LSE( [ q·k_lmk(c)·s + λ ,  q·μ[c,m]·s + b[c,m]  for m=1..M ] / τ )
```

- **视图 0(共享专家)**:μ_shared[c] = k_lmk(c)(HiLS 论文原式,不池化、不新增参数),
  偏置 λ 为**每层一个可学标量**(即 log 空间的共享专家权重);
- **视图 1..M(专用专家)**:现有多锚 Taylor 池化视图(锚初始化拉开,见 §4);
- τ 退火终点抬到 0.5–1.0(不做硬 max,防赢家通吃)。

### 1.3 为什么这是"共享专家结构"(DeepSeekMoE 语义)

- 共享视图常驻(每层、每块都参与包络),承担常规检索——等价于 HiLS 基线能力;
- 专用锚只有在**包络中胜过共享视图**时才影响选择——它们提供的是"单键压缩抓不到的多视角
  残差信息",λ 与锚参数共同决定其去留;
- λ 初始化使共享视图显著占优(如 λ=+2):起点≈纯 HiLS,结构为 HiLS 的严格超集,
  专用专家的价值可证伪(训完 λ/锚贡献可读出);
- 梯度主干变为 q·k_lmk:**选择与内容读取共享同一几何**(这正是 hils_lora 0.475 胜出的
  机制),NCI 检索监督有了可表达的通路。

### 1.4 实现注入点(已核实,无需改 TileLang kernel)

`ops/mass_retrieval_tilelang_v2.py` 入口处 `mu, bias = _MassSummaryFn.apply(anchors,
key_chunks, ...)` 在 torch 侧组装 Mu/Bias,下游 kernel 只按形状消费 M 个视图:
- 新签名增加 `shared_key [B,N,Hq,D]`(= k_lmk,GQA 展开到 Hq)与 `shared_bias` 标量;
- `_MassSummaryFn` 输出 M+1 个视图(M+1≤16 约束内:m2+1=3 或 m4+1=5);
- 反向 kernel(dmu/dbias)对共享视图自动生效,梯度流向 k_proj(LoRA 可训)与 λ;
- 路由统计(anchor_load/condH)**排除视图 0**;spec loss 弃用 MI 形式,换小系数边际
  均衡(0.01–0.05,仅对专用视图)或先置 0;
- `build_mass_retrieval_plan` 传入 k_lmk(key_source 的 chunk 末位 landmark 键)。

## 二、模型更换:Qwen3-0.6B → Olmo-3-1025-7B

- 数据血统匹配:dolma3_multiscale 的构建源是 `allenai/dolma3_mix-6T-1025-7B`
  ——**Olmo-3-1025-7B 的官方预训练混合语料**(build.log 证实),模型-数据天然配套;
- 本地已有权重 `allenai--Olmo-3-1025-7B`(8G)与兄弟版 Olmo-3-7B-Think(+已有 m8 MASS
  转换与 configs/config_mass_olmo3_7B_* 可参考);
- 7B 规模对"共享专家是否增量有效"的检验更有说服力(0.6B 上 hils_lora 已证明 LoRA+
  主通路可训,7B 验证可迁移性);
- 训练形态:LoRA r16 qkvo + 共享视图 λ + 专用锚 全部可训,backbone 冻结
  (单卡 A800 80G,mb2×accum4=global8,grad ckpt)。

## 三、数据更换与按模型规模取量

- 源:`data/dolma3_multiscale/`,5 档已分词样本(3B/10B/20B/42B/50B tokens,
  OLMo3 词表,parquet 含 input_ids list<int32> 文档);
- **7B 模型取 10B 档**(dolma3_mix_sample_10B):预算 20k 步 × 65,536 tok = 1.31B,
  池 10B ≈ 7.6 倍预算——与 0.6B 纪律(9.9B 池/1.31B 预算)一致;
- 格式:需打包为 HiLS uint32 mmap `.parquet.data`(文档间单 EOS,8192 打包),
  新脚本 `pack_tokenized_parquet.py`(跳过分词,纯重排);预计 ~40GB、<1h;
- 对照规模纪律(如后续换小模型):≤1.5B 模型取 3B 档,3–8B 取 10B 档,
  ≥13B 取 20B/42B 档。

## 四、训练配方(吸收此前全部教训)

| 项 | 值 | 依据 |
|---|---|---|
| 结构 | m2(+共享视图)起步;消融 m4 | M+1≤16;先最小可证伪 |
| 锚初始化 | offset 范数 ≈ 0.5–1×‖base‖(调 offset_init_std 至实测 cos(a0,a1)≈0) | 零样本 cos=1.0 的教训 |
| LSE 退火 | 4.0 → **0.5**(不再到 0.1) | 赢家通吃/早锁定 |
| spec loss | 弃用 MI;共享视图排除;专用视图边际均衡 0.01 或 0 | 装饰性分化教训 |
| NCI | 保留(MASS_NEEDLE_INFO_NCE=1, w=8) | 监督+表达力须同时到位 |
| LoRA | r16 α32 qkvo all | 与 hils_lora 对齐,表达力 |
| lr/步数 | 2e-5→2e-6 cosine,先 5,000 步试点(3.28 亿 tokens)再延至 20k | 探针 1–4k 步即可读出选择分化 |
| 数据混合 | ruler_0.05(与评测同构) | 管线复用 |
| 监控 | 每 500 步 anchor_load 分视图、λ 轨迹、NCI log-prob、探针 frac_qh | MoE 按类型利用率 |

## 五、评测与判据

1. 探针(needle 命中率/frac_qh):**共享视图命中份额**(视图 0 的包络胜出率)是核心新观测量
   ——专用锚胜出份额>0 且带来 frac_qh 提升 = 共享专家结构有效;
2. 绑定态判据:8K + topk=8(vs 随机 0.14);4K/n200 四任务与 hils_lora(0.6B)/纯 HiLS 对照;
3. 终点裁决:λ→0 或与纯 HiLS 打平 → 学习式块摘要证伪;multi_query/异质块超 HiLS → 多视角主张成立。

## 六、执行清单

1. [进行中] `pack_tokenized_parquet.py` 打包 dolma3 10B → `data/dolma3_1025_7B_10B_packed/`
2. Olmo-3-1025-7B → MASS-m2(+shared view)转换(参考 Olmo-3-7B-Think-m8 管线)
3. 共享视图实现(_MassSummaryFn + build_mass_retrieval_plan + 路由统计排除视图 0)
4. 冒烟:λ/锚梯度流、视图 0 命中份额>0、检查点一致
5. 训练 5k 试点 → 探针 → 20k 全程
6. 评测协议 §五

---

## 修订(2026-09-27 晚):缩小规模做快速验证

用户决策:不用 7B(太慢),先做 ≤200M 小规模验证。数据与模型相应调整:

| 项 | 原方案 | 修订后 |
|---|---|---|
| 模型 | Olmo-3-1025-7B 直接续训 | **Olmo-3-1025-7B 宽度裁剪 → 199.2M**(hidden 512 / inter 1280 / 4 头×head_dim128 保持 / 32 层与 3:1 SWA:full 交错全保留 / vocab 100,278 untied)+ dense recovery |
| 数据 | dolma3 10B 档 | **dolma3 3B 档**(预算 20k 步 1.31B tokens…小模型实际 recovery+实验预算更小,池 3B 足够) |
| 管线 | 转换即训 | 裁剪(importance/Minitron)→ dense recovery → 转 HiLS/MASS+shared → 共享专家臂 |

关键事实:
- dolma3 parquet 仅含 OLMo 词表 token id(无原文、HF 不通),不可转投 Qwen 词表
  → 小模型必须保 OLMo3 词表,只能走"裁剪 1025-7B"路线(与数据血统一致);
- 裁剪脚本 `utils/prune_olmo3_width.py`(自 prune_qwen3_width 适配:untied lm_head、
  无 q/k-norm、MHA 组=1、保留 YaRN8×/SWA/layer_types),自校验(教师/学生 CE + 参数量断言);
- 打包:`scripts/data/pack_tokenized_parquet.py` → `data/dolma3_1025_3B_packed/`
  (400M tokens/片 × ~8 片,EOS 100257 分隔);
- 已挂自动链:首个分片就绪 → GPU1 启动裁剪(教师 12×8192 校准)→
  产物 `allenai--Olmo-3-1025-200M-pruned`。

后续阶段(裁剪完成后):dense recovery(dolma3 3B,~0.5-1B tokens,预期 CE 从 ~10 降到
~3.x,参照 90M 管线)→ convert 到 olmo-hils/mass(+shared view 实现)→ 共享专家臂训练
(配方同 §四)→ 探针/绑定态评测(§五)。

---

## 事故记录与修复(2026-09-28):NCI 符号反转

90M shared 臂首跑 4714 步时的证据链:
- 日志:NCI 显示值(当时=8×log p)从 -2.66 恶化到 -34.8(1094 事件);lm 3.373 vs 基线同窗口 3.345(略差);
- 探针(GPU6,n=50):zeroshot90m hit 0.894/frac 0.5225 → shared@1000 0.814/0.397 → @2000 0.392/0.170
  → @4000 **0.096/0.064**(配对差 -0.53±0.05)——训练在主动把 needle 排除出 top-k;
- 根因:`_needle_info_nce` 返回了 log p(≤0)并加进总损失,最小化 ⇒ 压低 needle 概率。
  0.6B nci 臂带同 bug 却零效应(anchor-only 表达力死路,错误方向也推不动——其结论不变,且更强化);
  divmu 臂无 NCI,不受影响。

处置:stop 旧 run、废弃 checkpoints;修复 per_row = denom - numer(-log p ≥ 0);
冒烟确认 NCI 转正(21.2 ≈ 8×2.65,随机水平);重启 20k(PID 4007764,GPU4)。
流程改进:①NCI 首次触发时断言 -log p>0;②**1000 步内跑探针**列为标准动作(本次若早查可提前 ~20h 止损)。
MoE 修正有效性的判定待重启版 1000/2000/4000 步探针对比 zeroshot90m(frac 0.5225)。

---

## 90M 共享专家臂终评(2026-10-03,n=200,τ=0.5)

| 臂 | niah | mq | vt | macro3 |
|---|---|---|---|---|
| **shared noNCI@20k** | 0.2375(EM .155) | 0.2650 | **0.3284** | **0.2770** |
| shared NCI@20k | 0.2169(EM .025) | 0.2442 | 0.3222 | 0.2611 |
| mass_lora 基线@20k | 0.2225(EM .14) | 0.2631 | 0.2932 | 0.2596 |
| mass / hils_t / full@20k | ~0.21 | 0.20-0.21 | 0.197-0.211 | 0.203/0.207/0.210 |

探针@20k:topk32 frac 0.776(hit 0.973,随机 0.561);**topk8 frac 0.629(随机 0.140)**;
λ 终值 ≈2.0(初始 +2,几乎不动);共享份额 0.277。

**结论**:
1. 共享专家结构(noNCI)成为 90M 全场最佳:超最强基线 mass_lora +0.017、超 dense 上限 +0.067;
   增益集中于 vt(+0.035)与 niah(+0.015)——与"异质块需要多视图/共享摘要"假设一致。
2. NCI(w=8)净贡献为负:选择(frac 0.78 vs noNCI ~0.52)未在 4K 下兑换为下游(macro3 0.261<0.277);
   其价值在绑定态(topk8 0.629 vs 随机 0.140)与长上下文场景,需降权(1-2)/延后/或不用。
3. λ 稳定说明共享/专用平衡由分数几何承担,无需调偏置。
4. 论文叙事修正:MASS 的有效成分 = HiLS 主通路 + 共享视图结构;而非"可学习检索打分旁路"。

## B 臂终评(shared+lmkq+NCI w=2,2026-10-04 晚,n=200)

| 臂 | niah | EM | mq | vt | macro3 | 选择 topk32/8 |
|---|---|---|---|---|---|---|
| shared-noNCI(无lmkq) | 0.2375 | 0.155 | 0.2650 | 0.3284 | 0.2770 | ~0.52/随机 |
| **B:shared+lmkq+NCI w=2** | 0.2288 | 0.095 | 0.2650 | **0.3314** | **0.2751** | **0.776 / 0.664** |
| shared-NCI w=8(无lmkq) | 0.2169 | 0.025 | 0.2442 | 0.3222 | 0.2611 | 0.776 / 0.629 |
| mass_lora | 0.2225 | 0.14 | 0.2631 | 0.2932 | 0.2596 | ~0.58/随机 |

**核心结论:NCI 剂量响应存在甜点 w≈2**——选择能力在 w=2 即饱和(0.776/0.664,绑定态 rank 2.72=needle 常居前三),
而提取税几乎为零(macro3 与 noNCI 打平,Δ0.002≪噪声)。"低剂量检索监督以零下游代价兑换选择能力"成立;
EM 随剂量单调(0.155→0.095→0.025)刻画了部分召回↔全或无的形态谱。vt 0.3314 全场最高。
待 A 臂(lmkq 无NCI)与纯锚新协议对照补 lmkq 纯效应与共享结构净效应。
