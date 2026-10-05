# MASS 多 anchor 分化改进实验(2026-09-26 启动)

## 背景

探针结论(probe_mass_retrieval_0p6b_20260926):MASS 的 2-anchor 分化在向量空间完成
(cos(a0,a1)≈-0.1)但无功能——路由自信(condH→0)、负载恒 [0.5,0.5]、选择质量≈随机
(frac_qh 0.586 vs 随机 0.561),fixed0≈mass。根因:任务梯度饥饿 + spec loss 只奖励
路由置信/平衡(任意对称破缺即可达最优)+ 3k 步内早锁定。

## 新增代码(env 门控,默认全关,行为不变)

| 开关 | 作用 | 位置 |
|---|---|---|
| `MASS_NEEDLE_INFO_NCE=1` | needle-chunk 多正例 InfoNCE:检测 RULER needle(模板 token [' special',' magic',' numbers'],自然文本不出现),在检索层用参考打分路径(q·μ·scale+b,anchor-LSE,与 kernel 数学一致)对最后 64 个 query 行做 -log(Σneedle/Σeligible);梯度同时流向 anchors/q/k | `mass_attention.py` `_maybe_aux_losses/_needle_info_nce`;`modeling_qwen_mass.py` `_build_needle_ctx/forward` |
| `MASS_NCI_WEIGHT/_TAU/_QUERY_ROWS/_MAX_NEEDLES` | NCI 权重(默认 1.0,本实验 8.0 补偿 5% 稀疏)、温度、query 行数、正例上限 | 同上 |
| `MASS_SPEC_WARMUP_STEPS` | spec 权重 0→base 线性爬坡(训练循环设 `_mass_spec_weight_step`),防止分化在任务塑形前锁死 | `pretrain_with_ruler.py` + `modeling_qwen_mass.py` |
| `MASS_SPEC_DIVERSE_MU=1` | spec(MI 路由目标)整体替换为 μ 功能分化:mean cos²(μ₀,μ₁) 逐块/头,推向正交视图 | `mass_attention.py` `_maybe_aux_losses` |
| `MASS_NEEDLE_PATTERN` | 自定义检测短语(启动时对两个 needle 模板做逐 token 校验,失败即 raise) | `pretrain_with_ruler.py` |

训练循环同步:`loss_terms` 增加 `foundation_loss_mass_needle`(重构路径会绕过 outputs.loss,
必须显式加入)、tqdm/日志/wandb 增加 nci 显示、all_reduce 5 元组。

关键坑(已修):梯度检查点重算要求两次前向代码路径一致——needle ctx 必须保留到下一次
forward 开始再重置,否则 CheckpointError(99 vs 68 tensors)。

## 验证

- 单元:模板匹配、真实 RULER 样本 needle chunk 检测(与 numpy 基准一致)、NCE 数值
  (log-prob 量级)、梯度流(q/base/anchors 三路均非零)、cos² 初始≈0.85(reinit 后两视图已略分开)。
- 冒烟:14 步全特性无崩溃,step12 nci=-14.62 实际参与反向,spec warmup 0.00025→0.0035。

## 训练 run(均已启动,nohup + pid,crash-resume 同既有惯例)

| run | GPU | 配置增量(相对基线 run#2 anchor-only) | 输出 |
|---|---|---|---|
| nci | 0 | NCI(w=8, τ=1, 64 rows)+ spec warmup 4000 步 | `outputs/mass_qwen3_0p6b_anchor_nci_specwu_ft20000` |
| divmu | 5 | spec → cos²(μ) 功能分化(权重 1.0) | `outputs/mass_qwen3_0p6b_anchor_divmu_ft20000` |
| 基线 | — | run#2(已完成,spec 常数 1.0) | `outputs/mass_qwen3_0p6b_anchor_only_m2_r8_asym0p5_anneal_ft20000` |

日志:`logs/mass_anchor_{nci,divmu}_0p6b_ft20000.log`;pid:`pids/`。
其余(lr/batch/数据/冻结 pattern/退火)与 run#2 完全一致;启动脚本
`scripts/launch_mass_retrieval_improve_20260926.sh`。

## 完成后的评测协议(绑定态才有区分度)

1. 4K/n200 四任务 × {nci, divmu, 基线, fixed0}(固定 4K 只是连续性参照);
2. 探针重跑(frac_qh/命中率 vs 随机基线 32/57、anchor_load、condH、cos(μ));
3. **绑定态主判据:8K 输入 + topk=8 的评测与探针**——只有在那里,anchor 打分是否
   产生功能才能与 fixed0 拉开;判据 = nci/divmu 的 frac_qh 显著 > 随机(32/(130-8)≈0.26)
   且下游 niah/宏观超过基线与 fixed0。

## 中期探针结果(2026-09-27 18:10,nci@4300+/divmu@3050+ 时点)

50 样本 × 7 检索层;配对差为逐样本(远程 needle, n=43)95%CI。

### topk=32(4K 标准预算;随机基线 0.561)

| target | hit_rmt | frac_qh | 配对差 vs anchor1000 |
|---|---|---|---|
| zeroshot | 0.997 | 0.567 | — |
| anchor1000(基线) | 1.000 | 0.586 | — |
| nci1000/3000/4000 | 1.000 | 0.587 | +0.0018±0.0031 |
| divmu1000/3000 | 1.000 | 0.586 | +0.0012±0.0038 |

### topk=8(绑定态;随机基线 0.140)

| target | hit_rmt | frac_qh | 配对差 vs anchor1000 |
|---|---|---|---|
| zeroshot | 0.861 | 0.207 | -0.0081±0.0408 |
| anchor1000(基线) | 0.914 | 0.214 | — |
| nci4000 | 0.914 | 0.212 | -0.0030±0.0033 |
| divmu3000 | 0.914 | 0.213 | -0.0013±0.0052 |

### 判读

1. **两个干预在两种预算下都零效应**(CI 全含 0)。NCI 梯度确实存在且量级大
   (NCI 步 grad_norm 0.6-1.2 vs 常规 0.13,~1000 次事件、权重 8),但选择质量不动——
   与 spec loss 同参数下 1000 步内就把 condH 0.69→0.11 形成对照:**优化器能快速移动
   anchor 模块,NCI 的方向却表达不出来**。
2. 这把根因从"目标错位"上移到"参数化/表达力":anchor-only(114.8k,全局固定 rank-8
   offset,q/k 冻结)在功能上无法表达"提升 needle 相对干扰块的分数"。
3. 次级观察:divmu 臂无 MI spec loss 但 condH 仍降至 0.112——路由置信由 envelope-max
   结构与 LM 自身产生,spec loss 并非必要;cos² 0.845→0.830 缓降。
4. 两臂继续跑完(成本已沉没,终点做 n200 评测收尾确认),但机制问题已有答案。

### 下一步建议(按证据强度)

- 决定性实验:**mass_lora + NCI** vs mass_lora(LoRA 提供 q/k 表达力,检验"监督在可表达
  通路下是否生效";hils_lora 0.475 已证明 LoRA+landmark 接口可表达);
- 或接受结构性结论:anchor 旁路不可训,转向把打分耦合进 q·k 主通路(调制式)。
