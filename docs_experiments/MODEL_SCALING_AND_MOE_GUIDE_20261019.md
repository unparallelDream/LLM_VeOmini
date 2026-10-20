# gated_mla_hybrid 模型扩展与 MoE 指南

> 更新日期：2026-10-19。适用仓库：`/9950backfile/zhangem/LLM_VeOmni`（models/GatedMLAHybrid）。
> 硬约束：`hidden_size = num_heads × head_dim`、`qk_nope + qk_rope = head_dim`、`heads × v_head_dim = hidden`（config 构造时自动校验）。

## 1. Dense 扩展配置表（已全部冒烟验证）

| 参数 | 50M（现状） | 150M | 400M | 1B | 1.8B |
|---|---|---|---|---|---|
| hidden_size | 384 | 640 | 1024 | 1536 | 2048 |
| num_layers | 16 | 20 | 24 | 28 | 28 |
| heads × head_dim | 6×64 | 10×64 | 16×64 | 12×128 | 16×128 |
| qk_nope + qk_rope | 32+32 | 48+16 | 32+32 | 64+64 | 64+64 |
| intermediate (4×hidden) | 1536 | 2560 | 4096 | 6144 | 8192 |
| q/kv_lora_rank | 128/128 | 256/256 | 256/256 | 512/512 | 512/512 |
| window_size (SWA) | 128 | 256 | 512 | 1024 | 2048 |
| 词表 | 32000（锁死，见 §4） | 同 | 同 | 同 | 同 |

扩展原则（依据摘要）：宽度为主深度缓增（W\*~C^0.34 / D\*~C^0.12，与 Qwen3 实证一致）；MLA rank 绝对值缓涨、占比随规模下降（DeepSeek 从 16B 到 671B rank 完全不变的实证）；词表 32K 在 1B 内处于 compute-optimal 区间（Tao et al., NeurIPS 2024），3B+/中文/code 才需要扩。

冒烟实测（B4×S512，2 步，纯 bf16）：四档 loss 起点 10.50/10.56/10.69/10.81（≈ln 32000，正确），训练峰值显存 3.2/6.2/12.5/18.6 GB。显存预算结论：**24G 卡上限 ~1.4B（fp32-master+梯度检查点），40G 卡上限 ~2.3B**；更长序列受 GatedMLA 的 S² dense attention bias 支配（根治需换 flash 变体）。

## 2. MoE 部分新增

### 2.1 机制（tier-1 eager 参考实现，单机无 EP/TP）

- **路由**：`gate: Linear(hidden → E)`，softmax 后取 top-k，**路由权重重归一化**后加权求和各专家输出；
- **专家**：每个专家是一个独立 SwiGLU（宽度 `moe_intermediate_size`，默认 `intermediate_size // top_k`，**保证 active 参数与 dense 对应档位一致**）；可选共享专家（`shared_expert_intermediate_size > 0`，默认关闭）；
- **负载均衡损失**：switch 风格 `E · Σ_f (assign_frac_f × mean_prob_f)`，每层 MoE 存于 `self.last_balance_loss`，模型 forward 汇总后挂到 `outputs.aux_losses`，CausalLM 训练时以 `moe_balance_loss_weight`（默认 0.01）加进 CE loss——经由 veomni 的 `mean_global_loss` 多损失管线自动传递与归约；
- **已知固有行为**：小 batch 下未被路由到的专家该步梯度为 None（AdamW 自动跳过），属 MoE 正常现象，非 bug。

### 2.2 配置字段

| 字段 | 默认 | 说明 |
|---|---|---|
| `num_experts` | 0 | 0 = dense SwiGLU（原路径逐字节不变）；>1 启用 MoE；=1 被校验拒绝 |
| `num_experts_per_tok` | 2 | top-k，校验 1 ≤ k ≤ E |
| `moe_intermediate_size` | 0 | 0 → 自动取 `intermediate_size // k`（active 对齐 dense） |
| `shared_expert_intermediate_size` | 0 | >0 启用共享专家 |
| `moe_balance_loss_weight` | 0.01 | aux loss 权重 |

### 2.3 MoE 变体扩展表（E=8, top-2，active 与 dense 档对齐，参数量实测）

| 配置 | 宿主 dense 档（整个 MoE 模型的骨架来源，非专家本身） | expert 宽度 | **每专家参数**（3·d·i，单个 SwiGLU） | router/层 | **每层 MoE 块** | **总参数** | **active 参数** | 稀疏倍率 |
|---|---|---|---|---|---|---|---|---|
| `gated_mla_150m_moe` | 150M (d640) | 1280 | **2.46M**（3×640×1280） | 5.1K | 19.67M | **446M** | ≈151M | 3.0× |
| `gated_mla_400m_moe` | 400M (d1024) | 2048 | **6.29M**（3×1024×2048） | 8.2K | 50.34M | **1.32B** | ≈414M | 3.2× |
| `gated_mla_1b_moe` | 1B (d1536) | 3072 | **14.16M**（3×1536×3072） | 12.3K | 113.26M | **3.46B** | ≈1.08B | 3.2× |

说明：每专家 = 一个独立 SwiGLU（gate/up/down 三个投影，无 bias），参数 = 3 × hidden × expert 宽度；每层 MoE 块 = E 个专家 + router；active 构成 = top-2 专家 + 全部注意力 + embedding（tied）。同 active 下总参数约 3 倍是 E8/K2 的自然稀疏率。改 E 或 expert 宽度时按上表公式换算：`每专家 = 3·d·i`，`总参数 ≈ dense + (E−k)·3·d·i·L`。

### 2.3a 专家数变体矩阵（k=2 不变，专家宽度不变，active 不变）

| 变体 | E | 每层专家质量 | 150m 档总参 | 400m 档 | 1b 档 | 均衡损失权重 | 冒烟 |
|---|---|---|---|---|---|---|---|
| `_moe`（默认） | 8 | 8×单专家 | 446M | 1.32B | 3.46B | 0.01 | ✓ |
| `_moe_e10` | 10 | **10×单专家** | 0.54B | 1.62B | 4.25B | 0.01 | ✓(150m) |
| `_moe_e16` | 16 | 16×单专家 | 0.84B | 2.53B | 6.6B | 0.02 | ✓(150m) |

注：若要求"每层专家总质量 = 单专家的 8~10 倍"，即 E=8~10，用默认 E8 或 `_e10` 变体；"全模型总参数 = 单专家×8~10"在多层 MoE 下数学不成立（专家总质量 = E·L·单专家 ≥ 128×）。E16 档 1b 变体纯 bf16 静态显存 ≈40GB（80G 卡可跑）。

### 2.3.1 每专家完整 Dense 配置表（单个专家 = 独立 SwiGLU FFN）

| 维度 | 150m_moe 的专家 | 400m_moe 的专家 | 1b_moe 的专家 |
|---|---|---|---|
| 输入/输出宽度（= host hidden） | 640 | 1024 | 1536 |
| expert 中间宽度 i | 1280 | 2048 | 3072 |
| gate_proj | 640→1280，**0.819M** 参数 | 1024→2048，**2.097M** | 1536→3072，**4.719M** |
| up_proj | 640→1280，**0.819M** | 1024→2048，**2.097M** | 1536→3072，**4.719M** |
| down_proj | 1280→640，**0.819M** | 2048→1024，**2.097M** | 3072→1536，**4.719M** |
| 激活/偏置 | SwiGLU（SiLU 门控），无 bias | 同左 | 同左 |
| **每专家参数合计** | **2.46M** | **6.29M** | **14.16M** |
| 每层专家数 E / top-k | 8 / 2 | 8 / 2 | 8 / 2 |
| host 层数 | 20 | 24 | 28 |
| **等效 dense FFN 宽度（k·i）** | **2560**（= dense 150M 的 intermediate） | **4096**（= dense 400M） | **6144**（= dense 1B） |

要点：专家宽度按 `i = intermediate_dense / k` 反推，因此 **top-2 激活的专家联合宽度恰好等于对应 dense 档的 FFN 宽度**（active 容量严格对齐）；专家内部与 dense SwiGLU 逐算子同构，仅处于 host 模型中的位置不同（每层被 router 以重归一化的 top-2 概率加权求和）。

### 2.4 显存与训练预算影响

总参数决定静态显存（纯 bf16 ≈ 6 B/param 含优化器；fp32-master ≈ 16 B/param）：
- 150m_moe：纯 bf16 ≈ 2.7GB 静态 + 激活 → **24G 卡可跑**；fp32-master ≈ 7GB ✓；
- 400m_moe：纯 bf16 ≈ 8GB、fp32-master ≈ 21GB → 24G 勉强/40G 稳；
- 1b_moe（3.46B total）：纯 bf16 ≈ 21GB、fp32-master ≈ 55GB → **需 80G 卡（fp32-master）或 40G（纯 bf16+检查点）**。
路由 loop 的 eager 开销：每层 E 次小批量 GEMM，短程实验可接受；加速路线见 §2.6。

### 2.5 与其它机制的正交性（含 KDA×MoE 组合配置）

MoE（层内 FFN 替换）与 iHC/mHC（层间残差）、KDA 混合（注意力层替换）全部正交，config 可任意叠加；cache 协议修复与 generate 路径不受 MoE 影响。**KDA×MoE 组合 config 已建**（KDA 替换部分层的注意力 + 全部层的 FFN 换 MoE，即 Kimi Linear 式"线性注意力多数/混合 + MoE FFN"的同构元素）：

| 配置 | 结构 | KDA 层 | 每专家 | **总参数** | **active** |
|---|---|---|---|---|---|
| `kda_gated_mla_50m_moe` | d384×16L + E8K2(专家宽768) | 4/16（[3,7,11,15]） | 0.88M | **135M** | ≈50M |
| `kda_gated_mla_150m_moe` | d640×20L + E8K2(专家宽1280) | 5/20（[3,7,11,15,19]，按扩展表比例） | 2.46M | **449M** | ≈154M |

注意：组合配置含 fla KDA 层，训练需 `GATED_PURE_BF16=1` 与 `FLA_TILELANG=0`（smoke 脚本已默认）；更大规模组合（400m/1b）按 §2.3 公式与 §1 线性层比例表直接派生。

### 2.6 路线图

- **tier 1（已实现）**：eager top-k 路由 + 负载均衡损失，单机 DDP；
- **tier 2**：多卡 TP——专家 FFN 模块名加进切分计划（参照 qwen3_moe）；
- **tier 3**：多卡 EP——`--train.expert_parallel_size` 管线已就绪，ep_plan 参照 veomni `qwen3_moe/parallel_plan.py`（8 行：experts 三个投影 Shard(0)）；已知坑：EP+FSDP2 组合的 optimizer 参数过滤限制，EP 时优先 DDP；
- **性能加速**：eager 专家循环 → batched group-GEMM（veomni `ops/fused_moe/group_gemm` 现成）→ 真实吞吐需求出现时接入。

### 2.7 验证状态

- 单测：dense 路径（num_experts=0）不受影响 ✓；MoE 前向形状/损失有限/被采样专家全部有梯度 ✓；balance loss 数值正常（layer0 ≈ 1.09，接近均匀路由理论值 1.0）✓；top_k=E 退化路径 ✓；num_experts=1 被校验拒绝 ✓。期间修复两个实现 bug：① transformers 4.57 的 ModelOutput 不接受任意 kwargs，aux loss 改为构造后 setattr 挂载；② 路由权重经 `.float()` softmax 后未 cast 回模型 dtype，bf16 训练时 index_put 报 dtype 不匹配（CPU fp32 单测暴露不了，bf16 冒烟才触发）——已修，教训是 MoE 单测需补 bf16 用例。
- 三档冒烟（B4×S512，2 步，纯 bf16，全部通过）：

| 配置 | 起步 loss | grad_norm | 峰值显存 | DCP/HF |
|---|---|---|---|---|
| 150m_moe（446M total） | 10.71 | 6.1 | ~5 GB | ✓/✓ |
| 400m_moe（1.32B total） | 10.81 | 10.8 | ~11 GB | ✓/✓ |
| 1b_moe（3.46B total） | 10.97 | 17.0 | 26.5 GB | ✓/✓ |

（1b_moe 两次因共享 GPU 被其它进程抢占显存而中断，换卡重跑通过——共享集群上大显存任务注意。）产物：`tmp/scale_smoke_gated_mla_*_moe/checkpoints/global_step_2/hf_ckpt/`。

- KDA×MoE 组合冒烟（同口径，全部通过）：

| 配置 | 起步 loss | grad_norm | DCP/HF |
|---|---|---|---|
| kda_gated_mla_50m_moe（135M total） | 10.60 | 6.7 | ✓/✓ |
| kda_gated_mla_150m_moe（449M total） | 10.70 | 11.3 | ✓/✓ |
| gated_mla_150m_moe_e10（0.54B total） | 10.71 | 6.1 | ✓/✓ |
| gated_mla_150m_moe_e16（0.84B total） | 10.92 | 6.0 | ✓/✓ |

## 3. 复现命令

```bash
cd /9950backfile/zhangem/LLM_VeOmni
# dense 档
GPU=1 CONFIG=configs/gated_mla_hybrid/gated_mla_1b.json bash scripts/gated_mla_smoke.sh
# MoE 档
GPU=1 CONFIG=configs/gated_mla_hybrid/gated_mla_400m_moe.json bash scripts/gated_mla_smoke.sh
```

## 4. 附：暂不扩展的维度与触发条件

| 维度 | 现状 | 扩展触发条件 |
|---|---|---|
| 词表 | 32K | 3B+ 规模 / 中文 / code 语料 / 对齐 Qwen 生态评测（届时换 50K 级，配 per-char loss 桥接） |
| head_dim×heads 与 hidden 解耦 | 受硬约束绑定 | "v2 配方"一次性切换（连同 intermediate 系数变更） |
| 混合比（KDA:full） | 4:12（full 多数） | 1B 档放大验证时可切 3:1（KDA 多数，对齐 Kimi Linear） |
