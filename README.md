# LLM_VeOmni

基于 VeOmni 训练框架的 LLM 机制研究与扩展工作仓库。两条主线：

1. **上游线（HiLS / MASS）**：来自 [HiLS-Attention](https://github.com/Tencent-Hunyuan/HiLS-Attention)（论文 [arXiv:2607.02980](https://arxiv.org/abs/2607.02980)）的分层稀疏注意力 / 多锚点检索（MASS）训练与评测全栈，含 TileLang 内核；
2. **扩展线（GatedMLAHybrid，本仓库新增）**：从 LLM_Standardization（flame 栈）迁移并扩展的 gated_mla 模型族——**iHC/mHC 超连接、MoE（E8/E10/E16）、KDA 混合、规模扩展四档**，全部经冒烟验证。

> 环境约定：conda 环境 `hils_smoke_py311`（torch 2.6.0+cu124 / triton 3.2 / veomni）；fla 已装入（`--no-deps`，KDA 训练需 `FLA_TILELANG=0` + 纯 bf16）。

## 文件结构

```
LLM_VeOmni/
├── models/                          # 模型实现与 VeOmni 注册（MODELING/MODEL_CONFIG_REGISTRY）
│   ├── GatedMLAHybrid/              # ★ 扩展线模型族
│   │   ├── configuration_gated_mla_hybrid.py   # 配置：iHC/mHC/MoE 字段 + 三条硬约束校验
│   │   │                                       #   (hidden=heads×head_dim, nope+rope=head_dim, heads×v_head=hidden)
│   │   └── modeling_gated_mla_hybrid.py        # 建模：GatedMLA/SWA/HiLS占位/KDA 注意力
│   │                                           #   + IdentityHyperConnection / ManifoldHyperConnection
│   │                                           #   + MoEFFN(top-k路由+负载均衡损失) + DynamicCache 协议适配
│   ├── GatedGQAHybrid/              # ★ GQA 变体家族（model_type gated_gqa_hybrid；薄子类复用
│   │                                #   家族机制，baseline 固定 gated_gqa，KV=分组全维）
│   ├── FlashHiLS/                   # 上游：HiLS 注意力（qwen/olmo 两个 modeling + hils_forward）
│   ├── MASSIndex/                   # 上游：MASS 多锚检索（mass_attention/mass_retrieval 内核接口）
│   ├── FullAttn/ NSA/ DASHATTN/ HSA_Ultralong/ INFLLMV2/   # 上游：对照模型族
│   └── __init__.py                  # 注册入口（含 GatedMLAHybrid 导入；与上游同步时的豁免文件）
│
├── configs/
│   ├── gated_mla_hybrid/            # ★ 扩展线 config 矩阵（全部冒烟通过）
│   │   ├── gated_mla_50m.json       # 基线（50M, d384×16L, 6×64）
│   │   ├── {swa,hils,kda}_gated_mla_50m.json       # 混合注意力变体（4 线性层/16）
│   │   ├── {ihc,ihc_n2,mhc}_gated_mla_50m.json     # 超连接变体
│   │   ├── gated_mla_{150m,400m,1b,1p8b}.json      # 密集扩展四档
│   │   ├── gated_mla_{150m,400m,1b}_moe*.json      # MoE（E8 默认 / _e10 / _e16）
│   │   └── kda_gated_mla_{50m,150m}_moe.json       # KDA×MoE 组合
│   │   └── llama32k_tokenizer/      # 32K 词表 tokenizer（家族共用）
│   ├── gated_gqa_hybrid/           # ★ GQA 基线（gated_gqa_50m 6Q/2KV + _kv3 消融档；
│   │                               #   Qwen3.8-FN/MiniMax-M3 式配置，与 MLA 对照）
│   ├── mass_attention/ hils_attention/ naive_bsa/ nsa/ ...   # 上游各模型族配置
│   ├── training_recipes/            # VeOmni 训练 recipe YAML（fsdp2 等正式配方）
│   └── archive_qwen_small_scale/ olmo3_200M_cpt/ olmo3_7B/ olmo3_vocab/ ...
│
├── tasks/                           # VeOmni 训练任务入口
│   ├── pretrain_gated.py            # ★ gated_mla_hybrid 预训练（plaintext 管线；DDP-MP/纯bf16/fsdp2
│   │                                #   三种精度路径；GATED_PURE_BF16/FLA_TILELANG 开关）
│   └── pretrain_with_ruler.py / pretrain_mass_with_ruler.py / mass_profile_entry.py   # 上游 RULER 任务
│
├── scripts/
│   ├── gated_mla_smoke.sh           # ★ 冒烟/训练启动（CONFIG/GPU/STEPS/SEED/DP_MODE/WARMUP/SAVE_* 可覆盖）
│   ├── sync_from_hils.sh            # ★ 上游同步（--dry-run 预览；豁免 models/__init__.py 与 .gitignore）
│   └── eval/ cpt/ mass/ mid_train/ bench/ ...   # 上游脚本
│
├── ops/                             # TileLang/Triton 内核：HiLS/MASS 检索（flex_attn、mass_retrieval_v2、
│                                   #   topk_head_softmax 等；上游 10 月 kernel 修复已同步）
├── eval/                            # 评测：RULER / ppl 阶梯 / lm-eval / longbench / 对比分析
├── scripts_zem/                     # 实验启动脚本（含上游同步进来的 olmo200m 三臂续训 launch 系列）
├── unittests/ tests/                # 单测（MASS envelope、topk、训练推理一致性等）
├── docs_experiments/                # 实验文档
│   ├── MODEL_SCALING_AND_MOE_GUIDE_20261019.md   # ★ 扩展与 MoE 指南（扩展表/每专家参数/显存预算/路线图）
│   ├── hils_paper_arxiv_2607.02980.pdf           # 上游 HiLS 论文
│   └── DESIGN_SPEC_shared_expert.md / EVAL_* / MULTI_ANCHOR_* ...
├── utils/                           # 冻结参数、参数组、trainable checkpointer 等训练工具
├── data/ preprocess/                # veomni 数据管线代码与预处理
├── results/ assets/                 # 结果与图示
├── PROGRESS_*.md / KERNEL_ACCEL_REPORT.md          # 上游进度与内核加速报告
├── train.sh / train_dist.sh         # torchrun 启动器（单机/多机）
├── pyproject.toml / uv.lock / requirements.txt     # 依赖声明
├── outputs logs .cache              # → 本机共享资产软链（已 gitignore，不进版本库）
└── tmp/                             # 冒烟/消融产物（gitignore）
```

## 快速开始

```bash
# gated_mla_hybrid 冒烟（2 步 + HF 导出）
GPU=1 CONFIG=configs/gated_mla_hybrid/gated_mla_50m.json bash scripts/gated_mla_smoke.sh

# 换任意 config：MoE / KDA×MoE / iHC / mHC / 扩展档
GPU=1 CONFIG=configs/gated_mla_hybrid/gated_mla_400m_moe.json bash scripts/gated_mla_smoke.sh

# 短程训练（500 步，含 warmup/保存控制）
GPU=4 CONFIG=... STEPS=500 WARMUP=0.03 SEED=42 SAVE_STEPS=500 bash scripts/gated_mla_smoke.sh

# 单卡 fp32-master 路径（bf16 计算 + fp32 主权重）
DP_MODE=ddp GATED_PURE_BF16=0 bash scripts/gated_mla_smoke.sh
```

## 与上游的同步

上游 `HiLS-Attention-main`（本地目录）持续演进时：

```bash
bash scripts/sync_from_hils.sh --dry-run   # 预览差异
bash scripts/sync_from_hils.sh             # 拉入更新/新增，同步删除；豁免本仓库自维护文件
```

扩展线的机制细节、扩展配置表、每专家参数、显存预算与路线图见
**[docs_experiments/MODEL_SCALING_AND_MOE_GUIDE_20261019.md](docs_experiments/MODEL_SCALING_AND_MOE_GUIDE_20261019.md)**。
