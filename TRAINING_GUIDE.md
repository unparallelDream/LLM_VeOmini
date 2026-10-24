# 训练配置与新模型注册指南

> 适用仓库：LLM_VeOmni（VeOmni 训练框架 + gated 模型族）。
> 本文所有路径、参数、行为均经实际冒烟/消融验证。配套：[README.md](README.md)（项目结构）、[docs_experiments/MODEL_SCALING_AND_MOE_GUIDE_20261019.md](docs_experiments/MODEL_SCALING_AND_MOE_GUIDE_20261019.md)（扩展与 MoE）。

---

## 1. 训练配置：四个层次

### 1.1 层 1 —— 启动脚本环境变量（`scripts/gated_mla_smoke.sh`）

| 变量 | 默认 | 说明 |
|---|---|---|
| `CONFIG` | `configs/gated_mla_hybrid/gated_mla_50m.json` | 模型 config 路径（决定架构 + 全部机制开关） |
| `GPU` | 6 | 使用的卡号 |
| `STEPS` | 2 | 训练步数（冒烟 2、消融 500、正式 20000） |
| `SEED` | 42 | 随机种子 |
| `DATA` | `…/LLM_Standardization/data/smoke.jsonl` | 训练数据（jsonl/parquet；正式跑用 `/9950backfile/zhangem/data/fineweb_dataset/00X_00000.parquet`） |
| `TOKENIZER` | `configs/gated_mla_hybrid/llama32k_tokenizer` | 分词器目录 |
| `OUT` | `tmp/gated_smoke_<config名>` | 输出目录 |
| `SAVE_STEPS` | = STEPS | 存档间隔（正式跑建议 5000） |
| `SAVE_HF` | true | 最后一步是否导出 HF 格式 |
| `DP_MODE` | ddp | 数据并行模式（见 §1.4） |
| `INIT_DEVICE` / `ENABLE_FULL_SHARD` | cuda / false | fsdp2 相关（`meta`/`true` 仅多卡 fsdp2 用） |
| `WARMUP` | 0.5 | warmup 比例（正式跑改 0.03） |
| `GATED_PURE_BF16` | 1 | 精度开关（**KDA 配置必须 =1**） |
| `FLA_TILELANG` | 0 | fla 后端（**KDA 配置必须 =0**，本环境 tilelang 与 fla 不兼容） |
| `GQA_ATTN_BACKEND` | **flash** | GQA 注意力后端：flash（官方 FA 内核，sink=none 时生效，S8K 实测 3.7×）/ sdpa（全 sink 态通用）/ flex（长序列+sink，块稀疏）。virtual/first_tokens/增量/CPU 自动回退 sdpa |
| `GATED_COMPILE` | 0 | compile 注意力子模块（实测 50M@B4 稳态持平，当前规模不建议开） |

### 1.2 层 2 —— veomni 训练参数（脚本转译为 CLI）

三组 dataclass（`tasks/pretrain_gated.py` 的 `parse_args` 解析）：

- `--model.*`：`config_path` / `tokenizer_path` / `attn_implementation`（sdpa）
- `--data.*`：`train_path` / `data_type plaintext` / `max_seq_len` / `text_keys text` / `num_workers` / `drop_last`
- `--train.*`：`lr` `lr_min` `lr_decay_style` `lr_warmup_ratio` `max_grad_norm` `weight_decay` / `micro_batch_size` `global_batch_size` / `no_decay_params`（豁免名单：`norm bias embed alpha ihc_readout mhc_readout`）/ `save_steps` `ckpt_manager dcp` / `load_checkpoint_path`（断点续训）/ `use_wandb`

### 1.3 层 3 —— 模型 config JSON（架构 + 机制开关）

**结构字段**（`configuration` 构造时自动校验三条硬约束）：

```
hidden_size = num_attention_heads × head_dim
qk_nope_head_dim + qk_rope_head_dim = head_dim
num_attention_heads × v_head_dim = hidden_size        （GQA 家族另有：heads % num_key_value_heads == 0）
```

**机制开关（可任意叠加）**：`baseline`（gated_mla / gated_gqa / swa / hils / kda + `linear_layer_indices` 线性层位置）、`use_ihc`/`use_mhc`（超连接）、`num_experts`（MoE，0=dense）、`num_key_value_heads`（GQA 族）、**`sink_style`（sink 机制三态：`virtual`=gpt-oss 式每头可学习 β 进 softmax 分母【新默认，fp32 对照 1.9e-08】/ `first_tokens`=原 StreamingLLM 式首 token 突破窗口【存量 27 个 config 已固化此语义】/ `none`=无 sink，此时默认后端走官方 FA 内核）**。

### 1.4 层 4 —— 精度与并行路径（DP_MODE × GATED_PURE_BF16）

| 组合 | 行为 | 适用 |
|---|---|---|
| `ddp` + `PURE_BF16=1`（默认） | 全 bf16 + bf16 AdamW | 冒烟/消融；**KDA 必须** |
| `ddp` + `PURE_BF16=0` | DDP 原生 MixedPrecision：fp32 主权重、forward 真转 bf16、梯度 fp32 归约 | 单卡正式训练（数值更稳） |
| `fsdp2`（多卡） | fully_shard 分片 + MixedPrecisionPolicy | 多卡大模型（≥1.8B / 大 total 的 MoE） |

⚠ 本 veomni 版本 `fsdp_enabled` 要求 world_size>1：**单卡设 fsdp2 无效**（自动退回 ddp 语义且拒绝 `init_device=meta`）。

### 1.5 常用配方

```bash
# 冒烟（2 步 + HF 导出）
GPU=1 CONFIG=configs/gated_mla_hybrid/gated_mla_50m.json bash scripts/gated_mla_smoke.sh
# 任意变体：只换 CONFIG（MoE / GQA / iHC / mHC / KDA×MoE / 扩展档）
GPU=1 CONFIG=configs/gated_gqa_hybrid/gated_gqa_50m.json bash scripts/gated_mla_smoke.sh
# 消融（500 步）
GPU=4 CONFIG=… STEPS=500 SEED=42 WARMUP=0.03 SAVE_STEPS=500 bash scripts/gated_mla_smoke.sh
# 正式长跑（2 万步、全量数据分片、周期存档）
GPU=4 CONFIG=… STEPS=20000 WARMUP=0.03 SAVE_STEPS=5000 \
  DATA=/9950backfile/zhangem/data/fineweb_dataset/003_00000.parquet \
  OUT=tmp/formal_run bash scripts/gated_mla_smoke.sh
# 断点续训（在脚本追加参数）
#   --train.load_checkpoint_path <DCP checkpoint 目录>
```

**产物**：`$OUT/checkpoints/global_step_N/`（DCP）+ `…/hf_ckpt/`（HF 格式：config.json、model.safetensors、tokenizer——可直接 `from_pretrained`）。

---

## 2. 新模型注册：两种路线

### 路线 A：家族内新变体（改 baseline 分派，最轻）

适用于"新注意力/新 FFN 机制"，与现有家族共享一切。以 GQA 为例的三步：

1. **config 加开关**（`models/GatedMLAHybrid/configuration_gated_mla_hybrid.py`）：`BASELINES` 字典加 `"gated_gqa": ()`，需要新参数就加字段 + `__init__` 校验；
2. **modeling 加类**（同目录 `modeling_gated_mla_hybrid.py`）：写 `GQAAttention(nn.Module)`（约定：`forward(hidden_states, past_key_value=None, use_cache=False, position_ids=None, attention_mask=None, **kwargs)`，返回 `(output, cache_or_None)`；cache 用 legacy tuple，生成路径已被 DynamicCache 适配层覆盖），然后在 `HybridBlock.__init__` 的 baseline 分派里挂上；
3. **建 config JSON**（`configs/gated_mla_hybrid/` 或独立子目录）：拷最近似变体改字段。

### 路线 B：新独立家族（薄子类，独立 model_type）

适用于需要独立目录/独立 config 命名空间时。以 `GatedGQAHybrid` 为例的四件套：

```
models/GatedGQAHybrid/
├── __init__.py                       # ① VeOmni 注册（见下）
├── configuration_gated_gqa_hybrid.py # ② 薄子类：model_type + baseline 钉死
└── modeling_gated_gqa_hybrid.py      # ③ 薄子类：继承 ForCausalLM/Model
configs/gated_gqa_hybrid/*.json       # ④ model_type 与 architectures 指向新家族
```

**① `__init__.py` 注册模板**（VeOmni loader 的两个注册表）：

```python
from veomni.models.loader import MODELING_REGISTRY, MODEL_CONFIG_REGISTRY

@MODELING_REGISTRY.register("gated_gqa_hybrid")          # key = config 的 model_type
def register_gated_gqa_hybrid_modeling(architecture: str):
    from .modeling_gated_gqa_hybrid import GatedGQAHybridForCausalLM, GatedGQAHybridModel
    if "ForCausalLM" in architecture: return GatedGQAHybridForCausalLM
    if "Model" in architecture:       return GatedGQAHybridModel
    return GatedGQAHybridForCausalLM

@MODEL_CONFIG_REGISTRY.register("gated_gqa_hybrid")
def register_gated_gqa_hybrid_config():
    from .configuration_gated_gqa_hybrid import GatedGQAHybridConfig
    return GatedGQAHybridConfig
```

**②③ 薄子类要点**：config 子类重定义 `model_type`/`architectures` 并在 `__init__` 钉死 baseline；modeling 子类只改 `config_class`。机制（混合/iHC/MoE/cache）全部继承复用。若要接入 transformers Auto 体系（评测侧），再各加一行 `AutoConfig.register(...)` / `AutoModelForCausalLM.register(...)`。

**④ config JSON 必改字段**：`"model_type": "gated_gqa_hybrid"`、`"architectures": ["GatedGQAHybridForCausalLM"]`（loader 按 `architectures[0]` 选 ForCausalLM 类）。

最后把新家族挂进 `models/__init__.py` 的 import 列表与 `__all__`。

### 注册后如何被找到（调用链）

```
config JSON → model_type 查 MODEL_CONFIG_REGISTRY → config 类
           → architectures[0] 查 MODELING_REGISTRY → modeling 类
build_foundation_model → CustomizedModelingLoader.from_config → 实例化
```

---

## 3. 新模型验证清单（每步都是既有实践的固定动作）

1. **单测（CPU，分钟级）**
   - 构建即校验：非法配置被拒（如 GQA 的 heads 不整除、num_experts=1）；
   - 等价性：机制关闭时应退化为基线——同种子构建，`max|Δlogits|` 应为 **0.00e+00**（iHC α=0、mHC α=0、GQA 拆分均用过此标准）；
   - 梯度：全部新参数 grad 有限非 None；bf16 下再跑一次（fp32 单测暴露不了 dtype bug——MoE 的路由权重 dtype 教训）；
   - cache（若自定义）：legacy tuple 增量前向 + generate 等价。
2. **GPU 冒烟（2 步）**：起步 loss ≈ ln(vocab)（32K→10.37 附近）、grad_norm 有限、DCP+HF 导出成功。
3. **generate 等价性**：导出 ckpt 上 `model.generate` vs 手动全量重算 greedy 逐 token 一致。
4. **开销基准**（可选）：B4×S512 fwd+bwd 对比基线，记录 ms/iter。

```bash
# 冒烟验证命令
GPU=<空闲卡> CONFIG=configs/<族>/<新变体>.json bash scripts/gated_mla_smoke.sh
```

---

## 4. 已知约束速查

| 约束 | 内容 |
|---|---|
| KDA 依赖 | `FLA_TILELANG=0`（fla triton 后端）+ `GATED_PURE_BF16=1`（fp32 kernel 在本环境 triton 3.2 编不过） |
| 单卡 fsdp2 | 无效（veomni 的 fsdp_enabled 只看 world_size） |
| GatedMLA 长 sequence | S² dense attention bias 显存瓶颈（S8K 时 1B 模型 30GB+）；根治需换 flash 变体 |
| MoE 静态显存 | total 参数 × 6B（纯 bf16）/ ×16B（fp32 master）；1b_moe(3.46B) 需 80G 卡跑 fp32-master |
| triton 升级 | 别动共享环境；需要时 clone 环境验证（hils 内核 + fla 都依赖当前版本） |
| 生成评测 | 已修 DynamicCache 协议（recompute 模式）；SWA/KDA 层忽略 attention_mask → lm-eval 建议 batch_size=1 |
| 注意力后端路由 | 默认 flash：sink=none+GPU+全长 → 官方 FA（S8K 3.7×，等价 0.0002）；带 sink/增量/CPU → 自动 sdpa；flex=长序列+sink 选项。MagiAttention 组合层 vendored 仅供基准（官方 FA 上输出不正确，勿用于模型） |
| sink 语义 | 新默认 virtual（每头 β）；历史 config 已固化 first_tokens；500 步消融：ON 均值持平且种子方差低 4 倍 |
| 上游同步 | `bash scripts/sync_from_hils.sh [--dry-run]`；豁免：`models/__init__.py`、`.gitignore`、`README.md`；排除：archive_qwen_small_scale、三个已删上游 md |
