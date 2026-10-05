# MASS 项目迁移文档

> 适用于将 MASS（Multi-Tangent Attention 上包络）项目从当前节点迁移到新节点。
> 文档覆盖：MASS 方法原理、Qwen3-0.6B 模型、fineweb 数据与转化脚本、anchor MoE 全部配置项。

---

## 1. 项目概述

MASS 在 HiLS（分段 landmark 稀疏注意力）基础上，将原本的**单泰勒平面检索**升级为**多锚（Multi-Tangent）上包络检索**，借鉴 MoE 的专家分工思想让多个锚点（anchor）各自负责不同语义/位置的 landmark chunk。本项目为 Qwen3-0.6B 基座 + 冻结 backbone 的锚点训练研究。

当前研究核心：**激活锚点分工**——通过非对称初始化 + 温度退火 + 互信息特化损失，使 M 个锚点负载产生分化（实验见第 7 节）。

---

## 2. MASS 方法原理

### 2.1 从 HiLS 到 MASS

- **HiLS**：每 chunk 末尾放 landmark token，用单个 query 投影与 key chunk 的点积（单个泰勒平面）做 top-k chunk 检索，再用稀疏 value attention 聚合。
- **MASS**：把"一个泰勒平面"换成"M 个切线平面（锚）的上包络"：

```
score(q, chunk) = max_m ( q·mu[c,m] / sqrt(D) + b[c,m] )
```

其中 `mu[c,m]` 是每个 chunk c、每个锚 m 的切线均值（对 chunk 内 key 做 softmax 池化），`b[c,m]` 是对应熵偏置。

### 2.2 锚点生成（MultiTangentAnchorGenerator）

```python
anchors = RMSNorm( base_anchor + offsets )          # [B, N, M, Hq, D]
offsets = einsum("mr,hdr->mhd", anchor_embedding, offset_basis)
```

- `base_anchor`：HiLS 的 landmark query 投影（chunk 级）
- `anchor_embedding [M, offset_rank]` + `offset_basis [Hq, D, offset_rank]`：**低秩因子化的锚偏移**，是唯一可训练参数（每层 1 组）
- RMSNorm 归一化保证锚与 query 同尺度

### 2.3 评分与 top-k 选择

- **评分缩减**（`mass_score_reduction`）：
  - `max`：硬 max（不可微，argmax 无梯度）
  - `logsumexp`（LSE）：连续可微替代，`scores = tau * logsumexp(planes / tau)`，低分锚也能拿到真实 BP 信号（本项目的关键路径）
- **选择**：TileLang 内核按 `[B, L, Hkv, N]` 计算上包络候选分 → bitonic top-k → 排序索引；**索引本身 `mark_non_differentiable`**，只有分数参与梯度
- 未选中 chunk 无梯度（硬 top-k 截断）

### 2.4 特化损失（spec loss，互信息目标）

```
conditional_entropy = E_r[ -sum_m p(r,m) log p(r,m) ]     # 锚条件熵
marginal_entropy    = -sum_m load_m log load_m             # 锚边缘熵
specialization_loss = (cond_entropy - marginal_entropy + log M) / log M   # 归一化到 [0,1]
```

- **load_m**：softmax 路由概率的期望负载（LSE 路径下可微）
- 含义：uniform 坍塌（各锚完全混淆）→ 1；确信且全局均衡分工 → 0
- 反向中提供**精确 softmax Jacobian**：

```python
balance_grad = probs * (load_grad - expected) / (valid_count * tau)
```

### 2.5 锚点分工激活机制（研究关键）

三层设计，缺一不可：

| 机制 | 配置 | 作用 |
|---|---|---|
| 非对称初始化 | `mass_offset_init_std=2.0` | 打破 `0.02` 导致的熵鞍点（两锚相同→无梯度） |
| 高温退火 | `lse_temperature` 4.0 → 0.1 | 高温期 softmax 份额≈0.5，低分锚获得真实梯度；低温期锐化锁定分工 |
| spec loss 权重 | `MASS_SPECIALIZATION_LOSS_WEIGHT=1.0` | 特化信号强度 |

**实验结论**（ft20000 运行中）：spec 从 0.973（step300）单调降至 0.966（step2570）；v3 smoke300 在 step117（tau=2.5）达 0.944。已确认 **spec 分化由 tau 退火驱动而非训练步数**。

---

## 3. 迁移资产清单

所有路径均位于共享文件系统 `/9950backfile/zhangem/` 下（需同步保留目录结构）。

| 资产 | 路径 | 大小 | 必须 |
|---|---|---|---|
| 代码仓库 | `HiLS-Attention-main/` | ~150MB | 是 |
| Qwen3 基座（tokenizer） | `.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master` | 1.5G | 是 |
| Qwen3-MASS-m2（backbone 权重） | `.cache/modelscope/models/Qwen--Qwen3-0.6B-MASS-m2/snapshots/master` | 1.5G | 是 |
| fineweb 原始数据 | `data/fineweb_dataset/`（14 个 parquet） | 27G | 是 |
| fineweb 转化后数据 | `data/fineweb_qwen3_data/`（14 个 .parquet.data） | 37G | 是 |
| conda 环境 | `miniconda3/envs/hils_smoke_py311/` | ~10-15G | 是 |
| 训练输出/checkpoint | `outputs/` | 按需 | 可选 |
| 启动脚本 | `launch_asym_anneal.sh` | 小 | 是 |

> 提示：`fineweb_qwen3_data` 可由 `fineweb_dataset` + 转化脚本重新生成（见第 5 节），若带宽有限可只迁移原始 parquet。

### 3.1 conda 环境迁移注意

`hils_smoke_py311/bin/` 下脚本（pip、accelerate 等）的 **shebang 硬编码绝对路径**：

```bash
#!/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311/bin/python3.11
```

- 若新节点路径与迁移后一致（同路径挂载），无需处理
- 若路径变化，需修复：

```bash
NEW_ENV=/new/path/envs/hils_smoke_py311
grep -rl "/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311" "$NEW_ENV/bin" \
  | xargs sed -i "s|/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311|$NEW_ENV|g"
```

- 也可用 `conda env export > environment.yml` 在新节点重建（见第 4 节版本清单）

---

## 4. 环境依赖

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.11 | conda env: `hils_smoke_py311` |
| CUDA | 12.4 | `/usr/local/cuda-12.4`，需 `nvcc` 编译 TileLang |
| PyTorch | 2.6.0+cu124 | `torch 2.6.0+cu124` |
| tilelang | 0.1.7.post2+cuda.git53ea96c6 | **必须为 commit `53ea96c6`**（A800 上编译并缓存好内核） |
| veomni | 0.1.0 (git 441e1b24) | ByteDance-Seed/VeOmni，patch 混合精度/FusedMoe |
| transformers | 4.57.3 | Qwen3 架构 |
| flash-attn | 2.8.3 | 可选 |
| liger-kernel | 0.6.2 | fused cross-entropy |
| einops / pyarrow / datasets | 0.8.1 / 25.0.0 / 2.21.0 | 数据与张量操作 |
| modelscope | 1.39.1 | 模型缓存目录规范 |

运行前必须设置：

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/9950backfile/zhangem/HiLS-Attention-main:$PYTHONPATH
```

---

## 5. 数据（fineweb）与转化脚本

### 5.1 原始数据

- 来源：HuggingFace **FineWeb-Edu**（`000_00000.parquet` ~ 013 共 14 shard，每 shard ~726k docs）
- 路径：`/9950backfile/zhangem/data/fineweb_dataset/`

### 5.2 转化脚本

```
HiLS-Attention-main/scripts/data/tokenize_parquet_qwen.py
```

作用：parquet 文本 → Qwen tokenizer 分词 → **HiLS uint32 mmap 文件**（`.parquet.data`），文档间以单个 EOS 分隔。

用法：

```bash
cd HiLS-Attention-main
python scripts/data/tokenize_parquet_qwen.py \
  --input-dir  /9950backfile/zhangem/data/fineweb_dataset \
  --output-dir /9950backfile/zhangem/data/fineweb_qwen3_data \
  --tokenizer  /9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master \
  --text-column text \
  --batch-size 500 \
  --workers 2 \
  --rayon-threads 4
```

特性：
- 输出格式：`flat little-endian uint32; one EOS after every document`
- 幂等：按 `conversion_manifest.json` + tokenizer SHA256 校验跳过已完成 shard
- 每个 shard 原子写入（`.partial` → rename），产出 `conversion.log` / `conversion_manifest.json`

### 5.3 数据消费

训练时 `--data.train_path` 指向 `fineweb_qwen3_data` 目录，加载器为 `data.lazy_dataset.LazyChunkedLoader`（HiLS 原生 uint32 mmap 格式）。

关键数据参数：

```bash
--data.train_path /9950backfile/zhangem/data/fineweb_qwen3_data
--data.data_type ruler_0.05          # 5% RULER 合成任务混合（长上下文评测格式）
--data.datasets_type olmo3
--data.max_seq_len 8192
--data.train_size 327680000
--data.num_workers 8
--data.sort_files true
```

---

## 6. 训练启动

### 6.1 当前实验启动脚本

`/9950backfile/zhangem/launch_asym_anneal.sh`（ft20000，1 GPU）。入口统一走 `train.sh`（封装 torchrun）：

```bash
bash train.sh tasks/pretrain_mass_with_ruler.py \
  configs/training_recipes/pretrain_345M_8K_30B.yaml \
  --model.config_path configs/mass_attention/config_mass_qwen3_0p6B_8KA2K_RoPE_m2_r8_asym0p5_anneal.json \
  --model.model_path .../Qwen--Qwen3-0.6B-MASS-m2/snapshots/master \
  --model.tokenizer_path .../Qwen--Qwen3-0.6B/snapshots/master \
  --model.attn_implementation sdpa \
  --data.train_path .../data/fineweb_qwen3_data \
  --data.data_type ruler_0.05 --data.datasets_type olmo3 \
  --data.max_seq_len 8192 --data.train_size 327680000 \
  --train.max_steps 20000 --train.micro_batch_size 4 \
  --train.global_batch_size 8 --train.enable_gradient_checkpointing true \
  --train.output_dir .../outputs/mass_qwen3_0p6b_anchor_only_m2_r8_asym0p5_anneal_ft20000 \
  --train.save_steps 500 \
  --train.freeze_pattern '^(?!.*mass_anchor_generator\.(?:anchor_embedding|offset_basis)$).*' \
  --train.lr 2e-5 --train.lr_min 2e-6 --train.lr_warmup_ratio 0.03 \
  --train.lr_decay_style cosine --train.max_grad_norm 1.0
```

### 6.2 必设环境变量

| 环境变量 | 值 | 作用 |
|---|---|---|
| `MASS_ANCHOR_REINIT` | `1` | model 构建后按 `mass_offset_init_std` 重初始化锚点（打破对称） |
| `MASS_SPECIALIZATION_LOSS_WEIGHT` | `1.0` | spec loss 权重（覆盖 config） |
| `MASS_LSE_ANNEAL_START` | `4.0` | 温度退火起点（空则不退火） |
| `MASS_LSE_ANNEAL_END` | `0.1` | 温度退火终点 |
| `MASS_TWO_STAGE_BACKWARD` | `1` | 两级反传（shared-memory partial 内核，推荐） |
| `MASS_TRAINABLE_ONLY_CHECKPOINT` | `1` | 仅保存可训练参数 checkpoint（可选） |
| `CUDA_VISIBLE_DEVICES` | 空闲卡 | 显存分配 |
| `PYTHONPATH` | 仓库根目录 | 必须 |

### 6.3 Resume（断点续训）

```bash
--train.load_checkpoint_path <output_dir>/checkpoints/global_step_XXXX
```

恢复内容：model（含锚点分化）、optimizer、lr_scheduler、dataloader、RNG、global_step。
**温度退火自动续接**（公式基于 `global_step/20000`）；freeze_pattern 在 load 后重新应用，必须与原来一致。

---

## 7. Anchor MoE 配置项详解

### 7.1 JSON 配置（`configs/mass_attention/config_mass_qwen3_0p6B_8KA2K_RoPE_m2_r8_asym0p5_anneal.json`）

#### 基础模型（继承 Qwen3-0.6B）

| 字段 | 值 | 说明 |
|---|---|---|
| `architectures` | `["MASSForCausalLM"]` | 入口为 MASS 包装模型 |
| `model_type` | `qwen_mass` | 注册到 `modeling_registry` |
| `hidden_size` / `intermediate_size` | 1024 / 3072 | 0.6B 规模 |
| `num_hidden_layers` | 28 | 全部层替换为 MASSAttention |
| `num_attention_heads` / `num_key_value_heads` | 16 / 8 | GQA |
| `head_dim` | 128 | |
| `vocab_size` | 151936 | Qwen3 tokenizer |
| `torch_dtype` | `bfloat16` | 主 dtype |
| `tie_word_embeddings` | false | 本项目 embedding 与 lm_head 独立 |

#### HiLS 结构参数

| 字段 | 值 | 说明 |
|---|---|---|
| `chunk_size` | 64 | landmark chunk 长度 |
| `hils_topk` | 32 | 每 query 检索的 chunk 数（须为 2 的幂，TileLang 约束） |
| `hils_sliding_window` | 512 | 滑动窗口（SWA）长度 |
| `max_window_layers` | 28 | 全层 SWA |
| `full_attn_interleave` | 4 | 每 4 层做全注意力 |
| `insert_landmarks` / `adjust_lmk_pos` | true / true | 插入并修正 landmark 位置 |
| `enable_external_lmk_embed` | true | 外部 landmark embedding |
| `enable_prior_query` | true | 前序 query 前缀（扩展有效序列） |
| `mask_lmk_token` | true | landmark token 不参与注意力 |
| `apply_hils_rope` | true | landmark 位置 RoPE |
| `use_sliding_window` / `sliding_window` | true / 512 | 与 SWA 一致 |
| `enable_softmax1` | false | 关闭 softmax1 加权 |

#### MASS 锚点配置（本项目核心）

| 字段 | 值 | 说明 |
|---|---|---|
| `mass_num_anchors` (M) | **2** | 锚点数量（m2）。内核上限 16（`M > 16` 报错） |
| `mass_offset_rank` | **8** | 偏移低秩维度 `offset_basis [Hq, D, rank]` |
| `mass_anchor_rms_norm` | true | 锚点 RMSNorm 归一化 |
| `mass_offset_init_std` | **2.0** | 锚偏移初始化 std。**0.02 对称→熵鞍点无梯度；2.0 打破对称激活分工** |
| `mass_score_reduction` | `"logsumexp"` | LSE 连续可微评分（`max` 为硬 argmax 无梯度） |
| `mass_lse_temperature` | 0.1 | 退火终点 tau（config 初值，运行期被退火钩子覆盖） |
| `mass_lse_anneal_start` | 4.0 | 退火起点（与 env 一致） |
| `mass_lse_anneal_end` | 0.1 | 退火终点 |
| `mass_normalize_lse` | true | 分数减 `tau*log(M)`，消除 M 依赖偏置（不影响 top-k 与梯度） |
| `mass_specialization_loss_weight` | 1.0 | spec loss 权重（env 优先） |
| `mass_anchor_reinit` | true | 等价 `MASS_ANCHOR_REINIT=1` |
| `mass_two_stage_backward` | true | 两级反传（shared-memory partial，生产推荐） |

### 7.2 可训练参数与冻结模式

```bash
FREEZE_PATTERN='^(?!.*mass_anchor_generator\.(?:anchor_embedding|offset_basis)$).*'
```

- **冻结**：backbone 全部参数（Q/K/V/O、MLP、RMSNorm、landmark 投影、embedding）
- **可训练**（每层 2 个张量，7 层 × 2 = 14 个）：
  - `mass_anchor_generator.anchor_embedding [M=2, rank=8]`
  - `mass_anchor_generator.offset_basis [16, 128, 8]`
- 生效时机：model 构建后 + **checkpoint load 后重新应用**（两处均打印冻结汇总，日志应显示 "14 trainable parameter tensors"）

### 7.3 spec loss 计算与日志

- 计算位置：`models/MASSIndex/mass_attention.py`（`specialization_loss` 公式）
- 梯度入口：`ops/mass_retrieval_tilelang_v2.py` 的 `_MassEnvelopeTopKV2Fn`（LSE softmax Jacobian + 熵梯度）
- 日志：每 10 步打印 `[mass-specialization][step N]`，含 `spec` / `load_min-max` / `mean_layer_cv` / `dead_anchors` / 逐层 `anchors=[a0,a1]`

```text
spec: 0.9660x1, load: 0.482-0.518, mean_layer_cv: 0.0149, dead_anchors(<1%)=0
  layer=07 anchors=[0.5150, 0.4850]    # 分化：差距 0.03
  layer=11 anchors=[0.4821, 0.5179]    # 分化：差距 0.036
```

### 7.4 相关代码位置

| 文件 | 内容 |
|---|---|
| `models/MASSIndex/configuration_mass.py` | `MASSConfig` / `OLMoMASSConfig` |
| `models/MASSIndex/mass_index.py` | 参考实现：锚生成、LSE 评分、top-k、spec 公式 |
| `models/MASSIndex/mass_attention.py` | `MASSAttention`（替换 HiLS 检索路径） |
| `models/MASSIndex/modeling_qwen_mass.py` | `MASSForCausalLM` / `MASSModel`（Qwen 包装、spec loss 聚合） |
| `models/MASSIndex/modeling_registry.py` | 模型注册 |
| `ops/mass_retrieval_tilelang_v2.py` | 高性能检索内核 + autograd Function |
| `ops/mass_retrieval_tilelang.py` | v1 内核（`_MassSummaryFn`） |
| `tasks/pretrain_with_ruler.py` | 训练主循环（重初始化钩子 line 235、退火钩子 line 431） |
| `configs/mass_attention/` | 各 MASS 配置（m2/m4/m8、tau 变体） |
| `scripts/mass/` | 各规模训练脚本（200M / 0.6B / 7B） |

---

## 8. 迁移步骤 Checklist

1. **同步代码**：`HiLS-Attention-main/`（含 scripts、configs、ops、models）
2. **同步模型**：`Qwen--Qwen3-0.6B-MASS-m2`、`Qwen--Qwen3-0.6B`（各 1.5G）
3. **同步数据**：`fineweb_dataset`（27G）+ `fineweb_qwen3_data`（37G，或迁移后重跑转化脚本）
4. **同步环境**：conda env `hils_smoke_py311`（或 export/import 重建），确认 CUDA 12.4 + tilelang `53ea96c6` commit
5. **验证 GPU 空闲**：`nvidia-smi`，选择 `CUDA_VISIBLE_DEVICES`
6. **权限**：确认新节点用户对目标路径有读写权（NFS 共享时注意属主/组与 UID 映射）
7. **启动**：修改 `launch_asym_anneal.sh` 中 GPU/路径 → 运行 → 日志确认：
   - `[mass-reinit] re-initialized 7 anchor generators (offset_init_std=2.0)`
   - `14 trainable parameter tensors`（freeze_pattern 生效）
   - `[mass-anneal] step N/N lse_temperature=...`（退火启动）
8. **断点续训**：加 `--train.load_checkpoint_path .../checkpoints/global_step_XXXX`，确认 spec/load 衔接而非回到初始值

---

## 9. 常见问题

| 问题 | 原因 / 解决 |
|---|---|
| `ModuleNotFoundError: models` | torchrun 子进程缺 `PYTHONPATH`，脚本中显式 export |
| `kernel fwd input Q dtype expected bfloat16, got float32` | freeze_pattern 被 shell 单引号破坏 → 参数未冻结 → VeOmni 转 FP32；检查日志冻结汇总 |
| `spec` 恒 0.9979 / load 均匀 | 对称初始化（offset_std 太小）无梯度信号，需 2.0 级非对称 + 高温退火 |
| resume 后 spec 回到初始值 | `load_checkpoint_path` 应指向 `checkpoints/global_step_XXXX/` 目录而非内部文件 |
| 速度退化（如 41s→757s/it） | 同节点 GPU 被其他用户抢占，换卡或等释放 |
| `TileLang` 内核编译缓慢 | 首次编译按 shape 缓存（`_KERNEL_CACHE`），之后快；保持 shape 不变 |
