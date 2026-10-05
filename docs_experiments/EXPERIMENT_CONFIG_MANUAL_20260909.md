# MASS / HiLS 实验全量配置手册

**日期**:2026-09-09(覆盖 2026-08-29 至今全部实验)
**配套文档**:`TRAINING_AND_EVAL_REPORT_20260905.md`(结果与分析,本文只写配置)
**机器**:hy-a800-127-56,A800-SXM4-80GB ×8(多用户共享,全程有外部抢占)
**环境**:`/9950backfile/zhangem/miniconda3/envs/hils_smoke_py311`(python 3.11,torch≥2.4 带 DCP,无 scipy/peft——相关功能均自实现)
**代码仓**:`/9950backfile/zhangem/HiLS-Attention-main`(HiLS-Attention,arXiv 2607.02980,MASS 为其扩展)

---

## 一、硬件与环境约定(所有 run 共用)

| 项 | 值 |
|---|---|
| GPU 分配 | 单卡 DDP(world=1);历史占用:GPU0/2/3/6/7 |
| 共享环境变量 | `TMPDIR=/dev/shm/zhangem_tmp`(根分区满)、`PYTHONPYCACHEPREFIX=.../tmp/pycache`、`TOKENIZERS_PARALLELISM=false`、`TORCH_NCCL_AVOID_RECORD_STREAMS=1`、`PATH+=/usr/local/cuda-12.4/bin` |
| 训练入口 | `tasks/pretrain_mass_with_ruler.py`(注册 qwen_mass/qwen_hils + 可选 monkey-patch)→ `pretrain_with_ruler.main()` |
| 混合精度 | bf16 模型 + 可训练参数 fp32 主权重(非 fsdp2 路径);`MASS_PURE_BF16=1` 可跳过主权重转换(全参从头训练必用,TileLang kernel 只收 bf16) |
| 启动方式 | `torch.distributed.run --nproc-per-node=1 --standalone`,nohup 后台,PID 写 `*.pid` 文件 |

## 二、数据

| 项 | 值 |
|---|---|
| 来源 | FineWeb(已 tokenize 为 Qwen3 词表),37GB / 14 个 `.parquet.data` 分片(每片 2.8GB) |
| 数据池总量 | **990,206,632 tokens**(≈9.9 亿,~12.1 万条 8K 序列),路径 `data/fineweb_qwen3_data` |
| 每轮采样 | `--data.train_size 327680000`(**3.277 亿** tokens = 5000 步/遍) |
| 序列长度 | **8192**(olmo3 打包);landmark 由模型 forward 自动插入(63+1/块),注意力内实际 ~8321 位置 |
| 数据混合 | `--data.data_type ruler_0.05`:**~5% RULER 式合成 + ~95% fineweb**;`--data.sort_files true`;num_workers 8 |
| 每 run token 消耗 | 20,000 步 × 65,536 = **13.11 亿**(采样池过 4 遍 ≈ 全池 1.3 遍);15,000 步 = 9.83 亿 |

## 三、模型结构

| 项 | Qwen3-0.6B 系 | Qwen3-90M 系 | 100M/18L(基准测试用) |
|---|---|---|---|
| 来源 | 官方 Qwen3-0.6B(~36T tokens 预训练) | 0.6B 宽度裁剪 + recovery 0.98B tokens | 随机初始化(seed 42) |
| 层数 / hidden / inter | 28 / 1024 / 3072 | 28 / 320 / 1024 | 18 / 320 / **2432** |
| 注意力头(q/kv) | 16 / 8,head_dim 128,GQA=2 | 4 / 2,head_dim 128,GQA=2 | 4 / 2 |
| 唯一参数量 | ~596M(存储 751.8M,untied lm_head) | **89,932,608**(tie 口径;存储 138.6M) | **99.54M**(非嵌入 50.91M + 嵌入/head 48.63M) |
| 词表 | 151,936;输入侧嵌入 **151,968** 行(给 lmk_id 留位) | 同左;HiLS 版 lm_head 裁 151,936、tie=false | 同 0.6B 口径 |
| 位置编码 | RoPE theta=1e6,max_pos 40,960 | 同左 | 同左 |
| 检索层位置 | **[3,7,11,15,19,23,27] 共 7 层**(每 4 层 1 个),其余 21 层 SWA-512 | 同左 | 同左 |
| tie_word_embeddings | false(MASS 版) | MASS 版 true / HiLS 版 false | true(存储克隆 lm_head) |

## 四、稀疏注意力预算与 MASS/HiLS 机制(训练评测一致)

| 项 | 值 |
|---|---|
| chunk_size | 64(63 真实 + 1 块尾 landmark,lmk_id=151936,嵌入替换为 lmk_embed) |
| 局部滑窗(hils_sliding_window) | 512(仅检索层;非检索层 sliding_window=512 全程不变) |
| 远程 top-k(hils_topk) | 32 块 × 64 = 2048 tokens |
| **KV budget/query** | **2560**(512+2048);4K 输入下压缩率 ~61.5% |
| 值键掩码 | mask_lmk_token=true:块尾 landmark 不进 V、不进打分 |
| MASS anchor | M=2/层、offset rank 8;`anchor_embedding (M,8)` + `offset_basis (q头数,128,8)`;**0.6B:114,800 参数 / 90M:28,784 参数**(均 14 张量) |
| MASS 打分 | chunk 侧摘要(mu,bias)+query 侧门控双线性分解,O(L·d) 级;score_reduction=logsumexp;normalize_lse |
| LSE 温度退火 | 4.0 → 0.1 全程线性(env `MASS_LSE_ANNEAL_START/END`) |
| spec_loss | anchor 分化损失,权重 1.0(env `MASS_SPECIALIZATION_LOSS_WEIGHT`);two-stage backward(env) |
| 初始化 | anchor 全新随机 + `MASS_ANCHOR_REINIT=1` 重置(破坏对称鞍点);lmk_embed 初始化 = EOS 词向量 |

## 五、逐 run 完整配置

### 5.0 预处理阶段(非训练)

| 阶段 | 工具 | 产物 |
|---|---|---|
| 90M 宽度裁剪 | `utils/prune_qwen3_width.py`(importance 策略,校准 12×8192 fineweb 序列) | `Qwen--Qwen3-0.6B-pruned-90M`(裁后 CE 9.93 vs 教师 3.06) |
| MASS 移植 ×2 | `utils/convert_qwen3_to_mass.py`(严格逐张量复制+派生,anchor 随机初始化 seed 42) | `Qwen--Qwen3-0.6B-MASS-m2`、`Qwen--Qwen3-90M-MASS-m2-recovered` |
| HiLS 基座 ×2 | 自写脚本(剥离 14 anchor 张量 + qwen_hils config) | `Qwen--Qwen3-{0.6B,90M}-HiLS{,-recovered}`(90M 侧 embed 151968/lm_head 151936) |
| 100M/18L 随机初始化 | 自写脚本(MASSForCausalLM→bf16→safetensors,lm_head 克隆解 tie) | `Qwen--Qwen3-100M-MASS-m2-init` |

### 5.1 训练 run 总表

所有 run 共用:数据(§二)、batch(micro 4 × accum 2 = global 8 = 65,536 tok/步)、grad ckpt、clip 1.0、cosine、warmup 3%、no_decay(norm/bias/embed)、include_frozen_params_in_optimizer=false、save_steps 500、DDP、bf16 混合精度。

| # | run | 起止 | 初始化 | 可训练参数(占比) | lr→lr_min | 步数 | tokens | 事件 |
|---|---|---|---|---|---|---|---|---|
| 1 | 90M recovery(全参,dense) | 8/29–9/3 | pruned-90M | **89.93M(100%)** | 3e-4→3e-5 | 15,000 | 9.83 亿 | ✅ 9/3 09:42,末段 loss 3.24 |
| 2 | 0.6B anchor-FT(冻结) | 9/2–9/5 | 0.6B-MASS-m2 | **114,800(0.019%)** | 2e-5→2e-6 | 20,000 | 13.11 亿 | ✅ 9/5,lm 5.12,spec 0.152;9/2 崩溃一次 |
| 3 | 90M anchor-FT(冻结) | 9/3–9/6 | 90M-MASS-m2-recovered | **28,784(0.032%)** | 2e-5→2e-6 | 20,000 | 13.11 亿 | ✅ 9/6,lm 3.514,spec 0.118;9/5 崩溃于 12609 恢复于 12500 |
| 4 | 0.6B hils 对照(冻结) | 9/5–9/7 | 0.6B-HiLS | **1,024(lmk_embed)** | 2e-5→2e-6 | 20,000 | 13.11 亿 | ✅ 9/7,loss 5.449 |
| 5 | 90M hils 对照(冻结) | 9/5–9/6 | 90M-HiLS-recovered | **320(lmk_embed)** | 2e-5→2e-6 | 20,000 | 13.11 亿 | ✅ 9/6,loss 3.490 |
| 6 | 0.6B anchor+LoRA | 9/6–(9/13) | 0.6B-MASS-m2 | **114,800 + 4,587,520 LoRA = 4,702,320(0.79%)** | 2e-5→2e-6 | 20,000 | 13.11 亿 | 🔄 48%;lm 2.87(已低于 dense 参照) |
| 7 | 90M anchor+LoRA | 9/6–9/9 | 90M-MASS-m2-recovered | **28,784 + 1,261,568 = 1,290,352(1.44%)** | 2e-5→2e-6 | 20,000 | 13.11 亿 | ✅ 9/9,lm 3.374;9/7 崩溃于 12619 恢复于 12500;**评测 macro4 0.3436 超 full** |
| 8 | 90M hils+LoRA(归因对照) | 9/9–(9/10) | 90M-HiLS-recovered | **320 + 1,261,568 = 1,261,888(1.40%)** | 2e-5→2e-6 | 20,000 | 13.11 亿 | 🔄 7% |

注:冻结类 run 的 freeze_pattern:anchor 版 `^(?!.*mass_anchor_generator\.(?:anchor_embedding|offset_basis)$).*`;hils 版 `^(?!.*model\.lmk_embed$).*`;LoRA 版额外豁免 `\.lora_[AB]\.weight`。

### 5.2 LoRA 机制细节(utils/lora_adapt.py + MASS_LORA_* 环境变量)

| 项 | 值 |
|---|---|
| 结构 | `out = base(x) + (α/r)·B(A(x))`;A=Kaiming,B=零初始化(**注入后 step-0 位级等价**,已验证) |
| 超参 | rank 16,alpha 32(scale 2.0) |
| 目标 | 全部 28 层的 q/k/v/o(`MASS_LORA_SCOPE=all`);112 个 Linear |
| 参数量 | 0.6B:4,587,520;90M:1,261,568 |
| 设备/属性 | 跟随 base 权重设备创建(fp32);代理 `weight/bias/in_features/out_features` 属性(模型代码直接访问 `o_proj.weight.dtype`) |
| 命名 | `<orig>.lora_A/B.weight`,freeze_pattern 豁免,trainable-only checkpoint 自动收录 |
| 评测加载 | `mass_lora` 变体注入同款 LoRA 后 `load_state_dict(strict=False)`,校验 238(+tie 重复 lm_head 剔除) |

### 5.3 Checkpoint 体系

| 项 | 值 |
|---|---|
| 格式 | veomni DCP(`__0_*.distcp` + `extra_state`/目录),每 500 步 |
| 两种模式 | 全量(326 张量,~1.2–2.4GB/存档)与 **trainable-only**(`MASS_TRAINABLE_ONLY_CHECKPOINT=1`:仅 requires_grad 张量,KB–MB 级,含优化器/数据状态,恢复时 backbone 从 model_path 重载) |
| 提取工具 | `extract_anchor_only_ckpt.py`(全量 DCP → 14 anchor 张量 anchor-only DCP,往返校验) |
| 崩溃恢复 | 同命令 + `--train.load_checkpoint_path <最近存档>`;三次实战(#2/#3/#7),损失 ≤500 步 |

## 六、评测协议(eval_ruler_qwen_mass_bsa_4k.py)

### 6.1 共用条件

| 项 | 值 |
|---|---|
| 输入 | 原始 4096 tokens → landmark 扩展 4161(63+1/块);position_ids 带补偿;fill_id=模型 lmk_id(**曾为 100278'拼',已修复**) |
| 预算 | 等预算 2560(§四);eval 内置三重校验(样本哈希/预算/输入布局全变体相等) |
| 样本/种子/精度 | 每任务 n=50,seed 42,bf16,sm_scale=1/√128 |
| 指标 | token accuracy(主)、exact match(辅);吞吐按扩展长度/稳态中位延迟 |
| 任务 | 0 single_niah(7 位魔法数字单针)/ 1 multi_query(6 变量、2 查询)/ 2 variable_tracking(6 变量、2 跳链)/ 3 FWE(Zipf α=2.0、2000 词表、答 top-3 高频词;token acc 有区分度,EM 全 0 不用) |

### 6.2 变体定义(9 个)

| 变体 | 块打分 | 加载 | 可学习参数 |
|---|---|---|---|
| mass | 2-anchor 低秩摘要(训练后) | 基座 + anchor-only ckpt | 114,800/28,784 |
| fixed0 | anchor0 复制消融 | 同上(评测时塌缩) | 0 |
| dsa | 逐 token `log Σ exp(q·k)`(NSA 式,O(L²)) | 仅基座 | 0 |
| bsa | 块均值 `q·mean(k_c)`(O(L·N),自实现,掩码/eligibility 与 dsa 严格对齐) | 仅基座 | 0 |
| hils | HiLS landmark 键打分(原生,零样本) | 基座(HiLSAttention 换装,权重逐张量校验) | 0 |
| hils_t | 同上 + 训练后 lmk_embed | 基座 + lmk ckpt(校验恰含 `model.lmk_embed`) | 320/1024 |
| full | 稠密因果 SDPA(块尾 landmark 键掩掉;预算上报 4160≈100%) | 仅基座(DenseRetrievalAttention 换装) | 0 |
| mass_lora | mass + 同款 LoRA 注入 | 基座 + 238 张量 adapter | 1.29M/4.70M |
| (hils_lora | 待 #8 完成) | | 1.26M |

### 6.3 专项评测

| 评测 | 配置 | 产物 |
|---|---|---|
| 预算扫描(0.6B) | 6 档(window,topk)=(512,4)(512,2)(256,2)(128,1)(64,1)(64,0),实际 18.5/15.4/9.2/4.6/3.1/1.5%;topk 受 MASS kernel 约束取 2 的幂;b1 档 mass/fixed0 不可跑(topk=0 TVM 编译错) | `budget_sweep_0p6b/` + sweep_summary.json |
| 训练轨迹 | 匹配步数 checkpoint 评测(90M 7 点 ×2 / 0.6B 5 点 ×2) | `eval_trajectory_*` |
| 完整四任务(90M) | 8 变体 × 任务 0–3 | §5.12 of 报告 |
| SM/吞吐基准(100M/18L) | 随机初始化从零训练 30 步;mb 4/8/16/24/32;双卡 A/B/C 相位 | bench_*.log |

## 七、基建补丁清单(全部 env 门控或向后兼容,默认行为不变)

| 位置 | 改动 | 门控 |
|---|---|---|
| `tasks/pretrain_with_ruler.py` | fp32 主权重转换旁路;DDP 后属性统一 `raw_model = getattr(model,"module",model)`(anchor 重置与 LSE 退火读 config 处) | `MASS_PURE_BF16=1` / 无害重构 |
| `tasks/pretrain_mass_with_ruler.py` | LoRA 注入 hook(包 build_foundation_model) | `MASS_LORA_ADAPT=1`(+RANK/ALPHA/TARGETS/SCOPE) |
| `utils/lora_adapt.py`(新) | LoRALinear + inject_lora(零初始化等价/设备跟随/属性代理,单元验证) | — |
| `models/NaiveBSA/*` | `score_mode="exact|mean"`(mean=块均值打分,双实现掩码对齐) | config `bsa_score_mode` |
| `eval/eval_ruler_qwen_mass_bsa_4k.py` | dsa/bsa/hils/hils_t/full/mass_lora 变体;fill_id 修复;task 3 映射 | — |
| `data/data_transform.py` | FWE 的 zeta 纯 python 兜底 | ImportError 自动 |
| `utils/trainable_checkpointer.py`(原有) | trainable-only 原子存档 | `MASS_TRAINABLE_ONLY_CHECKPOINT=1`(仓库原生) |

## 八、脚本索引

| 用途 | 脚本 |
|---|---|
| 冻结 anchor 训练 | `launch_mass90m_anchor_ft.sh`(+`_resume12500.sh`)、0.6B 由 `launch_ft20000.sh`(先前存在) |
| hils 对照 | `launch_hils_lmkembed_ft.sh 0p6b|90m` |
| LoRA 臂 | `launch_anchor_lora_kv_ft.sh 0p6b|90m`(+`_resume12500.sh`)、`launch_hils_lora_90m_ft.sh` |
| 全参 A/B(备而未发) | `launch_0p6b_fullparam_ab.sh mass|hils` |
| 评测 | `launch_eval_0p6b_anneal_ft20000_ruler4k.sh`、`launch_eval_90m_7variants.sh`、`launch_eval_0p6b_budget_sweep.sh`、`launch_eval_trajectory_mass_vs_hils.sh` |
| 基准 | `bench_100m_18l_sm.sh`、`bench_2gpu.sh`、`bench_bigbatch.sh` |
| 工具 | `extract_anchor_only_ckpt.py` |

## 九、勘误(相对此前对话/文档)

1. **0.6B anchor 参数量是 114,800**(offset_basis 随 16 q 头放大),此前部分表述误用 90M 的 28,784;§5.1 表已按模型分别标注。
2. `extra_state` 是目录不是文件(曾致守卫误判)。
3. eval 的 landmark fill 曾为 100278('拼')而非 lmk_id——修复后 dsa/bsa 逐位不变、mass/fixed0 噪声级移动。
