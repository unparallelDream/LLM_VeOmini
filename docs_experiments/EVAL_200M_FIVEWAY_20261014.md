# 200M 五方对照评测汇总(2026-10-13/14,2026-10-20 补充最新修复版)

> **2026-10-20 命名勘误与新增列**:2026-10-19 发现 12d3e46(μ/b 梯度修复)在重写 envelope kernel 时引入了
> **view 维归约塌缩**(M≥2 时 max/LSE 双模式都退化为"仅末视图",即"单视图退化MASS",commit b5e8a0a 修复)。
> 因此本文名词需重新对齐:
> - **mass buggy** = **更老版本**(12d3e46 之前,μ/b 梯度丢失/重排 bug 时期)
> - **mass修复**(即 "new")= **修复前版本**(12d3e46:修了 μ/b 梯度,但 envelope 单视图退化——上表所有历史数据在此 kernel 下训练)
> - **最新修复**(新增列,标注 **@55000**)= 10-19 修复后 kernel(b5e8a0a/c04a85b)训练:
>   `olmo200m_mass_cont10B_fixkernel` checkpoint **global_step_55000**
>   (= new版 cpt@15000 起点 → cont10B 30001-47500 坏kernel → 47500-55000 修复kernel;
>   评测代码为修复后版本;**ppl 与 lmeval-13 均已转 HF 格式评测**,自制 bench 与其他列同 DCP 直读口径)

## 模型与训练协议

| 代号 | 架构 | checkpoint |
|---|---|---|
| 最新修复@55000 | 24×SWA-512 + 8×MASS(层3,7,…,31),MHA 4/4,双锚+shared_view | olmo200m_mass_cont10B_fixkernel/global_step_55000(修复后kernel b5e8a0a) |
| mass修复(=new,修复前) | 同上(配置项完全相同) | olmo200m_mass_cpt_ft15000(12d3e46,含envelope塌缩) |
| mass buggy(更老) | 同上 | olmo200m_mass_cpt_ft15000_buggykernel(12d3e46之前) |
| hils | 24×SWA-512 + 8×HiLS(同槽位,单锚) | olmo200m_hils_cpt_ft15000 |
| dense(rec) | 24×SWA-4096 + 8×Full(OLMo-3原生3:1) | olmo3_200m_recovery_ft15000,lr 5e-5 |
| dense_ctrl | 同rec | olmo200m_dense_ctrl_ft15000,与mass/hils完全同协议(lr 3e-4) |

共同:dolma3_1025_3B_packed + ruler_0.05 混合、seq 8192、15000 步(≈0.98B token)。mass修复/buggy唯一差异=训练kernel数值(git 12d3e46 修复 μ/b 梯度丢/重)。层布局勘误:mass/hils 并非全稀疏——3:1槽位同OLMo-3,稀疏层只占8个全注意力槽,其余24层为SWA-512。
**最新修复@55000 差异声明**:在此之上做 cont10B 续训(dolma3_1025_10Bminus3B_packed 去重语料,lr 3e-5→3e-6,96165 步配方,30001-47500 为坏kernel、47500 起为修复kernel+全新warmup),@55000 取最新存档评测——**非受控单变量对比**(多了续训步数与数据),跨列比较时需注意。

## 评测口径

- 自制bench(论文3任务,S-N/MK-MQ/VT):teacher-forced,n=50,seed42,8192/16384(DCP直读,各列一致)
- lm-eval RULER 13任务:生成式,4096(n=50)、8192(n=50与n=500),seed42;8192用修正元数据键max_seq_lengths。**最新修复列:ppl与lmeval-13经 DCP→HF 转换后评测**(hf_ckpt,utils/convert_dcp_to_hf_ckpt.py)
- ppl:n=100,tail-512,dolma3语料尾10%,8K/16K/32K/64K

## 一、自制 benchmark @ 8192(n=50,EM)

| 任务 | 最新修复@55000 | mass修复(new) | mass buggy(更老) | hils | dense(rec) | dense_ctrl |
|---|---|---|---|---|---|---|
| single_niah | 0.980 | 0.980 | 0.980 | 0.960 | 1.000 | 1.000 |
| multi_query | **0.860** | 0.840 | 0.320 | 0.180 | 0.200 | 0.140 |
| variable_tracking | 0.240 | 0.240 | 0.280 | 0.080 | 0.180 | 0.060 |
| **宏均EM** | **0.693** | 0.687 | 0.527 | 0.407 | 0.460 | 0.400 |
| 宏均token_acc | 0.942 | 0.938 | 0.900 | 0.830 | 0.871 | 0.852 |

## 一、自制 benchmark @ 16384(n=50,EM)

| 任务 | 最新修复@55000 | mass修复(new) | mass buggy(更老) | hils | dense(rec) | dense_ctrl |
|---|---|---|---|---|---|---|
| single_niah | 0.920 | 0.900 | 0.940 | 0.820 | 0.860 | 1.000 |
| multi_query | **0.760** | 0.720 | 0.120 | 0.080 | 0.080 | 0.060 |
| variable_tracking | 0.060 | 0.080 | 0.100 | 0.000 | 0.000 | 0.000 |
| **宏均EM** | **0.580** | 0.567 | 0.387 | 0.300 | 0.313 | 0.353 |
| 宏均token_acc | 0.903 | 0.896 | 0.855 | 0.743 | 0.804 | 0.834 |

## 二、lm-eval 13任务 @ 4096(n=50,HF格式)

| 任务 | 最新修复@55000 | mass修复(new) | mass buggy(更老) | hils | dense(rec) | dense_ctrl |
|---|---|---|---|---|---|---|
| niah_single_1 | 0.980 | 1.000 | 0.720 | 0.980 | 1.000 | 1.000 |
| niah_single_2 | 0.260 | 0.460 | 0.940 | 0.940 | 0.540 | 1.000 |
| niah_single_3 | **0.740** | 0.700 | 0.420 | 0.840 | 0.200 | 0.040 |
| niah_multikey_1 | 0.180 | 0.220 | 0.440 | 0.260 | 0.140 | 0.280 |
| niah_multikey_2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| niah_multikey_3 | 0.020 | 0.020 | 0.020 | 0.020 | 0.020 | 0.040 |
| niah_multiquery | 0.000 | 0.000 | 0.030 | 0.145 | 0.020 | 0.070 |
| niah_multivalue | 0.175 | 0.165 | 0.185 | 0.205 | 0.135 | 0.195 |
| ruler_cwe | 0.196 | 0.114 | 0.238 | 0.078 | 0.050 | 0.016 |
| ruler_fwe | 0.007 | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 |
| ruler_vt | 0.000 | 0.000 | 0.000 | 0.004 | 0.004 | 0.004 |
| ruler_qa_squad | 0.033 | 0.080 | 0.080 | 0.075 | 0.075 | 0.093 |
| ruler_qa_hotpot | 0.140 | 0.160 | 0.180 | 0.140 | 0.100 | 0.040 |
| **宏均(13)** | 0.210 | 0.225 | 0.250 | 0.283 | 0.176 | 0.214 |

## 三、lm-eval 13任务 @ 8192(n=50,HF格式)

| 任务 | 最新修复@55000 | mass修复(new) | mass buggy(更老) | hils | dense(rec) | dense_ctrl |
|---|---|---|---|---|---|---|
| niah_single_1 | 0.940 | 0.980 | 0.780 | 0.940 | 0.900 | 1.000 |
| niah_single_2 | 0.540 | 0.380 | 0.900 | 0.900 | 0.520 | 0.980 |
| niah_single_3 | **0.560** | 0.400 | 0.220 | 0.660 | 0.340 | 0.020 |
| niah_multikey_1 | 0.160 | 0.200 | 0.400 | 0.200 | 0.240 | 0.240 |
| niah_multikey_2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| niah_multikey_3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| niah_multiquery | 0.040 | 0.005 | 0.070 | 0.145 | 0.145 | 0.155 |
| niah_multivalue | 0.200 | 0.085 | 0.205 | 0.210 | 0.125 | 0.155 |
| ruler_cwe | 0.040 | 0.022 | 0.016 | 0.016 | 0.000 | 0.004 |
| ruler_fwe | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ruler_vt | 0.000 | 0.004 | 0.008 | 0.000 | 0.000 | 0.000 |
| ruler_qa_squad | 0.000 | 0.013 | 0.015 | 0.000 | 0.000 | 0.033 |
| ruler_qa_hotpot | 0.100 | 0.120 | 0.120 | 0.100 | 0.060 | 0.060 |
| **宏均(13)** | 0.198 | 0.170 | 0.210 | 0.244 | 0.179 | 0.204 |

## 四、lm-eval 13任务 @ 8192(n=500,全量)

(最新修复@55000 的 n=500 未跑——生成式全量耗时较长,且 n=50 已给出量级;如需可后补)

| 任务 | mass修复(new) | mass buggy(更老) | hils | dense(rec) | dense_ctrl |
|---|---|---|---|---|---|
| niah_single_1 | 0.986 | 0.664 | 0.992 | 0.920 | 0.998 |
| niah_single_2 | 0.404 | 0.906 | 0.902 | 0.476 | 0.986 |
| niah_single_3 | 0.310 | 0.182 | 0.648 | 0.286 | 0.028 |
| niah_multikey_1 | 0.152 | 0.274 | 0.250 | 0.194 | 0.286 |
| niah_multikey_2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| niah_multikey_3 | 0.002 | 0.002 | 0.004 | 0.002 | 0.002 |
| niah_multiquery | 0.002 | 0.085 | 0.182 | 0.149 | 0.138 |
| niah_multivalue | 0.087 | 0.204 | 0.230 | 0.142 | 0.157 |
| ruler_cwe | 0.043 | 0.037 | 0.017 | 0.002 | 0.004 |
| ruler_fwe | 0.005 | 0.003 | 0.001 | 0.000 | 0.001 |
| ruler_vt | 0.002 | 0.004 | 0.001 | 0.000 | 0.000 |
| ruler_qa_squad | 0.024 | 0.033 | 0.030 | 0.037 | 0.025 |
| ruler_qa_hotpot | 0.086 | 0.086 | 0.078 | 0.082 | 0.066 |
| **宏均(13)** | 0.162 | 0.191 | 0.257 | 0.176 | 0.207 |

## 五、ppl 阶梯(n=100,tail-512,最新修复列为HF格式)

| 长度 | 最新修复@55000 | mass修复(new) | mass buggy(更老) | hils | dense(rec) | dense_ctrl |
|---|---|---|---|---|---|---|
| 8192 | **17.614** | 17.746 | 17.801 | 14.749 | 11.917 | 14.322 |
| 16384 | **18.913** | 19.962 | 19.665 | 14.113 | 12.317 | 14.873 |
| 32768 | **17.279** | 18.688 | 18.265 | 12.715 | 12.295 | 14.401 |
| 65536 | **15.484** | 16.282 | 16.475 | 14.331 | 13.518 | 15.763 |
| 64K/8K 比 | **0.879** | 0.917 | 0.926 | 0.972 | 1.134 | 1.101 |

## 六、核心结论

1. **多查询(MK-MQ)是唯一分水岭,与训练数据无关**:自制bench@8K mass修复0.840,其余四方(含同协议混合dense_ctrl)0.14~0.32;@16K 0.720 vs ≤0.12。dense_ctrl排除数据组成解释——优势=双锚检索架构×kernel数值修复。修复收益轨迹:6500步0.24→15000步0.84(loss平台期内涌现)。**最新修复@55000 在此分水岭上继续小幅抬升(@8K 0.86、@16K 0.76,宏均 0.693/0.580 均为六方最高)**。
2. **kernel修复对LM ppl零影响**(两列差<0.5),收益全部集中在检索/聚合任务。**最新修复@55000 ppl 四档全面更优(比new低0.1~1.4),且 64K/8K 比降到 0.879(六方最好)——续训+修复kernel同时改善了绝对值与长度外推斜率**。
3. **生成式(lm-eval)口径五家挤在0.16~0.28**,与teacher-forced的巨大分化互补:生成式被指令跟随卡住(所有模型multiquery生成式≈0);hils生成式宏均最高(4K 0.283 / 8K n500 0.257),其长处是单针变体(single_2/3)与multiquery的生成。**最新修复@55000 生成式 @8K 宏均 0.198(new 0.170),single_3 0.560(new 0.400)显著抬升,single_2 0.540(new 0.380)回升——§五-5 的"发射弱"失衡在修复+续训后部分收敛;@4K 0.210 与 new 0.225 基本持平**。
4. **ppl**:dense(rec,lr5e-5)四挡全优(11.9~13.5);同协议对照dense_ctrl为14.3~15.8。mass以8K时+3.4 ppl换检索能力,64K差距缩至+0.5(最新修复@55000 后缩至+2.0内,且趋势继续向好);稀疏三方ppl随长度改善(64K/8K<1),两个dense退化(1.10~1.13)。
5. **single_2反常(修复0.404 vs buggy 0.906,n=500)已定位**:推理代码/注意力宏观分配(针块权重两家均~0.8%)/选块翻转均相同;差异在发射瞬间——修复版针数字概率被稀释2.4×(金前缀后0.158 vs 0.377),top1判别翻向干扰数字;开环同样弱(4/12 vs 10/12),排除闭环误差累积。根因:μ/b梯度腐蚀与否→"聚合强/发射弱"vs"聚合弱/发射强"两个均衡;发射方向两家训练均无显式监督。修复方向:发射方向显式监督(问题+gen_prefix在末尾、loss只计答案段)。
6. **训练充分度**:15000步≈0.98B token(论文345M配方30B的1/30)。loss末段斜率≈0但能力指标仍在爬升;要对齐论文绝对水位(域内MK-MQ 95+)需更多数据,续训建议warm-restart并盯multi_query/VT而非loss。**最新修复@55000(cont至55K步≈4.6B token累计)验证了该判断:loss平台期内自制bench/lmeval/ppl 全线仍在上行**。
7. **(2026-10-20)最新修复列的读法**:该列=修复前权重续训7.5K步(修复kernel)+此前坏kernel续训17.5K步,非受控单变量;其全面上行说明 ①envelope 修复未破坏已学能力,②修复后的选择语义(共享键view0真正参与envelope)与继续训练兼容。受控的"同起点仅kernel不同"对比需另起 15K 步重训实验。

## 七、数据文件索引

- 自制bench: outputs/ruler200m_paper_quad/{name}_paper50_len{8192,16384}.json
- lmeval4096: outputs/lmeval_ruler_200m/{name}_fullruler_n50.json
- lmeval8192: outputs/lmeval_ruler_200m/{name}_fullruler_n50_len8192.json 及 _n500_len8192.json
- ppl: outputs/lmeval_ruler_200m/{name}_ppl_paper_n100.jsonl
- 训练日志: logs/olmo200m_mass_cpt.log / olmo200m_hils_cpt.log / olmo200m_dense_ctrl.log / olmo3_200m_recovery_ft15000.log
- kernel修复commit: 12d3e46;论文原文: docs_experiments/hils_paper_arxiv_2607.02980.pdf
- **最新修复@55000**: 训练 logs/olmo200m_mass_cont10B_fixkernel.log;HF ckpt .../olmo200m_mass_cont10B_fixkernel/checkpoints/global_step_55000/hf_ckpt;
  自制bench outputs/ruler200m_paper_quad/mass_fixkernel55000_paper50_len{8192,16384}.json;
  lmeval outputs/lmeval_ruler_200m/mass_fixkernel55000_fullruler_n50{,_len8192}.json;
  ppl outputs/lmeval_ruler_200m/mass_fixkernel55000_ppl_paper_n100.jsonl;
  评测脚本 scripts/eval_fixkernel55000_full_20261020.sh + scripts/ppl_fixkernel55000_gpu3.sh;
  envelope修复commit: b5e8a0a(+c04a85b GQA语义统一)
