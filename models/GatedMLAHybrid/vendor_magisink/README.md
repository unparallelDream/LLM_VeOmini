# Vendored MagiAttention sink 组合层（仅供基准研究，勿用于模型）

来源：SandAI-org/MagiAttention（Apache-2.0），vendor 了 fa2_func_with_sink 及其纯 Python 依赖。

状态（2026-10-24，torch 2.6 + 官方 flash_attn 2.8.3）：
- 可导入、可运行、计时有效（组合层开销 = +44%~133% vs 官方 FA 无 sink）；
- **输出正确性不成立**：β=0 时与官方 FA 输出不一致（小例 fp32 手算对照确认）——
  组合层内部约定耦合其 fork 的 FA 2.8.4（本机编译失败，见 /tmp/build_magi_fa.log），
  对 vanilla FA 的 LSE 返回假设不成立；
- 本仓库模型一律使用 models/GatedMLAHybrid 内自研的 virtual sink
  （SDPA 虚拟列实现，fp32 手算对照 1.9e-08，见 GQAAttention._virtual_sink_attention）。

改动记录：*_compiled 包装换 eager（dynamo 在 vendored 上下文追踪失败）；
utils/dtype.py 的 FP4 dtype 加 getattr 守卫（torch 2.6 无该 dtype）。
