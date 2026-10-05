"""单卡 torch.profiler:复现 v8 训练配置的 MASS 204M 前向+反向 kernel 热点分析"""
import os
import sys
import json

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# 与训练一致的 veomni ops 环境
os.environ.setdefault("USE_GROUP_GEMM", "1")
os.environ.setdefault("USE_LIGER_KERNEL", "1")

import torch
import torch.profiler as profiler

BASE = "/9950backfile/zhangem/HiLS-Attention-main"
sys.path.insert(0, BASE)

import veomni  # noqa: F401  触发 ops patch(与训练一致)
import models.MASSIndex.modeling_registry  # noqa: F401  注册 qwen_mass
from models.MASSIndex.configuration_mass import MASSConfig
from models.MASSIndex.modeling_qwen_mass import MASSForCausalLM

cfg_dict = json.load(
    open(f"{BASE}/configs/mass_attention/config_mass_attn_200M_m4_r8.json")
)
cfg = MASSConfig(**{k: v for k, v in cfg_dict.items() if k != "architectures"})

torch.manual_seed(0)
model = MASSForCausalLM(cfg).to(torch.bfloat16).cuda()
model.train()

B, L = 4, 8192  # 训练 micro_batch=4, max_seq_len=8192(模型内 landmark 扩展 -> q_len 8322)
input_ids = torch.randint(0, cfg.vocab_size, (B, L), device="cuda")
labels = input_ids.clone()
attn_mask = torch.ones(B, L, dtype=torch.bool, device="cuda")

# ---- 预热 ----
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(input_ids=input_ids, labels=labels, attention_mask=attn_mask, use_cache=False)
    out.loss.backward()
    model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
print(f"[profile] 预热完成, q_len 实际={input_ids.shape}, 开始 profile...", flush=True)

# ---- profile: 1 步 forward + backward ----
with torch.autocast("cuda", dtype=torch.bfloat16):
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA, profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        out = model(input_ids=input_ids, labels=labels, attention_mask=attn_mask, use_cache=False)
        out.loss.backward()
        torch.cuda.synchronize()

# 1) 总耗时
total_cuda = sum(e.self_device_time_total for e in prof.key_averages())
total_cpu = sum(e.self_cpu_time_total for e in prof.key_averages())
print(f"\n===== 总耗时 =====\nCUDA total: {total_cuda / 1e6:.3f} s, CPU total: {total_cpu / 1e3:.3f} ms")
print(f"device self time top 事件数: {len(prof.key_averages())}")

# 2) kernel 热点 top 30
print("\n===== kernel 热点 (self CUDA time, top 30) =====")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))

# 3) MASS/检索/tilelang 相关 kernel 聚合占比
mass_keys = ["mass", "retrieval", "anchor", "summary", "envelope", "tilelang", "block_M", "topk", "landmark", "lmk"]
evs = prof.key_averages()
mass_evs = [e for e in evs if any(k in e.key.lower() for k in mass_keys)]
mass_cuda = sum(e.self_device_time_total for e in mass_evs)
print(f"\n===== MASS 相关 kernel 聚合 =====\n"
      f"候选 {len(mass_evs)} 个, self CUDA = {mass_cuda/1e6:.3f} s ({mass_cuda/total_cuda*100:.1f}%)")
for e in sorted(mass_evs, key=lambda x: -x.self_device_time_total)[:25]:
    print(f"  {e.key[:90]:<92} {e.self_device_time_total/1e6:8.3f} ms  x{e.count}")

# 4) 非 MASS 的 top 15(对照:纯 transformer 开销)
print("\n===== 非 MASS 相关 kernel top 15 =====")
others = [e for e in evs if not any(k in e.key.lower() for k in mass_keys)]
for e in sorted(others, key=lambda x: -x.self_device_time_total)[:15]:
    print(f"  {e.key[:90]:<92} {e.self_device_time_total/1e6:8.3f} ms  x{e.count}")

# 5) 按 CUDA 事件(含 input/output shape)看 tilelang 检索
print("\n===== tilelang/检索类 CUDA 事件明细 =====")
for e in prof.events()[:0]:  # 占位避免空
    pass
tl = [e for e in prof.events() if any(k in (e.name or "").lower() for k in ["tilelang", "mass", "retrieval", "anchor"])]
seen = {}
for e in tl:
    key = (e.name, str(e.input_shapes))
    seen.setdefault(key, [0, 0])
    seen[key][0] += e.self_device_time_total
    seen[key][1] += 1
for (name, shapes), (t, n) in sorted(seen.items(), key=lambda x: -x[1][0])[:20]:
    print(f"  {name[:80]:<82} {t/1e6:8.3f} ms x{n}  {shapes[:120]}")

print("\n[profile] 完成")

# 6) MASS 分段耗时(record_function 区域内的 CUDA 时间)
rf_keys = ["MASS_PROJ_QKV", "MASS_SWA", "MASS_RETRIEVE", "MASS_MIX", "MASS_SPARSE_ATTN", "MASS_OUT", "MASS_ROPED"]
import collections
seg = collections.defaultdict(lambda: [0.0, 0, 0.0])
for e in prof.events():
    if not (e.name or "").startswith("MASS_"):
        continue
    seg[e.name][0] += e.self_device_time_total
    seg[e.name][1] += 1
    seg[e.name][2] += e.device_time_total
print("\n===== MASS 分段耗时 (self CUDA) =====")
for k, (s, n, tot) in sorted(seg.items(), key=lambda x: -x[1][0]):
    print(f"  {k:<20} self {s/1e6:8.3f} s   total {tot/1e6:8.3f} s  x{n}")
import collections

def owner_frame(e):
    """返回事件 stack 中第一个项目内/站点包的文件:行号 帧名"""
    for fn, ln, name in (e.stack or []):
        if "HiLS-Attention-main" in fn or "veomni" in fn or "site-packages" in fn:
            return f"{fn.split('/')[-1]}:{ln} {name}"
    return "(no stack)"

targets = ["kernel_kernel", "aten::mul", "aten::index", "aten::gather", "aten::copy_",
           "aten::index_add_", "elementwise_kernel", "aten::sum", "aten::add_",
           "hils_bwd_dqkv", "hils_block_M_reuse", "fwd_recompute", "bwd_kernel", "_FlexAttnTL"]
for t in targets:
    evs_t = [e for e in prof.events() if t in (e.name or "")]
    if not evs_t:
        continue
    agg = collections.defaultdict(lambda: [0.0, 0])
    for e in evs_t:
        fr = owner_frame(e)
        agg[fr][0] += e.self_device_time_total
        agg[fr][1] += 1
    tot = sum(v[0] for v in agg.values())
    print(f"\n===== {t} 归属 (总 {tot/1e6:.1f} ms) =====")
    for fr, (tt, n) in sorted(agg.items(), key=lambda x: -x[1][0])[:6]:
        print(f"  {fr[:110]:<112} {tt/1e6:8.1f} ms x{n} ({tt/tot*100:.0f}%)")
