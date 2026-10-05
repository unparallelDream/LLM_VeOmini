"""梯度探针：验证 MASS v2 检索算子中 LSE 温度对"低分 anchor"梯度的影响。

测试1（真实 TileLang kernel）：spec loss backward 后梯度是否到达 anchors，
  以及 per-anchor 梯度范数对比 vs lse_temperature。
测试2（PyTorch 数学复算）：低分 plane 从 scoring loss 获得的梯度占比 vs tau。
"""
import math
import os
import sys

import torch

sys.path.insert(0, "/9950backfile/zhangem/HiLS-Attention-main")

from ops.mass_retrieval_tilelang import _MassSummaryFn
from ops.mass_retrieval_tilelang_v2 import mass_retrieval_tilelang_v2

# ---- 训练同款尺寸（B=1 全序列 smoke）----
B, L, N, S, H, Hkv, M, D, topk = 1, 8322, 130, 64, 16, 8, 2, 128, 32
SM = 1.0 / math.sqrt(D)
LOG2 = math.log(2)

torch.manual_seed(0)
dev = "cuda:0"


def spec_loss(cond_ent, anchor_load):
    marg_ent = -(anchor_load * torch.log(anchor_load.clamp_min(1e-30))).sum()
    return (cond_ent - marg_ent + LOG2) / LOG2


# 保存 _MassSummaryFn 输出（mu, bias），用于 PyTorch 复算 planes
_orig_fwd = _MassSummaryFn.forward
_mu_bias = {}


def _patched_fwd(ctx, anchors, keys, sm_scale, groups):
    out = _orig_fwd(ctx, anchors, keys, sm_scale, groups)
    _mu_bias["mu"] = out[0].detach()
    _mu_bias["bias"] = out[1].detach()
    return out


_MassSummaryFn.forward = _patched_fwd

q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=dev)
keys = torch.randn(B, N, S, Hkv, D, dtype=torch.bfloat16, device=dev)
anchors = torch.randn(B, N, M, H, D, dtype=torch.bfloat16, device=dev)

print("=== 测试1: spec-loss backward, per-anchor 梯度（真实 kernel）===", flush=True)
for tau in [0.1, 4.0]:
    anchors.requires_grad_(True)
    indices, scores, anchor_load, cond_ent = mass_retrieval_tilelang_v2(
        q,
        keys,
        anchors,
        num_key_value_heads=Hkv,
        topk=topk,
        chunk_size=S,
        window_size=0,
        sm_scale=SM,
        score_reduction="logsumexp",
        lse_temperature=tau,
    )
    # 联合 loss：模拟训练（lm 路径用 -scores.mean() 代表 scoring 监督）
    loss = spec_loss(cond_ent, anchor_load)
    loss.backward()
    ga = anchors.grad.float()
    per_anchor = ga.square().sum(dim=(0, 1, 3, 4)).sqrt()  # [M]
    hi, lo = per_anchor.max(), per_anchor.min()
    ratio = (lo / hi).item() if hi > 0 else 0.0
    print(
        f"tau={tau}: spec={loss.item():.4f} load={anchor_load.detach().cpu().tolist()}"
        f" cond_ent={cond_ent.item():.4f} per_anchor_grad_norm="
        f"{[round(x, 6) for x in per_anchor.detach().cpu().tolist()]}"
        f" low/high_ratio={ratio:.6f}",
        flush=True,
    )
    anchors.grad = None
    # 每个 anchor 在 M 维上的梯度符号一致性（分工信号强弱）
    sign = torch.sign(ga.sum(dim=(0, 1, 3, 4)))
    print(f"    sign_per_anchor={sign.tolist()}", flush=True)

print("=== 测试2: 低分 plane 从 scoring 获得的梯度占比（PyTorch 数学）===", flush=True)
mu = _mu_bias["mu"]  # [B,N,H,M,D]
bias = _mu_bias["bias"]  # [B,N,H,M]
q_f = q.float()
# planes[b, l, n, h, m] = (q·mu) * sm + bias
planes = torch.einsum("blhd,bnhmd->blnhm", q_f, mu.float()) * SM + bias.float().unsqueeze(1)
# 真实训练中 topk 选择基于 scores（LSE over M 后取 top-K）；这里近似取全部 N
for tau in [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]:
    p = torch.softmax(planes / tau, dim=-1)  # [B,L,N,H,M]
    p_sorted, _ = p.sort(dim=-1, descending=True)
    hi_p, lo_p = p_sorted[..., 0], p_sorted[..., -1]
    # 低分 plane 从 scores 项获得的梯度占比 = p_low / (p_high + p_low)（LSE 导数）
    share = (lo_p / (hi_p + lo_p)).clamp(0, 1)
    mask = hi_p > 1e-4
    vals = share[mask].flatten()
    n = vals.numel()
    if n > 2_000_000:
        idx = torch.randperm(n, device=vals.device)[:2_000_000]
        vals = vals[idx]
    med = vals.median().item() if vals.numel() else float("nan")
    p90 = torch.sort(vals).values[int(vals.numel() * 0.9)].item() if vals.numel() else float("nan")
    print(
        f"tau={tau}: low-plane-grad-share mean={vals.mean().item():.5f}"
        f" median={med:.5f} p90={p90:.5f} n={n}",
        flush=True,
    )

print("=== 测试3: spec 项（balance+entropy）对低分 plane 的梯度贡献 vs tau ===", flush=True)
# 用 PyTorch 数学复算 spec 反传分量（与 kernel backward 公式一致）
probs = torch.softmax(planes.float(), dim=-1)  # tau=1 基准
for tau in [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]:
    p = torch.softmax(planes.float() / tau, dim=-1)
    valid = p[..., 0] > 1e-4
    # entropy 梯度: -p_j*(log p_j + H)/tau 的幅度（模拟 grad_cond_ent=1）
    H_ent = -(p * p.clamp_min(1e-30).log()).sum(-1, keepdim=True)
    ent_grad = -p * (p.clamp_min(1e-30).log() + H_ent) / tau
    p_sorted = p.sort(dim=-1, descending=True).values
    hi_p, lo_p = p_sorted[..., 0], p_sorted[..., -1]
    ent_grad_sorted = ent_grad.sort(dim=-1, descending=True).values
    # 低分 plane 的 entropy 梯度占比（绝对值）
    share = (ent_grad_sorted[..., -1].abs() / (ent_grad_sorted[..., 0].abs() + ent_grad_sorted[..., -1].abs() + 1e-30))
    vals = share[valid].flatten()
    n = vals.numel()
    if n > 2_000_000:
        idx = torch.randperm(n, device=vals.device)[:2_000_000]
        vals = vals[idx]
    med = vals.median().item() if vals.numel() else float("nan")
    print(
        f"tau={tau}: low-plane entropy-grad share mean={vals.mean().item():.5f}"
        f" median={med:.5f} n={n}",
        flush=True,
    )
