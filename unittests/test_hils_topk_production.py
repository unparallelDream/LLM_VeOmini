"""单锚(Arm A / HiLS 基线)检索 kernel 产线形状正确性核查。

路径:ops/topk_head_softmax.py::online_softmax_topk_head(lse→加权topk→sort→重算)
+ 纯 PyTorch straight-through backward。自带测试只覆盖 G=8/block32/window32;
本测试按产线配置跑:G=1、block_size=64、window=512、topk=32、L=8322、S=130,
覆盖 200M(h_kv=4)与 7B(h_kv=32)两档,forward+backward 均对拍
ref_softmax_topk_max_pooling(自带 torch 参考实现,可 autograd)。

用法: python -m unittests.test_hils_topk_production
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ops.topk_head_softmax import (  # noqa: E402
    online_softmax_topk_head,
    ref_softmax_topk_max_pooling,
)


def run_case(tag, h_kv, L=8322, S=130, D=128, G=1, topk=32, block=64, window=512, seed=7):
    torch.manual_seed(seed)
    B = 1
    dev = "cuda"
    q0 = torch.randn(B, L, h_kv, G, D, dtype=torch.bfloat16, device=dev)
    k0 = torch.randn(B, S, h_kv, D, dtype=torch.bfloat16, device=dev)
    # lse_swa 量级贴近产线(swa 窗口 LSE)
    lse0 = (torch.randn(B, L, h_kv, G, dtype=torch.bfloat16, device=dev) * 2 + 4).float()

    # ---- 参考前向(不可训练张量)与参考反向(可训练副本 autograd) ----
    ref_idx, ref_scores = ref_softmax_topk_max_pooling(
        q0.detach(), k0.detach(), lse0.detach(), topk, block, window, is_causal=True
    )

    q = q0.clone().requires_grad_(True)
    k = k0.clone().requires_grad_(True)
    qr = q0.clone().float().requires_grad_(True)
    kr = k0.clone().float().requires_grad_(True)

    fused_idx, fused_scores = online_softmax_topk_head(
        q, k, lse0.clone(), topk, block_size=block, window_size=window,
        is_causal=True, is_training=True,
    )
    ref_idx2, ref_scores2 = ref_softmax_topk_max_pooling(
        qr, kr, lse0.detach().clone(), topk, block, window, is_causal=True
    )

    # 前向:索引一致率(有效位)与分数误差
    valid = fused_idx.ge(0)
    match = (fused_idx.long() == ref_idx.long()).float()[valid].mean().item() if valid.any() else float("nan")
    fs = fused_scores.view(B, L, h_kv, G, topk).float()
    rs = ref_scores2.float()  # 可微参考的分数(前向比较与反向都可用)
    finite = torch.isfinite(fs) & torch.isfinite(rs)  # 无效槽位(-inf)不参与
    sc_err = ((fs - rs).abs()[finite]).max().item()
    sc_scale = rs[finite].abs().max().item()

    # 反向:同权重(仅有限槽位)对 fused 与 ref 各做一次加权求和后 backward
    w = torch.where(finite, torch.randn_like(fs), torch.zeros_like(fs))
    (fs * w).sum().backward()
    (rs * w).sum().backward()

    def _rel(a, b):
        return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()

    rq, rk = _rel(q.grad.float(), qr.grad), _rel(k.grad.float(), kr.grad)

    print(
        f"[{tag}] h_kv={h_kv} G={G} | idx_match={match*100:.4f}%  "
        f"score_maxdiff={sc_err:.4f}(scale {sc_scale:.1f})  "
        f"rel(gq)={rq:.4f} rel(gk)={rk:.4f}"
    )
    ok = match > 0.999 and sc_err < 0.05 and rq < 0.02 and rk < 0.02
    print(f"        {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


if __name__ == "__main__":
    ok = True
    ok &= run_case("200M 产线", h_kv=4)
    ok &= run_case("7B 产线", h_kv=32)
    ok &= run_case("GQA 参照(自带测试形状)", h_kv=2, L=4096, S=64, G=8, topk=16, block=32, window=32)
    print("单锚检索 kernel 核查:", "全部通过" if ok else "存在失败!")
    sys.exit(0 if ok else 1)
