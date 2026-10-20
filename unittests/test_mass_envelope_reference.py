"""Envelope/selected-scores kernels vs an independent torch reference.

2026-10-19: the envelope kernel's T.serial(mp) accumulation over an
alloc_local scalar inside T.Parallel silently collapsed to the m=mp-1
plane (verified: old kernel == last plane bit-exact, off by ~3.7 vs
reference).  The fix unrolls the loop in Python.  This test pins BOTH
forward kernels against an independent reference so a kernel-vs-kernel
regression test can never mask a shared lowering bug again.
"""
import math
import sys

import pytest
import torch

sys.path.insert(0, "/9950backfile/zhangem/HiLS-Attention-main")
from ops.mass_retrieval_tilelang_v2 import (
    mass_envelope_candidates_kernel,
    mass_selected_scores_kernel,
)

SHAPES = [
    # (B, L, H, Hkv, M, D)  chunk=64, window=512
    (1, 1024, 4, 4, 2, 128),    # 200M production: G=1, M=2
    (1, 1024, 8, 4, 2, 128),    # G=2
    (1, 1024, 4, 4, 4, 128),    # M=4 verified 2026-10-19
    (1, 1024, 4, 4, 8, 128),    # M=8 verified 2026-10-19
    # NOTE: odd mp (M=3) is NOT supported — tilelang free-mode layout
    # inference rejects the block_n*3 GEMM ("no available layout found").
    # Even/power-of-two M in {2,4,8} verified exact vs reference.
]


def _reference(Q, Mu, Bias, sm_scale, tau, use_lse):
    B, L, H, D = Q.shape
    _, N, _, M, _ = Mu.shape
    q, mu, b = Q.float(), Mu.float(), Bias.float()
    planes = torch.einsum("blhd,bnhmd->blhnm", q, mu) * sm_scale \
        + b.permute(0, 2, 1, 3).unsqueeze(1)
    if use_lse:
        m = planes.max(-1, keepdim=True).values
        env = m.squeeze(-1) + tau * torch.logsumexp((planes - m) / tau, dim=-1)
    else:
        env = planes.max(-1).values
    return env  # [B,L,H,N]


@pytest.mark.parametrize("use_lse", [True, False])
@pytest.mark.parametrize("B,L,H,Hkv,M,D", SHAPES)
def test_envelope_matches_reference(B, L, H, Hkv, M, D, use_lse):
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    N, C, WIN = L // 64, 64, 512
    sm, tau = 1.0 / math.sqrt(D), 0.5
    torch.manual_seed(0)
    Q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    Mu = torch.randn(B, N, H, M, D, dtype=torch.bfloat16, device="cuda") * 0.5
    Bias = torch.randn(B, N, H, M, dtype=torch.float32, device="cuda") * 0.3

    f = mass_envelope_candidates_kernel(
        B=B, L=L, N=N, H=H, Hkv=Hkv, M=M, D=D, chunk_size=C, window_size=WIN,
        sm_scale=sm, use_logsumexp=use_lse, lse_temperature=tau,
    )
    out = f(Q, Mu, Bias)
    got = out[0] if isinstance(out, (tuple, list)) else out
    groups = H // Hkv
    ref = _reference(Q, Mu, Bias, sm, tau, use_lse)  # [B,L,H,N]

    ar = torch.arange(L, device="cuda")
    qlimit = ((ar - WIN + 1) // C).clamp(min=0)
    mask = torch.arange(N, device="cuda")[None, :] < qlimit[:, None]  # [L,N]
    for hkv in range(Hkv):
        ref_h = ref[0, :, hkv * groups:(hkv + 1) * groups, :]  # [L,G,N]
        got_h = got[0, :, hkv, :].float()  # [L,N]
        exp_h = ref_h.masked_fill(~mask.unsqueeze(1), float("-inf")).max(1).values \
            if groups > 1 else ref_h[:, 0].masked_fill(~mask, float("-inf"))
        valid = torch.isfinite(exp_h)
        diff = (got_h[valid] - exp_h[valid]).abs().max().item()
        assert diff < 1e-2, f"envelope mismatch (lse={use_lse}, M={M}, G={groups}): {diff}"


@pytest.mark.parametrize("B,L,H,Hkv,M,D", SHAPES)
def test_selected_scores_matches_reference(B, L, H, Hkv, M, D):
    """selected_scores kernel shares the serial-loop+alloc_local pattern;
    check score = max_m <q, Mu[idx]> per (group, selected chunk)."""
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    N, K = L // 64, 4
    sm = 1.0 / math.sqrt(D)
    torch.manual_seed(1)
    Q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    Mu = torch.randn(B, N, H, M, D, dtype=torch.bfloat16, device="cuda") * 0.5
    # random distinct indices per (B, L, Hkv)
    idx = torch.stack([torch.randperm(N, device="cuda")[:K] for _ in range(B * L * Hkv)])
    idx = idx.view(B, L, Hkv, K).int()

    Bias = torch.randn(B, N, H, M, dtype=torch.float32, device="cuda") * 0.3
    f = mass_selected_scores_kernel(B=B, L=L, N=N, H=H, Hkv=Hkv, M=M, D=D, topk=K, sm_scale=sm)
    out = f(Q, Mu, Bias, idx)
    got = out[0] if isinstance(out, (tuple, list)) else out  # [B,L,H,topk]

    groups = H // Hkv
    q, mu, b = Q.float(), Mu.float(), Bias.float()
    for hkv in range(Hkv):
        for g in range(groups):
            h = hkv * groups + g
            sel = idx[0, :, hkv, :].long()  # [L,K]
            mu_sel = mu[0][:, h][sel]        # [L,K,M,D]
            b_sel = b[0][:, h][sel]          # [L,K,M]
            dots = torch.einsum("lkd,lkmd->lkm", q[0, :, h, :].unsqueeze(1).expand(L, K, D), mu_sel) * sm + b_sel
            ref = dots.max(-1).values        # [L,K] max over m
            got_h = got[0, :, h, :].float()  # [L,K]
            diff = (got_h - ref).abs().max().item()
            assert diff < 1e-2, f"selected_scores mismatch (M={M}, G={groups}): {diff}"
