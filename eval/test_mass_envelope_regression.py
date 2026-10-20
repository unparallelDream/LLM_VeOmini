"""MASS envelope kernel regression tests (2026-10-19).

Two bugs are pinned down here; both must stay fixed:

1. View-reduction collapse (introduced by the 2026-10-10 tiled rewrite,
   fixed 2026-10-19): for M >= 2 anchor views the envelope kernel returned
   the LAST view's raw plane for BOTH max and logsumexp modes, because the
   ``T.serial(mp)`` accumulation over an ``alloc_local`` scalar inside
   ``T.Parallel`` did not lower correctly.  The kernel now unrolls the view
   loop at trace time.
2. GQA prefill/decode selection fork (fixed 2026-10-19): the TileLang
   candidates kernel selects per KV-head group with a HARD MAX over the
   group's query heads; ``_decode_retrieval`` / ``_reference_retrieval_prefill``
   used a temperature-LSE over heads and thus selected different chunks than
   training for every groups > 1 config.  Both PyTorch paths now use the
   hard max.  (groups == 1 keeps both formulas identical by construction.)

Run: CUDA_VISIBLE_DEVICES=0 python eval/test_mass_envelope_regression.py
"""
import math

import torch

from ops.mass_retrieval_tilelang_v2 import mass_envelope_candidates_kernel


def _planes(q, mu, bias, sm_scale, tau):
    planes = torch.einsum("blhd,bnhmd->blnhm", q.float(), mu.float()) * sm_scale + bias.float()[:, None]
    return tau * torch.logsumexp(planes / tau, dim=4)  # per-head envelope [B,L,N,H]


def _mask_window(per_head, window, chunk):
    B, L, N, H = per_head.shape
    rows = torch.arange(L, device=per_head.device)
    lim = ((rows - window + 1) // chunk).clamp_min(0)
    elig = torch.arange(N, device=per_head.device)[None, None, :] < lim[:, None]
    return per_head.masked_fill(~elig[..., None], float("-inf"))


def test_view_reduction_no_collapse():
    """M=2: max mode must return max over views, lse mode the LSE over views."""
    device = "cuda"
    D, chunk, window, L = 128, 64, 64, 128
    N = L // chunk
    H = Hkv = 8  # MHA isolates the view reduction
    torch.manual_seed(0)
    q = torch.randn(1, L, H, D, dtype=torch.bfloat16, device=device)
    mu = torch.randn(1, N, H, 2, D, dtype=torch.bfloat16, device=device)
    bias = (torch.randn(1, N, H, 2, device=device) * 0.5).float()
    sm = 1 / math.sqrt(D)
    t, n, hkv = 127, 0, 0
    views = [
        torch.dot(q[0, t, hkv].float(), mu[0, n, hkv, m].float()).item() * sm
        + bias[0, n, hkv, m].item()
        for m in range(2)
    ]
    exp_max, exp_lse = max(views), math.log(sum(math.exp(v) for v in views))
    for use_lse, expect in ((False, exp_max), (True, exp_lse)):
        got = mass_envelope_candidates_kernel(
            1, L, N, H, Hkv, 2, D, chunk, window, sm, use_lse, 1.0
        )(q, mu, bias)[0, t, hkv, n].item()
        assert abs(got - expect) < 1e-2, f"view reduction collapsed: mode(lse={use_lse}) got {got:.4f}, expect {expect:.4f}"
    print("[PASS] view reduction (M=2, max & lse)")


def test_gqa_group_reduction_is_max():
    """GQA: kernel candidates must equal hard-max over the group's q heads."""
    device = "cuda"
    D, chunk, window, L, M, tau = 128, 64, 512, 4096, 2, 1.0
    N = L // chunk
    sm = 1 / math.sqrt(D)
    for H, Hkv in ((16, 4), (16, 8), (16, 16)):
        groups = H // Hkv
        torch.manual_seed(0)
        q = torch.randn(1, L, H, D, dtype=torch.bfloat16, device=device)
        mu = torch.randn(1, N, H, M, D, dtype=torch.bfloat16, device=device)
        bias = (torch.randn(1, N, H, M, device=device) * 0.5).float()
        cand = mass_envelope_candidates_kernel(
            1, L, N, H, Hkv, M, D, chunk, window, sm, True, tau
        )(q, mu, bias)
        per = _mask_window(_planes(q, mu, bias, sm, tau), window, chunk)
        g = per.view(1, L, N, Hkv, groups)
        ref_max = g.max(-1).values.permute(0, 1, 3, 2)
        ref_lse = (tau * torch.logsumexp(g / tau, -1)).permute(0, 1, 3, 2)
        fin = torch.isfinite(cand)
        dmax = (cand - ref_max).abs()[fin].max().item()
        dlse = (cand - ref_lse).abs()[fin].max().item()
        assert dmax < 1e-4, f"kernel != max_g for H={H} Hkv={Hkv}: {dmax}"
        if groups > 1:
            assert dlse > 0.1, f"expected max/lse fork to be visible for groups={groups}"
        else:
            assert dlse < 1e-4, "groups=1 must make max_g and lse_g identical"
        print(f"[PASS] GQA group reduction H={H} Hkv={Hkv} groups={groups} (==max_g, |lse_g| diff {dlse:.3f})")


if __name__ == "__main__":
    test_view_reduction_no_collapse()
    test_gqa_group_reduction_is_max()
    print("ALL REGRESSION TESTS PASSED")
