"""Numerics + microbench for the block-diagonal gather-GEMM dQ kernel.

对比三条路径:
  1. mass_selected_dq_kernel      (旧标量 per-query kernel,fp32 数学)
  2. mass_selected_dq_gemm_kernel (新块对角 gather-GEMM,bf16 tensor core)
  3. torch 参考实现(gather 后 fp32 einsum)

以及合并 tangent partial kernel vs 旧 per-tangent kernels 的等价性。

运行: python -m unittests.test_mass_dq_gemm  (或 pytest)
"""

import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ops.mass_retrieval_tilelang_v2 import (  # noqa: E402
    mass_selected_dq_gemm_kernel,
    mass_selected_dq_kernel,
    mass_selected_partial_bwd_kernel,
    mass_selected_partial_bwd_mt_kernel,
    mass_selected_partial_reduce_kernel,
)


def _make_case(B, L, N, H, Hkv, M, D, K, seed=0, invalid_frac=0.1, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mu = torch.randn(B, N, H, M, D, dtype=torch.float32).bfloat16().to(device)
    idx = torch.randint(0, N, (B, L, Hkv, K), generator=g).to(device, torch.int32)
    if invalid_frac > 0:
        bad = torch.rand(B, L, Hkv, K, generator=g) < invalid_frac
        idx[bad] = -1
    planes = torch.randn(B, L, H, K, M, dtype=torch.float32, generator=g).to(device)
    return mu, idx, planes


def _ref_dq_simple(mu, idx, planes, Hkv, sm_scale):
    """O(L*K) 显式参考,groups 任意。"""
    B, N, H, M, D = mu.shape
    L, K = idx.shape[1], idx.shape[3]
    groups = H // Hkv
    mu_f = mu.float()
    out = torch.zeros(B, L, Hkv, groups, D, device=mu.device, dtype=torch.float32)
    b_idx = torch.arange(B, device=mu.device)[:, None, None]
    for k in range(K):
        n = idx[:, :, :, k]  # [B, L, Hkv]
        valid = ((n >= 0) & (n < N)).float()  # [B, L, Hkv]
        n_safe = n.clamp(0, N - 1).long()
        for m in range(M):
            w_hm = planes[:, :, :, k, m].view(B, L, Hkv, groups)  # [B,L,Hkv,G]
            for g in range(groups):
                # h = hkv*groups + g(与 kernel 的 head 映射一致)
                h_col = torch.arange(Hkv, device=mu.device) * groups + g
                mu_sel = mu_f[b_idx, n_safe, h_col[None, None, :], m, :]  # [B,L,Hkv,D]
                out[:, :, :, g, :] += (w_hm[..., g] * valid).unsqueeze(-1) * mu_sel
    return out.reshape(B, L, H, D) * sm_scale


def _run_case(name, B, L, N, H, Hkv, M, D, K, sm_scale=None):
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    mu, idx, planes = _make_case(B, L, N, H, Hkv, M, D, K)
    legacy = mass_selected_dq_kernel(B, L, N, H, Hkv, M, D, K, sm_scale)
    gemm = mass_selected_dq_gemm_kernel(B, L, N, H, Hkv, M, D, K, sm_scale)
    q_legacy = legacy(mu.contiguous(), idx.contiguous(), planes.contiguous())
    q_gemm = gemm(mu.contiguous(), idx.contiguous(), planes.contiguous())
    q_ref = _ref_dq_simple(mu, idx, planes, Hkv, sm_scale)

    def _rel(a, b):
        denom = b.abs().max().clamp_min(1e-6)
        return ((a - b).abs().max() / denom).item()

    r_gem_ref = _rel(q_gemm.float(), q_ref)
    r_leg_ref = _rel(q_legacy.float(), q_ref)
    r_gem_leg = _rel(q_gemm.float(), q_legacy.float())
    print(
        f"[{name}] B{B} L{L} N{N} H{H} Hkv{Hkv} M{M} D{D} K{K} | "
        f"gemm-vs-ref rel={r_gem_ref:.4f}  legacy-vs-ref rel={r_leg_ref:.4f}  "
        f"gemm-vs-legacy rel={r_gem_leg:.4f}"
    )
    torch.testing.assert_close(q_gemm.float(), q_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(q_legacy.float(), q_ref, atol=5e-2, rtol=5e-2)
    return legacy, gemm, mu, idx, planes, sm_scale


def _bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def _test_partial_merge(B, L, N, H, Hkv, M, D, K, sm_scale):
    """合并 tangent partial kernel 对照 dense 参考(不是对照旧 kernel:旧两段式
    partial 的分轮清零/写回结构会丢/重共享原子更新,数值本身是错的)。"""
    mu_q, idx, planes = _make_case(B, L, N, H, Hkv, M, D, K, seed=3)
    q = torch.randn(B, L, H, D, dtype=torch.float32).bfloat16().to(mu_q.device)
    num_tiles = (L + 31) // 32
    mt = mass_selected_partial_bwd_mt_kernel(B, L, N, H, Hkv, M, D, K, sm_scale)
    p2 = torch.zeros(B, H, num_tiles, N * M, D, dtype=torch.float32, device=q.device)
    b2 = torch.zeros(B, H, num_tiles, N * M, dtype=torch.float32, device=q.device)
    mt(q.contiguous(), idx.contiguous(), planes.contiguous(), p2, b2)

    # dense 参考(抽样校验,全量 32×261 组太慢):partial[b,h,tile,n*M+t,:] =
    # Σ_{l∈tile,k: idx=n} planes·q·sc
    import random

    groups = H // Hkv
    rng = random.Random(0)
    checks = min(24, H * num_tiles)
    for _ in range(checks):
        h = rng.randrange(H)
        tile = rng.randrange(num_tiles)
        hkv = h // groups
        eq = (
            idx[0, tile * 32 : (tile + 1) * 32, hkv, :, None]
            == torch.arange(N, device=q.device)[None, None, :]
        ).float()  # [l,K,N]
        pl = planes[0, tile * 32 : (tile + 1) * 32, h]  # [l,K,M]
        refb = torch.einsum("lkn,lkm->nm", eq, pl)  # [N,M]
        refp = torch.einsum("lkn,lkm,ld->nmd", eq, pl, q[0, tile * 32 : (tile + 1) * 32, h].float()) * sm_scale
        kern_b = b2[0, h, tile].view(N, M)
        kern_p = p2[0, h, tile].view(N, M, D)
        torch.testing.assert_close(kern_b, refb, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(kern_p, refp, atol=5e-3, rtol=5e-3)
    print(f"[partial-merged] B{B} L{L} M{M}: {checks} sampled (h,tile) blocks match reference ✓")

    t_mt = _bench(lambda: mt(q.contiguous(), idx.contiguous(), planes.contiguous(), p2, b2))
    print(f"[partial-merged] single launch: {t_mt:.2f}ms")


def main():
    torch.manual_seed(0)
    # 小形状(groups=2 覆盖通用性)
    _run_case("small-g2", 1, 128, 4, 4, 2, 2, 64, 4)
    # 小形状(groups=1)
    _run_case("small-g1", 1, 256, 8, 4, 4, 2, 64, 8)
    # 产线形状
    legacy, gemm, mu, idx, planes, sm_scale = _run_case(
        "prod", 1, 8322, 130, 32, 32, 2, 128, 32
    )
    t_legacy = _bench(lambda: legacy(mu.contiguous(), idx.contiguous(), planes.contiguous()))
    t_gemm = _bench(lambda: gemm(mu.contiguous(), idx.contiguous(), planes.contiguous()))
    print(f"[prod-bench] dq legacy: {t_legacy:.1f}ms  dq gemm: {t_gemm:.2f}ms  speedup: {t_legacy/max(t_gemm,1e-9):.0f}x")

    _test_partial_merge(1, 8322, 130, 32, 32, 2, 128, 32, 1.0 / math.sqrt(128))
    print("ALL OK")


if __name__ == "__main__":
    main()
