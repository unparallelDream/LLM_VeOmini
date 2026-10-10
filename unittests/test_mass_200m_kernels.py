"""200M 形状的 MASS kernel 同类问题核查。

产线 200M(config_mass_olmo3_200M_8KA2K_RoPE_m2_sharedview_lmkq.json):
L=8322, N=130, H=4, Hkv=4(G=1), M=2, D=128, topk=32,chunk=64。

核查项:
1. 旧两段式 partial(legacy)在该形状是否同样数值错误(与 dense 参考比);
2. 修复后的合并版 partial 是否正确;
3. dq 旧标量 kernel 与新 gather-GEMM 的正确性与耗时。

用法: python unittests/test_mass_200m_kernels.py
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

# 200M 产线形状
B, L, N, H, Hkv, M, D, K = 1, 8322, 130, 4, 4, 2, 128, 32
SM_SCALE = 1.0 / math.sqrt(D)


def make_inputs(seed=11):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mu = torch.randn(B, N, H, M, D, dtype=torch.float32, generator=g).bfloat16().cuda()
    q = torch.randn(B, L, H, D, dtype=torch.float32, generator=g).bfloat16().cuda()
    idx = torch.randint(0, N, (B, L, Hkv, K), generator=g).to(torch.int32).cuda()
    bad = torch.rand(B, L, Hkv, K, generator=g) < 0.1
    idx[bad] = -1
    planes = torch.randn(B, L, H, K, M, dtype=torch.float32, generator=g).cuda()
    return mu, q, idx, planes


def dense_reference(mu_f, q, idx, planes):
    arH = torch.arange(H, device="cuda")
    arB = torch.arange(B, device="cuda")
    # dmu / dbias(H==Hkv,G=1)
    eq = (idx[0].permute(1, 0, 2)[..., None] == torch.arange(N, device="cuda")).float()  # [H,L,K,N]
    ref_mu = (torch.einsum("hlkn,lhkm,lhd->nhmd", eq, planes[0], q[0].float()) * SM_SCALE).unsqueeze(0)
    ref_bias = torch.einsum("hlkn,lhkm->nhm", eq, planes[0]).unsqueeze(0)
    # dq
    ref_dq = torch.zeros(B, L, H, D, device="cuda")
    for k in range(K):
        n = idx[0, :, :, k]                       # [L, Hkv]
        valid = ((n >= 0) & (n < N)).float()      # [L, Hkv]
        ns = n.clamp(0, N - 1).long()
        for m in range(M):
            mu_sel = mu_f[0][ns, arH[None, :], m, :]     # [L, H, D]
            ref_dq[0] += (planes[0, :, :, k, m] * valid).unsqueeze(-1) * mu_sel
    return ref_mu, ref_bias, ref_dq * SM_SCALE


def bench(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


if __name__ == "__main__":
    torch.manual_seed(0)
    mu, q, idx, planes = make_inputs()
    mu_f = mu.float()
    nt = (L + 31) // 32
    ref_mu, ref_bias, ref_dq = dense_reference(mu_f, q, idx, planes)

    print(f"== 200M 形状 B{B} L{L} N{N} H{H} Hkv{Hkv} M{M} D{D} K{K} ==")

    # 1) 旧两段式 partial(保留的 legacy kernel)
    p1 = torch.zeros(B, H, nt, N * M, D, device="cuda")
    b1 = torch.zeros(B, H, nt, N * M, device="cuda")
    legacy_ks = [mass_selected_partial_bwd_kernel(B, L, N, H, Hkv, M, D, K, SM_SCALE, t) for t in range(M)]
    for kern in legacy_ks:
        kern(q, idx, planes, p1, b1)
    mu1, bias1 = mass_selected_partial_reduce_kernel(B, L, N, H, M, D)(p1, b1)
    r_mu = ((mu1.float() - ref_mu).abs() > 0.05).float().mean().item() * 100
    r_b = ((bias1.float() - ref_bias).abs() > 0.05).float().mean().item() * 100
    print(f"[legacy partial]  bad_mu={r_mu:.2f}%  bad_bias={r_b:.2f}%  "
          f"max|dbias|={(bias1.float() - ref_bias).abs().max().item():.2f}")

    # 2) 修复后的合并 partial
    p2 = torch.zeros_like(p1); b2 = torch.zeros_like(b1)
    mt = mass_selected_partial_bwd_mt_kernel(B, L, N, H, Hkv, M, D, K, SM_SCALE)
    mt(q, idx, planes, p2, b2)
    mu2, bias2 = mass_selected_partial_reduce_kernel(B, L, N, H, M, D)(p2, b2)
    r_mu2 = ((mu2.float() - ref_mu).abs() > 0.05).float().mean().item() * 100
    r_b2 = ((bias2.float() - ref_bias).abs() > 0.05).float().mean().item() * 100
    print(f"[fixed  partial]  bad_mu={r_mu2:.2f}%  bad_bias={r_b2:.2f}%  "
          f"max|dbias|={(bias2.float() - ref_bias).abs().max().item():.4f}")
    assert r_mu2 < 0.1 and r_b2 < 0.1, "修复版 partial 在 200M 形状不正确!"

    # 3) dq 两版
    dq_legacy_k = mass_selected_dq_kernel(B, L, N, H, Hkv, M, D, K, SM_SCALE)
    dq_gemm_k = mass_selected_dq_gemm_kernel(B, L, N, H, Hkv, M, D, K, SM_SCALE)
    dql = dq_legacy_k(mu, idx, planes)
    dqg = dq_gemm_k(mu, idx, planes)
    rel_l = ((dql.float() - ref_dq).abs().max() / ref_dq.abs().max().clamp_min(1e-6)).item()
    rel_g = ((dqg.float() - ref_dq).abs().max() / ref_dq.abs().max().clamp_min(1e-6)).item()
    t_l = bench(lambda: dq_legacy_k(mu, idx, planes))
    t_g = bench(lambda: dq_gemm_k(mu, idx, planes))
    print(f"[dq legacy]  rel={rel_l:.4f}  {t_l:7.1f}ms")
    print(f"[dq gemm  ]  rel={rel_g:.4f}  {t_g:6.2f}ms  speedup={t_l / t_g:.0f}x")
    assert rel_g < 0.02, "dq gemm 在 200M 形状不正确!"

    # 4) partial 计时
    t_leg = bench(lambda: [k(q, idx, planes, p1, b1) for k in legacy_ks])
    t_mt = bench(lambda: mt(q, idx, planes, p2, b2))
    print(f"[partial] legacy(2 launches)={t_leg:.1f}ms  merged={t_mt:.1f}ms")
    print("200M 形状核查完成")
