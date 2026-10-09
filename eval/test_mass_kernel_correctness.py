"""MASS tilelang kernel correctness unit test vs PyTorch reference.

Synthetic inputs, three sequence lengths. The reference recomputes chunk
selection with the same mu/bias (from _MassSummaryFn) and compares the
kernel's returned indices + selected scores.
"""

import math
import sys

import torch

sys.path.insert(0, "/9950backfile/zhangem/HiLS-Attention-main")
from ops.mass_retrieval_tilelang_v2 import mass_retrieval_tilelang_v2, _MassSummaryFn


def reference_selection(queries, key_chunks, anchors, *, topk, chunk_size, window_size,
                        sm_scale, tau, normalize, shared_key, shared_bias):
    B, L, H, D = queries.shape
    _, N, S, Hkv, _ = key_chunks.shape
    _, _, M, _, _ = anchors.shape
    groups = H // Hkv
    with torch.no_grad():
        mu, bias = _MassSummaryFn.apply(anchors, key_chunks, sm_scale, groups)
        if mu.shape[2] == anchors.shape[2]:          # came as [B,N,M,H,D] -> normalize to [B,N,H,M,D]
            mu = mu.permute(0, 1, 3, 2, 4).contiguous()
            bias = bias.permute(0, 1, 3, 2).contiguous()
        if shared_key is not None:
            shared_mu = shared_key.repeat_interleave(groups, dim=2).unsqueeze(3).to(mu.dtype).contiguous()
            shared_b = shared_bias.to(dtype=bias.dtype).view(1, 1, 1, 1).expand(B, N, H, 1).contiguous()
            mu = torch.cat([shared_mu, mu[:, :, :, 1:, :]], dim=3).contiguous()
            bias = torch.cat([shared_b, bias[:, :, :, 1:]], dim=3).contiguous()
        q = queries.float()                       # [B,L,H,D]
        planes = torch.einsum("blhd,bnhmd->blnhm", q, mu.float()) * sm_scale + bias.float().unsqueeze(1)
        scores = tau * torch.logsumexp(planes / tau, dim=4)          # [B,L,N,H]
        if normalize:
            scores = scores - tau * math.log(mu.shape[4])
        scores = scores.permute(0, 1, 3, 2)                          # [B,L,H,N]
        # window mask: chunk n eligible iff n >= (l - window + 1)//chunk
        l_idx = torch.arange(L, device=q.device).view(1, L, 1, 1)
        n_idx = torch.arange(N, device=q.device).view(1, 1, 1, N)
        limit = (l_idx - window_size + 1) // chunk_size if window_size > 0 else torch.zeros_like(l_idx)
        scores = scores.masked_fill(n_idx >= limit, float("-inf"))  # 远程块=窗口之前(n<limit)
        k = min(topk, N)
        ref_idx = scores.topk(k, dim=-1).indices.sort(dim=-1).values      # [B,L,H,k]
        ref_sc = torch.gather(scores, -1, ref_idx)
        return ref_idx, ref_sc, k


def run_case(L_raw, tag, H=32, Hkv=32):
    torch.manual_seed(0)
    B, D, S, M, topk, chunk, window, tau = 1, 128, 64, 2, 32, 64, 512, 3.73
    N = L_raw // S
    L = N * S
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda") * 0.1
    kc = torch.randn(B, N, S, Hkv, D, dtype=torch.bfloat16, device="cuda") * 0.1
    an = torch.randn(B, N, M, H, D, dtype=torch.bfloat16, device="cuda") * 0.1
    sk = torch.randn(B, N, Hkv, D, dtype=torch.bfloat16, device="cuda") * 0.1
    sb = torch.randn(1, dtype=torch.bfloat16, device="cuda")
    ref_idx, ref_sc, k = reference_selection(
        q, kc, an, topk=topk, chunk_size=chunk, window_size=window,
        sm_scale=1 / math.sqrt(D), tau=tau, normalize=True, shared_key=sk, shared_bias=sb,
    )
    try:
        out = mass_retrieval_tilelang_v2(
            q, kc, an, num_key_value_heads=Hkv, topk=topk, chunk_size=chunk,
            window_size=window, q_offset=0, sm_scale=1 / math.sqrt(D),
            score_reduction="logsumexp", lse_temperature=tau, two_stage_backward=True,
            shared_key=sk, shared_bias=sb,
        )
        indices, sel_scores = out[0], out[1]
        print(f"[{tag}] kernel ran: indices{tuple(indices.shape)} sel{tuple(sel_scores.shape)} ref k={k}", flush=True)
        ker_idx = indices.sort(dim=-1).values
        if ker_idx.shape[-1] != k:
            print(f"[{tag}] !! k mismatch kernel {ker_idx.shape[-1]} vs ref {k}")
            return f"{tag}:K_MISMATCH"
        # 只比较有限区间(窗口内才有意义),按 chunk id 集合重合率
        ker_idx_p = ker_idx.permute(0, 1, 3, 2) if ker_idx.dim() == 4 else ker_idx   # 对齐 [B,L,H,k]
        # 只统计"足量可选块"的行:limit>=k+8(避开窗口内无远程块的头部),且 ref 分数有限
        l_ar = torch.arange(L, device=q.device).view(L, 1)
        lim_rows = ((l_ar - window + 1) // chunk) if window > 0 else torch.full_like(l_ar, float(N))
        row_ok = (lim_rows >= k + 8).view(1, L, 1) & torch.isfinite(ref_sc).all(-1)
        if row_ok.sum() == 0:
            inter = float("nan")
        else:
            eq = (ker_idx_p.unsqueeze(-1) == ref_idx.unsqueeze(-2)).any(-1).float()
            inter = eq[row_ok].mean().item()
        print(f"    有效行占比 {row_ok.float().mean().item():.3f}", flush=True)
        sel_p = sel_scores.permute(0, 1, 3, 2) if sel_scores.dim() == 4 else sel_scores
        if row_ok.sum() > 0:
            sc_err = (sel_p.float()[row_ok] - ref_sc.float()[row_ok]).abs().mean().item()
            sc_mag = ref_sc.float()[row_ok].abs().mean().item()
        else:
            sc_err = sc_mag = float("nan")
        sc_mag = ref_sc.float().abs().mean().item()
        print(f"[{tag}] idx集合重合率 {inter:.4f}  score平均误差 {sc_err:.4f} (量级 {sc_mag:.3f})", flush=True)
        verdict = "OK" if inter > 0.98 and sc_err < 0.05 * max(sc_mag, 1e-3) else "WRONG"
        print(f"[{tag}] VERDICT: {verdict}", flush=True)
        return f"{tag}:{verdict}"
    except Exception as exc:
        print(f"[{tag}] kernel FAILED: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
        return f"{tag}:CRASH"


def stage_test(L_raw, tag, H=32, Hkv=32):
    """阶段二分:只测候选 kernel vs 逐头参考(LSE+窗口),消除下游语义干扰。"""
    import math as _m
    torch.manual_seed(0)
    B, D, S, M, topk, chunk, window, tau = 1, 128, 64, 2, 32, 64, 512, 3.73
    N = L_raw // S
    L = N * S
    groups = H // Hkv
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda") * 0.1
    kc = torch.randn(B, N, S, Hkv, D, dtype=torch.bfloat16, device="cuda") * 0.1
    an = torch.randn(B, N, M, H, D, dtype=torch.bfloat16, device="cuda") * 0.1
    sk = torch.randn(B, N, Hkv, D, dtype=torch.bfloat16, device="cuda") * 0.1
    sb = torch.randn(1, dtype=torch.bfloat16, device="cuda")
    sm = 1 / _m.sqrt(D)
    with torch.no_grad():
        mu, bias = _MassSummaryFn.apply(an, kc, sm, groups)
        if mu.shape[2] == M:
            mu = mu.permute(0, 1, 3, 2, 4).contiguous(); bias = bias.permute(0, 1, 3, 2).contiguous()
        shared_mu = sk.repeat_interleave(groups, dim=2).unsqueeze(3).to(mu.dtype).contiguous()
        shared_b = sb.to(dtype=bias.dtype).view(1, 1, 1, 1).expand(B, N, H, 1).contiguous()
        mu = torch.cat([shared_mu, mu[:, :, :, 1:, :]], dim=3).contiguous()
        bias = torch.cat([shared_b, bias[:, :, :, 1:]], dim=3).contiguous()
        # 参考:逐头 candidates
        planes = torch.einsum("blhd,bnhmd->blnhm", q.float(), mu.float()) * sm + bias.float().unsqueeze(1)
        ref = tau * torch.logsumexp(planes / tau, dim=4)          # [B,L,N,H]
        l_ar = torch.arange(L, device="cuda").view(1, L, 1, 1)
        n_ar = torch.arange(N, device="cuda").view(1, 1, N, 1)
        limit = (l_ar - window + 1) // chunk
        ref = ref.masked_fill(n_ar >= limit, float("-inf")).permute(0, 1, 3, 2)  # [B,L,H,N]
    from ops.mass_retrieval_tilelang_v2 import mass_envelope_candidates_kernel
    kern = mass_envelope_candidates_kernel(B, L, N, H, Hkv, M, D, chunk, window, sm, True, tau)
    got = kern(q.contiguous(), mu.contiguous(), bias.contiguous())  # [B,L,Hkv,N]
    mask = torch.isfinite(ref)
    diff = (got.float() - ref.float()).abs()[mask]
    print(f"[{tag}] candidates 有限点 {mask.sum().item()} 平均差 {diff.mean().item():.5f} 最大 {diff.max().item():.5f} ref量级 {ref[mask].abs().mean().item():.3f}", flush=True)
    verdict = "OK" if diff.max().item() < 0.02 * max(ref[mask].abs().mean().item(), 1e-3) else "WRONG"
    print(f"[{tag}] STAGE1(candidates) VERDICT: {verdict}", flush=True)
    return verdict


def consistency_test(L_raw=4096):
    """H=4 与 H=32 共享前4头内容,两跑的对应头选择必须一致(语义无关判据)。"""
    import math as _m
    torch.manual_seed(7)
    B, D, S, M, topk, chunk, window, tau = 1, 128, 64, 2, 32, 64, 512, 3.73
    N = L_raw // S; L = N * S
    sm = 1 / _m.sqrt(D)
    q4 = (torch.randn(B, L, 4, D, dtype=torch.bfloat16, device="cuda") * 0.1)
    mu4 = (torch.randn(B, N, 4, M, D, dtype=torch.bfloat16, device="cuda") * 0.1)
    bias4 = torch.randn(B, N, 4, M, device="cuda", dtype=torch.float32)
    q32 = torch.zeros(B, L, 32, D, dtype=torch.bfloat16, device="cuda"); q32[:, :, :4] = q4
    mu32 = torch.zeros(B, N, 32, M, D, dtype=torch.bfloat16, device="cuda"); mu32[:, :, :4] = mu4
    b32 = torch.zeros(B, N, 32, M, device="cuda", dtype=torch.float32); b32[:, :, :4] = bias4
    from ops.mass_retrieval_tilelang_v2 import mass_envelope_candidates_kernel
    cands = {}
    for tag, (q, mu, bias, H) in [("H4", (q4, mu4, bias4, 4)), ("H32", (q32, mu32, b32, 32))]:
        kern = mass_envelope_candidates_kernel(B, L, N, H, H, M, D, chunk, window, sm, True, tau)
        cands[tag] = kern(q.contiguous(), mu.contiguous(), bias.contiguous())
    c4, c32 = cands["H4"].float(), cands["H32"].float()[:, :, :4]
    fin = torch.isfinite(c4) & torch.isfinite(c32)
    print(f"candidates 前4头: 有限点 {fin.sum().item()} 其中相等(±1e-3) {( (c4-c32).abs()[fin] < 1e-3 ).float().mean().item():.4f} 平均差 {(c4-c32).abs()[fin].mean().item():.6f} NaN占比 {( ~torch.isfinite(c32) ).float().mean().item():.5f}", flush=True)
    from ops.mass_retrieval_tilelang_v2 import mass_bitonic_topk_kernel, mass_sort_selected_indices_kernel
    tops = {}
    for tag, c in [("H4", cands["H4"]), ("H32", cands["H32"])]:
        Hkv = c.shape[2]
        tk = mass_bitonic_topk_kernel(1, L, N, Hkv, topk)
        tops[tag] = tk(c.contiguous())
    t4, t32 = tops["H4"], tops["H32"][:, :, :4]
    print(f"topk原始输出 前4头相同率 {(t4 == t32).float().mean().item():.4f}", flush=True)
    srt4 = mass_sort_selected_indices_kernel(1, L, 4, topk)(t4.contiguous())
    srt32 = mass_sort_selected_indices_kernel(1, L, 32, topk)(tops["H32"].contiguous())[:, :, :4]
    print(f"sort后 前4头相同率 {(srt4 == srt32).float().mean().item():.4f}", flush=True)
    outs = {}
    for tag, (q, mu, bias, H) in [("H4", (q4, mu4, bias4, 4)), ("H32", (q32, mu32, b32, 32))]:
        out = mass_retrieval_tilelang_v2(
            q.contiguous(), torch.randn(B, N, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.1,
            torch.randn(B, N, M, H, D, dtype=torch.bfloat16, device="cuda") * 0.1,
            num_key_value_heads=H, topk=topk, chunk_size=chunk, window_size=window,
            q_offset=0, sm_scale=sm, score_reduction="logsumexp",
            lse_temperature=tau, two_stage_backward=True, shared_key=None, shared_bias=None)
        outs[tag] = (out[0], out[1], H)   # indices [B,L,Hkv,K], scores
    i4, s4, _ = outs["H4"]; i32, s32, _ = outs["H32"]
    idx_match = (i4 == i32[:, :, :4]).float().mean().item()
    sc_diff = (s4.float() - s32[:, :, :4].float()).abs().mean().item()
    print(f"一致性: 前4头 indices 相同率 {idx_match:.4f}, scores 平均差 {sc_diff:.5f}", flush=True)
    print("CONSISTENCY:", "OK" if idx_match > 0.999 and sc_diff < 1e-3 else "BROKEN(H=32 执行期损坏实锤)", flush=True)


def nan_hunt(L_raw=2048):
    """隔离 scores 的 NaN:逐段复现 epilogue,定位产生 NaN 的环节。"""
    import math as _m
    from ops.mass_retrieval_tilelang_v2 import (
        mass_envelope_candidates_kernel, mass_bitonic_topk_kernel,
        mass_sort_selected_indices_kernel)
    from ops.topk_head_softmax import recompute_topk_max_pooling_scores_kernel
    torch.manual_seed(3)
    B, H, D, S, M, topk, chunk, window, tau = 1, 32, 128, 64, 2, 32, 64, 512, 3.73
    N = L_raw // S; L = N * S; groups = 1; sm = 1 / _m.sqrt(D)
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda") * 0.1
    mu = torch.randn(B, N, H, M, D, dtype=torch.bfloat16, device="cuda") * 0.1
    bias = torch.randn(B, N, H, M, device="cuda", dtype=torch.float32)
    ck = mass_envelope_candidates_kernel(B, L, N, H, H, M, D, chunk, window, sm, True, tau)
    tk = mass_bitonic_topk_kernel(B, L, N, H, topk)
    sk = mass_sort_selected_indices_kernel(B, L, H, topk)
    cand = ck(q.contiguous(), mu.contiguous(), bias.contiguous())
    idx = sk(tk(cand).contiguous()).contiguous()          # [B,L,H,topk]
    print("indices NaN/负:", (idx < 0).float().mean().item(), "重复行占比检查", flush=True)
    tangent = torch.arange(M, device="cuda", dtype=torch.int32)
    expanded = (idx.unsqueeze(-1) * M + tangent).reshape(B, L, H, topk * M).contiguous()
    valid = expanded.ge(0); safe = expanded.clamp(0, N * M - 1)
    mu_flat = mu.permute(0, 1, 3, 2, 4).reshape(B, N * M, H, D).contiguous()
    q_grouped = q.reshape(B, L, H, groups, D).contiguous()
    rk = recompute_topk_max_pooling_scores_kernel(B, h_kv=H, groups=groups, head_dim=D,
        topk=topk * M, seq_len=L, s_len=N * M, is_training=True, sm_scale=sm, per_qhead_lmks=True)
    dots = rk(q_grouped, mu_flat, expanded)
    print("dots NaN占比:", torch.isnan(dots.float()).mean().item(), "inf:", torch.isinf(dots.float()).mean().item(), flush=True)
    bias_flat = bias.permute(0, 1, 3, 2).reshape(B, N * M, H).permute(0, 2, 1).contiguous()
    ids_h = safe.permute(0, 2, 1, 3).long()
    sel_bias = torch.gather(bias_flat.unsqueeze(2).expand(B, H, L, N * M), 3, ids_h).permute(0, 2, 1, 3)
    allsc = (dots.reshape(B, L, H, topk * M) + sel_bias).masked_fill(~valid, float("-inf"))
    planes = allsc.reshape(B, L, H, topk, M)
    routes = idx.ge(0)
    scores = tau * torch.logsumexp(planes / tau, dim=-1)
    print("scores NaN:", torch.isnan(scores).mean().item(), "-inf:", (scores == float("-inf")).float().mean().item(), flush=True)
    # 对照:对每个被选 chunk 手工算 LSE 分数,与 kernel 链对比有限点
    ref_pl = torch.einsum("blhd,bnhmd->blnhm", q.float(), mu.float()) * sm + bias.float().unsqueeze(1)
    fin = routes & torch.isfinite(scores)
    sel_idx = idx.clamp(0, N - 1)
    ref_sc = tau * torch.logsumexp(
        torch.gather(ref_pl, 2, sel_idx[..., None].expand(-1, -1, -1, -1, M)).permute(0,1,2,3,4), dim=-1)
    d = (scores[fin] - ref_sc[fin]).abs()
    print(f"选中分数 vs 参考: n={fin.sum().item()} 平均差 {d.mean().item():.5f} 最大 {d.max().item():.5f}", flush=True)


def clean_full(L_raw=2048):
    """干净整链(无 shared、autograd 关),看真实输出健康度。"""
    import math as _m
    torch.manual_seed(3)
    B, H, D, S, M, topk, chunk, window, tau = 1, 32, 128, 64, 2, 32, 64, 512, 3.73
    N = L_raw // S; L = N * S
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda") * 0.1
    kc = torch.randn(B, N, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.1
    an = torch.randn(B, N, M, H, D, dtype=torch.bfloat16, device="cuda") * 0.1
    with torch.no_grad():
        out = mass_retrieval_tilelang_v2(q, kc, an, num_key_value_heads=H, topk=topk,
            chunk_size=chunk, window_size=window, q_offset=0, sm_scale=1/_m.sqrt(D),
            score_reduction="logsumexp", lse_temperature=tau, two_stage_backward=True,
            shared_key=None, shared_bias=None)
    idx, sc = out[0], out[1]
    print(f"整链 indices: 负值率 {(idx<0).float().mean().item():.4f} 范围 [{idx.min().item()},{idx.max().item()}]", flush=True)
    print(f"整链 scores : NaN {torch.isnan(sc.float()).mean().item():.5f} ±inf {torch.isinf(sc.float()).mean().item():.5f}", flush=True)


if __name__ == "__main__":
    import os
    if os.environ.get("CLEANFULL"):
        clean_full(2048)
    elif os.environ.get("NANHUNT"):
        nan_hunt(2048)
    elif os.environ.get("CONSIST"):
        consistency_test(4096)
    elif os.environ.get("STAGE1"):
        r1 = stage_test(4096, "4K-H32", 32, 32)
        r2 = stage_test(4096, "4K-H4", 4, 4)
        r3 = stage_test(8192, "8K-H32", 32, 32)
        print("STAGE1 SUMMARY:", [r1, r2, r3], flush=True)
    elif os.environ.get("CASE200M"):
        results = [run_case(4096, "4K-H4(200M)", H=4, Hkv=4)]
    else:
        results = [run_case(8192, "8K(train)"), run_case(4096, "4K(eval)"), run_case(1024, "1K(short)")]
    print("SUMMARY:", results, flush=True)
