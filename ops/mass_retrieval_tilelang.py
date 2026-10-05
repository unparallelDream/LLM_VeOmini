"""Memory-bounded TileLang retrieval for MASS attention.

The public operation replaces the materializing PyTorch chain
``build_mass_tangent_summary -> upper_envelope_scores -> mass_topk_reference``.
The tangent summary is compact, and the upper-envelope kernel streams planes
directly into Top-K, so no [B,L,N,M,Hq] tensor is allocated.
"""

import math

import torch
import tilelang
import tilelang.language as T


_PASSES = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(out_idx=[2, 3, 4], pass_configs=_PASSES)
def mass_tangent_summary_fwd_kernel(BN, S, H, Hkv, M, D, sm_scale, threads=128):
    """Build all M tangent summaries without expanding K over H or M."""
    dtype, accum_dtype, MP = "bfloat16", "float", 16
    groups = H // Hkv
    anchor_shape = [BN, M, H, D]
    key_shape = [BN, S, Hkv, D]
    mu_shape = [BN, H, M, D]
    bias_shape = [BN, H, M]
    prob_shape = [BN, H, M, S]

    @T.prim_func
    def kernel(
        Anchors: T.Tensor(anchor_shape, dtype),
        Keys: T.Tensor(key_shape, dtype),
        Mu: T.Tensor(mu_shape, dtype),
        Bias: T.Tensor(bias_shape, accum_dtype),
        Prob: T.Tensor(prob_shape, accum_dtype),
    ):
        with T.Kernel(BN, H, threads=threads) as (bn, h):
            a = T.alloc_shared([MP, D], dtype)
            k = T.alloc_shared([S, D], dtype)
            p = T.alloc_shared([MP, S], dtype)
            o = T.alloc_shared([MP, D], dtype)
            logits = T.alloc_fragment([MP, S], accum_dtype)
            out = T.alloc_fragment([MP, D], accum_dtype)
            row_max = T.alloc_fragment([MP], accum_dtype)
            row_sum = T.alloc_fragment([MP], accum_dtype)
            ent_terms = T.alloc_fragment([MP, S], accum_dtype)
            entropy = T.alloc_fragment([MP], accum_dtype)

            for m, d in T.Parallel(MP, D):
                a[m, d] = T.if_then_else(m < M, Anchors[bn, m, h, d], T.Cast(dtype, 0.0))
            for s, d in T.Parallel(S, D):
                k[s, d] = Keys[bn, s, h // groups, d]

            T.clear(logits)
            T.gemm(a, k, logits, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            for m, s in T.Parallel(MP, S):
                logits[m, s] = T.if_then_else(
                    (m >= M) or (s == S - 1), -T.infinity(accum_dtype), logits[m, s] * sm_scale
                )
            T.fill(row_max, -T.infinity(accum_dtype))
            T.reduce_max(logits, row_max, dim=1, clear=True)
            for m, s in T.Parallel(MP, S):
                logits[m, s] = T.if_then_else(
                    m < M, T.exp2((logits[m, s] - row_max[m]) * 1.44269504), 0.0
                )
            T.fill(row_sum, 0.0)
            T.reduce_sum(logits, row_sum, dim=1, clear=True)
            for m, s in T.Parallel(MP, S):
                logits[m, s] = T.if_then_else(m < M, logits[m, s] / row_sum[m], 0.0)
                ent_terms[m, s] = T.if_then_else(
                    logits[m, s] > 0.0, logits[m, s] * T.log2(logits[m, s]) * 0.6931471805599453, 0.0
                )
            T.fill(entropy, 0.0)
            T.reduce_sum(ent_terms, entropy, dim=1, clear=True)
            T.copy(logits, p)
            T.clear(out)
            T.gemm(p, k, out, policy=T.GemmWarpPolicy.FullRow)
            T.copy(out, o)

            for m, s in T.Parallel(M, S):
                Prob[bn, h, m, s] = T.Cast(accum_dtype, p[m, s])
            for m in T.Parallel(M):
                Bias[bn, h, m] = -entropy[m]
            for m, d in T.Parallel(M, D):
                Mu[bn, h, m, d] = o[m, d]
    return kernel


@tilelang.jit(out_idx=[6, 7], pass_configs=_PASSES)
def mass_tangent_summary_bwd_kernel(BN, S, H, Hkv, M, D, sm_scale, threads=128):
    dtype, accum_dtype, MP = "bfloat16", "float", 16
    groups = H // Hkv
    anchor_shape = [BN, M, H, D]
    key_shape = [BN, S, Hkv, D]
    mu_shape = [BN, H, M, D]
    bias_shape = [BN, H, M]
    prob_shape = [BN, H, M, S]
    grad_key_shape = [BN, S, H, D]

    @T.prim_func
    def kernel(
        Anchors: T.Tensor(anchor_shape, dtype),
        Keys: T.Tensor(key_shape, dtype),
        Prob: T.Tensor(prob_shape, accum_dtype),
        Bias: T.Tensor(bias_shape, accum_dtype),
        GradMu: T.Tensor(mu_shape, dtype),
        GradBias: T.Tensor(bias_shape, accum_dtype),
        GradAnchors: T.Tensor(anchor_shape, accum_dtype),
        GradKeysExpanded: T.Tensor(grad_key_shape, accum_dtype),
    ):
        with T.Kernel(BN, H, threads=threads) as (bn, h):
            a = T.alloc_shared([MP, D], dtype)
            k = T.alloc_shared([S, D], dtype)
            gm = T.alloc_shared([MP, D], dtype)
            p = T.alloc_shared([MP, S], accum_dtype)
            dl_shared = T.alloc_shared([MP, S], dtype)
            dp = T.alloc_fragment([MP, S], accum_dtype)
            delta_terms = T.alloc_fragment([MP, S], accum_dtype)
            delta = T.alloc_fragment([MP], accum_dtype)
            dl = T.alloc_fragment([MP, S], accum_dtype)
            ga = T.alloc_fragment([MP, D], accum_dtype)

            for m, d in T.Parallel(MP, D):
                if m < M:
                    a[m, d] = Anchors[bn, m, h, d]
                    gm[m, d] = GradMu[bn, h, m, d]
                else:
                    a[m, d] = T.Cast(dtype, 0.0)
                    gm[m, d] = T.Cast(dtype, 0.0)
            for s, d in T.Parallel(S, D):
                k[s, d] = Keys[bn, s, h // groups, d]
            for m, s in T.Parallel(MP, S):
                p[m, s] = T.if_then_else(m < M, Prob[bn, h, m, s], 0.0)

            T.clear(dp)
            T.gemm(gm, k, dp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            for m, s in T.Parallel(MP, S):
                delta_terms[m, s] = p[m, s] * dp[m, s]
            T.fill(delta, 0.0)
            T.reduce_sum(delta_terms, delta, dim=1, clear=True)
            for m, s in T.Parallel(MP, S):
                if (m < M) and (p[m, s] > 0.0):
                    log_p = T.log2(p[m, s]) * 0.6931471805599453
                    dl[m, s] = p[m, s] * (
                        dp[m, s] - delta[m] - GradBias[bn, h, m] * (log_p + Bias[bn, h, m])
                    )
                else:
                    dl[m, s] = 0.0
            T.copy(dl, dl_shared)
            T.clear(ga)
            T.gemm(dl_shared, k, ga, policy=T.GemmWarpPolicy.FullRow)
            for m, d in T.Parallel(M, D):
                GradAnchors[bn, m, h, d] = ga[m, d] * sm_scale
            for s, d in T.Parallel(S, D):
                value = T.alloc_local([1], accum_dtype)
                value[0] = 0.0
                for m in T.serial(M):
                    value[0] += (
                        sm_scale * T.Cast(accum_dtype, dl_shared[m, s]) * T.Cast(accum_dtype, a[m, d])
                        + p[m, s] * T.Cast(accum_dtype, gm[m, d])
                    )
                GradKeysExpanded[bn, s, h, d] = value[0]
    return kernel


class _MassSummaryFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, anchors, keys, sm_scale, groups):
        B, N, M, H, D = anchors.shape
        S, Hkv = keys.shape[2:4]
        a = anchors.reshape(B * N, M, H, D).contiguous()
        k = keys.reshape(B * N, S, Hkv, D).contiguous()
        fwd = mass_tangent_summary_fwd_kernel(B * N, S, H, Hkv, M, D, sm_scale)
        mu, bias, prob = fwd(a, k)
        ctx.save_for_backward(a, k, prob, bias)
        ctx.shape = (B, N, M, H, Hkv, D, S)
        ctx.sm_scale, ctx.groups = sm_scale, groups
        return mu.reshape(B, N, H, M, D), bias.reshape(B, N, H, M)

    @staticmethod
    def backward(ctx, grad_mu, grad_bias):
        a, k, prob, bias = ctx.saved_tensors
        B, N, M, H, Hkv, D, S = ctx.shape
        if grad_mu is None:
            grad_mu = torch.zeros((B, N, H, M, D), device=a.device, dtype=a.dtype)
        if grad_bias is None:
            grad_bias = torch.zeros((B, N, H, M), device=a.device, dtype=torch.float32)
        bwd = mass_tangent_summary_bwd_kernel(B * N, S, H, Hkv, M, D, ctx.sm_scale)
        da, dk_expanded = bwd(
            a,
            k,
            prob,
            bias,
            grad_mu.reshape(B * N, H, M, D).to(a.dtype).contiguous(),
            grad_bias.reshape(B * N, H, M).float().contiguous(),
        )
        dk = dk_expanded.to(k.dtype).reshape(B, N, S, Hkv, ctx.groups, D).sum(dim=4)
        return da.to(a.dtype).reshape(B, N, M, H, D), dk, None, None


@tilelang.jit(out_idx=[3, 4, 5], pass_configs=_PASSES)
def mass_envelope_topk_fwd_kernel(
    B, L, N, H, Hkv, M, D, topk, chunk_size, window_size, sm_scale, block_l=64, threads=64
):
    """Stream upper-envelope planes directly into per-KV-head Top-K."""
    dtype, accum_dtype, idx_dtype, MP = "bfloat16", "float", "int32", 16
    groups = H // Hkv
    q_shape = [B, L, H, D]
    mu_shape = [B, N, H, M, D]
    bias_shape = [B, N, H, M]
    index_shape = [B, L, Hkv, topk]
    score_shape = [B, L, H, topk]

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, dtype),
        Mu: T.Tensor(mu_shape, dtype),
        Bias: T.Tensor(bias_shape, accum_dtype),
        Indices: T.Tensor(index_shape, idx_dtype),
        Scores: T.Tensor(score_shape, accum_dtype),
        Winners: T.Tensor(score_shape, idx_dtype),
    ):
        with T.Kernel(tilelang.cdiv(L, block_l), Hkv, B, threads=threads) as (bx, by, bz):
            base_l, hkv, batch = bx * block_l, by, bz
            tx = T.get_thread_binding()
            q = T.alloc_shared([block_l, D], dtype)
            mu = T.alloc_shared([MP, D], dtype)
            plane_shared = T.alloc_shared([block_l, MP], accum_dtype)
            planes = T.alloc_fragment([block_l, MP], accum_dtype)
            top_scores = T.alloc_local([topk], accum_dtype)
            top_indices = T.alloc_local([topk], idx_dtype)
            candidate = T.alloc_local([1], accum_dtype)
            T.fill(top_scores, -T.infinity(accum_dtype))
            T.fill(top_indices, -1)

            global_end = base_l + block_l
            block_limit = T.if_then_else(
                window_size > 0,
                (global_end - window_size) // chunk_size,
                (global_end - 1) // chunk_size,
            )
            loop_limit = T.min(N, T.max(block_limit, 0))
            for n in T.serial(loop_limit):
                candidate[0] = -T.infinity(accum_dtype)
                for g in T.serial(groups):
                    h = hkv * groups + g
                    for l, d in T.Parallel(block_l, D):
                        q[l, d] = T.if_then_else(base_l + l < L, Q[batch, base_l + l, h, d], T.Cast(dtype, 0.0))
                    for m, d in T.Parallel(MP, D):
                        mu[m, d] = T.if_then_else(m < M, Mu[batch, n, h, m, d], T.Cast(dtype, 0.0))
                    T.clear(planes)
                    T.gemm(q, mu, planes, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(planes, plane_shared)
                    T.sync_threads()
                    if (tx < block_l) and (base_l + tx < L):
                        tq = base_l + tx
                        query_limit = T.if_then_else(
                            window_size > 0,
                            (tq - window_size + 1) // chunk_size,
                            tq // chunk_size,
                        )
                        if n < query_limit:
                            group_best = T.alloc_local([1], accum_dtype)
                            group_best[0] = -T.infinity(accum_dtype)
                            for m in T.serial(M):
                                group_best[0] = T.max(
                                    group_best[0], plane_shared[tx, m] * sm_scale + Bias[batch, n, h, m]
                                )
                            candidate[0] = T.max(candidate[0], group_best[0])
                    T.sync_threads()

                if (tx < block_l) and (base_l + tx < L) and (candidate[0] > top_scores[topk - 1]):
                    moving = T.alloc_var("bool")
                    moving = True
                    for kk in T.serial(topk):
                        pos = topk - 1 - kk
                        if moving:
                            if (pos > 0) and (candidate[0] > top_scores[pos - 1]):
                                top_scores[pos] = top_scores[pos - 1]
                                top_indices[pos] = top_indices[pos - 1]
                            else:
                                top_scores[pos] = candidate[0]
                                top_indices[pos] = n
                                moving = False

            if (tx < block_l) and (base_l + tx < L):
                # Sort the small candidate list by chunk id for sparse V-attention.
                for unused in T.serial(topk):
                    for pos in T.serial(topk - 1):
                        left = T.if_then_else(top_indices[pos] < 0, N, top_indices[pos])
                        right = T.if_then_else(top_indices[pos + 1] < 0, N, top_indices[pos + 1])
                        if left > right:
                            swap = top_indices[pos]
                            top_indices[pos] = top_indices[pos + 1]
                            top_indices[pos + 1] = swap
                tq = base_l + tx
                for pos in T.serial(topk):
                    Indices[batch, tq, hkv, pos] = top_indices[pos]

                # Different queries select different chunks.  Recompute only K
                # selected planes scalarly inside this same launch (K << N).
                for g in T.serial(groups):
                    h = hkv * groups + g
                    for pos in T.serial(topk):
                        n = top_indices[pos]
                        best = T.alloc_local([1], accum_dtype)
                        winner = T.alloc_local([1], idx_dtype)
                        best[0], winner[0] = -T.infinity(accum_dtype), 0
                        if n >= 0:
                            for m in T.serial(M):
                                dot = T.alloc_local([1], accum_dtype)
                                dot[0] = 0.0
                                for d in T.serial(D):
                                    dot[0] += T.Cast(accum_dtype, Q[batch, tq, h, d]) * T.Cast(
                                        accum_dtype, Mu[batch, n, h, m, d]
                                    )
                                value = dot[0] * sm_scale + Bias[batch, n, h, m]
                                if value > best[0]:
                                    best[0], winner[0] = value, m
                        Scores[batch, tq, h, pos] = best[0]
                        Winners[batch, tq, h, pos] = winner[0]
    return kernel


class _MassEnvelopeTopKFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, queries, mu, bias, kernel, groups, sm_scale):
        indices, scores, winners = kernel(queries.contiguous(), mu.contiguous(), bias.contiguous())
        ctx.save_for_backward(queries, mu, indices, winners)
        ctx.groups, ctx.sm_scale = groups, sm_scale
        ctx.mark_non_differentiable(indices)
        return indices, scores

    @staticmethod
    def backward(ctx, unused_grad_indices, grad_scores):
        queries, mu, indices, winners = ctx.saved_tensors
        B, L, H, D = queries.shape
        N, M, K = mu.shape[1], mu.shape[3], indices.shape[-1]
        grad_q = torch.zeros_like(queries, dtype=torch.float32)
        grad_mu_rows = torch.zeros((B * H * N * M, D), device=mu.device, dtype=torch.float32)
        grad_b_rows = torch.zeros((B * H * N * M,), device=mu.device, dtype=torch.float32)
        mu_heads = mu.permute(0, 2, 1, 3, 4).reshape(B, H, N * M, D)
        row_base = torch.arange(B * H, device=mu.device).reshape(B, H, 1, 1) * (N * M)

        # Larger blocks amortize gather/index_add launch overhead while keeping
        # the selected-only backward workspace below 1 GiB at microbatch=2.
        query_block = 1024
        for start in range(0, L, query_block):
            end = min(start + query_block, L)
            qn = end - start
            idx = indices[:, start:end].repeat_interleave(ctx.groups, dim=2).permute(0, 2, 1, 3).long()
            win = winners[:, start:end].permute(0, 2, 1, 3).long()
            valid = idx.ge(0) & idx.lt(N)
            flat = idx.clamp(0, N - 1) * M + win.clamp(0, M - 1)
            selected_mu = torch.gather(
                mu_heads.unsqueeze(2).expand(B, H, qn, N * M, D),
                3,
                flat.unsqueeze(-1).expand(B, H, qn, K, D),
            ).float()
            grad = grad_scores[:, start:end].permute(0, 2, 1, 3).float().masked_fill(~valid, 0.0)
            q = queries[:, start:end].permute(0, 2, 1, 3).float()
            grad_q[:, start:end] = ((grad.unsqueeze(-1) * selected_mu).sum(3) * ctx.sm_scale).permute(0, 2, 1, 3)
            rows = (row_base + flat).reshape(-1)
            valid_flat = valid.reshape(-1)
            values = (grad.unsqueeze(-1) * q.unsqueeze(3) * ctx.sm_scale).reshape(-1, D)
            grad_mu_rows.index_add_(0, rows[valid_flat], values[valid_flat])
            grad_b_rows.index_add_(0, rows[valid_flat], grad.reshape(-1)[valid_flat])

        grad_mu = grad_mu_rows.reshape(B, H, N, M, D).permute(0, 2, 1, 3, 4)
        grad_bias = grad_b_rows.reshape(B, H, N, M).permute(0, 2, 1, 3)
        return grad_q.to(queries.dtype), grad_mu.to(mu.dtype), grad_bias, None, None, None


_KERNEL_CACHE = {}


def mass_retrieval_tilelang(
    queries,
    key_chunks,
    anchors,
    *,
    num_key_value_heads,
    topk,
    chunk_size,
    window_size,
    q_offset=0,
    sm_scale=None,
):
    """Return [B,L,Hkv,K] indices and [B,L,Hq,K] selected scores."""
    if q_offset != 0:
        raise NotImplementedError("MASS TileLang retrieval currently supports full-sequence training only")
    if any(t.dtype != torch.bfloat16 for t in (queries, key_chunks, anchors)):
        raise TypeError("MASS TileLang retrieval requires BF16 queries, keys and anchors")
    B, L, H, D = queries.shape
    Bk, N, S, Hkv, Dk = key_chunks.shape
    Ba, Na, M, Ha, Da = anchors.shape
    if (Bk, Ba, Na, Ha, Da, Dk, Hkv, S) != (B, B, N, H, D, D, num_key_value_heads, chunk_size):
        raise ValueError("incompatible MASS retrieval tensor shapes")
    if H % Hkv or M > 16:
        raise ValueError("Hq must divide Hkv and MASS supports at most 16 anchors")
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    groups = H // Hkv
    mu, bias = _MassSummaryFn.apply(anchors, key_chunks, sm_scale, groups)
    cache_key = (B, L, N, H, Hkv, M, D, topk, chunk_size, window_size, sm_scale)
    if cache_key not in _KERNEL_CACHE:
        _KERNEL_CACHE[cache_key] = mass_envelope_topk_fwd_kernel(
            B, L, N, H, Hkv, M, D, topk, chunk_size, window_size, sm_scale
        )
    return _MassEnvelopeTopKFn.apply(queries, mu, bias, _KERNEL_CACHE[cache_key], groups, sm_scale)
