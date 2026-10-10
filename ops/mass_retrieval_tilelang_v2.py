"""High-throughput multi-kernel TileLang retrieval for MASS attention.

The operation deliberately uses several GPU kernels behind one autograd op.
It keeps only the compact ``[B, L, Hkv, N]`` envelope buffer and never
materializes the much larger ``[B, L, N, M, Hq]`` plane tensor.
"""

import math
import os

import torch
import tilelang
import tilelang.language as T

from ops.mass_retrieval_tilelang import _MassSummaryFn
from ops.topk_head_softmax import recompute_topk_max_pooling_scores_kernel


_PASSES = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}

_SORT_PASSES = {
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(out_idx=[3], pass_configs=_PASSES)
def mass_envelope_candidates_kernel(
    B,
    L,
    N,
    H,
    Hkv,
    M,
    D,
    chunk_size,
    window_size,
    sm_scale,
    use_logsumexp,
    lse_temperature,
    block_l=32,
    block_n=16,
    threads=128,
):
    """Compute one upper-envelope score per query/KV-head/chunk.

    Tiling N*M together turns the many tiny ``L x M`` products in v1 into
    tensor-core-friendly ``block_l x (block_n*M)`` GEMMs.
    """
    dtype, accum_dtype = "bfloat16", "float"
    groups = H // Hkv
    mp = M
    q_shape = [B, L, H, D]
    mu_shape = [B, N, H, M, D]
    bias_shape = [B, N, H, M]
    candidate_shape = [B, L, Hkv, N]

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, dtype),
        Mu: T.Tensor(mu_shape, dtype),
        Bias: T.Tensor(bias_shape, accum_dtype),
        Candidates: T.Tensor(candidate_shape, accum_dtype),
    ):
        with T.Kernel(tilelang.cdiv(L, block_l), Hkv, B, threads=threads) as (bx, by, bz):
            base_l, hkv, batch = bx * block_l, by, bz
            q_all = T.alloc_shared([groups, block_l, D], dtype)
            q_group = T.alloc_shared([block_l, D], dtype)
            mu = T.alloc_shared([block_n * mp, D], dtype)
            planes_shared = T.alloc_shared([block_l, block_n * mp], accum_dtype)
            best_shared = T.alloc_shared([block_l, block_n], accum_dtype)
            planes = T.alloc_fragment([block_l, block_n * mp], accum_dtype)

            for g, l, d in T.Parallel(groups, block_l, D):
                tq = base_l + l
                q_all[g, l, d] = T.if_then_else(
                    tq < L, Q[batch, tq, hkv * groups + g, d], T.Cast(dtype, 0.0)
                )

            for n_block in T.serial(tilelang.cdiv(N, block_n)):
                base_n = n_block * block_n
                T.fill(best_shared, -T.infinity(accum_dtype))
                for g in T.serial(groups):
                    h = hkv * groups + g
                    for l, d in T.Parallel(block_l, D):
                        q_group[l, d] = q_all[g, l, d]
                    for nn, m, d in T.Parallel(block_n, mp, D):
                        n = base_n + nn
                        mu[nn * mp + m, d] = T.if_then_else(
                            n < N, Mu[batch, n, h, m, d], T.Cast(dtype, 0.0)
                        )
                    T.sync_threads()
                    T.clear(planes)
                    T.gemm(q_group, mu, planes, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(planes, planes_shared)
                    T.sync_threads()
                    for l, nn in T.Parallel(block_l, block_n):
                        tq, n = base_l + l, base_n + nn
                        if (tq < L) and (n < N):
                            group_max = T.alloc_local([1], accum_dtype)
                            group_score = T.alloc_local([1], accum_dtype)
                            group_max[0] = -T.infinity(accum_dtype)
                            for m in T.serial(mp):
                                value = planes_shared[l, nn * mp + m] * sm_scale + Bias[batch, n, h, m]
                                group_max[0] = T.max(group_max[0], value)
                            if use_logsumexp:
                                exp_sum = T.alloc_local([1], accum_dtype)
                                exp_sum[0] = 0.0
                                for m in T.serial(mp):
                                    value = planes_shared[l, nn * mp + m] * sm_scale + Bias[batch, n, h, m]
                                    exp_sum[0] += T.exp((value - group_max[0]) / lse_temperature)
                                group_score[0] = group_max[0] + lse_temperature * T.log(exp_sum[0])
                            else:
                                group_score[0] = group_max[0]
                            best_shared[l, nn] = T.max(best_shared[l, nn], group_score[0])
                    T.sync_threads()

                for l, nn in T.Parallel(block_l, block_n):
                    tq, n = base_l + l, base_n + nn
                    if (tq < L) and (n < N):
                        query_limit = T.if_then_else(
                            window_size > 0,
                            (tq - window_size + 1) // chunk_size,
                            tq // chunk_size,
                        )
                        Candidates[batch, tq, hkv, n] = T.if_then_else(
                            n < query_limit, best_shared[l, nn], -T.infinity(accum_dtype)
                        )
    return kernel


@tilelang.jit(out_idx=[1], pass_configs=_SORT_PASSES)
def mass_bitonic_topk_kernel(B, L, N, Hkv, topk, block_l=8, threads=128):
    """Fully sort the small N-vector in shared memory and retain its Top-K."""
    accum_dtype, idx_dtype = "float", "int32"
    npad = 1 << (N - 1).bit_length()
    stages = int(math.log2(npad))
    candidate_shape = [B, L, Hkv, N]
    index_shape = [B, L, Hkv, topk]

    @T.prim_func
    def kernel(
        Candidates: T.Tensor(candidate_shape, accum_dtype),
        Indices: T.Tensor(index_shape, idx_dtype),
    ):
        with T.Kernel(tilelang.cdiv(L, block_l), Hkv, B, threads=threads) as (bx, by, bz):
            base_l, hkv, batch = bx * block_l, by, bz
            score_shared = T.alloc_shared([block_l, npad], accum_dtype)
            index_shared = T.alloc_shared([block_l, npad], idx_dtype)

            for l, n in T.Parallel(block_l, npad):
                tq = base_l + l
                if (tq < L) and (n < N):
                    value = Candidates[batch, tq, hkv, n]
                    score_shared[l, n] = value
                    index_shared[l, n] = T.if_then_else(value > -T.infinity(accum_dtype), n, -1)
                else:
                    score_shared[l, n] = -T.infinity(accum_dtype)
                    index_shared[l, n] = -1
            T.sync_threads()

            k_step = T.alloc_var(idx_dtype)
            j_step = T.alloc_var(idx_dtype)
            score_i = T.alloc_var(accum_dtype)
            score_j = T.alloc_var(accum_dtype)
            index_i = T.alloc_var(idx_dtype)
            index_j = T.alloc_var(idx_dtype)
            k_step = 2
            for unused in T.serial(stages):
                j_step = k_step // 2
                while j_step > 0:
                    for l, i in T.Parallel(block_l, npad):
                        tq = base_l + l
                        ixj = i ^ j_step
                        if (tq < L) and (ixj > i) and (ixj < npad):
                            score_i, score_j = score_shared[l, i], score_shared[l, ixj]
                            index_i, index_j = index_shared[l, i], index_shared[l, ixj]
                            ascending = (i & k_step) == 0
                            swap = T.alloc_var("bool")
                            swap = T.if_then_else(ascending, score_i > score_j, score_i < score_j)
                            if swap:
                                score_shared[l, i], score_shared[l, ixj] = score_j, score_i
                                index_shared[l, i], index_shared[l, ixj] = index_j, index_i
                    T.sync_threads()
                    j_step = j_step // 2
                k_step = k_step * 2

            for l, k in T.Parallel(block_l, topk):
                tq = base_l + l
                # topk may exceed npad (short sequences: N=16 -> npad=16 <
                # topk=32); emit -1 for the missing slots instead of reading
                # index_shared at negative offsets (illegal shared-memory access).
                if tq < L:
                    Indices[batch, tq, hkv, k] = T.if_then_else(
                        k < npad, index_shared[l, npad - 1 - k], -1
                    )
    return kernel


@tilelang.jit(out_idx=[1], pass_configs=_SORT_PASSES)
def mass_sort_selected_indices_kernel(B, L, Hkv, topk, block_l=16, threads=64):
    """Sort selected chunk ids for the downstream sparse-V attention kernel."""
    idx_dtype = "int32"
    stages = int(math.log2(topk))
    index_shape = [B, L, Hkv, topk]
    invalid_key = 0x7FFFFFFF

    @T.prim_func
    def kernel(
        IndicesIn: T.Tensor(index_shape, idx_dtype),
        IndicesOut: T.Tensor(index_shape, idx_dtype),
    ):
        with T.Kernel(tilelang.cdiv(L, block_l), Hkv, B, threads=threads) as (bx, by, bz):
            base_l, hkv, batch = bx * block_l, by, bz
            idx_shared = T.alloc_shared([block_l, topk], idx_dtype)
            for l, k in T.Parallel(block_l, topk):
                tq = base_l + l
                idx_shared[l, k] = T.if_then_else(tq < L, IndicesIn[batch, tq, hkv, k], -1)
            T.sync_threads()

            k_step = T.alloc_var(idx_dtype)
            j_step = T.alloc_var(idx_dtype)
            val_i = T.alloc_var(idx_dtype)
            val_j = T.alloc_var(idx_dtype)
            key_i = T.alloc_var(idx_dtype)
            key_j = T.alloc_var(idx_dtype)
            k_step = 2
            for unused in T.serial(stages):
                j_step = k_step // 2
                while j_step > 0:
                    for l, i in T.Parallel(block_l, topk):
                        tq = base_l + l
                        ixj = i ^ j_step
                        if (tq < L) and (ixj > i) and (ixj < topk):
                            val_i, val_j = idx_shared[l, i], idx_shared[l, ixj]
                            key_i = T.if_then_else(val_i >= 0, val_i, invalid_key)
                            key_j = T.if_then_else(val_j >= 0, val_j, invalid_key)
                            ascending = (i & k_step) == 0
                            swap = T.alloc_var("bool")
                            swap = T.if_then_else(ascending, key_i > key_j, key_i < key_j)
                            if swap:
                                idx_shared[l, i], idx_shared[l, ixj] = val_j, val_i
                    T.sync_threads()
                    j_step = j_step // 2
                k_step = k_step * 2

            for l, k in T.Parallel(block_l, topk):
                tq = base_l + l
                if tq < L:
                    IndicesOut[batch, tq, hkv, k] = idx_shared[l, k]
    return kernel


@tilelang.jit(out_idx=[4, 5], pass_configs=_PASSES)
def mass_selected_scores_kernel(B, L, N, H, Hkv, M, D, topk, sm_scale, threads=128):
    """Recompute K selected scores in parallel, retaining the winning tangent."""
    dtype, accum_dtype, idx_dtype = "bfloat16", "float", "int32"
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
        with T.Kernel(L, Hkv, B, threads=threads) as (tq, hkv, batch):
            q_shared = T.alloc_shared([groups, D], dtype)
            for g, d in T.Parallel(groups, D):
                q_shared[g, d] = Q[batch, tq, hkv * groups + g, d]
            T.sync_threads()
            for g, k in T.Parallel(groups, topk):
                h = hkv * groups + g
                n = Indices[batch, tq, hkv, k]
                best = T.alloc_local([1], accum_dtype)
                winner = T.alloc_local([1], idx_dtype)
                best[0], winner[0] = -T.infinity(accum_dtype), 0
                if (n >= 0) and (n < N):
                    for m in T.serial(M):
                        dot = T.alloc_local([1], accum_dtype)
                        dot[0] = 0.0
                        for d in T.serial(D):
                            dot[0] += T.Cast(accum_dtype, q_shared[g, d]) * T.Cast(
                                accum_dtype, Mu[batch, n, h, m, d]
                            )
                        value = dot[0] * sm_scale + Bias[batch, n, h, m]
                        if value > best[0]:
                            best[0], winner[0] = value, m
                Scores[batch, tq, h, k] = best[0]
                Winners[batch, tq, h, k] = winner[0]
    return kernel


@tilelang.jit(out_idx=[3], pass_configs=_PASSES)
def mass_selected_dq_kernel(B, L, N, H, Hkv, M, D, topk, sm_scale, threads=128):
    dtype, accum_dtype, idx_dtype = "bfloat16", "float", "int32"
    groups = H // Hkv
    mu_shape = [B, N, H, M, D]
    index_shape = [B, L, Hkv, topk]
    plane_grad_shape = [B, L, H, topk, M]
    q_shape = [B, L, H, D]

    @T.prim_func
    def kernel(
        Mu: T.Tensor(mu_shape, dtype),
        Indices: T.Tensor(index_shape, idx_dtype),
        GradPlanes: T.Tensor(plane_grad_shape, accum_dtype),
        GradQ: T.Tensor(q_shape, accum_dtype),
    ):
        with T.Kernel(L, Hkv, B, threads=threads) as (tq, hkv, batch):
            partial = T.alloc_shared([groups, M, D], accum_dtype)
            for g, m, d in T.Parallel(groups, M, D):
                h = hkv * groups + g
                value = T.alloc_local([1], accum_dtype)
                value[0] = 0.0
                for k in T.serial(topk):
                    n = Indices[batch, tq, hkv, k]
                    if (n >= 0) and (n < N):
                        value[0] += GradPlanes[batch, tq, h, k, m] * T.Cast(
                            accum_dtype, Mu[batch, n, h, m, d]
                        )
                partial[g, m, d] = value[0]
            T.sync_threads()
            for g, d in T.Parallel(groups, D):
                value = T.alloc_local([1], accum_dtype)
                value[0] = 0.0
                for m in T.serial(M):
                    value[0] += partial[g, m, d]
                GradQ[batch, tq, hkv * groups + g, d] = value[0] * sm_scale
    return kernel


@tilelang.jit(out_idx=[3], pass_configs=_PASSES)
def mass_selected_dq_gemm_kernel(
    B, L, N, H, Hkv, M, D, topk, sm_scale, block_l=16, block_j=16, threads=128
):
    """Tensor-core dQ via a block-diagonal gather-GEMM.

    GradQ[b,l,h,:] = sm_scale * Σ_{k,m} GradPlanes[b,l,h,k,m] · Mu[b,idx[l,hkv,k],h,m,:]

    Same gather-materialization family as the forward's
    recompute_topk_max_pooling_scores_kernel: every (query-row, selected j)
    Mu row is staged into shared memory, and the per-row reduction becomes a
    block-diagonal GEMM with weights W[l, l'*block_j+j] = grad[l, j] if
    l'==l else 0.  The GEMM carries a block_l-fold redundancy, but tensor
    cores make it far cheaper than the scalar per-query kernel above, which
    profiled at 422ms/call (L=8322, N=130, K=32, M=2, D=128) against a ~3ms
    memory roofline.  Env MASS_DQ_LEGACY=1 restores the scalar kernel.
    """
    dtype, accum_dtype, idx_dtype = "bfloat16", "float", "int32"
    groups = H // Hkv
    j_total = topk * M
    mu_shape = [B, N, H, M, D]
    index_shape = [B, L, Hkv, topk]
    plane_grad_shape = [B, L, H, topk, M]
    q_shape = [B, L, H, D]

    @T.prim_func
    def kernel(
        Mu: T.Tensor(mu_shape, dtype),
        Indices: T.Tensor(index_shape, idx_dtype),
        GradPlanes: T.Tensor(plane_grad_shape, accum_dtype),
        GradQ: T.Tensor(q_shape, accum_dtype),
    ):
        with T.Kernel(tilelang.cdiv(L, block_l), Hkv, B, threads=threads) as (bx, hkv, batch):
            base_l = bx * block_l
            g_shared = T.alloc_shared([block_l * block_j, D], dtype)
            w_shared = T.alloc_shared([block_l, block_l * block_j], dtype)
            acc = T.alloc_fragment([block_l, D], accum_dtype)
            for g in T.serial(groups):
                h = hkv * groups + g
                T.clear(acc)
                for j_tile in T.serial((j_total + block_j - 1) // block_j):
                    base_j = j_tile * block_j
                    for l, jj, d in T.Parallel(block_l, block_j, D):
                        tq = base_l + l
                        j = base_j + jj
                        row = l * block_j + jj
                        if (tq < L) and (j < j_total):
                            n = Indices[batch, tq, hkv, j // M]
                            if (n >= 0) and (n < N):
                                g_shared[row, d] = Mu[batch, n, h, j % M, d]
                            else:
                                g_shared[row, d] = T.Cast(dtype, 0.0)
                        else:
                            g_shared[row, d] = T.Cast(dtype, 0.0)
                    for l, l2, jj in T.Parallel(block_l, block_l, block_j):
                        tq = base_l + l
                        j = base_j + jj
                        col = l2 * block_j + jj
                        if (l2 == l) and (tq < L) and (j < j_total):
                            w_shared[l, col] = T.Cast(
                                dtype, GradPlanes[batch, tq, h, j // M, j % M]
                            )
                        else:
                            w_shared[l, col] = T.Cast(dtype, 0.0)
                    T.sync_threads()
                    T.gemm(
                        w_shared,
                        g_shared,
                        acc,
                        transpose_B=False,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.sync_threads()
                for l, d in T.Parallel(block_l, D):
                    tq = base_l + l
                    if tq < L:
                        GradQ[batch, tq, h, d] = acc[l, d] * sm_scale
    return kernel


@tilelang.jit(pass_configs=_PASSES)
def mass_selected_dmu_dbias_kernel(
    B, L, N, H, Hkv, M, D, topk, sm_scale, block_l=16, threads=256
):
    """Accumulate selected gradients, coalescing equal rows across nearby queries."""
    dtype, accum_dtype, idx_dtype = "bfloat16", "float", "int32"
    groups = H // Hkv
    q_shape = [B, L, H, D]
    mu_shape = [B, N, H, M, D]
    bias_shape = [B, N, H, M]
    index_shape = [B, L, Hkv, topk]
    plane_grad_shape = [B, L, H, topk, M]

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, dtype),
        Indices: T.Tensor(index_shape, idx_dtype),
        GradPlanes: T.Tensor(plane_grad_shape, accum_dtype),
        GradMu: T.Tensor(mu_shape, accum_dtype),
        GradBias: T.Tensor(bias_shape, accum_dtype),
    ):
        with T.Kernel(tilelang.cdiv(L, block_l), Hkv * M, B, threads=threads) as (
            bx,
            hm,
            batch,
        ):
            base_l = bx * block_l
            hkv, tangent = hm // M, hm % M
            for g, k, d in T.Parallel(groups, topk, D):
                h = hkv * groups + g
                prev_n = T.alloc_local([1], idx_dtype)
                accum = T.alloc_local([1], accum_dtype)
                accum_bias = T.alloc_local([1], accum_dtype)
                prev_n[0], accum[0], accum_bias[0] = -1, 0.0, 0.0
                for l in T.serial(block_l):
                    tq = base_l + l
                    n = T.if_then_else(tq < L, Indices[batch, tq, hkv, k], -1)
                    if n != prev_n[0]:
                        if (prev_n[0] >= 0) and (prev_n[0] < N):
                            T.atomic_add(
                                GradMu[batch, prev_n[0], h, tangent, d], accum[0] * sm_scale
                            )
                            if d == 0:
                                T.atomic_add(
                                    GradBias[batch, prev_n[0], h, tangent], accum_bias[0]
                                )
                        prev_n[0], accum[0], accum_bias[0] = n, 0.0, 0.0
                    if (n >= 0) and (n < N):
                        grad = GradPlanes[batch, tq, h, k, tangent]
                        accum[0] += grad * T.Cast(accum_dtype, Q[batch, tq, h, d])
                        if d == 0:
                            accum_bias[0] += grad
                if (prev_n[0] >= 0) and (prev_n[0] < N):
                    T.atomic_add(
                        GradMu[batch, prev_n[0], h, tangent, d], accum[0] * sm_scale
                    )
                    if d == 0:
                        T.atomic_add(
                            GradBias[batch, prev_n[0], h, tangent], accum_bias[0]
                        )
    return kernel


@tilelang.jit(pass_configs=_PASSES)
def mass_selected_partial_bwd_kernel(
    B,
    L,
    N,
    H,
    Hkv,
    M,
    D,
    topk,
    sm_scale,
    tangent,
    block_q=32,
    threads=256,
    topk_tile=8,
):
    """Accumulate dMu/dBias into one dense shared-memory table per query tile.

    All high-contention updates stay in shared memory.  The global output is a
    compact ``num_query_tiles x (N*M)`` partial table, never an LxNxMxH plane.
    """
    dtype, accum_dtype, idx_dtype = "bfloat16", "float", "int32"
    groups = H // Hkv
    nm = N * M
    num_tiles = tilelang.cdiv(L, block_q)
    q_shape = [B, L, H, D]
    index_shape = [B, L, Hkv, topk]
    plane_grad_shape = [B, L, H, topk, M]
    partial_shape = [B, H, num_tiles, nm, D]
    partial_bias_shape = [B, H, num_tiles, nm]

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, dtype),
        Indices: T.Tensor(index_shape, idx_dtype),
        GradPlanes: T.Tensor(plane_grad_shape, accum_dtype),
        Partials: T.Tensor(partial_shape, accum_dtype),
        PartialBias: T.Tensor(partial_bias_shape, accum_dtype),
    ):
        with T.Kernel(num_tiles, H, B, threads=threads) as (tile, h, batch):
            base_q = tile * block_q
            hkv = h // groups
            # One tangent per block reduces the shared table from N*M rows to
            # N rows.  For OLMo3 (N=130,D=128) this is ~65 KiB and fits A800;
            # the old all-tangent table would require >500 KiB.
            partial = T.alloc_shared([N, D], accum_dtype)
            partial_bias = T.alloc_shared([N], accum_dtype)
            # Keep layout inference independent of nm.  In the real model the
            # prior-query prefix makes L=8322 and N=130, hence nm=520 rather
            # than the power-of-two 512 used by the standalone benchmark.
            row_tile = 8
            for rb in T.serial(tilelang.cdiv(N, row_tile)):
                for rr, d in T.Parallel(row_tile, D):
                    row = rb * row_tile + rr
                    if row < N:
                        partial[row, d] = 0.0
                        if d == 0:
                            partial_bias[row] = 0.0
            T.sync_threads()

            # TileLang cannot infer a legal free-mode layout when production
            # topk=32 is flattened together with D=64 (2048 parallel lanes).
            # Process eight selected chunks per round.  Global memory only
            # receives the compact, conflict-free partial table below.
            for q_local, k, d in T.Parallel(block_q, topk, D):
                tq = base_q + q_local
                if (tq < L) and (k < topk):
                    n = Indices[batch, tq, hkv, k]
                    if (n >= 0) and (n < N):
                        grad = GradPlanes[batch, tq, h, k, tangent]
                        T.atomic_add(
                            partial[n, d],
                            grad * T.Cast(accum_dtype, Q[batch, tq, h, d]) * sm_scale,
                        )
                        if d == 0:
                            T.atomic_add(partial_bias[n], grad)
            T.sync_threads()

            for rb in T.serial(tilelang.cdiv(N, row_tile)):
                for rr, d in T.Parallel(row_tile, D):
                    n = rb * row_tile + rr
                    if n < N:
                        row = n * M + tangent
                        Partials[batch, h, tile, row, d] = partial[n, d]
                        if d == 0:
                            PartialBias[batch, h, tile, row] = partial_bias[n]
    return kernel


@tilelang.jit(pass_configs=_PASSES)
def mass_selected_partial_bwd_mt_kernel(
    B,
    L,
    N,
    H,
    Hkv,
    M,
    D,
    topk,
    sm_scale,
    block_q=32,
    threads=256,
    topk_tile=8,
):
    """Merged-tangent partial accumulation: one launch covers all M tangents.

    Same [N, D] shared partial table and accumulation body as
    mass_selected_partial_bwd_kernel; the tangent folds into the grid's head
    axis (by = h*M + tangent) so the M per-tangent launches collapse into one
    kernel object and one launch.  The body is kept character-identical to
    the per-tangent kernel: wrapping the accumulate loop in an outer
    T.serial(tangent) instead miscompiles shared-memory atomics under
    free-mode layout inference, which is why the tangent lives in the grid.
    Env MASS_PARTIAL_LEGACY=1 restores the per-tangent kernels.
    """
    dtype, accum_dtype, idx_dtype = "bfloat16", "float", "int32"
    groups = H // Hkv
    num_tiles = tilelang.cdiv(L, block_q)
    q_shape = [B, L, H, D]
    index_shape = [B, L, Hkv, topk]
    plane_grad_shape = [B, L, H, topk, M]
    partial_shape = [B, H, num_tiles, N * M, D]
    partial_bias_shape = [B, H, num_tiles, N * M]

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, dtype),
        Indices: T.Tensor(index_shape, idx_dtype),
        GradPlanes: T.Tensor(plane_grad_shape, accum_dtype),
        Partials: T.Tensor(partial_shape, accum_dtype),
        PartialBias: T.Tensor(partial_bias_shape, accum_dtype),
    ):
        with T.Kernel(num_tiles, H * M, B, threads=threads) as (tile, ht, batch):
            h = ht // M
            tangent = ht % M
            base_q = tile * block_q
            hkv = h // groups
            # One tangent per block keeps the shared table at [N, D] (~65 KiB
            # for OLMo3 N=130, D=128).
            partial = T.alloc_shared([N, D], accum_dtype)
            partial_bias = T.alloc_shared([N], accum_dtype)
            # 清零/写回必须整块 Parallel(N, D):旧实现按 row_tile=8 分轮的写法
            # 在 free-mode 布局推导下会与累加段对同一 shared 表的原子访问产生
            # 冲突的布局假设,导致丢/重更新(计数测试可差 10+)。
            for n, d in T.Parallel(N, D):
                partial[n, d] = 0.0
                if d == 0:
                    partial_bias[n] = 0.0
            T.sync_threads()

            # 累加段:Parallel(q, d) + k 内层 serial(与已验证的 dq kernel 同构)。
            for q_local, d in T.Parallel(block_q, D):
                tq = base_q + q_local
                if tq < L:
                    for k in T.serial(topk):
                        n = Indices[batch, tq, hkv, k]
                        if (n >= 0) and (n < N):
                            grad = GradPlanes[batch, tq, h, k, tangent]
                            T.atomic_add(
                                partial[n, d],
                                grad * T.Cast(accum_dtype, Q[batch, tq, h, d]) * sm_scale,
                            )
                            if d == 0:
                                T.atomic_add(partial_bias[n], grad)
            T.sync_threads()

            for n, d in T.Parallel(N, D):
                row = n * M + tangent
                Partials[batch, h, tile, row, d] = partial[n, d]
                if d == 0:
                    PartialBias[batch, h, tile, row] = partial_bias[n]
    return kernel


@tilelang.jit(out_idx=[2, 3], pass_configs=_PASSES)
def mass_selected_partial_reduce_kernel(B, L, N, H, M, D, block_q=32, threads=128):
    """Reduce query-tile partials directly into compact MASS gradients."""
    dtype, accum_dtype = "bfloat16", "float"
    nm = N * M
    num_tiles = tilelang.cdiv(L, block_q)
    partial_shape = [B, H, num_tiles, nm, D]
    partial_bias_shape = [B, H, num_tiles, nm]
    grad_mu_shape = [B, N, H, M, D]
    grad_bias_shape = [B, N, H, M]

    @T.prim_func
    def kernel(
        Partials: T.Tensor(partial_shape, accum_dtype),
        PartialBias: T.Tensor(partial_bias_shape, accum_dtype),
        GradMu: T.Tensor(grad_mu_shape, dtype),
        GradBias: T.Tensor(grad_bias_shape, accum_dtype),
    ):
        with T.Kernel(nm, H, B, threads=threads) as (row, h, batch):
            n, m = row // M, row % M
            for d in T.Parallel(D):
                value = T.alloc_local([1], accum_dtype)
                value[0] = 0.0
                for tile in T.serial(num_tiles):
                    value[0] += Partials[batch, h, tile, row, d]
                GradMu[batch, n, h, m, d] = T.Cast(dtype, value[0])
            if T.get_thread_binding() == 0:
                bias_value = T.alloc_local([1], accum_dtype)
                bias_value[0] = 0.0
                for tile in T.serial(num_tiles):
                    bias_value[0] += PartialBias[batch, h, tile, row]
                GradBias[batch, n, h, m] = bias_value[0]
    return kernel


class _MassEnvelopeTopKV2Fn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        queries,
        mu,
        bias,
        kernels,
        groups,
        sm_scale,
        use_two_stage_backward,
        use_logsumexp,
        lse_temperature,
    ):
        (
            candidate_kernel,
            topk_kernel,
            sort_kernel,
            recompute_kernel,
            dq_kernel,
            direct_kernel,
            partial_kernel,
            reduce_kernel,
        ) = kernels
        candidates = candidate_kernel(queries.contiguous(), mu.contiguous(), bias.contiguous())
        if os.environ.get("MASS_SYNC_KERNELS", "0") == "1":
            torch.cuda.synchronize()   # debug: force stream sync between kernels
        indices_raw = topk_kernel(candidates)
        if os.environ.get("MASS_SYNC_KERNELS", "0") == "1":
            torch.cuda.synchronize()
        indices = sort_kernel(indices_raw)
        if os.environ.get("MASS_SYNC_KERNELS", "0") == "1":
            torch.cuda.synchronize()

        # Reuse the tensor-core recompute layout from topk_head_softmax.py.
        # Each selected chunk contributes M compact tangent rows; after the
        # GEMM, add the tangent bias and reduce M to the winning plane.
        B, L, H, D = queries.shape
        N, M, K = mu.shape[1], mu.shape[3], indices.shape[-1]
        Hkv = indices.shape[2]
        tangent = torch.arange(M, device=indices.device, dtype=torch.int32)
        expanded = (indices.unsqueeze(-1) * M + tangent).reshape(B, L, Hkv, K * M).contiguous()
        valid_expanded = expanded.ge(0)
        safe_expanded = expanded.clamp(0, N * M - 1)
        mu_flat = mu.permute(0, 1, 3, 2, 4).reshape(B, N * M, H, D).contiguous()
        q_grouped = queries.reshape(B, L, Hkv, groups, D).contiguous()
        dots = recompute_kernel(q_grouped, mu_flat, expanded)
        if os.environ.get("MASS_DEBUG_SHAPES", "0") == "1":
            _dbg = lambda name, t: print(
                f"[dbg] {name}: nan={torch.isnan(t.float()).float().mean().item():.5f} "
                f"inf={torch.isinf(t.float()).float().mean().item():.5f}", flush=True)
            _dbg("indices_raw", indices_raw.float())
            _dbg("indices(sorted)", indices.float())
            _dbg("dots", dots)

        bias_flat = bias.permute(0, 1, 3, 2).reshape(B, N * M, H).permute(0, 2, 1).contiguous()
        ids_h = safe_expanded.repeat_interleave(groups, dim=2).permute(0, 2, 1, 3).long()
        selected_bias = torch.gather(
            bias_flat.unsqueeze(2).expand(B, H, L, N * M), 3, ids_h
        ).permute(0, 2, 1, 3)
        valid_h = valid_expanded.repeat_interleave(groups, dim=2)
        all_scores = (dots.reshape(B, L, H, K * M) + selected_bias).masked_fill(
            ~valid_h, float("-inf")
        )
        planes = all_scores.reshape(B, L, H, K, M)
        valid_routes = indices.ge(0).repeat_interleave(groups, dim=2)
        valid_count = valid_routes.sum(dtype=torch.float32).clamp_min(1.0)
        if use_logsumexp:
            # The selected score and load use the same continuous router.  In
            # contrast to hard max + STE, every finite anchor plane receives
            # its exact LogSumExp derivative in backward.
            scores = lse_temperature * torch.logsumexp(planes / lse_temperature, dim=-1)
            if os.environ.get("MASS_DEBUG_SHAPES", "0") == "1":
                _vr = indices.ge(0).repeat_interleave(groups, dim=2)
                _poison = (_vr & (torch.isinf(scores) | torch.isnan(scores))).float().sum().item()
                _nvalid = _vr.float().sum().item()
                print(f"[dbg] valid_routes={_nvalid:.0f} poisoned_valid_routes={_poison:.0f}", flush=True)
            safe_planes = planes.masked_fill(~valid_routes.unsqueeze(-1), 0.0)
            routing_probs = torch.softmax(safe_planes / lse_temperature, dim=-1)
            routing_probs = routing_probs.masked_fill(~valid_routes.unsqueeze(-1), 0.0)
            anchor_load = routing_probs.float().sum(dim=(0, 1, 2, 3)) / valid_count
            routing_probs_f32 = routing_probs.float()
            conditional_entropy = -(
                routing_probs_f32
                * routing_probs_f32.clamp_min(torch.finfo(torch.float32).tiny).log()
            ).sum(dim=-1)
            conditional_entropy = conditional_entropy.masked_fill(
                ~valid_routes, 0.0
            ).sum() / valid_count
        else:
            scores, winners = planes.max(dim=-1)
            routing_probs = torch.nn.functional.one_hot(winners, num_classes=M).to(torch.float32)
            routing_probs = routing_probs.masked_fill(~valid_routes.unsqueeze(-1), 0.0)
            anchor_load = routing_probs.sum(dim=(0, 1, 2, 3)) / valid_count
            conditional_entropy = planes.new_zeros((), dtype=torch.float32)

        # BF16 halves the saved-router footprint.  It is used only as a
        # backward multiplier; scores and load are accumulated in FP32 above.
        ctx.save_for_backward(queries, mu, indices, routing_probs.to(torch.bfloat16))
        ctx.groups, ctx.sm_scale = groups, sm_scale
        ctx.use_logsumexp, ctx.lse_temperature = use_logsumexp, lse_temperature
        ctx.dq_kernel = dq_kernel
        ctx.direct_kernel = direct_kernel
        ctx.partial_kernel, ctx.reduce_kernel = partial_kernel, reduce_kernel
        ctx.use_two_stage_backward = use_two_stage_backward
        ctx.mark_non_differentiable(indices)
        return indices, scores, anchor_load, conditional_entropy

    @staticmethod
    def backward(
        ctx,
        unused_grad_indices,
        grad_scores,
        grad_anchor_load,
        grad_conditional_entropy,
    ):
        queries, mu, indices, routing_probs = ctx.saved_tensors
        B, L, H, D = queries.shape
        N, M, K = mu.shape[1], mu.shape[3], indices.shape[-1]
        Hkv = indices.shape[2]
        probs = routing_probs.float()
        valid_routes = indices.ge(0).repeat_interleave(ctx.groups, dim=2)
        if grad_scores is None:
            grad_scores_f32 = torch.zeros((B, L, H, K), device=queries.device, dtype=torch.float32)
        else:
            grad_scores_f32 = grad_scores.float()
        # PyTorch may pass an expanded zero-stride gradient when the loss is a
        # scalar reduction.  TileLang kernels consume raw contiguous storage.
        # Materialize here before multiplying by routing probabilities;
        # otherwise the forward-view stride can be misinterpreted as a factor.
        if grad_scores_f32.shape != (B, L, H, K):
            grad_scores_f32 = grad_scores_f32.expand(B, L, H, K)
        grad_planes = grad_scores_f32.unsqueeze(-1) * probs

        if grad_anchor_load is not None:
            valid_count = valid_routes.sum(dtype=torch.float32).clamp_min(1.0)
            load_grad = grad_anchor_load.float()
            if ctx.use_logsumexp:
                # Exact softmax Jacobian for load_m = mean_r p(r,m).
                expected = (probs * load_grad).sum(dim=-1, keepdim=True)
                balance_grad = probs * (load_grad - expected)
                balance_grad = balance_grad / (valid_count * ctx.lse_temperature)
                grad_planes = grad_planes + balance_grad.masked_fill(
                    ~valid_routes.unsqueeze(-1), 0.0
                )
            else:
                # Preserve the legacy hard-max straight-through balance path.
                centered = load_grad - load_grad.mean()
                winners = probs.argmax(dim=-1)
                hard_balance = centered[winners] / valid_count
                grad_planes = grad_planes + hard_balance.unsqueeze(-1) * probs

        if grad_conditional_entropy is not None and ctx.use_logsumexp:
            valid_count = valid_routes.sum(dtype=torch.float32).clamp_min(1.0)
            safe_probs = probs.clamp_min(torch.finfo(torch.float32).tiny)
            route_entropy = -(probs * safe_probs.log()).sum(dim=-1, keepdim=True)
            # For z = plane / tau and H(p) = -sum p log p:
            #   dH/dplane_j = -p_j * (log p_j + H) / tau.
            entropy_grad = -probs * (safe_probs.log() + route_entropy)
            entropy_grad = entropy_grad * (
                grad_conditional_entropy.float()
                / (valid_count * ctx.lse_temperature)
            )
            grad_planes = grad_planes + entropy_grad.masked_fill(
                ~valid_routes.unsqueeze(-1), 0.0
            )

        grad_planes = grad_planes.contiguous()
        if grad_planes.shape != (B, L, H, K, M):
            raise RuntimeError(f"unexpected MASS plane-gradient shape: {tuple(grad_planes.shape)}")
        if ctx.use_two_stage_backward:
            # dQ is a selected-row gather/reduction, so compute it directly in
            # TileLang.  This removes the remaining blocked PyTorch
            # gather/nonzero/index path from the optimized backward.
            grad_q = ctx.dq_kernel(
                mu.contiguous(),
                indices,
                grad_planes,
            )
            num_tiles = tilelang.cdiv(L, 32)
            partials = torch.empty(
                (B, H, num_tiles, N * M, D),
                device=queries.device,
                dtype=torch.float32,
            )
            partial_bias = torch.empty(
                (B, H, num_tiles, N * M),
                device=queries.device,
                dtype=torch.float32,
            )
            for partial_kernel in ctx.partial_kernel:
                partial_kernel(
                    queries.contiguous(),
                    indices,
                    grad_planes,
                    partials,
                    partial_bias,
                )
            grad_mu, grad_bias = ctx.reduce_kernel(partials, partial_bias)
        else:
            grad_q = ctx.dq_kernel(mu.contiguous(), indices, grad_planes)
            grad_mu = torch.zeros_like(mu, dtype=torch.float32)
            grad_bias = torch.zeros(
                (B, N, H, M), device=mu.device, dtype=torch.float32
            )
            ctx.direct_kernel(
                queries.contiguous(), indices, grad_planes, grad_mu, grad_bias
            )
            # The legacy direct atomic kernel replicates each update across
            # its 256-thread block under TileLang free-mode lowering.  Keep
            # this fallback numerically equivalent to the preferred partial
            # path; production uses MASS_TWO_STAGE_BACKWARD=1.
            grad_mu.mul_(1.0 / 256.0)
            grad_bias.mul_(1.0 / 256.0)
        return (
            grad_q.to(queries.dtype),
            grad_mu.to(mu.dtype),
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
        )


_KERNEL_CACHE = {}


def mass_retrieval_tilelang_v2(
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
    score_reduction="max",
    lse_temperature=1.0,
    two_stage_backward=None,
    shared_key=None,
    shared_bias=None,
):
    """Return MASS indices, scores, anchor load and conditional entropy.

    ``shared_key``/``shared_bias`` (shared-expert view): the HiLS landmark
    key ``[B, N, Hkv, D]`` (bf16) and a scalar logit bias join the envelope
    as fixed view 0 ahead of the M specialized anchor views.  The concat
    happens between ``_MassSummaryFn`` and the envelope kernels, so autograd
    splits gradients: specialized views flow back through the summary
    backward, the shared view flows straight into ``shared_key`` (k_proj)
    and the ``shared_view_bias`` parameter.  anchor_load/conditional_entropy
    then have M+1 entries with view 0 = shared share.
    """
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
        raise ValueError("Hq must be divisible by Hkv and MASS supports at most 16 anchors")
    if topk & (topk - 1):
        raise ValueError("MASS v2 requires power-of-two topk")
    if score_reduction not in {"max", "logsumexp"}:
        raise ValueError(f"unsupported MASS score reduction: {score_reduction!r}")
    if score_reduction == "logsumexp" and lse_temperature <= 0:
        raise ValueError("MASS LogSumExp temperature must be positive")
    if shared_key is not None and shared_bias is None:
        raise ValueError("shared_bias is required together with shared_key")
    if shared_key is not None:
        if tuple(shared_key.shape) != (B, N, Hkv, D):
            raise ValueError(
                f"shared_key must be [B,N,Hkv,D]={(B, N, Hkv, D)}, got {tuple(shared_key.shape)}"
            )
        if M + 1 > 16:
            raise ValueError("MASS supports at most 16 envelope views (shared + anchors)")
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    groups = H // Hkv
    use_logsumexp = score_reduction == "logsumexp"
    mu, bias = _MassSummaryFn.apply(anchors, key_chunks, sm_scale, groups)
    if shared_key is not None:
        # Shared-expert REPLACE mode: keep the kernel-visible view count at
        # num_anchors (the summary/envelope kernels only compile reliably at
        # the production M values, e.g. M=2).  View 0 of the tangent summary
        # is swapped out-of-place for the HiLS landmark key + learnable logit
        # bias; view 1..M-1 remain the specialized anchor views.  All kernels
        # still see M views, so no kernel recompiles for new shapes.
        shared_mu = (
            shared_key.repeat_interleave(groups, dim=2).unsqueeze(3).to(mu.dtype).contiguous()
        )
        shared_b = (
            shared_bias.to(dtype=bias.dtype).view(1, 1, 1, 1).expand(B, N, H, 1).contiguous()
        )
        mu = torch.cat([shared_mu, mu[:, :, :, 1:, :]], dim=3).contiguous()
        bias = torch.cat([shared_b, bias[:, :, :, 1:]], dim=3).contiguous()
    cache_key = (
        B,
        L,
        N,
        H,
        Hkv,
        M,
        D,
        topk,
        chunk_size,
        window_size,
        sm_scale,
        use_logsumexp,
        float(lse_temperature),
    )
    kernels = _KERNEL_CACHE.get(cache_key)
    if kernels is None:
        if os.environ.get("MASS_DEBUG_SHAPES", "0") == "1":
            print(f"[mass-v2] compiling cache_key={cache_key}", flush=True)
        kernels = (
            mass_envelope_candidates_kernel(
                B,
                L,
                N,
                H,
                Hkv,
                M,
                D,
                chunk_size,
                window_size,
                sm_scale,
                use_logsumexp,
                float(lse_temperature),
            ),
            mass_bitonic_topk_kernel(B, L, N, Hkv, topk),
            mass_sort_selected_indices_kernel(B, L, Hkv, topk),
            recompute_topk_max_pooling_scores_kernel(
                B,
                h_kv=Hkv,
                groups=groups,
                head_dim=D,
                topk=topk * M,
                seq_len=L,
                s_len=N * M,
                is_training=True,
                sm_scale=sm_scale,
                per_qhead_lmks=True,
            ),
            # dQ: block-diagonal gather-GEMM by default (29% of GPU kernel
            # time in the 2026-10-10 trace came from the scalar kernel);
            # MASS_DQ_LEGACY=1 restores it.
            mass_selected_dq_gemm_kernel(B, L, N, H, Hkv, M, D, topk, sm_scale)
            if os.environ.get("MASS_DQ_LEGACY", "0") != "1"
            else mass_selected_dq_kernel(B, L, N, H, Hkv, M, D, topk, sm_scale),
            mass_selected_dmu_dbias_kernel(B, L, N, H, Hkv, M, D, topk, sm_scale),
            # Partial accumulation: merged-tangent single launch by default;
            # MASS_PARTIAL_LEGACY=1 restores the per-tangent kernels.
            (
                mass_selected_partial_bwd_mt_kernel(B, L, N, H, Hkv, M, D, topk, sm_scale),
            )
            if os.environ.get("MASS_PARTIAL_LEGACY", "0") != "1"
            else tuple(
                mass_selected_partial_bwd_kernel(
                    B, L, N, H, Hkv, M, D, topk, sm_scale, tangent
                )
                for tangent in range(M)
            ),
            mass_selected_partial_reduce_kernel(B, L, N, H, M, D),
        )
        _KERNEL_CACHE[cache_key] = kernels
    # The N-row partial path is being retained for kernel development, but the
    # production LSE run currently uses the numerically verified direct path.
    if two_stage_backward is None:
        two_stage_backward = os.environ.get("MASS_TWO_STAGE_BACKWARD", "0") == "1"
    use_two_stage_backward = (
        bool(two_stage_backward)
        # The partial kernel assigns one tangent per block, so only N rows are
        # resident in shared memory.  Include its bias table in the guard.
        and N * (D + 1) <= 40 * 1024
    )
    return _MassEnvelopeTopKV2Fn.apply(
        queries,
        mu,
        bias,
        kernels,
        groups,
        sm_scale,
        use_two_stage_backward,
        use_logsumexp,
        float(lse_temperature),
    )
