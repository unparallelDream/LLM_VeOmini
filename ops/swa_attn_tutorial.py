"""
滑动窗口注意力（Sliding Window Attention）Flash Attention 教学版
================================================================

从 `ops/flex_attn_tilelang.py` 精简而来的 TileLang Flash Attention 前向 kernel。

相对原版删减的内容（教学目的）：
1. 掩码只保留「滑动窗口 + 因果」：
       keep = (kv_idx >= q_idx - window_size + 1) & (kv_idx <= q_idx)
   删除了 mask_lmk（landmark 块末尾 token 掩码）与 expand_to_chunk
   （窗口左边界扩展到所在 chunk 起点）两套分支。
2. 只支持训练形态（q_len == kv_len）：删除了 use_cache 推理路径、
   KVStart、q_offset 等缓存相关逻辑。
3. 只支持 MHA（heads_q == heads_kv）：删除了 GQA groups 逻辑。
4. 只提供前向，删除了全部反向 kernel。

保留的核心机制（Flash Attention 的本质）：
- Q / K / V 分块（tile）加载进共享内存，用 T.gemm 完成 QK^T 与 P·V 两次矩阵乘；
- 逐块在线 softmax（online softmax）：维护 running max / running sum，
  用 log2 域（exp2 / log2）计算以贴近硬件指令，避免每步除法；
- 非对齐序列长度的尾部处理：所有全局读写都带 tq/ts < seq_len 守卫，
  OOB 槽位在共享内存中填 0、再经掩码置为 -inf（K 侧）/ 乘 0（V 侧），
  保证不越界、不产生 NaN。

运行：
    python ops/swa_attn_tutorial.py
"""

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(
    out_idx=[3, 4],  # fwd(Q, K, V, Output, Lse) 中 Output / Lse 是第 4、5 个参数
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def swa_attn_fwd(
    batch,
    heads,
    seq_len,
    dim_qk,
    dim_v,
    window_size,
    block_M,
    block_N,
    sm_scale=None,
):
    """滑动窗口注意力的 Flash Attention 前向 kernel。

    网格：grid = (ceildiv(seq_len, block_M), heads, batch)
        - bx: 当前处理的 Q 行块（block_M 行）
        - by: 注意力头
        - bz: batch 样本

    每个线程块只扫描与本 Q 块有掩码交集的 KV 块，即区间
        [max(0, q_block_start - window_size + 1), min(q_block_end, seq_len))
    窗口外的 KV 块根本不会被加载 / 计算 —— 这正是滑动窗口注意力
    相对全注意力节省计算量的关键。

    `sm_scale` 是自然对数域的 softmax 缩放（默认 1/sqrt(dim_qk)）。
    返回的 Lse 是 log2 域的 log-sum-exp，由 wrapper 转回自然对数。
    """
    # 1.44269504 = log2(e)：把自然对数域的 scale 换算到 log2 域，
    # 这样 exp2 / log2 可以直接替代 exp / ln。
    _base = (1.0 / dim_qk) ** 0.5 if sm_scale is None else float(sm_scale)
    scale = _base * 1.44269504

    dtype = "bfloat16"
    accum_dtype = "float"

    @T.prim_func
    def fwd(
        Q: T.Tensor([batch, seq_len, heads, dim_qk], dtype),  # type: ignore
        K: T.Tensor([batch, seq_len, heads, dim_qk], dtype),  # type: ignore
        V: T.Tensor([batch, seq_len, heads, dim_v], dtype),  # type: ignore
        Output: T.Tensor([batch, seq_len, heads, dim_v], dtype),  # type: ignore
        Lse: T.Tensor([batch, seq_len, heads], accum_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=128) as (bx, by, bz):
            # ---- 共享内存 / 寄存器缓冲 ----
            Q_shared = T.alloc_shared([block_M, dim_qk], dtype)
            K_shared = T.alloc_shared([block_N, dim_qk], dtype)
            V_shared = T.alloc_shared([block_N, dim_v], dtype)
            O_shared = T.alloc_shared([block_M, dim_v], dtype)
            lse_out_shared = T.alloc_shared([block_M], accum_dtype)

            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)   # QK^T 分数
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)    # bf16 化的 P，供 gemm
            acc_o = T.alloc_fragment([block_M, dim_v], accum_dtype)     # 累积输出 O
            scores_max = T.alloc_fragment([block_M], accum_dtype)       # 当前 running max
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)  # 上一块结束时的 max
            scores_scale = T.alloc_fragment([block_M], accum_dtype)     # 新旧 max 的缩放因子
            scores_sum = T.alloc_fragment([block_M], accum_dtype)       # 本块的 exp 和
            logsum = T.alloc_fragment([block_M], accum_dtype)           # 累积 softmax 分母

            # ---- 加载本块 Q：尾部 OOB 行填 0（GEMM 输出为 0，之后掩码置 -inf） ----
            for i, j in T.Parallel(block_M, dim_qk):
                tq = bx * block_M + i
                if tq < seq_len:
                    Q_shared[i, j] = Q[bz, tq, by, j]
                else:
                    Q_shared[i, j] = T.Cast(dtype, 0)

            # ---- online softmax 初始化 ----
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            # ---- 本 Q 块能看到的 KV 范围（窗口 + 因果，钳到合法区间） ----
            q_block_start = bx * block_M
            q_block_end = (bx + 1) * block_M
            left_kv = T.max(q_block_start - window_size + 1, 0)  # 窗口左边界
            right_kv = T.min(q_block_end, seq_len)               # 因果右边界
            loop_st = T.floordiv(left_kv, block_N)
            loop_ed = T.ceildiv(right_kv, block_N)

            # ---- 逐 KV 块扫描 ----
            for k in T.Pipelined(loop_st, loop_ed, num_stages=1):
                # 加载本 KV 块：尾部 OOB 行填 0（K 侧掩码置 -inf，V 侧乘 0 后无贡献）
                for i, j in T.Parallel(block_N, dim_qk):
                    ts = k * block_N + i
                    if ts < seq_len:
                        K_shared[i, j] = K[bz, ts, by, j]
                    else:
                        K_shared[i, j] = T.Cast(dtype, 0)
                for i, j in T.Parallel(block_N, dim_v):
                    ts = k * block_N + i
                    if ts < seq_len:
                        V_shared[i, j] = V[bz, ts, by, j]
                    else:
                        V_shared[i, j] = T.Cast(dtype, 0)

                # ---- S = Q @ K^T（共享内存直喂 warp 级 GEMM） ----
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # ---- 掩码：滑动窗口 + 因果 + 越界守卫，被 mask 的位置置 -inf ----
                for i, j in T.Parallel(block_M, block_N):
                    q_idx = bx * block_M + i
                    kv_idx = k * block_N + j
                    acc_s[i, j] = T.if_then_else(
                        (kv_idx >= q_idx - window_size + 1)  # 窗口左边界
                        & (kv_idx <= q_idx)                   # 因果（不能看未来）
                        & (kv_idx < seq_len)                  # KV 尾部 OOB 列
                        & (q_idx < seq_len),                  # Q 尾部 OOB 行
                        acc_s[i, j],
                        -T.infinity(accum_dtype),
                    )

                # ---- online softmax 更新（log2 域，共 6 步） ----
                # 1) 记下旧 max，求出本块的行 max
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                # 2) 旧累计量缩放因子 = exp2(prev_max*scale - new_max*scale)
                #    守卫：整行还没有任何可见 token 时 max == -inf，
                #    (-inf) - (-inf) 是 NaN，此时取 1.0（acc_o 本来就是 0）。
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.if_then_else(
                        scores_max[i] > -1e30,
                        T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale),
                        1.0,
                    )

                # 3) 重缩放已累积的输出：O *= scale（Flash Attention 的关键一步，
                #    用乘法修正旧块在旧 max 下的 softmax 归一化）
                for i, j in T.Parallel(block_M, dim_v):
                    acc_o[i, j] *= scores_scale[i]

                # 4) 本块概率 P = exp2(S*scale - new_max*scale)，同样守卫空行填 0
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(
                        scores_max[i] > -1e30,
                        T.exp2(acc_s[i, j] * scale - scores_max[i] * scale),
                        0.0,
                    )

                # 5) O += P @ V（P 经 bf16 副本喂 GEMM）
                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                # 6) 累积分母：logsum = logsum * scale + sum(P)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

            # ---- 收尾：O = acc_o / logsum；lse = log2(logsum) + max * scale ----
            for i, j in T.Parallel(block_M, dim_v):
                acc_o[i, j] = T.if_then_else(logsum[i] > 0, acc_o[i, j] / logsum[i], 0.0)
            for i in T.Parallel(block_M):
                logsum[i] = T.if_then_else(
                    logsum[i] > 0,
                    T.log2(logsum[i]) + scores_max[i] * scale,
                    -T.infinity(accum_dtype),
                )

            # ---- 结果经共享内存写回全局（跳过尾部 OOB 行） ----
            T.copy(acc_o, O_shared)
            T.copy(logsum, lse_out_shared)
            for i, j in T.Parallel(block_M, dim_v):
                tq = bx * block_M + i
                if tq < seq_len:
                    Output[bz, tq, by, j] = O_shared[i, j]
            for i in T.Parallel(block_M):
                tq = bx * block_M + i
                if tq < seq_len:
                    Lse[bz, tq, by] = lse_out_shared[i]

    return fwd


# ---------------------------------------------------------------------------
# 公共接口
# ---------------------------------------------------------------------------
def swa_attn(q, k, v, window_size: int, sm_scale: float = None):
    """滑动窗口注意力（教学版 Flash Attention）。

    Args:
        q: (B, S, H, D_qk)，要求 contiguous 的 bfloat16
        k: (B, S, H, D_qk)
        v: (B, S, H, D_v)
        window_size: 滑动窗口大小（token 数），q 可看到 [q-w+1, q] 的 key
        sm_scale: 自然对数域 softmax 缩放，默认 1/sqrt(D_qk)

    Returns:
        o:   (B, S, H, D_v)
        lse: (B, S, H) 自然对数域 log-sum-exp
    """
    batch, seq_len, heads, dim_qk = q.shape
    dim_v = v.shape[-1]
    # 与原版一致的分块配置：Q 块 128 行、KV 块 64 列
    block_M, block_N = 128, 64

    mod = swa_attn_fwd(
        batch, heads, seq_len, dim_qk, dim_v,
        window_size, block_M, block_N,
        sm_scale=sm_scale,
    )
    o, lse_log2 = mod(q, k, v)
    # kernel 内部是 log2 域，这里转回自然对数域（ln(2) ≈ 0.69314718）
    ln2 = 0.6931471805599453
    return o, lse_log2 * ln2


# ---------------------------------------------------------------------------
# 纯 PyTorch 参考实现（float32 精确计算），用于对照测试
# ---------------------------------------------------------------------------
def swa_attn_ref(q, k, v, window_size: int, sm_scale: float = None):
    """朴素实现：直接构造 (S, S) 掩码 + 标准 softmax，只用于验证 kernel。"""
    batch, seq_len, heads, dim_qk = q.shape
    scale = (1.0 / dim_qk) ** 0.5 if sm_scale is None else sm_scale

    scores = torch.einsum("bshd,bthd->bhst", q.float(), k.float()) * scale
    q_idx = torch.arange(seq_len, device=q.device).view(seq_len, 1)
    kv_idx = torch.arange(seq_len, device=q.device).view(1, seq_len)
    keep = (kv_idx >= q_idx - window_size + 1) & (kv_idx <= q_idx)
    scores = scores.masked_fill(~keep, float("-inf"))

    p = torch.softmax(scores, dim=-1)
    o = torch.einsum("bhst,bthd->bshd", p, v.float())
    lse = torch.logsumexp(scores, dim=-1).permute(0, 2, 1)  # (B, S, H)
    return o.to(q.dtype), lse


# ---------------------------------------------------------------------------
# 对照测试
# ---------------------------------------------------------------------------
def _run_case(name, batch, heads, seq_len, dim_qk, dim_v, window_size, sm_scale=None):
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    q = torch.randn(batch, seq_len, heads, dim_qk, device=device, dtype=dtype) * 0.5
    k = torch.randn(batch, seq_len, heads, dim_qk, device=device, dtype=dtype) * 0.5
    v = torch.randn(batch, seq_len, heads, dim_v, device=device, dtype=dtype) * 0.5

    o, lse = swa_attn(q, k, v, window_size, sm_scale=sm_scale)
    o_ref, lse_ref = swa_attn_ref(q, k, v, window_size, sm_scale=sm_scale)

    o_ref_f, o_f = o_ref.float(), o.float()
    o_abs = (o_ref_f - o_f).abs().max().item()
    o_ratio = (o_ref_f - o_f).square().mean().sqrt().item() / (o_ref_f.square().mean().sqrt().item() + 1e-12)
    lse_abs = (lse_ref.float() - lse.float()).abs().max().item()

    print(f"  {name:<16s} B={batch} H={heads} S={seq_len} D={dim_qk} W={window_size:<5d} "
          f"| o_abs={o_abs:.5f} o_ratio={o_ratio:.6f} lse_abs={lse_abs:.5f}")

    assert o_ratio < 6e-3, f"[{name}] o 相对误差过大: {o_ratio}"
    assert lse_abs < 5e-2, f"[{name}] lse 绝对误差过大: {lse_abs}"
    return True


if __name__ == "__main__":
    # 覆盖：常规 / 序列长度不对齐 / 窗口大于序列（退化为全因果）/
    # 序列短于单个块 / 不同头数与维度 / 自定义 sm_scale / batch > 1
    cases = [
        ("basic",       dict(batch=1, heads=8,  seq_len=512,  dim_qk=64,  dim_v=64,  window_size=128)),
        ("long_tail",   dict(batch=1, heads=8,  seq_len=1000, dim_qk=64,  dim_v=64,  window_size=256)),
        ("full_causal", dict(batch=1, heads=4,  seq_len=256,  dim_qk=64,  dim_v=64,  window_size=4096)),
        ("tiny_seq",    dict(batch=1, heads=8,  seq_len=64,   dim_qk=64,  dim_v=64,  window_size=32)),
        ("mid_tail",    dict(batch=1, heads=8,  seq_len=384,  dim_qk=64,  dim_v=64,  window_size=128)),
        ("wide_head",   dict(batch=1, heads=8,  seq_len=512,  dim_qk=128, dim_v=128, window_size=128)),
        ("small_win",   dict(batch=1, heads=8,  seq_len=512,  dim_qk=64,  dim_v=64,  window_size=32)),
        ("batch2",      dict(batch=2, heads=4,  seq_len=512,  dim_qk=64,  dim_v=64,  window_size=64)),
        ("scale0p1",    dict(batch=1, heads=8,  seq_len=512,  dim_qk=64,  dim_v=64,  window_size=128, sm_scale=0.1)),
        ("scale2p0",    dict(batch=1, heads=8,  seq_len=512,  dim_qk=64,  dim_v=64,  window_size=128, sm_scale=2.0)),
    ]

    print(f"\n滑动窗口注意力教学版 kernel vs torch 参考（{len(cases)} 个用例）")
    print("=" * 110)
    for name, cfg in cases:
        _run_case(name, **cfg)
    print("=" * 110)
    print("全部通过")
