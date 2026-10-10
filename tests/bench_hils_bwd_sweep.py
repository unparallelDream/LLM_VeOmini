"""hils_bwd_dqkv 参数扫描基准(产线形状)+ 数值一致性校验。

产线配置: B=1, L=q=kv=8322, HQ=32, Hkv=32(G=1), D=Dv=128, block_size=64,
topk S=32, num_weights=33(SWA 列在最后)。调用方 mass_attention.py 对 G=1 退化
出 threads=64;本脚本验证更大 threads / block_M 的收益与数值一致性。

索引语义:同 chunk_size=64 的 query 块内共享索引(HiLS 原生语义,可与
hils_torch_ref 对拍;MASS 的每-query 索引由 ownership 掩码扩展,不影响调度)。

用法: python tests/bench_hils_bwd_sweep.py
"""

import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ops.hils_fwd_bwd_head import (  # noqa: E402
    HiLS_block_M_head,
    hils_torch_ref,
    hierarchical_sparse_attention_bwd_dqkv_block_M_inverse,
)

B, L, HQ, H, D, BS, S, NW = 1, 8322, 32, 32, 128, 64, 32, 33
NCH = L // BS  # 130


def make_inputs(seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, L, HQ, D, generator=g).bfloat16().cuda()
    k = torch.randn(B, NCH * BS, H, D, generator=g).bfloat16().cuda()
    v = torch.randn(B, NCH * BS, H, D, generator=g).bfloat16().cuda()
    # 权重同样按 chunk 共享(参考实现 hils_torch_ref 要求 [B, Bq, H, S] 粒度)
    w_chunk = torch.rand(B, NCH, HQ, NW, generator=g)
    w_chunk[:, :, :, -1] = 0.2
    w_chunk[:, :, :, :-1] = (
        w_chunk[:, :, :, :-1] / w_chunk[:, :, :, :-1].sum(-1, keepdim=True) * 0.8
    )
    w_full = w_chunk.repeat_interleave(BS, dim=1)
    if w_full.shape[1] < L:
        w_full = torch.cat(
            [w_full, w_full[:, -1:, :, :].expand(-1, L - w_full.shape[1], -1, -1)], dim=1
        )
    w = w_full.bfloat16().cuda()
    # 同一 64-query chunk 内共享 32 个不重复块索引(L=8322=130*64+2,尾部补最后一块)
    per_chunk = torch.stack(
        [torch.randperm(NCH, generator=g)[:S] for _ in range(B * NCH * H)]
    ).view(B, NCH, H, S)
    idx_full = per_chunk.repeat_interleave(BS, dim=1)
    if idx_full.shape[1] < L:
        pad = idx_full[:, -1:, :, :].expand(-1, L - idx_full.shape[1], -1, -1)
        idx_full = torch.cat([idx_full, pad], dim=1)
    idx = idx_full.to(torch.int32).cuda()
    return q, k, v, w, idx


def verify_fwd(o, q, k, v, w, idx, nblk=8):
    o_ref = hils_torch_ref(
        q[:, : nblk * BS].float(), k.float(), v.float(),
        w[:, : nblk * BS : BS, :, :S].float(),  # chunk 共享权重 → 每块首行 [B,Bq,HQ,S]
        idx[:, : nblk * BS],
        chunk_size=BS, sm_scale=1.0 / math.sqrt(D), block_q=BS,
        mask_last_token=True,
    )
    o_k = o[:, : nblk * BS].float()
    rel = ((o_k - o_ref).abs().max() / o_ref.abs().max()).item()
    return rel


def run_config(cfg_name, threads_fwd, threads_bwd, block_m, inputs, ref=None):
    q0, k0, v0, w0, idx = inputs
    q = q0.clone().requires_grad_(True)
    k = k0.clone().requires_grad_(True)
    v = v0.clone().requires_grad_(True)
    w = w0.clone().requires_grad_(True)
    dg = torch.Generator(device="cuda").manual_seed(123)
    do = torch.randn(B, L, HQ, D, generator=dg, device="cuda").bfloat16()

    def fwd_bwd():
        o = HiLS_block_M_head(
            q, k, v, weights=w, indices=idx,
            block_size=BS, block_M=block_m,
            num_threads_fwd=threads_fwd, num_threads_bwd=threads_bwd,
            mask_last_token=True, is_training=True,
        )
        (o.float() * do.float()).sum().backward()
        return o

    o = fwd_bwd()
    fwd_rel = verify_fwd(o.detach(), q0, k0, v0, w0, idx)
    for _ in range(2):
        fwd_bwd()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        fwd_bwd()
    torch.cuda.synchronize()
    total = (time.perf_counter() - t0) / 5 * 1e3

    grads = (q.grad / 8, k.grad / 8, v.grad / 8, w.grad / 8)  # 8 次 backward 累积
    note = ""
    if ref is not None:
        # 相对判据(按各张量自身量级):dw 量级可达 60+,绝对 atol 会误报
        rels = []
        for a, b in zip(grads, ref):
            scale = b.float().abs().max().clamp_min(1e-6)
            rels.append(((a.float() - b.float()).abs().max() / scale).item())
        assert all(r < 0.02 for r in rels), rels
        note = " rel(dq,dk,dv,dw)=" + ",".join("%.1e" % r for r in rels) + " ✓"
    print(f"[{cfg_name:16s}] fwd+bwd={total:7.1f}ms  fwd_rel={fwd_rel:.4f}{note}", flush=True)
    return total, grads


def main():
    torch.manual_seed(0)
    inputs = make_inputs()
    print(f"q{tuple(inputs[0].shape)} k{tuple(inputs[1].shape)} idx{tuple(inputs[4].shape)} "
          f"(N={NCH} chunks, topk={S}, 每 chunk 共享索引)", flush=True)

    _, ref = run_config("prod(64/64,M16)", 64, 64, 16, inputs)
    run_config("f128/b128,M16", 128, 128, 16, inputs, ref)
    run_config("f128/b256,M16", 128, 256, 16, inputs, ref)
    run_config("f256/b256,M16", 256, 256, 16, inputs, ref)
    print("done")


if __name__ == "__main__":
    main()
