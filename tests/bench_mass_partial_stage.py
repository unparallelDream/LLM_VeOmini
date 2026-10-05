"""Benchmark the production-shape MASS first-stage partial kernel."""

import gc
import math
import os
import pathlib
import statistics
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ops.mass_retrieval_tilelang_v2 import (
    mass_selected_partial_bwd_kernel,
    mass_selected_partial_reduce_kernel,
)


B = int(os.environ.get("MASS_TEST_BATCH", "4"))
L = int(os.environ.get("MASS_TEST_SEQ_LEN", "8322"))
S, H, Hkv, M, D, K = 64, 12, 3, 4, 64, 32
N = L // S
BLOCK_Q = int(os.environ.get("MASS_PARTIAL_BLOCK_Q", "256"))
ROW_PARTS = [1]
REPEATS = int(os.environ.get("MASS_BENCH_REPEATS", "5"))
TOPK_TILE = int(os.environ.get("MASS_PARTIAL_TOPK_TILE", "8"))


def timed_call(kernel, *args):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    outputs = kernel(*args)
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop), outputs


torch.manual_seed(19)
q = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16)
indices = torch.randint(N, (B, L, Hkv, K), device="cuda", dtype=torch.int32)
winners = torch.randint(M, (B, L, H, K), device="cuda", dtype=torch.int32)
grad_scores = torch.randn(B, L, H, K, device="cuda", dtype=torch.float32)
reduce_kernel = mass_selected_partial_reduce_kernel(B, L, N, H, M, D, block_q=BLOCK_Q)

reference = None
for row_parts in ROW_PARTS:
    partial_kernel = mass_selected_partial_bwd_kernel(
        B,
        L,
        N,
        H,
        Hkv,
        M,
        D,
        K,
        1.0 / math.sqrt(D),
        block_q=BLOCK_Q,
        topk_tile=TOPK_TILE,
    )
    # Compile and warm both stages before collecting events.
    partials, partial_bias = partial_kernel(q, indices, winners, grad_scores)
    grad_mu, grad_bias = reduce_kernel(partials, partial_bias)
    torch.cuda.synchronize()

    if reference is None:
        reference = (grad_mu.clone(), grad_bias.clone())
        max_mu_diff = max_bias_diff = 0.0
    else:
        max_mu_diff = (grad_mu.float() - reference[0].float()).abs().max().item()
        max_bias_diff = (grad_bias - reference[1]).abs().max().item()

    del grad_mu, grad_bias
    reduce_times = []
    for _ in range(REPEATS):
        elapsed, outputs = timed_call(reduce_kernel, partials, partial_bias)
        reduce_times.append(elapsed)
        del outputs
    del partials, partial_bias
    times = []
    for _ in range(REPEATS):
        elapsed, outputs = timed_call(partial_kernel, q, indices, winners, grad_scores)
        times.append(elapsed)
        del outputs

    print(
        f"row_parts={row_parts} block_q={BLOCK_Q} topk_tile={TOPK_TILE} "
        f"partial_ms_median={statistics.median(times):.3f} "
        f"partial_ms_min={min(times):.3f} samples={','.join(f'{value:.3f}' for value in times)} "
        f"reduce_ms_median={statistics.median(reduce_times):.3f} "
        f"two_stage_ms={statistics.median(times) + statistics.median(reduce_times):.3f} "
        f"max_mu_diff={max_mu_diff:.6g} max_bias_diff={max_bias_diff:.6g}",
        flush=True,
    )
    del partial_kernel
    gc.collect()
    torch.cuda.empty_cache()
