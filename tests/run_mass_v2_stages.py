import time

import torch

from ops.mass_retrieval_tilelang_v2 import (
    mass_bitonic_topk_kernel,
    mass_envelope_candidates_kernel,
    mass_selected_dmu_dbias_kernel,
    mass_selected_dq_kernel,
    mass_selected_scores_kernel,
    mass_sort_selected_indices_kernel,
)


B, L, N, H, Hkv, M, D, K, S, W = 1, 128, 2, 4, 2, 2, 64, 2, 64, 64
scale = D ** -0.5
q = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16)
mu = torch.randn(B, N, H, M, D, device="cuda", dtype=torch.bfloat16)
bias = torch.randn(B, N, H, M, device="cuda", dtype=torch.float32)
unused = torch.empty(1, device="cuda", dtype=torch.float32)


def run(name, fn):
    print(f"START {name}", flush=True)
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    print(f"DONE {name} {time.perf_counter() - start:.3f}s", flush=True)
    return value


candidate_kernel = mass_envelope_candidates_kernel(B, L, N, H, Hkv, M, D, S, W, scale)
candidates = run("candidates", lambda: candidate_kernel(q, mu, bias))
topk_kernel = mass_bitonic_topk_kernel(B, L, N, Hkv, K)
indices_raw = run("topk", lambda: topk_kernel(candidates))
sort_kernel = mass_sort_selected_indices_kernel(B, L, Hkv, K)
indices = run("sort", lambda: sort_kernel(indices_raw))
score_kernel = mass_selected_scores_kernel(B, L, N, H, Hkv, M, D, K, scale)
scores, winners = run("scores", lambda: score_kernel(q, mu, bias, indices))
grad_scores = torch.randn_like(scores)
dq_kernel = mass_selected_dq_kernel(B, L, N, H, Hkv, M, D, K, scale)
run("dq", lambda: dq_kernel(mu, indices, winners, grad_scores, unused))
dmu_kernel = mass_selected_dmu_dbias_kernel(B, L, N, H, Hkv, M, D, K, scale)
run("dmu", lambda: dmu_kernel(q, indices, winners, grad_scores, unused))
