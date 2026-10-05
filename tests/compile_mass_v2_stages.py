import time

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
stages = [
    ("candidates", lambda: mass_envelope_candidates_kernel(B, L, N, H, Hkv, M, D, S, W, scale)),
    ("topk", lambda: mass_bitonic_topk_kernel(B, L, N, Hkv, K)),
    ("sort", lambda: mass_sort_selected_indices_kernel(B, L, Hkv, K)),
    ("scores", lambda: mass_selected_scores_kernel(B, L, N, H, Hkv, M, D, K, scale)),
    ("dq", lambda: mass_selected_dq_kernel(B, L, N, H, Hkv, M, D, K, scale)),
    ("dmu", lambda: mass_selected_dmu_dbias_kernel(B, L, N, H, Hkv, M, D, K, scale)),
]
for name, build in stages:
    print(f"START {name}", flush=True)
    start = time.perf_counter()
    build()
    print(f"DONE {name} {time.perf_counter() - start:.3f}s", flush=True)
