import time

import torch

from ops.topk_head_softmax import recompute_topk_max_pooling_scores_kernel


B, L, N, H, Hkv, G, M, D, K = 2, 8192, 128, 12, 3, 4, 4, 64, 32
scale = D ** -0.5
q = torch.randn(B, L, Hkv, G, D, device="cuda", dtype=torch.bfloat16)
mu = torch.randn(B, N, H, M, D, device="cuda", dtype=torch.bfloat16)
bias = torch.randn(B, N, H, M, device="cuda", dtype=torch.float32)
indices = torch.randint(0, N, (B, L, Hkv, K), device="cuda", dtype=torch.int32)
tangent_ids = torch.arange(M, device="cuda", dtype=torch.int32)
expanded = (indices.unsqueeze(-1) * M + tangent_ids).reshape(B, L, Hkv, K * M).contiguous()
mu_flat = mu.permute(0, 1, 3, 2, 4).reshape(B, N * M, H, D).contiguous()
bias_flat = bias.permute(0, 1, 3, 2).reshape(B, N * M, H).permute(0, 2, 1).contiguous()

kernel = recompute_topk_max_pooling_scores_kernel(
    B,
    h_kv=Hkv,
    groups=G,
    head_dim=D,
    topk=K * M,
    seq_len=L,
    s_len=N * M,
    is_training=True,
    sm_scale=scale,
    per_qhead_lmks=True,
)


def run():
    dots = kernel(q, mu_flat, expanded)
    ids_h = expanded.repeat_interleave(G, dim=2).permute(0, 2, 1, 3).long()
    gathered_bias = torch.gather(
        bias_flat.unsqueeze(2).expand(B, H, L, N * M), 3, ids_h
    ).permute(0, 2, 1, 3)
    scores = (dots.reshape(B, L, H, K * M) + gathered_bias).reshape(B, L, H, K, M)
    return scores.max(dim=-1)


start = time.perf_counter()
values, winners = run()
torch.cuda.synchronize()
print(f"first={time.perf_counter() - start:.6f}s", flush=True)
for _ in range(2):
    run()
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(5):
    values, winners = run()
torch.cuda.synchronize()
print(f"warm={(time.perf_counter() - start) / 5:.6f}s values={tuple(values.shape)}", flush=True)
