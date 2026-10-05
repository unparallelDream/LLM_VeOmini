"""Check the no-atomic partial stage against a PyTorch index_add reference."""

import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ops.mass_retrieval_tilelang_v2 import (
    mass_selected_partial_bwd_kernel,
    mass_selected_partial_reduce_kernel,
)


torch.manual_seed(37)
B, L, N, H, Hkv, M, D, K = 1, 257, 8, 4, 2, 4, 64, 4
groups = H // Hkv
scale = 1.0 / math.sqrt(D)
q = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16)
# argsort produces the same uniqueness invariant as top-k chunk selection.
indices = torch.rand(B, L, Hkv, N, device="cuda").argsort(dim=-1)[..., :K].to(torch.int32)
winners = torch.randint(M, (B, L, H, K), device="cuda", dtype=torch.int32)
grad_scores = torch.randn(B, L, H, K, device="cuda", dtype=torch.float32)

partial_kernel = mass_selected_partial_bwd_kernel(
    B, L, N, H, Hkv, M, D, K, scale, block_q=128
)
reduce_kernel = mass_selected_partial_reduce_kernel(B, L, N, H, M, D, block_q=128)
partials, partial_bias = partial_kernel(q, indices, winners, grad_scores)
grad_mu, grad_bias = reduce_kernel(partials, partial_bias)

indices_h = indices.repeat_interleave(groups, dim=2).long()
dest = (
    torch.arange(B, device="cuda")[:, None, None, None] * (N * H * M)
    + indices_h * (H * M)
    + torch.arange(H, device="cuda")[None, None, :, None] * M
    + winners.long()
)
contrib = (
    q.float().unsqueeze(3) * grad_scores.unsqueeze(-1) * scale
).reshape(-1, D)
dest_flat = dest.reshape(-1)
ref_mu = torch.zeros(B * N * H * M, D, device="cuda", dtype=torch.float32)
ref_bias = torch.zeros(B * N * H * M, device="cuda", dtype=torch.float32)
ref_mu.index_add_(0, dest_flat, contrib)
ref_bias.index_add_(0, dest_flat, grad_scores.reshape(-1))
ref_mu = ref_mu.reshape(B, N, H, M, D).to(torch.bfloat16)
ref_bias = ref_bias.reshape(B, N, H, M)

mu_diff = (grad_mu.float() - ref_mu.float()).abs()
bias_diff = (grad_bias - ref_bias).abs()
print(
    f"mu_max_abs={mu_diff.max().item():.6g} mu_mean_abs={mu_diff.mean().item():.6g} "
    f"bias_max_abs={bias_diff.max().item():.6g} bias_mean_abs={bias_diff.mean().item():.6g}",
    flush=True,
)
assert torch.equal(grad_mu, ref_mu)
assert torch.allclose(grad_bias, ref_bias, atol=2e-5, rtol=2e-5)

