import torch

from ops.mass_retrieval_tilelang import mass_retrieval_tilelang
from ops.mass_retrieval_tilelang_v2 import mass_retrieval_tilelang_v2


torch.manual_seed(29)
B, L, N, S, H, Hkv, M, D, K = 1, 256, 4, 64, 4, 2, 2, 64, 4
q0 = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16)
k0 = torch.randn(B, N, S, Hkv, D, device="cuda", dtype=torch.bfloat16)
a0 = torch.randn(B, N, M, H, D, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(B, L, H, K, device="cuda", dtype=torch.float32)


def run(fn):
    q = q0.clone().requires_grad_(True)
    keys = k0.clone().requires_grad_(True)
    anchors = a0.clone().requires_grad_(True)
    indices, scores = fn(
        q,
        keys,
        anchors,
        num_key_value_heads=Hkv,
        topk=K,
        chunk_size=S,
        window_size=S,
    )
    torch.where(torch.isfinite(scores), scores * weight, 0.0).sum().backward()
    return indices, scores, (q.grad, keys.grad, anchors.grad)


i1, s1, g1 = run(mass_retrieval_tilelang)
i2, s2, g2 = run(mass_retrieval_tilelang_v2)
valid = torch.isfinite(s1) & torch.isfinite(s2)
print(f"indices_equal={torch.equal(i1, i2)}")
print(f"scores_max_abs={(s1[valid] - s2[valid]).abs().max().item():.6g}")
for name, left, right in zip(("q", "keys", "anchors"), g1, g2):
    diff = (left.float() - right.float()).abs()
    denom = left.float().abs().mean().clamp_min(1e-8)
    print(
        f"grad_{name}_max_abs={diff.max().item():.6g} "
        f"grad_{name}_mean_rel={(diff.mean() / denom).item():.6g}"
    )
