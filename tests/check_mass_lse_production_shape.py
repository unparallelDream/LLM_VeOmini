import math
import time

import torch

from ops.mass_retrieval_tilelang_v2 import mass_retrieval_tilelang_v2


def main():
    torch.manual_seed(11)
    B, L, N, S, H, Hkv, M, D, K = 1, 8322, 130, 64, 32, 32, 8, 128, 32
    device = "cuda"
    q = torch.randn(B, L, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    keys = torch.randn(B, N, S, Hkv, D, device=device, dtype=torch.bfloat16)
    anchors = torch.randn(B, N, M, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    indices, scores, load, conditional_entropy = mass_retrieval_tilelang_v2(
        q,
        keys,
        anchors,
        num_key_value_heads=Hkv,
        topk=K,
        chunk_size=S,
        window_size=512,
        sm_scale=1.0 / math.sqrt(D),
        score_reduction="logsumexp",
        lse_temperature=1.0,
    )
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - start
    finite = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
    marginal_entropy = -(load * load.clamp_min(torch.finfo(torch.float32).tiny).log()).sum()
    specialization_loss = (
        conditional_entropy - marginal_entropy + math.log(M)
    ) / math.log(M)
    loss = finite.mean() + 0.01 * specialization_loss
    start = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    backward_seconds = time.perf_counter() - start

    # Report warm execution separately from first-process compilation.
    q2 = q.detach().clone().requires_grad_(True)
    anchors2 = anchors.detach().clone().requires_grad_(True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    _, scores2, load2, conditional_entropy2 = mass_retrieval_tilelang_v2(
        q2,
        keys,
        anchors2,
        num_key_value_heads=Hkv,
        topk=K,
        chunk_size=S,
        window_size=512,
        sm_scale=1.0 / math.sqrt(D),
        score_reduction="logsumexp",
        lse_temperature=1.0,
    )
    torch.cuda.synchronize()
    warm_forward_seconds = time.perf_counter() - start
    loss2 = torch.where(torch.isfinite(scores2), scores2, torch.zeros_like(scores2)).mean()
    marginal_entropy2 = -(
        load2 * load2.clamp_min(torch.finfo(torch.float32).tiny).log()
    ).sum()
    specialization_loss2 = (
        conditional_entropy2 - marginal_entropy2 + math.log(M)
    ) / math.log(M)
    loss2 = loss2 + 0.01 * specialization_loss2
    start = time.perf_counter()
    loss2.backward()
    torch.cuda.synchronize()
    warm_backward_seconds = time.perf_counter() - start
    per_anchor = anchors.grad.float().abs().sum(dim=(0, 1, 3, 4))
    assert torch.isfinite(scores[torch.isfinite(scores)]).all()
    assert torch.isfinite(q.grad).all() and torch.isfinite(anchors.grad).all()
    assert torch.all(per_anchor > 0), per_anchor
    print("MASS_LSE_PRODUCTION_SHAPE_OK")
    print("shapes", tuple(indices.shape), tuple(scores.shape), tuple(load.shape))
    print("load", [round(x, 6) for x in load.detach().cpu().tolist()])
    print("conditional_entropy", round(float(conditional_entropy.detach()), 6))
    print("marginal_entropy", round(float(marginal_entropy.detach()), 6))
    print("specialization_loss", round(float(specialization_loss.detach()), 6))
    print("anchor_grad_l1", [round(x, 6) for x in per_anchor.detach().cpu().tolist()])
    print("peak_allocated_gib", round(torch.cuda.max_memory_allocated() / 2**30, 3))
    print("forward_seconds", round(forward_seconds, 4))
    print("backward_seconds", round(backward_seconds, 4))
    print("warm_forward_seconds", round(warm_forward_seconds, 4))
    print("warm_backward_seconds", round(warm_backward_seconds, 4))


if __name__ == "__main__":
    main()
