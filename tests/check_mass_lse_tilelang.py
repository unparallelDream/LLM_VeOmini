import math

import torch

from ops.mass_retrieval_tilelang_v2 import mass_retrieval_tilelang_v2
import ops.mass_retrieval_tilelang_v2 as mass_v2


def main():
    print("TEST_START", flush=True)
    torch.manual_seed(7)
    device = "cuda"
    # Use the production-aligned S/D/M multiples required by the summary MMA.
    B, L, N, S, H, Hkv, M, D, K = 1, 192, 3, 64, 4, 4, 8, 128, 2
    q = torch.randn(B, L, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    keys = torch.randn(B, N, S, Hkv, D, device=device, dtype=torch.bfloat16)
    anchors = torch.randn(B, N, M, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    anchors_ref = anchors.detach().clone().requires_grad_(True)
    scale = 1.0 / math.sqrt(D)
    captured = {}
    original_summary_backward = mass_v2._MassSummaryFn.backward

    def capture_summary_backward(ctx, grad_mu, grad_bias):
        captured["grad_mu"] = None if grad_mu is None else grad_mu.detach().clone()
        captured["grad_bias"] = None if grad_bias is None else grad_bias.detach().clone()
        return original_summary_backward(ctx, grad_mu, grad_bias)

    mass_v2._MassSummaryFn.backward = staticmethod(capture_summary_backward)
    print("CALL_RETRIEVAL", flush=True)
    indices, scores, load, conditional_entropy = mass_retrieval_tilelang_v2(
        q,
        keys,
        anchors,
        num_key_value_heads=Hkv,
        topk=K,
        chunk_size=S,
        window_size=S,
        sm_scale=scale,
        score_reduction="logsumexp",
        lse_temperature=1.0,
    )
    print("RETRIEVAL_DONE", flush=True)
    assert indices.shape == (B, L, Hkv, K)
    assert scores.shape == (B, L, H, K)
    assert load.shape == (M,)
    assert conditional_entropy.ndim == 0
    torch.testing.assert_close(load.sum(), torch.ones((), device=device), atol=2e-4, rtol=2e-4)
    finite_scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
    marginal_entropy = -(load * load.clamp_min(torch.finfo(torch.float32).tiny).log()).sum()
    specialization_loss = (
        conditional_entropy - marginal_entropy + math.log(M)
    ) / math.log(M)
    loss = finite_scores.mean() + 0.01 * specialization_loss
    loss.backward()
    print("BACKWARD_DONE", flush=True)
    mass_v2._MassSummaryFn.backward = staticmethod(original_summary_backward)

    # PyTorch reference for the same fixed selected chunks.  Top-K indices are
    # intentionally treated as non-differentiable in both paths.
    logits = torch.einsum(
        "bnmhd,bnshd->bnmsh", anchors_ref.float(), keys.float()
    ) * scale
    logits[:, :, :, -1, :] = float("-inf")
    pool_probs = torch.softmax(logits, dim=3)
    mu_ref = torch.einsum("bnmsh,bnshd->bnmhd", pool_probs, keys.float())
    log_probs = torch.log_softmax(logits, dim=3)
    bias_ref = -(
        pool_probs
        * torch.where(torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs))
    ).sum(dim=3)
    mu_ref.retain_grad()
    bias_ref.retain_grad()
    planes = torch.einsum("blhd,bnmhd->blnmh", q_ref.float(), mu_ref) * scale
    planes = (planes + bias_ref[:, None]).permute(0, 1, 4, 2, 3)
    valid = indices.ge(0)
    safe_indices = indices.clamp(0, N - 1).long()
    selected = torch.gather(
        planes,
        3,
        safe_indices.unsqueeze(-1).expand(B, L, H, K, M),
    ).masked_fill(~valid.unsqueeze(-1), float("-inf"))
    scores_ref = torch.logsumexp(selected, dim=-1)
    safe_selected = selected.masked_fill(~valid.unsqueeze(-1), 0.0)
    routing_ref = torch.softmax(safe_selected, dim=-1).masked_fill(
        ~valid.unsqueeze(-1), 0.0
    )
    load_ref = routing_ref.sum(dim=(0, 1, 2, 3)) / valid.sum().clamp_min(1)
    conditional_entropy_ref = -(
        routing_ref
        * routing_ref.clamp_min(torch.finfo(torch.float32).tiny).log()
    ).sum(dim=-1)
    conditional_entropy_ref = conditional_entropy_ref.masked_fill(
        ~valid, 0.0
    ).sum() / valid.sum().clamp_min(1)
    marginal_entropy_ref = -(
        load_ref * load_ref.clamp_min(torch.finfo(torch.float32).tiny).log()
    ).sum()
    specialization_loss_ref = (
        conditional_entropy_ref - marginal_entropy_ref + math.log(M)
    ) / math.log(M)
    loss_ref = torch.where(
        torch.isfinite(scores_ref), scores_ref, torch.zeros_like(scores_ref)
    ).mean() + 0.01 * specialization_loss_ref
    loss_ref.backward()

    mu_kernel = captured["grad_mu"].float()
    mu_expected = mu_ref.grad.permute(0, 1, 3, 2, 4).float()
    bias_kernel = captured["grad_bias"].float()
    bias_expected = bias_ref.grad.permute(0, 1, 3, 2).float()
    print(
        "grad_mu_diag",
        float(mu_kernel.abs().max()),
        float(mu_expected.abs().max()),
        float((mu_kernel - mu_expected).abs().max()),
        float(torch.linalg.vector_norm(mu_kernel)),
        float(torch.linalg.vector_norm(mu_expected)),
    )
    print(
        "grad_bias_diag",
        float(bias_kernel.abs().max()),
        float(bias_expected.abs().max()),
        float((bias_kernel - bias_expected).abs().max()),
        float(torch.linalg.vector_norm(bias_kernel)),
        float(torch.linalg.vector_norm(bias_expected)),
    )
    torch.testing.assert_close(
        mu_kernel,
        mu_expected,
        atol=4e-3,
        rtol=1.5e-1,
    )
    torch.testing.assert_close(
        bias_kernel,
        bias_expected,
        atol=4e-3,
        rtol=1.5e-1,
    )

    torch.testing.assert_close(scores, scores_ref, atol=4e-2, rtol=4e-2, equal_nan=True)
    torch.testing.assert_close(load, load_ref, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(
        conditional_entropy,
        conditional_entropy_ref,
        atol=3e-3,
        rtol=3e-3,
    )
    torch.testing.assert_close(q.grad.float(), q_ref.grad.float(), atol=3e-3, rtol=8e-2)
    torch.testing.assert_close(
        anchors.grad.float(), anchors_ref.grad.float(), atol=4e-3, rtol=1.5e-1
    )
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert anchors.grad is not None and torch.isfinite(anchors.grad).all()
    per_anchor = anchors.grad.float().abs().sum(dim=(0, 1, 3, 4))
    assert torch.all(per_anchor > 0), per_anchor
    print("MASS_LSE_TILELANG_OK")
    print("indices", tuple(indices.shape), "scores", tuple(scores.shape))
    print("load", [round(x, 6) for x in load.detach().cpu().tolist()])
    print("conditional_entropy", round(float(conditional_entropy.detach()), 6))
    print("marginal_entropy", round(float(marginal_entropy.detach()), 6))
    print("specialization_loss", round(float(specialization_loss.detach()), 6))
    print("anchor_grad_l1", [round(x, 6) for x in per_anchor.detach().cpu().tolist()])
    print("reference_match", True)


if __name__ == "__main__":
    main()
