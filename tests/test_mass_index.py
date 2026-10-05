import math

import torch

from models.MASSIndex.mass_index import (
    MASSIndexConfig,
    MultiTangentAnchorGenerator,
    build_mass_tangent_summary,
    mass_topk_reference,
    upper_envelope_scores,
)


def _manual_pool(anchor, keys):
    logits = torch.einsum("bnhd,bnshd->bnsh", anchor.float(), keys.float()) / math.sqrt(anchor.shape[-1])
    logits[:, :, -1, :] = float("-inf")
    p = torch.softmax(logits, dim=2)
    mu = torch.einsum("bnsh,bnshd->bnhd", p, keys.float())
    log_p = torch.log_softmax(logits, dim=2)
    b = -(p * torch.where(torch.isfinite(log_p), log_p, torch.zeros_like(log_p))).sum(dim=2)
    return mu, b


def test_single_anchor_matches_single_tangent_pooling():
    torch.manual_seed(0)
    base = torch.randn(2, 3, 4, 8)
    keys = torch.randn(2, 3, 5, 4, 8, dtype=torch.bfloat16)
    cfg = MASSIndexConfig(num_anchors=1, offset_rank=2, anchor_rms_norm=False)
    generator = MultiTangentAnchorGenerator(4, 8, cfg)
    with torch.no_grad():
        generator.anchor_embedding.zero_()
        generator.offset_basis.zero_()

    summary = build_mass_tangent_summary(base, keys, generator)
    expected_mu, expected_b = _manual_pool(base, keys)
    torch.testing.assert_close(summary.mu[:, :, 0].float(), expected_mu, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(summary.b[:, :, 0], expected_b, atol=1e-5, rtol=1e-5)


def test_upper_envelope_returns_best_anchor():
    q = torch.tensor([[[[1.0, 0.0]]]])
    mu = torch.tensor([[[[[1.0, 0.0]], [[0.0, 1.0]]]]])
    b = torch.tensor([[[[0.0], [2.0]]]])
    scores, winner = upper_envelope_scores(q, mu, b, sm_scale=1.0)
    assert scores.item() == 2.0
    assert winner.item() == 1


def test_logsumexp_envelope_matches_reference_and_reaches_low_planes():
    q = torch.tensor([[[[1.0, 0.0]]]])
    mu = torch.tensor(
        [[[[[1.0, 0.0]], [[0.0, 1.0]], [[-2.0, 0.0]]]]],
        requires_grad=True,
    )
    b = torch.tensor([[[[0.0], [2.0], [-1.0]]]], requires_grad=True)
    temperature = 0.7
    scores, winner = upper_envelope_scores(
        q,
        mu,
        b,
        sm_scale=1.0,
        score_reduction="logsumexp",
        lse_temperature=temperature,
    )
    planes = torch.tensor([1.0, 2.0, -3.0])
    expected = temperature * torch.logsumexp(planes / temperature, dim=0)
    torch.testing.assert_close(scores.squeeze(), expected)
    assert winner.item() == 1

    scores.sum().backward()
    # The lowest-scoring plane still receives a non-zero scoring derivative.
    assert torch.all(b.grad.squeeze() > 0)
    assert mu.grad[0, 0, 2].abs().sum() > 0


def test_offsets_receive_gradient_and_gqa_topk_is_causal():
    torch.manual_seed(1)
    base = torch.randn(1, 4, 4, 4, requires_grad=True)
    keys = torch.randn(1, 4, 4, 2, 4)
    generator = MultiTangentAnchorGenerator(4, 4, MASSIndexConfig(num_anchors=3, offset_rank=2))
    summary = build_mass_tangent_summary(base, keys, generator)
    query = torch.randn(1, 4, 4, 4)
    envelope, winner = upper_envelope_scores(query, summary.mu, summary.b)
    envelope.sum().backward()
    assert generator.offset_basis.grad is not None
    assert generator.offset_basis.grad.abs().sum() > 0

    result = mass_topk_reference(
        envelope.detach(), winner.detach(), num_key_value_heads=2, topk=2, chunk_size=1, window_size=0
    )
    assert result.indices.shape == (1, 4, 2, 2)
    # With no retrieval window, query 0 has no eligible historical chunk.
    assert torch.equal(result.indices[:, 0], torch.full((1, 2, 2), -1, dtype=torch.long))
