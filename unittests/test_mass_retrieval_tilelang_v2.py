import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="TileLang requires CUDA")

from models.MASSIndex.mass_index import mass_topk_reference
from ops.mass_retrieval_tilelang_v2 import mass_retrieval_tilelang_v2


def _reference(q, keys, anchors, hkv, topk, chunk_size, window_size):
    groups = q.shape[2] // hkv
    scale = 1.0 / math.sqrt(q.shape[-1])
    key_heads = keys.repeat_interleave(groups, dim=3)
    logits = torch.einsum("bnmhd,bnshd->bnmhs", anchors.float(), key_heads.float()) * scale
    logits[..., -1] = float("-inf")
    p = torch.softmax(logits, dim=-1)
    mu = torch.einsum("bnmhs,bnshd->bnmhd", p, key_heads.float()).to(keys.dtype)
    log_p = torch.log_softmax(logits, dim=-1)
    bias = -(p * torch.where(torch.isfinite(log_p), log_p, 0.0)).sum(-1)
    planes = torch.einsum("blhd,bnmhd->blnmh", q.float(), mu.float()) * scale
    envelope, winners = (planes + bias[:, None]).max(dim=3)
    result = mass_topk_reference(
        envelope,
        winners,
        num_key_value_heads=hkv,
        topk=topk,
        chunk_size=chunk_size,
        window_size=window_size,
    )
    return result.indices, result.selected_scores


def test_fused_mass_forward_and_backward():
    torch.manual_seed(7)
    B, L, N, S, H, Hkv, M, D, K = 1, 128, 2, 64, 4, 2, 2, 64, 2
    q0 = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16)
    keys0 = torch.randn(B, N, S, Hkv, D, device="cuda", dtype=torch.bfloat16)
    anchors0 = torch.randn(B, N, M, H, D, device="cuda", dtype=torch.bfloat16)
    q_ref, q = q0.clone().requires_grad_(True), q0.clone().requires_grad_(True)
    keys_ref, keys = keys0.clone().requires_grad_(True), keys0.clone().requires_grad_(True)
    anchors_ref = anchors0.clone().requires_grad_(True)
    anchors = anchors0.clone().requires_grad_(True)
    ref_indices, ref_scores = _reference(q_ref, keys_ref, anchors_ref, Hkv, K, S, S)
    indices, scores, _, _ = mass_retrieval_tilelang_v2(
        q,
        keys,
        anchors,
        num_key_value_heads=Hkv,
        topk=K,
        chunk_size=S,
        window_size=S,
    )
    torch.testing.assert_close(indices.long(), ref_indices)
    valid = indices.repeat_interleave(H // Hkv, dim=2).ge(0)
    torch.testing.assert_close(scores[valid], ref_scores[valid], atol=8e-2, rtol=8e-2)
    weight = torch.randn_like(scores)
    torch.where(torch.isfinite(ref_scores), ref_scores * weight, 0.0).sum().backward()
    torch.where(torch.isfinite(scores), scores * weight, 0.0).sum().backward()
    for tensor, reference in zip((q, keys, anchors), (q_ref, keys_ref, anchors_ref)):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
        torch.testing.assert_close(
            tensor.grad.float(), reference.grad.float(), atol=2.5e-1, rtol=2.5e-1
        )
