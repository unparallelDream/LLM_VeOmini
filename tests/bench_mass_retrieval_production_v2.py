"""One-shot production-shape compile/memory check; not part of pytest."""

import os
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ops.mass_retrieval_tilelang_v2 import mass_retrieval_tilelang_v2


B = int(os.environ.get("MASS_TEST_BATCH", "1"))
L = int(os.environ.get("MASS_TEST_SEQ_LEN", "8192"))
S, H, Hkv, M, D, K = 64, 12, 3, 4, 64, 32
N = L // S
torch.manual_seed(11)
torch.cuda.reset_peak_memory_stats()
q = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
keys = torch.randn(B, N, S, Hkv, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
anchors = torch.randn(B, N, M, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
def run_once():
    indices, scores, _, _ = mass_retrieval_tilelang_v2(
        q,
        keys,
        anchors,
        num_key_value_heads=Hkv,
        topk=K,
        chunk_size=S,
        window_size=512,
    )
    loss = torch.where(torch.isfinite(scores), scores, 0.0).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    return indices, scores


compile_start = time.perf_counter()
indices, scores = run_once()
compile_elapsed = time.perf_counter() - compile_start
q.grad = keys.grad = anchors.grad = None
start = time.perf_counter()
indices, scores = run_once()
elapsed = time.perf_counter() - start
print(
    f"batch={B} compile_elapsed={compile_elapsed:.3f}s elapsed={elapsed:.3f}s "
    f"peak_gib={torch.cuda.max_memory_allocated() / 2**30:.3f} "
    f"indices={tuple(indices.shape)} scores={tuple(scores.shape)} "
    f"grad_q={q.grad.abs().mean().item():.6g} "
    f"grad_k={keys.grad.abs().mean().item():.6g} "
    f"grad_a={anchors.grad.abs().mean().item():.6g}"
)
