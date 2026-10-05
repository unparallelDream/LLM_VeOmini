import os, sys
import torch

sys.path.insert(0, '/9950backfile/zhangem/HiLS-Attention-main')
from ops.mass_retrieval_tilelang_v2 import mass_retrieval_tilelang_v2

torch.manual_seed(42)
B, L, H, Hkv, D, M, S, N, topk = 4, 8322, 16, 8, 128, 2, 64, 130, 32
queries = torch.randn(B, L, H, D, dtype=torch.bfloat16, device='cuda', requires_grad=True)
key_chunks = torch.randn(B, N, S, Hkv, D, dtype=torch.bfloat16, device='cuda', requires_grad=True)
anchors = torch.randn(B, N, M, H, D, dtype=torch.bfloat16, device='cuda', requires_grad=True)
kw = dict(num_key_value_heads=Hkv, topk=topk, chunk_size=S, window_size=512,
          sm_scale=128 ** -0.5, score_reduction='logsumexp', lse_temperature=2.05)

def run(env):
    os.environ['MASS_TWO_STAGE_BACKWARD'] = env
    for p in (queries, key_chunks, anchors):
        if p.grad is not None:
            p.grad = None
    r = mass_retrieval_tilelang_v2(queries, key_chunks, anchors, **kw)
    scores, anchor_load, cent = r[1], r[2], r[3]
    loss = scores.float().square().mean() + anchor_load.float().square().mean()
    if cent is not None:
        loss = loss + cent.float() * 0.1
    loss.backward()
    return (queries.grad.float().cpu(), key_chunks.grad.float().cpu(), anchors.grad.float().cpu())

g0 = run('0')
print('direct path done')
g1 = run('1')
print('two-stage path done')

names = ['grad_queries', 'grad_key_chunks', 'grad_anchors']
for n, a, b in zip(names, g0, g1):
    md = (a - b).abs().max().item()
    mx = a.abs().max().item()
    denom = a.abs().clamp_min(1e-6)
    rel = ((a - b).abs() / denom).median().item()
    print(f'{n}: max_abs_diff={md:.3e}, max|grad|={mx:.3e}, median_rel_err={rel:.3e}')
