import sys, time
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

def fwd():
    return mass_retrieval_tilelang_v2(queries, key_chunks, anchors, **kw)

print('warmup (JIT)...')
r = fwd(); torch.cuda.synchronize()
print(f'fwd returns: indices{tuple(r[0].shape)} scores{tuple(r[1].shape)} load{None if r[2] is None else tuple(r[2].shape)} ent{None if r[3] is None else tuple(r[3].shape)}')

fwds = []
for _ in range(4):
    torch.cuda.synchronize(); t0 = time.time()
    r = fwd()
    torch.cuda.synchronize(); fwds.append((time.time() - t0) * 1000)
print('fwd wall ms:', [round(t, 1) for t in fwds])

bwds = []
for _ in range(4):
    for p in (queries, key_chunks, anchors):
        if p.grad is not None:
            p.grad = None
    r = fwd()
    scores, anchor_load = r[1], r[2]
    torch.cuda.synchronize(); t0 = time.time()
    loss = scores.float().square().mean() + anchor_load.float().square().mean()
    loss.backward()
    torch.cuda.synchronize(); bwds.append((time.time() - t0) * 1000)
print('bwd wall ms:', [round(t, 1) for t in bwds])
print(f'-> per-layer retrieval fwd+bwd median: {(sorted(fwds)[1]+sorted(bwds)[1]):.0f} ms')
