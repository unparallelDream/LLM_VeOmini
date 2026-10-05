import os, sys, time, math
import torch

sys.path.insert(0, '/9950backfile/zhangem/HiLS-Attention-main')
from ops.hils_fwd_bwd_head import HiLS_block_M_head, build_block_indices_block_M

torch.manual_seed(42)
B, q_len, kv_len, H, HQ, D, S, block_size = 4, 8322, 8322, 8, 16, 128, 32, 64
dtype, device, block_M = torch.bfloat16, 'cuda', 8
G = HQ // H
scale = 1.0 / math.sqrt(D)

block_indices = build_block_indices_block_M(
    B=B, SEQ_LEN=q_len, H=H, S=S, block_size=block_size,
    overlap_ratio=0.8, block_M=block_M, device=device, kv_len=kv_len,
)

def make_tensors():
    Q = torch.randn((B, q_len, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, kv_len, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, kv_len, H, D), dtype=dtype, device=device, requires_grad=True)
    W = torch.randn((B, q_len, HQ, S), dtype=dtype, device=device, requires_grad=True)
    return Q, K, V, W

Q, K, V, W = make_tensors()
g = torch.randn((B, q_len, HQ, D), dtype=dtype, device=device)

def run_once():
    for p in (Q, K, V, W):
        if p.grad is not None:
            p.grad = None
    torch.cuda.synchronize(); t0 = time.time()
    o = HiLS_block_M_head(Q, K, V, W, block_indices, block_size=block_size,
                          sm_scale=scale, block_M=block_M, is_training=True)
    torch.cuda.synchronize(); fwd = (time.time() - t0) * 1000
    torch.cuda.synchronize(); t0 = time.time()
    o.backward(g)
    torch.cuda.synchronize(); bwd = (time.time() - t0) * 1000
    return fwd, bwd, o

mode = sys.argv[1] if len(sys.argv) > 1 else 'base'
print(f'[mode={mode}] compiling + warmup (first call includes JIT)...')
try:
    f, b, o = run_once()
    print(f'[mode={mode}] warmup ok  fwd={f:.1f}ms bwd={b:.1f}ms')
except Exception as e:
    import traceback; traceback.print_exc()
    print(f'[mode={mode}] COMPILE/RUN FAILED: {type(e).__name__}: {e}')
    sys.exit(1)

times = [run_once()[:2] for _ in range(7)]
fwds = sorted(t for t, _ in times); bwds = sorted(b for _, b in times)
fwd = fwds[len(fwds) // 2]; bwd = bwds[len(bwds) // 2]
print(f'[mode={mode}] median fwd={fwd:.1f}ms bwd={bwd:.1f}ms total={fwd + bwd:.1f}ms')
print(f'[mode={mode}] all fwd={[round(t,1) for t in fwds]} all bwd={[round(b,1) for b in bwds]}')

if mode == 'base':
    torch.save({'o': o.detach().float().cpu(),
                'gQ': Q.grad.detach().float().cpu(), 'gK': K.grad.detach().float().cpu(),
                'gV': V.grad.detach().float().cpu(), 'gW': W.grad.detach().float().cpu()},
               '/tmp/hils_base.pt')
    print('[mode=base] baseline saved to /tmp/hils_base.pt')
else:
    base = torch.load('/tmp/hils_base.pt', weights_only=False)
    checks = [('o', o.detach().float().cpu(), base['o']),
              ('gQ', Q.grad.detach().float().cpu(), base['gQ']),
              ('gK', K.grad.detach().float().cpu(), base['gK']),
              ('gV', V.grad.detach().float().cpu(), base['gV']),
              ('gW', W.grad.detach().float().cpu(), base['gW'])]
    for n, cur, ref in checks:
        md = (cur - ref).abs().max().item()
        rel = md / (ref.abs().max().item() + 1e-12)
        print(f'[mode={mode}] {n}: max_abs_diff={md:.3e} rel={rel:.3e}')
