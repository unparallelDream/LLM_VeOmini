import json, sys, time, statistics
import torch
from types import SimpleNamespace
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, '/9950backfile/zhangem/HiLS-Attention-main')
from models.MASSIndex.mass_attention import MASSAttention
from models.MASSIndex.mass_index import MASSIndexConfig

CFG = '/9950backfile/zhangem/HiLS-Attention-main/configs/mass_attention/config_mass_qwen3_0p6B_8KA2K_RoPE_m2_r8_asym0p5_anneal.json'
with open(CFG) as f:
    j = json.load(f)
skip = {'architectures', 'model_type', 'torch_dtype', 'rope_scaling', 'use_cache',
        'use_sliding_window', 'max_position_embeddings', 'num_hidden_layers',
        'intermediate_size', 'vocab_size', 'rms_norm_eps', 'hidden_act',
        'tie_word_embeddings', 'attention_bias', 'attention_dropout',
        'bos_token_id', 'eos_token_id', 'pad_token_id'}
cfg = SimpleNamespace(**{k: v for k, v in j.items() if k not in skip})
for k, v in dict(layerwise_lmkq_norm=False, lmk_q_lora_dim=-1, shared_q_c=False).items():
    if not hasattr(cfg, k):
        setattr(cfg, k, v)

mass_cfg = MASSIndexConfig(
    num_anchors=int(j['mass_num_anchors']), offset_rank=int(j['mass_offset_rank']),
    anchor_rms_norm=bool(j['mass_anchor_rms_norm']), offset_init_std=float(j['mass_offset_init_std']),
    score_reduction=j['mass_score_reduction'], lse_temperature=float(j['mass_lse_temperature']),
    normalize_lse=bool(j['mass_normalize_lse']),
)

torch.manual_seed(42)
layer = MASSAttention(cfg, 0, norm_cls=torch.nn.RMSNorm, mass_config=mass_cfg).to(torch.bfloat16).cuda()
B, L, D, hd = 4, 8322, 1024, 128
hidden = torch.randn(B, L, D, dtype=torch.bfloat16, device='cuda', requires_grad=True)
freqs = (1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))).to('cuda')
pos = torch.arange(L, dtype=torch.float32, device='cuda')
angles = torch.outer(pos, freqs)
cos = torch.cat([angles.cos(), angles.cos()], dim=-1).unsqueeze(0).to(torch.bfloat16)
sin = torch.cat([angles.sin(), angles.sin()], dim=-1).unsqueeze(0).to(torch.bfloat16)
pos_emb = (cos, sin)

print('warming up (JIT compile)...')
with torch.no_grad():
    layer(hidden, position_embeddings=pos_emb)
torch.cuda.synchronize()

def run():
    if hidden.grad is not None:
        hidden.grad = None
    o = layer(hidden, position_embeddings=pos_emb)[0]
    o.float().pow(2).mean().backward()

print('=== torch.profiler (fwd+bwd) ===')
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    run()
torch.cuda.synchronize()
print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=28))

ts = []
for _ in range(3):
    torch.cuda.synchronize(); t0 = time.time()
    run()
    torch.cuda.synchronize(); ts.append(time.time() - t0)
print(f'=== single layer fwd+bwd wall time (median of 3): {statistics.median(ts):.2f} s ===')
print(f'-> 28 layers x 2 microbatch: {statistics.median(ts)*56:.1f} s per step (excl. other ops)')
