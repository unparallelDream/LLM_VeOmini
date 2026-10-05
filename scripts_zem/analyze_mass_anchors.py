#!/usr/bin/env python
"""Extract MASS anchor parameters (weights + Adam states) from DCP checkpoints
and report balance statistics across the M=4 anchors."""
import io
import pickle
import sys
import tempfile
import os

import torch


def load_metadata(ckpt_dir):
    with open(os.path.join(ckpt_dir, ".metadata"), "rb") as f:
        return pickle.load(f)


def load_block(ckpt_dir, relative_path, offset, length):
    with open(os.path.join(ckpt_dir, relative_path), "rb") as f:
        f.seek(offset)
        block = f.read(length)
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    try:
        tmp.write(block)
        tmp.close()
        return torch.load(tmp.name, map_location="cpu", weights_only=False)
    finally:
        os.unlink(tmp.name)


def extract_tensor(ckpt_dir, md, fqn):
    """Reassemble a full tensor from its DCP chunks."""
    meta = md.state_dict_metadata.get(fqn)
    if meta is None:
        return None
    full = torch.zeros(meta.size, dtype=torch.float32)
    for ch in meta.chunks:
        key = None
        for k in md.storage_data.keys():
            if k.fqn == fqn and k.offset is not None and list(k.offset) == list(ch.offsets):
                key = k
                break
        if key is None:
            raise RuntimeError(f"no storage info for {fqn} chunk {ch.offsets}")
        si = md.storage_data[key]
        chunk = load_block(ckpt_dir, si.relative_path, si.offset, si.length).float()
        # slice into place
        sl = tuple(slice(o, o + s) for o, s in zip(ch.offsets, ch.sizes))
        full[sl] = chunk
    return full


def main():
    ckpt_dir = sys.argv[1]
    md = load_metadata(ckpt_dir)
    fqns = [k for k in md.state_dict_metadata.keys()
            if "self_attn.mass_anchor_generator" in k
            and (k.startswith("model.") or k.startswith("optimizer.state."))]
    weights = sorted(k for k in fqns if ".exp_avg" not in k and ".step" not in k and k.startswith("model."))
    exp_avg = sorted(k for k in fqns if ".exp_avg" in k)
    print(f"checkpoint: {ckpt_dir}")
    print(f"mass anchor 张量: {len(weights)} (weights) + {len(exp_avg)} (exp_avg)")

    layers = sorted({int(k.split("layers.")[1].split(".")[0]) for k in weights})
    n_anchor, n_emb = 4, 8
    print(f"\n=== 每层 anchor_embedding 行范数 ||e_m|| (期望随训练分化) ===")
    print(f"{'layer':>5} | " + " | ".join(f"a{m} L2" for m in range(n_anchor)) + " | " + " | ".join(f"a{m} exp_avg" for m in range(n_anchor)))
    all_offs_norm = []
    all_emb_norm = []
    all_expavg_norm = []
    for li in layers:
        emb = extract_tensor(ckpt_dir, md, f"model.model.layers.{li}.self_attn.mass_anchor_generator.anchor_embedding")
        basis = extract_tensor(ckpt_dir, md, f"model.model.layers.{li}.self_attn.mass_anchor_generator.offset_basis")
        ea = extract_tensor(ckpt_dir, md, f"model.model.layers.{li}.self_attn.mass_anchor_generator.anchor_embedding.exp_avg")
        if emb is None or basis is None:
            print(f"layer {li}: MISSING"); continue
        # offsets[m] = emb[m] @ basis^T  → shape [M, Hq, D]
        offsets = torch.einsum("mr,hdr->mhd", emb, basis)
        off_norms = offsets.norm(dim=(-2, -1))      # per-anchor perturbation magnitude
        emb_norms = emb.norm(dim=-1)                 # per-anchor embedding row norm
        ea_norms = ea.norm(dim=-1) if ea is not None else torch.zeros(n_anchor)
        all_offs_norm.append(off_norms)
        all_emb_norm.append(emb_norms)
        all_expavg_norm.append(ea_norms)
        print(f"{li:>5} | " + " | ".join(f"{v:6.4f}" for v in off_norms.tolist()) + " | " + " | ".join(f"{v:6.4f}" for v in ea_norms.tolist()))

    offs = torch.stack(all_offs_norm)   # [L, M]
    embs = torch.stack(all_emb_norm)
    eas = torch.stack(all_expavg_norm)
    print(f"\n=== 汇总(20 层) ===")
    for m in range(n_anchor):
        print(f"anchor{m}: 偏移范数 mean={offs[:, m].mean():.4f} std={offs[:, m].std():.4f} "
              f"| embedding 行范数 mean={embs[:, m].mean():.4f} std={embs[:, m].std():.4f} "
              f"| exp_avg 范数 mean={eas[:, m].mean():.4f} std={eas[:, m].std():.4f}")
    # 每层内 anchor 偏移范数的极差比(不平衡指标)
    ratios = offs.max(dim=1).values / offs.min(dim=1).values
    print(f"\n每层 anchor 偏移范数 max/min 比: mean={ratios.mean():.2f}, max={ratios.max():.2f}, min={ratios.min():.2f}")
    # anchor 间相似度: 每层 4 个 offsets [M,H,D] 的余弦相似矩阵
    cos_sims = []
    for li in layers:
        basis = extract_tensor(ckpt_dir, md, f"model.model.layers.{li}.self_attn.mass_anchor_generator.offset_basis")
        emb = extract_tensor(ckpt_dir, md, f"model.model.layers.{li}.self_attn.mass_anchor_generator.anchor_embedding")
        offsets = torch.einsum("mr,hdr->mhd", emb, basis).flatten(1)
        norms = offsets.norm(dim=1, keepdim=True)
        sim = (offsets @ offsets.T) / (norms @ norms.T).clamp_min(1e-9)
        cos_sims.append(sim)
    cs = torch.stack(cos_sims)
    print(f"\n=== anchor 偏移间余弦相似度(20 层均值) ===")
    print("        " + " ".join(f"a{m:>7}" for m in range(n_anchor)))
    for i in range(n_anchor):
        print(f"a{i}: " + " ".join(f"{cs[:, i, j].mean():8.4f}" for j in range(n_anchor)))
    # 每层偏置项(offsets 总能量 vs 初始化)
    print(f"\n初始化参照: offset_basis std=0.02, anchor_embedding std=0.02 → 期望偏移范数≈{0.02 * (8) ** 0.5 * 0.02 * (64 * 8) ** 0.5:.3f}(理论) ")


if __name__ == "__main__":
    main()
