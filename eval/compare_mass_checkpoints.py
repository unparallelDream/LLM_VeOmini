#!/usr/bin/env python3
"""Compare raw and effective MASS anchor weights across DCP checkpoints."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch
import torch.nn.functional as F


def stats(x: torch.Tensor) -> dict[str, float]:
    y = x.float()
    return {
        "norm": float(y.norm()),
        "rms": float(y.square().mean().sqrt()),
        "abs_max": float(y.abs().max()),
        "mean": float(y.mean()),
        "std": float(y.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-a", required=True)
    parser.add_argument("--step-b", required=True)
    args = parser.parse_args()

    from veomni.checkpoint import ckpt_to_state_dict

    a = {k: v.detach().cpu().float() for k, v in ckpt_to_state_dict(args.step_a).items()}
    b = {k: v.detach().cpu().float() for k, v in ckpt_to_state_dict(args.step_b).items()}
    if set(a) != set(b):
        raise RuntimeError(
            f"checkpoint key mismatch: only_a={sorted(set(a)-set(b))}, "
            f"only_b={sorted(set(b)-set(a))}"
        )

    by_layer: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    rows = []
    for key in sorted(a):
        layer = key.split(".layers.", 1)[1].split(".", 1)[0]
        leaf = key.rsplit(".", 1)[-1]
        by_layer[layer][f"a_{leaf}"] = a[key]
        by_layer[layer][f"b_{leaf}"] = b[key]
        delta = b[key] - a[key]
        rows.append(
            {
                "key": key,
                "shape": list(a[key].shape),
                "exact_equal": bool(torch.equal(a[key], b[key])),
                "a": stats(a[key]),
                "b": stats(b[key]),
                "delta": stats(delta),
                "relative_delta_norm": float(delta.norm() / a[key].norm().clamp_min(1e-30)),
                "cosine_a_b": float(F.cosine_similarity(a[key].flatten(), b[key].flatten(), dim=0)),
            }
        )

    effective = []
    for layer in sorted(by_layer, key=int):
        tensors = by_layer[layer]
        ea, eb = tensors["a_anchor_embedding"], tensors["b_anchor_embedding"]
        ba, bb = tensors["a_offset_basis"], tensors["b_offset_basis"]
        # [M,R] x [H,D,R] -> [M,H,D], exactly as the training model.
        oa = torch.einsum("mr,hdr->mhd", ea, ba)
        ob = torch.einsum("mr,hdr->mhd", eb, bb)
        delta = ob - oa
        pair_a = torch.cdist(oa.flatten(1), oa.flatten(1))
        pair_b = torch.cdist(ob.flatten(1), ob.flatten(1))
        offdiag = ~torch.eye(ea.shape[0], dtype=torch.bool)
        row_cos_a_b = F.cosine_similarity(oa.flatten(1), ob.flatten(1), dim=1)

        # Rank/singular values reveal whether eight landmarks collapse into a
        # much smaller number of effective directions.
        sva = torch.linalg.svdvals(oa.flatten(1))
        svb = torch.linalg.svdvals(ob.flatten(1))
        effective.append(
            {
                "layer": int(layer),
                "effective_offset_a": stats(oa),
                "effective_offset_b": stats(ob),
                "effective_delta": stats(delta),
                "relative_effective_delta_norm": float(delta.norm() / oa.norm().clamp_min(1e-30)),
                "effective_cosine_a_b": float(F.cosine_similarity(oa.flatten(), ob.flatten(), dim=0)),
                "per_anchor_cosine_a_b": [float(x) for x in row_cos_a_b],
                "anchor_offset_norms_a": [float(x) for x in oa.flatten(1).norm(dim=1)],
                "anchor_offset_norms_b": [float(x) for x in ob.flatten(1).norm(dim=1)],
                "pairwise_distance_a_mean": float(pair_a[offdiag].mean()),
                "pairwise_distance_a_min": float(pair_a[offdiag].min()),
                "pairwise_distance_b_mean": float(pair_b[offdiag].mean()),
                "pairwise_distance_b_min": float(pair_b[offdiag].min()),
                "singular_values_a": [float(x) for x in sva],
                "singular_values_b": [float(x) for x in svb],
            }
        )

    result = {
        "step_a": args.step_a,
        "step_b": args.step_b,
        "key_count": len(a),
        "all_raw_tensors_changed": all(not row["exact_equal"] for row in rows),
        "raw_parameters": rows,
        "effective_offsets": effective,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
