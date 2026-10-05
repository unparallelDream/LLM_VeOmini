"""Extract an anchor-only DCP checkpoint from a full-model DCP checkpoint.

pretrain_mass_with_ruler.py saves trainable-only checkpoints only when
MASS_TRAINABLE_ONLY_CHECKPOINT=1; runs launched without it (e.g. the
asym0p5_anneal ft20000 pair) store the full model.  eval_ruler_*_bsa_4k.py
requires the checkpoint to contain exactly the anchor tensors, so this
script re-writes the 14 anchor tensors in the layout ckpt_to_state_dict
returns ({"model": {name: tensor}}).

Usage: python extract_anchor_only_ckpt.py <src_ckpt_dir> <dst_ckpt_dir>
"""

import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter

from veomni.checkpoint import ckpt_to_state_dict


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"Refusing non-empty output: {dst}")

    full = ckpt_to_state_dict(src)
    anchors = {k: v for k, v in full.items() if ".mass_anchor_generator." in k}
    if not anchors:
        raise RuntimeError("no mass_anchor_generator tensors found in source")
    print(f"[extract] {len(full)} tensors -> {len(anchors)} anchor tensors")
    for k in sorted(anchors):
        print(f"  {k} {tuple(anchors[k].shape)}")

    dst.mkdir(parents=True)
    dcp.save(
        {"model": anchors},
        storage_writer=FileSystemWriter(
            dst, thread_count=1, single_file_per_rank=True, sync_files=True
        ),
    )

    back = ckpt_to_state_dict(dst)
    if set(back) != set(anchors):
        raise RuntimeError(
            f"round-trip key mismatch: {sorted(set(back) ^ set(anchors))}"
        )
    for k in anchors:
        if not torch.equal(back[k], anchors[k]):
            raise RuntimeError(f"round-trip tensor mismatch for {k}")
    print(f"[extract] verified anchor-only checkpoint at {dst}")


if __name__ == "__main__":
    main()
