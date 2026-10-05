#!/usr/bin/env python3
"""Pack already-tokenized parquet docs into HiLS uint32 mmap shards.

Input: parquet with columns (input_ids: list<int32>, num_tokens: int32) — e.g.
data/dolma3_multiscale/dolma3_mix_sample_10B.parquet built from
allenai/dolma3_mix-6T-1025-7B.  Output: flat little-endian uint32
``.parquet.data`` files, one EOS (default 100257, OLMo-3) after every
document, plus a manifest.json — byte-compatible with what
tokenize_parquet_qwen.py produces and LazyChunkedLoader consumes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="tokenized parquet file")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--eos-id", type=int, default=100257)
    ap.add_argument("--shard-tokens", type=int, default=500_000_000)
    ap.add_argument("--batch-rows", type=int, default=8192)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "manifest.json"

    pf = pq.ParquetFile(args.input)
    eos = np.asarray([args.eos_id], dtype="<u4")
    carry = np.empty(0, dtype="<u4")
    shard_idx = 0
    total_docs = 0
    total_tokens = 0
    shards = {}
    t0 = time.time()

    for batch in pf.iter_batches(batch_size=args.batch_rows, columns=["input_ids"]):
        ids_list = batch.column(0).to_pylist()
        parts = []
        for ids in ids_list:
            if not ids:
                continue
            parts.append(np.asarray(ids, dtype="<u4"))
            parts.append(eos)
            total_docs += 1
        if parts:
            carry = np.concatenate([carry] + parts) if carry.size else np.concatenate(parts)
        while carry.size >= args.shard_tokens:
            shard_path = outdir / f"{shard_idx:03d}_00000.parquet.data"
            tmp = shard_path.with_name(shard_path.name + ".incomplete")
            take = carry[: args.shard_tokens]
            carry = carry[args.shard_tokens :]
            with open(tmp, "wb") as stream:
                stream.write(take.tobytes())
            tmp.rename(shard_path)
            shard_tokens = int(take.size)
            total_tokens += shard_tokens
            shards[shard_path.name] = {"documents": -1, "tokens": shard_tokens}
            print(
                f"[shard {shard_idx}] {shard_path.name} tokens={shard_tokens:,} "
                f"(total {total_tokens / 1e9:.3f}B, "
                f"{total_tokens / max(1e-9, time.time() - t0) / 1e6:.1f}M tok/s)",
                flush=True,
            )
            shard_idx += 1

    # flush remainder as the final shard
    if carry.size > 0:
        shard_path = outdir / f"{shard_idx:03d}_00000.parquet.data"
        tmp = shard_path.with_name(shard_path.name + ".incomplete")
        with open(tmp, "wb") as f:
            f.write(carry.tobytes())
        tmp.rename(shard_path)
        total_tokens += int(carry.size)
        shards[shard_path.name] = {"documents": -1, "tokens": int(carry.size)}
        print(f"[final] {shard_path.name} tokens={carry.size:,}", flush=True)

    manifest = {
        "source": str(args.input),
        "eos_id": args.eos_id,
        "format": "flat little-endian uint32; one EOS after every document",
        "total_documents": total_docs,
        "total_tokens": total_tokens,
        "shards": shards,
    }
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(
        f"ALL COMPLETE shards={len(shards)} docs={total_docs:,} "
        f"tokens={total_tokens:,} -> {outdir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
