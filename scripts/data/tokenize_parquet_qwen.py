#!/usr/bin/env python3
"""Tokenize parquet text shards into HiLS uint32 mmap files.

Each input ``NAME.parquet`` becomes ``NAME.parquet.data``. Documents are
concatenated and separated with exactly one tokenizer EOS token, matching the
format consumed by ``data.lazy_dataset.LazyChunkedLoader``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--rayon-threads", type=int, default=4)
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Optional parquet basenames, for example 000_00000.parquet",
    )
    parser.add_argument("--max-rows", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_fingerprint(tokenizer_dir: Path) -> dict[str, Any]:
    names = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]
    files = {
        name: sha256_file(tokenizer_dir / name)
        for name in names
        if (tokenizer_dir / name).is_file()
    }
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"path": str(tokenizer_dir.resolve()), "sha256": digest, "files": files}


def atomic_json_dump(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".partial")
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def tokenize_shard(job: tuple[str, str, str, str, int, int | None]) -> dict[str, Any]:
    input_name, output_dir, tokenizer_dir, text_column, batch_size, max_rows = job
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    input_path = Path(input_name)
    final_path = Path(output_dir) / f"{input_path.name}.data"
    partial_path = Path(str(final_path) + ".partial")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError(f"A fast tokenizer is required, got {type(tokenizer).__name__}")
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer has no eos_token_id")

    parquet = pq.ParquetFile(input_path)
    if text_column not in parquet.schema_arrow.names:
        raise KeyError(f"Missing column {text_column!r} in {input_path}")

    documents = 0
    tokens = 0
    batches = 0
    started = time.time()
    with partial_path.open("wb", buffering=16 * 1024 * 1024) as stream:
        for record_batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=[text_column],
            use_threads=False,
        ):
            texts = record_batch.column(0).to_pylist()
            if max_rows is not None:
                remaining = max_rows - documents
                if remaining <= 0:
                    break
                texts = texts[:remaining]
            texts = [text if isinstance(text, str) else "" for text in texts]
            encoded = tokenizer(
                texts,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]

            count = sum(len(ids) + 1 for ids in encoded)
            packed = np.empty(count, dtype="<u4")
            cursor = 0
            for ids in encoded:
                end = cursor + len(ids)
                packed[cursor:end] = ids
                packed[end] = tokenizer.eos_token_id
                cursor = end + 1
            packed.tofile(stream)

            documents += len(encoded)
            tokens += count
            batches += 1
            if batches % 20 == 0:
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"[{input_path.name}] batches={batches:,} docs={documents:,} "
                    f"tokens={tokens:,} rate={tokens / elapsed:,.0f} tok/s",
                    flush=True,
                )
            if max_rows is not None and documents >= max_rows:
                break
        stream.flush()
        os.fsync(stream.fileno())

    expected_size = tokens * 4
    actual_size = partial_path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {partial_path}: {actual_size} != {expected_size}"
        )
    os.replace(partial_path, final_path)
    elapsed = time.time() - started
    result = {
        "source": str(input_path.resolve()),
        "source_size": input_path.stat().st_size,
        "output": str(final_path.resolve()),
        "output_size": actual_size,
        "documents": documents,
        "tokens": tokens,
        "elapsed_seconds": elapsed,
    }
    print(
        f"[{input_path.name}] COMPLETE docs={documents:,} tokens={tokens:,} "
        f"elapsed={elapsed / 60:.1f} min",
        flush=True,
    )
    return result


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.rayon_threads < 1 or args.batch_size < 1:
        raise ValueError("workers, rayon-threads and batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["RAYON_NUM_THREADS"] = str(args.rayon_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    fingerprint = tokenizer_fingerprint(args.tokenizer)
    manifest_path = args.output_dir / "conversion_manifest.json"
    manifest: dict[str, Any] = {
        "format": "flat little-endian uint32; one EOS after every document",
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "tokenizer": fingerprint,
        "text_column": args.text_column,
        "shards": {},
    }
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as stream:
            old_manifest = json.load(stream)
        if old_manifest.get("tokenizer", {}).get("sha256") != fingerprint["sha256"]:
            raise RuntimeError("Existing output manifest uses a different tokenizer")
        manifest = old_manifest

    paths = sorted(args.input_dir.glob("*.parquet"))
    if args.include:
        wanted = set(args.include)
        paths = [path for path in paths if path.name in wanted]
        missing = wanted.difference(path.name for path in paths)
        if missing:
            raise FileNotFoundError(f"Requested shards not found: {sorted(missing)}")
    if not paths:
        raise FileNotFoundError(f"No parquet shards found under {args.input_dir}")

    jobs = []
    for path in paths:
        previous = manifest["shards"].get(path.name)
        output_path = args.output_dir / f"{path.name}.data"
        valid_previous = (
            args.max_rows is None
            and previous is not None
            and output_path.is_file()
            and output_path.stat().st_size == previous.get("tokens", -1) * 4
            and path.stat().st_size == previous.get("source_size", -1)
        )
        if valid_previous:
            print(f"[{path.name}] SKIP verified completed shard", flush=True)
            continue
        jobs.append(
            (
                str(path),
                str(args.output_dir),
                str(args.tokenizer),
                args.text_column,
                args.batch_size,
                args.max_rows,
            )
        )

    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(tokenize_shard, job): Path(job[0]).name for job in jobs}
            try:
                for future in as_completed(futures):
                    name = futures[future]
                    result = future.result()
                    manifest["shards"][name] = result
                    atomic_json_dump(manifest_path, manifest)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    total_documents = sum(x["documents"] for x in manifest["shards"].values())
    total_tokens = sum(x["tokens"] for x in manifest["shards"].values())
    manifest["total_documents"] = total_documents
    manifest["total_tokens"] = total_tokens
    manifest["completed_shards"] = len(manifest["shards"])
    manifest["updated_at_unix"] = time.time()
    atomic_json_dump(manifest_path, manifest)
    print(
        f"ALL COMPLETE shards={len(manifest['shards'])} docs={total_documents:,} "
        f"tokens={total_tokens:,}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
