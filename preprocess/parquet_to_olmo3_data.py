"""Stream Parquet text shards into HiLS/OLMo3 ``uint32`` ``.data`` shards.

The project loader memory-maps raw uint32 token IDs from files ending in
``.data``.  This converter intentionally mirrors the repository's JSONL
preprocessor: encode the ``text`` field and append one EOS token per row.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def convert_file(
    source: Path,
    destination: Path,
    tokenizer,
    *,
    batch_rows: int,
    flush_tokens: int,
) -> tuple[int, int]:
    """Convert one shard atomically, returning (documents, token_ids)."""
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if destination.exists() and destination.stat().st_size > 0:
        return 0, destination.stat().st_size // np.dtype(np.uint32).itemsize

    temporary.unlink(missing_ok=True)
    docs = 0
    tokens = 0
    buffer: list[int] = []
    reader = pq.ParquetFile(source)
    tokenizer.model_max_length = int(1e30)  # training later windows this stream
    with temporary.open("wb") as output:
        batch_number = 0
        for row_group in range(reader.num_row_groups):
            try:
                batches = reader.iter_batches(
                    batch_size=batch_rows, columns=["text"], row_groups=[row_group]
                )
                for record_batch in batches:
                    batch_number += 1
                    texts = [text for text in record_batch.column(0).to_pylist() if isinstance(text, str) and text.strip()]
                    # Fast tokenizers process a batch in Rust.  This preserves
                    # the repository preprocessor's default special-token
                    # behavior while avoiding a Python call per document.
                    encoded_batch = tokenizer(texts)["input_ids"] if texts else []
                    for ids in encoded_batch:
                        if tokenizer.eos_token_id is not None:
                            ids.append(tokenizer.eos_token_id)
                        buffer.extend(ids)
                        docs += 1
                        tokens += len(ids)
                        if len(buffer) >= flush_tokens:
                            np.asarray(buffer, dtype=np.uint32).tofile(output)
                            buffer.clear()
                    if batch_number % 100 == 0:
                        print(f"  batches={batch_number:,} documents={docs:,} tokens={tokens:,}", flush=True)
            except OSError as error:
                print(f"  skipping corrupt row_group={row_group}: {error}", flush=True)
        if buffer:
            np.asarray(buffer, dtype=np.uint32).tofile(output)
    temporary.replace(destination)
    return docs, tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tokenizer", default="configs/olmo3_vocab")
    parser.add_argument("--batch-rows", type=int, default=512)
    parser.add_argument("--flush-tokens", type=int, default=1_000_000)
    parser.add_argument("--limit-files", type=int)
    parser.add_argument("--skip-name", action="append", default=[], help="Parquet filename to exclude")
    parser.add_argument("--only-name", action="append", default=[], help="Parquet filename to include")
    args = parser.parse_args()

    skip_names = set(args.skip_name)
    only_names = set(args.only_name)
    files = [
        path
        for path in sorted(args.input_dir.glob("*.parquet"))
        if path.name not in skip_names and (not only_names or path.name in only_names)
    ]
    if args.limit_files is not None:
        files = files[: args.limit_files]
    if not files:
        raise FileNotFoundError(f"No Parquet files in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    total_docs = total_tokens = 0
    for number, source in enumerate(files, start=1):
        destination = args.output_dir / f"{source.name}.data"
        print(f"[{number}/{len(files)}] {source.name}", flush=True)
        docs, tokens = convert_file(
            source,
            destination,
            tokenizer,
            batch_rows=args.batch_rows,
            flush_tokens=args.flush_tokens,
        )
        total_docs += docs
        total_tokens += tokens
        print(f"  documents={docs:,} tokens={tokens:,} output={destination}", flush=True)
    print(f"complete documents={total_docs:,} tokens={total_tokens:,}", flush=True)


if __name__ == "__main__":
    main()
