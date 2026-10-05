"""Convert flat uint32 token streams (.numpy) into per-document Parquet.

流式读取 .numpy token 流, 按 eos 切分成文档, 每行输出一个文档的
input_ids (list<int32>, 含文档末尾的 eos) 与 num_tokens。
flat buffer + offsets 向量化构建行, 避免 Python 逐 token 转换。
"""

import os
import time
import argparse
from collections import deque

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([
    pa.field("input_ids", pa.list_(pa.int32())),
    pa.field("num_tokens", pa.int32()),
])


def flush_batch(writer, doc_slices, doc_lens):
    flat = np.concatenate(doc_slices).astype(np.int32, copy=False)
    offsets = np.zeros(len(doc_slices) + 1, dtype=np.int32)
    np.cumsum(doc_lens, out=offsets[1:])
    ids = pa.ListArray.from_arrays(pa.array(offsets), pa.array(flat))
    table = pa.Table.from_arrays(
        [ids, pa.array(np.asarray(doc_lens, dtype=np.int32))],
        names=["input_ids", "num_tokens"])
    writer.write_table(table)
    return int(offsets[-1])


def take_batch(doc_slices, doc_lens, row_group_docs, max_group_tokens):
    """从头取一批: 行数达 row_group_docs 或 token 量达 max_group_tokens。"""
    n, acc = 0, 0
    for l in doc_lens:
        if n >= row_group_docs or (acc + l > max_group_tokens and n > 0):
            break
        acc += l
        n += 1
    return max(n, 1)


def convert(in_path, out_path, eos_id, row_group_docs=32768,
            max_group_tokens=32 * 1024 * 1024, compression="zstd",
            limit_docs=None):
    stream = np.memmap(in_path, dtype=np.uint32, mode="r")
    total_tokens = stream.size
    writer = pq.ParquetWriter(out_path, SCHEMA, compression=compression)
    doc_slices, doc_lens = deque(), deque()
    written_docs = written_tokens = 0
    carry = np.empty(0, dtype=np.uint32)
    t0 = time.time()
    chunk = 64 * 1024 * 1024
    pos = 0
    try:
        while pos < total_tokens:
            if carry.size:
                block = np.concatenate([carry, stream[pos:pos + chunk]])
            else:
                block = stream[pos:pos + chunk]
            pos += chunk - carry.size
            ends = np.flatnonzero(block == eos_id)
            doc_start = 0
            for e in ends:
                doc_slices.append(block[doc_start:e + 1])
                doc_lens.append(int(e + 1 - doc_start))
                doc_start = e + 1
            carry = block[doc_start:].copy()
            while doc_slices:
                n = take_batch(doc_slices, doc_lens, row_group_docs,
                               max_group_tokens)
                take_s = [doc_slices.popleft() for _ in range(n)]
                take_l = [doc_lens.popleft() for _ in range(n)]
                written_tokens += flush_batch(writer, take_s, take_l)
                written_docs += n
                if written_docs % 1_000_000 < row_group_docs:
                    rate = written_tokens / max(time.time() - t0, 1e-6)
                    print(f"{written_docs:,} docs, {written_tokens:,} tokens "
                          f"({rate / 1e6:.1f}M tok/s)", flush=True)
                if limit_docs and written_docs >= limit_docs:
                    return written_docs, written_tokens
        if doc_slices:
            n = len(doc_slices)
            take_s = [doc_slices.popleft() for _ in range(n)]
            take_l = [doc_lens.popleft() for _ in range(n)]
            written_tokens += flush_batch(writer, take_s, take_l)
            written_docs += n
    finally:
        writer.close()
    return written_docs, written_tokens


def main():
    parser = argparse.ArgumentParser(
        "Convert flat uint32 .numpy token stream to per-document parquet")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eos_token_id", type=int, default=100257)
    parser.add_argument("--row_group_docs", type=int, default=32768)
    parser.add_argument("--compression", default="zstd",
                        choices=["zstd", "snappy", "none"])
    parser.add_argument("--limit_docs", type=int, default=None,
                        help="only convert the first N docs (smoke test)")
    args = parser.parse_args()

    docs, tokens = convert(args.input, args.output, args.eos_token_id,
                           row_group_docs=args.row_group_docs,
                           compression=args.compression,
                           limit_docs=args.limit_docs)
    print(f"done: {docs:,} docs, {tokens:,} tokens -> {args.output} "
          f"({os.path.getsize(args.output) / 1e9:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
