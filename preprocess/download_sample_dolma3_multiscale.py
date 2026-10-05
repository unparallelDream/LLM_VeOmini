"""Chunked download + multi-scale sampling for the dolma3 mix repo.

对 allenai/dolma3_mix-6T-1025-7B 做一次下载遍历, 同时产出多个 token 规模的
嵌套采样子集 (例如 3B ⊂ 10B ⊂ 20B ⊂ 50B):

  1. 通过 HF API 列出 repo 全部 .jsonl.zst 分片及大小 (不下载数据),
     按文件大小比例为每个分片分配最大规模下的 token 配额。
     各较小规模的配额按同一比例缩小, 因此各规模是嵌套前缀子集,
     与分别独立运行 build_olmo3_datasets_weighted 等价 (差 <1 token/分片)。
  2. --download_budget_gb: 服务器到 hf-mirror 实测仅 ~1MB/s, 全量 4TB 不可行。
     此模式按大小加权无放回抽取分片, 直到压缩字节预算用尽; 配额在选中
     分片集合上重新按比例计算, 各子语料期望占比不变, 下载量按需缩减。
  3. worker 循环: hf_hub_download 单个分片到 /dev/shm 下的 HF 缓存 ->
     zstd 流式解压 + tokenize 到该分片配额为止 -> uint32 写入 NFS staging ->
     删除已下载分片。多个 worker 间下载与 tokenize 自然流水线化。
  4. 全部分片完成后, 按确定顺序对每个规模拼接各 staging 分片的前
     quota_s 个 token, 生成最终 <scale>.numpy。

断点续跑: staging 目录下 progress.jsonl 以分片路径为键记录已完成项,
重启后自动跳过 (跨 --download_budget_gb 变更安全, staging 文件名含路径哈希)。
"""

import os
import io
import sys
import json
import time
import hashlib
import signal
import argparse
import multiprocessing as mp

import numpy as np
import requests
import zstandard as zstd
from urllib.parse import urlsplit, parse_qs
from huggingface_hub import HfApi
from transformers import AutoTokenizer

IO_CHUNK = 64 * 1024 * 1024


def list_repo_shards(repo, endpoint, token, limit=None):
    # 直接走 API 分页: hf-mirror 的 Link(next) 头指向 huggingface.co,
    # huggingface_hub.paginate 会跟随该绝对 URL 导致翻页失败, 这里只取
    # cursor 参数并始终请求镜像 host。cursor 必须经 parse_qs 解码,
    # 否则 requests 二次编码后服务端视为非法并重复返回第一页(死循环)。
    session = requests.Session()
    headers = {"authorization": f"Bearer {token}"} if token else {}
    url = f"{endpoint}/api/datasets/{repo}/tree/main"
    cursor = None
    seen_cursors = set()
    shards = []
    page = 0
    while True:
        page += 1
        params = {"recursive": "true", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        r = None
        for attempt in range(30):
            try:
                r = session.get(url, params=params, headers=headers,
                                timeout=(10, 60))
                r.raise_for_status()
                break
            except Exception:
                if attempt == 29:
                    raise
                wait = min(10 * (attempt + 1), 60)
                print(f"[page {page} retry {attempt + 1}/30] sleep {wait}s",
                      flush=True)
                time.sleep(wait)
        for entry in r.json():
            if (isinstance(entry, dict) and entry.get("path", "").endswith(".jsonl.zst")
                    and entry.get("size", 0)):
                shards.append((entry["path"], int(entry["size"])))
                if limit and len(shards) >= limit:
                    break
        if page % 10 == 0:
            print(f"listed {len(shards)} shards so far (page {page})",
                  flush=True)
        if limit and len(shards) >= limit:
            break
        nxt = r.links.get("next", {}).get("url")
        if not nxt:
            break
        cursor = parse_qs(urlsplit(nxt).query).get("cursor", [None])[0]
        if cursor is None or cursor in seen_cursors or page >= 500:
            raise RuntimeError(
                f"pagination stuck at page {page} "
                f"(cursor={str(cursor)[:24]}...)")
        seen_cursors.add(cursor)
        time.sleep(0.3)
    shards.sort(key=lambda x: x[0])
    if limit:
        shards = shards[:limit]
    return shards


def sample_shards(shards, budget_bytes, seed):
    sizes = np.array([s for _, s in shards], dtype=np.float64)
    rng = np.random.default_rng(seed)
    order = rng.choice(len(shards), size=len(shards), replace=False,
                       p=sizes / sizes.sum())
    picked, total = [], 0
    for i in order:
        if total >= budget_bytes:
            break
        picked.append(i)
        total += shards[i][1]
    picked.sort()
    return [shards[i] for i in picked]


def compute_quotas(shards, scales):
    sizes = np.array([s for _, s in shards], dtype=np.float64)
    weights = sizes / sizes.sum()
    # floor 按比例配额; 不做余数再分配, 以保证各规模严格嵌套
    # (相对目标规模的缺口 <= 分片数个 token)
    return {s: np.floor(weights * s).astype(np.int64) for s in scales}


def staging_name(rel_path):
    return hashlib.sha1(rel_path.encode()).hexdigest()[:20] + ".data"


def tokenize_zst_prefix(path, tokenizer, max_tokens):
    eos = tokenizer.eos_token_id
    tokens = []
    with open(path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8", errors="ignore") as tr:
                accum = ""
                for line in tr:
                    if len(tokens) >= max_tokens:
                        break
                    if not line.strip():
                        continue
                    accum += line
                    try:
                        obj = json.loads(accum)
                    except json.JSONDecodeError:
                        continue
                    accum = ""
                    enc = tokenizer.encode(obj["text"])
                    if eos is not None:
                        enc.append(eos)
                    remain = max_tokens - len(tokens)
                    tokens.extend(enc[:remain])
    return tokens


def _remove_hub_blob(snapshot_path):
    try:
        real = os.path.realpath(snapshot_path)
        os.remove(snapshot_path)
        if real != snapshot_path and os.path.exists(real):
            os.remove(real)
    except OSError:
        pass


def worker(task_q, result_q, cfg):
    os.environ["HF_HUB_CACHE"] = cfg["hub_cache"]
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        os.nice(19)
    except OSError:
        pass
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    tokenizer = AutoTokenizer.from_pretrained(cfg["vocab_dir"])
    api = HfApi(endpoint=cfg["endpoint"], token=cfg["token"])

    while True:
        item = task_q.get()
        if item is None:
            return
        idx, rel_path, quota = item
        staged = os.path.join(cfg["staging_dir"], staging_name(rel_path))
        try:
            local = None
            for attempt in range(5):
                try:
                    local = api.hf_hub_download(cfg["repo"], rel_path,
                                                repo_type="dataset")
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    wait = 60 * (attempt + 1)
                    print(f"[dl retry {attempt + 1}/5] {rel_path}: sleep {wait}s",
                          flush=True)
                    time.sleep(wait)
            tokens = tokenize_zst_prefix(local, tokenizer, int(quota))
            tmp = staged + ".tmp"
            np.array(tokens, dtype=np.uint32).tofile(tmp)
            os.replace(tmp, staged)
            _remove_hub_blob(local)
            result_q.put((idx, rel_path, len(tokens), None))
        except Exception as e:  # noqa: BLE001
            result_q.put((idx, rel_path, -1, f"{type(e).__name__}: {e}"))


def load_progress(staging_dir):
    done = {}
    prog = os.path.join(staging_dir, "progress.jsonl")
    if os.path.exists(prog):
        with open(prog, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[rec["path"]] = rec["tokens"]
    return done


def run_download_stage(shards, pending, cfg):
    print(f"pending shards: {len(pending)}", flush=True)
    if not pending:
        return

    ctx = mp.get_context("spawn")
    task_q = ctx.Queue()
    result_q = ctx.Queue()
    procs = []
    for _ in range(min(cfg["num_workers"], len(pending))):
        p = ctx.Process(target=worker, args=(task_q, result_q, cfg))
        p.daemon = True
        p.start()
        procs.append(p)

    prog_path = os.path.join(cfg["staging_dir"], "progress.jsonl")
    t0 = time.time()
    finished = 0
    try:
        it = iter(pending)
        for _ in range(len(procs)):
            try:
                task_q.put(next(it))
            except StopIteration:
                break
        with open(prog_path, "a", encoding="utf-8") as prog:
            while finished < len(pending):
                idx, rel_path, n_tokens, err = result_q.get()
                if err is not None:
                    print(f"[ERROR] shard {idx} ({rel_path}): {err}", flush=True)
                    for p in procs:
                        p.terminate()
                    sys.exit(1)
                finished += 1
                prog.write(json.dumps({"path": rel_path, "tokens": n_tokens}) + "\n")
                prog.flush()
                rate = finished / max(time.time() - t0, 1e-6)
                eta = (len(pending) - finished) / max(rate, 1e-6) / 3600
                print(f"[{finished}/{len(pending)}] {rel_path} "
                      f"tokens={n_tokens:,} eta={eta:.1f}h", flush=True)
                try:
                    task_q.put(next(it))
                except StopIteration:
                    pass
    finally:
        for _ in procs:
            try:
                task_q.put_nowait(None)
            except Exception:
                pass
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
    print(f"download+tokenize stage done: {finished} shards", flush=True)


def build_scale_outputs(shards, quotas, actual, staging_dir, output_dir, scales):
    for scale in sorted(scales):
        label = int(scale // 10**9)
        out_path = os.path.join(output_dir, f"dolma3_mix_sample_{label}B.numpy")
        total = 0
        with open(out_path, "wb") as fo:
            for i, (rel_path, _) in enumerate(shards):
                n = min(int(quotas[scale][i]), actual.get(rel_path, 0))
                if n <= 0:
                    continue
                sf = os.path.join(staging_dir, staging_name(rel_path))
                remaining = n * 4
                with open(sf, "rb") as fi:
                    while remaining > 0:
                        buf = fi.read(min(IO_CHUNK, remaining))
                        if not buf:
                            raise RuntimeError(
                                f"staging file truncated: {sf} "
                                f"(need {remaining} more bytes)")
                        fo.write(buf)
                        remaining -= len(buf)
                total += n
        print(f"built {out_path}: {total:,} tokens "
              f"({os.path.getsize(out_path) / 1e9:.1f} GB)", flush=True)


def main():
    parser = argparse.ArgumentParser(
        "Chunked-download multi-scale sampling of dolma3 mix (nested subsets)")
    parser.add_argument("--repo", default="allenai/dolma3_mix-6T-1025-7B")
    parser.add_argument("--vocab_dir", required=True)
    parser.add_argument("--scales", required=True, type=int, nargs="+",
                        help="token quotas, e.g. 3000000000 10000000000")
    parser.add_argument("--download_budget_gb", type=float, default=None,
                        help="only download a size-weighted random shard subset "
                             "up to this many compressed GB (for slow uplinks); "
                             "changing it between runs is resume-safe")
    parser.add_argument("--shm_dir", required=True,
                        help="tmpfs working dir for transient shard downloads, e.g. /dev/shm/xxx")
    parser.add_argument("--staging_dir", required=True,
                        help="persistent dir for per-shard tokenized staging + progress.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit_files", type=int, default=None,
                        help="only process the first N shards (smoke test)")
    parser.add_argument("--refresh_list", action="store_true",
                        help="re-fetch and overwrite the cached shard list")
    parser.add_argument("--list_cache", default=None,
                        help="shared shard-list cache path (smoke and full run "
                             "should pass the same path)")
    args = parser.parse_args()

    hub_cache = os.path.join(args.shm_dir, "hf_cache")
    os.makedirs(hub_cache, exist_ok=True)
    os.makedirs(args.staging_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # 列表缓存: 冒烟与正式运行必须共用同一份(--list_cache), 避免重复翻页
    list_cache = args.list_cache or os.path.normpath(
        os.path.join(args.staging_dir, "..", "shard_list.json"))
    shards = None
    if os.path.exists(list_cache) and not args.refresh_list:
        with open(list_cache, "r", encoding="utf-8") as f:
            shards = [tuple(x) for x in json.load(f)]
        print(f"loaded cached shard list: {len(shards)}", flush=True)
    else:
        for attempt in range(5):
            try:
                shards = list_repo_shards(args.repo, args.endpoint, args.token,
                                          args.limit_files)
                break
            except Exception:
                if attempt == 4:
                    raise
                print(f"[list retry {attempt + 1}/5] retrying in 60s",
                      flush=True)
                time.sleep(60)
        tmp = list_cache + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump([list(x) for x in shards], f)
        os.replace(tmp, list_cache)
        print(f"cached shard list -> {list_cache}", flush=True)
    if args.limit_files:
        shards = shards[:args.limit_files]
    print(f"repo shards: {len(shards)}, "
          f"total {sum(s for _, s in shards) / 1e12:.2f} TB (compressed)", flush=True)

    if args.download_budget_gb:
        shards = sample_shards(shards, args.download_budget_gb * 1e9, args.seed)
        print(f"shard-sampled: {len(shards)} shards, "
              f"{sum(s for _, s in shards) / 1e9:.1f} GB (compressed)", flush=True)

    scales = sorted(args.scales)
    quotas = compute_quotas(shards, scales)
    max_scale = scales[-1]

    cfg = {
        "repo": args.repo, "vocab_dir": args.vocab_dir, "endpoint": args.endpoint,
        "token": args.token, "hub_cache": hub_cache,
        "staging_dir": args.staging_dir, "num_workers": args.num_workers,
    }

    done = load_progress(args.staging_dir)
    pending = [(i, shards[i][0], int(quotas[max_scale][i]))
               for i in range(len(shards))
               if quotas[max_scale][i] > 0 and shards[i][0] not in done]
    run_download_stage(shards, pending, cfg)

    actual = load_progress(args.staging_dir)
    build_scale_outputs(shards, quotas, actual, args.staging_dir,
                        args.output_dir, scales)


if __name__ == "__main__":
    main()
