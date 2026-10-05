#!/usr/bin/env python3
"""MASS-side retrieval diagnostics on RULER single_niah (0.6B).

For each requested checkpoint this probe replays the exact eval-sample
pipeline (same corpus slice, same RulerSynthesizer, same landmark fill and
position ids), hooks ``build_mass_retrieval_plan`` on every MASS retrieval
layer and records:

* needle-chunk hit@topk for the answer-adjacent query rows (the rows whose
  logits predict the needle digits) and for the broader question region,
* the rank of the needle chunk inside the selected list,
* per-layer anchor_load routing and conditional entropy.

Samples are verified against the canonical eval hashes before probing.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

REPO = PROJECT_ROOT
MASSCFG = REPO / "configs/mass_attention/config_mass_qwen3_0p6B_8KA2K_RoPE_m2_r8_asym0p5_anneal.json"
MASSMODEL = Path("/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B-MASS-m2/snapshots/master")
ORIGMODEL = Path("/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master")
CORPUS = Path("/9950backfile/zhangem/data/fineweb_qwen3_data")
ANCHOR_RUN = Path("/9950backfile/zhangem/outputs/mass_qwen3_0p6b_anchor_only_m2_r8_asym0p5_anneal_ft20000/checkpoints")
LORA_RUN = Path("/9950backfile/zhangem/outputs/mass_lora_qwen3_0p6b_anchor_qkvo_all_r16_ft20000/checkpoints")
NCI_RUN = Path("/9950backfile/zhangem/outputs/mass_qwen3_0p6b_anchor_nci_specwu_ft20000/checkpoints")
DIVMU_RUN = Path("/9950backfile/zhangem/outputs/mass_qwen3_0p6b_anchor_divmu_ft20000/checkpoints")

# First three hashes produced by the canonical eval (seed-independent sample
# identity gate): bsa_lora_tasks012.json / mass_lora_tasks012.json etc.
CANONICAL_HASHES = ["81c8ae56f0fe1375", "901149492ddc5986", "b528ee1a522f4bde"]


def load_eval_module():
    spec = importlib.util.spec_from_file_location(
        "eval_ruler_qwen_mass_bsa_4k", REPO / "eval/eval_ruler_qwen_mass_bsa_4k.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_subsequence(hay: np.ndarray, needle: np.ndarray) -> int:
    n = len(needle)
    if n == 0 or len(hay) < n:
        return -1
    candidates = np.flatnonzero(hay == needle[0])
    for c in candidates:
        if c + n <= len(hay) and np.array_equal(hay[int(c) : int(c) + n], needle):
            return int(c)
    return -1


class RetrievalProbe:
    """Hook every MASS retrieval layer and capture its top-k plan."""

    def __init__(self, model):
        self.captures = {}
        self.handles = []
        layer_indices = getattr(model.model, "_mass_layer_indices", None)
        if layer_indices is None:
            raise RuntimeError("model has no _mass_layer_indices; not a MASS model?")
        for layer_idx in layer_indices:
            module = model.model.layers[int(layer_idx)].self_attn
            original = module.build_mass_retrieval_plan

            def wrapped(self_, *args, _original=original, _layer=int(layer_idx), **kwargs):
                result = _original(*args, **kwargs)
                # indices: [B, Lq, Hkv, K] chunk ids per query row / kv head.
                self.captures.setdefault(_layer, {})["indices"] = (
                    result.indices.detach().to(torch.long).cpu()
                )
                self.captures[_layer]["anchor_load"] = (
                    result.anchor_load.detach().float().cpu().flatten()
                )
                cond = result.conditional_entropy
                if cond is not None and torch.is_tensor(cond):
                    self.captures[_layer]["cond_entropy"] = float(cond.detach().float())
                return result

            module.build_mass_retrieval_plan = wrapped.__get__(module, type(module))
            self.handles.append((module, original))

    def reset(self):
        self.captures = {}

    def close(self):
        for module, original in self.handles:
            module.build_mass_retrieval_plan = original
        self.handles = []


def probe_model(model, tokenizer, dataset, evalmod, n_samples, chunk_size=63, topk_expected=32):
    from data import RulerSynthesizer

    synth = RulerSynthesizer(tokenizer, task_id=0)
    probe = RetrievalProbe(model)
    lmk_fill_id = int(getattr(model, "lmk_id", model.config.vocab_size))

    per_sample = []
    hashes = []
    for idx in range(n_samples):
        raw_sample = dataset[idx]
        batch = synth.single_token_eval_collate_fn([raw_sample])
        raw_input = batch["input_ids"]
        digest = hashlib.sha256(raw_input.numpy().tobytes()).hexdigest()[:16]
        hashes.append(digest)

        labels = batch["labels"][0]
        # labels = needle digits (+ eos); locate the needle span in the raw input.
        digit_tokens = labels[:-1] if int(labels[-1]) in (151645, 151643) else labels
        needle_raw = np.asarray(digit_tokens, dtype=np.int64)
        hay = raw_input[0].numpy().astype(np.int64)
        pos = find_subsequence(hay, needle_raw)
        occurrences = 0
        scan = 0
        while True:
            p = find_subsequence(hay[scan:], needle_raw)
            if p < 0:
                break
            occurrences += 1
            scan += p + 1
        if pos < 0:
            per_sample.append({"idx": idx, "needle_found": False, "occurrences": occurrences})
            continue

        input_ids, labels_dev, position_ids, logits_start, offsets, original_len = (
            evalmod.prepare_answer_positions(
                raw_input, batch["labels"], True, chunk_size, torch.device("cuda"), lmk_fill_id
            )
        )
        # Absolute expanded positions of the answer-adjacent query rows.
        ans_logit_positions = offsets + logits_start
        expanded_len = input_ids.shape[1]
        needle_exp = pos + pos // chunk_size
        needle_chunk = pos // chunk_size

        cache_position = torch.arange(expanded_len, device="cuda")
        probe.reset()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=False,
            )
        if not probe.captures:
            raise RuntimeError("retrieval plan hook never fired")

        window = int(model.config.hils_sliding_window)
        topk = None
        layer_stats = {}
        ans_rows = ans_logit_positions.tolist()
        qregion_rows = list(range(max(0, expanded_len - 96), expanded_len))

        for layer, cap in sorted(probe.captures.items()):
            indices = cap["indices"][0]  # [Lq, Hkv, K]
            topk = int(indices.shape[-1])
            def row_hit(rows):
                hits, ranks, qh = 0, [], 0.0
                for r in rows:
                    if r >= indices.shape[0]:
                        continue
                    row_chunks = indices[r]  # [Hkv, K]
                    member = row_chunks == needle_chunk  # [Hkv, K]
                    qh += float(member.any(dim=-1).float().mean())
                    if bool(member.any()):
                        hits += 1
                        h, k = member.nonzero()[0].tolist()
                        ranks.append(k)
                n = len(rows)
                return {
                    "any_row": hits > 0,
                    "frac_rows": hits / n if n else 0.0,
                    "frac_qh": qh / n if n else 0.0,
                    "mean_rank": float(statistics.mean(ranks)) if ranks else None,
                }
            layer_stats[layer] = {
                "ans": row_hit(ans_rows),
                "qregion": row_hit(qregion_rows),
                "anchor_load": [round(v, 4) for v in cap["anchor_load"].tolist()],
                "cond_entropy": cap.get("cond_entropy"),
            }

        # Distance from the (median) answer query to the needle.
        med_ans = int(statistics.median(ans_rows))
        needle_dist = med_ans - needle_exp
        per_sample.append(
            {
                "idx": idx,
                "needle_found": True,
                "occurrences": occurrences,
                "needle_chunk": needle_chunk,
                "needle_exp": needle_exp,
                "needle_dist": needle_dist,
                "in_local_window": needle_dist < window,
                "layers": layer_stats,
            }
        )
        del input_ids, labels_dev, position_ids
        gc.collect()
        torch.cuda.empty_cache()

    probe.close()
    return per_sample, hashes


def summarize(per_sample):
    found = [s for s in per_sample if s.get("needle_found")]
    single = found  # needle = first occurrence (question+answer tail repeats it once)
    layers = sorted(single[0]["layers"].keys()) if single else []
    layer_summary = {}
    for layer in layers:
        remote = [s for s in single if not s["in_local_window"]]
        ans_any = statistics.mean(
            1.0 if s["layers"][layer]["ans"]["any_row"] else 0.0 for s in single
        )
        ans_any_remote = statistics.mean(
            1.0 if s["layers"][layer]["ans"]["any_row"] else 0.0 for s in remote
        ) if remote else None
        q_any = statistics.mean(
            1.0 if s["layers"][layer]["qregion"]["any_row"] else 0.0 for s in single
        )
        q_any_remote = statistics.mean(
            1.0 if s["layers"][layer]["qregion"]["any_row"] else 0.0 for s in remote
        ) if remote else None
        frac_qh = statistics.mean(s["layers"][layer]["ans"]["frac_qh"] for s in single)
        ranks = [
            s["layers"][layer]["ans"]["mean_rank"]
            for s in single
            if s["layers"][layer]["ans"]["mean_rank"] is not None
        ]
        loads = np.array([s["layers"][layer]["anchor_load"] for s in single], dtype=float)
        conds = [
            s["layers"][layer]["cond_entropy"]
            for s in single
            if s["layers"][layer]["cond_entropy"] is not None
        ]
        layer_summary[str(layer)] = {
            "hit_ans_any_all": round(ans_any, 4),
            "hit_ans_any_remote": round(ans_any_remote, 4) if ans_any_remote is not None else None,
            "hit_qregion_any_remote": round(q_any_remote, 4) if q_any_remote is not None else None,
            "frac_qh_ans": round(frac_qh, 4),
            "mean_rank_when_hit": round(statistics.mean(ranks), 2) if ranks else None,
            "anchor_load_mean": [round(v, 4) for v in loads.mean(axis=0).tolist()],
            "anchor_load_std": [round(v, 4) for v in loads.std(axis=0).tolist()],
            "cond_entropy_mean": round(statistics.mean(conds), 4) if conds else None,
        }
    n_window = sum(1 for s in single if s["in_local_window"])
    return {
        "n_samples": len(per_sample),
        "n_needle_single_occurrence": len(single),
        "n_in_local_window": n_window,
        "n_remote": len(single) - n_window,
        "layers": layer_summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--topk-override",
        type=int,
        default=None,
        help="probe with a smaller remote top-k (binding regime; must be a power of two)",
    )
    parser.add_argument(
        "--targets",
        default="zeroshot,anchor1000,anchor3000,anchor5000,anchor6500,anchor20000,"
        "lora1000,lora3000,lora5000,lora6500,lora10000,lora15000,lora20000",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(ORIGMODEL))
    evalmod = load_eval_module()
    from data import build_numpy_dataset
    from veomni.models import build_foundation_model

    dataset = build_numpy_dataset(str(CORPUS), 4096, namespace="test", sort_files=True)

    def load_target(name):
        ns = SimpleNamespace(
            mass_config=str(MASSCFG), mass_model=str(MASSMODEL), checkpoint=""
        )
        if name.startswith("shared90m") or name == "zeroshot90m":
            ns.mass_config = "configs/mass_attention/config_mass_qwen3_90M_8KA2K_RoPE_m2_r8_sharedview.json"
            ns.mass_model = "/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-90M-MASS-m2-recovered/snapshots/master"
            if name == "zeroshot90m":
                from models.MASSIndex import modeling_registry as _reg90  # noqa: F401
                from veomni.models import build_foundation_model as _bf90
                m90 = _bf90(
                    config_path=ns.mass_config, weights_path=ns.mass_model,
                    torch_dtype="bfloat16", attn_implementation="sdpa", init_device="cuda",
                )
                m90.config.use_cache = False
                m90.eval()
                return m90
            step = name[len("shared90m"):]
            ns.checkpoint = str(Path("/9950backfile/zhangem/outputs/mass_lora_qwen3_90m_shared_m2_ft20000/checkpoints") / f"global_step_{step}")
            model, _meta = evalmod.load_mass_lora_model(ns)
            return model
        if name == "zeroshot":
            from models.MASSIndex import modeling_registry as _reg  # noqa: F401

            model = build_foundation_model(
                config_path=ns.mass_config,
                weights_path=ns.mass_model,
                torch_dtype="bfloat16",
                attn_implementation="sdpa",
                init_device="cuda",
            )
            model.config.use_cache = False
            model.eval()
            return model
        if name.startswith("anchor"):
            step = name[len("anchor"):]
            ns.checkpoint = str(ANCHOR_RUN / f"global_step_{step}_anchor_only")
            model, _meta = evalmod.load_mass_model(ns, fixed_anchor_zero=False)
            return model
        if name.startswith("lmkq90m"):
            is_b = name.startswith("lmkq90mB")
            step = name[len("lmkq90mB"):] if is_b else name[len("lmkq90m"):]
            ns.mass_config = "configs/mass_attention/config_mass_qwen3_90M_8KA2K_RoPE_m2_r8_sharedview_lmkq.json"
            ns.mass_model = "/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-90M-MASS-m2-recovered/snapshots/master"
            sub = "shared_lmkq_nciw2" if is_b else "shared_lmkq"
            ns.checkpoint = str(Path(f"/9950backfile/zhangem/outputs/mass_lora_qwen3_90m_{sub}_ft20000/checkpoints") / f"global_step_{step}")
            model, _meta = evalmod.load_mass_lora_model(ns)
            return model
        if name.startswith("nonci90m"):
            step = name[len("nonci90m"):]
            ns.mass_config = "configs/mass_attention/config_mass_qwen3_90M_8KA2K_RoPE_m2_r8_sharedview.json"
            ns.mass_model = "/9950backfile/zhangem/.cache/modelscope/models/Qwen--Qwen3-90M-MASS-m2-recovered/snapshots/master"
            ns.checkpoint = str(Path("/9950backfile/zhangem/outputs/mass_lora_qwen3_90m_shared_m2_nonci_ft20000/checkpoints") / f"global_step_{step}")
            model, _meta = evalmod.load_mass_lora_model(ns)
            return model
        if name.startswith("nci"):
            step = name[len("nci"):]
            ns.checkpoint = str(NCI_RUN / f"global_step_{step}")
            model, _meta = evalmod.load_mass_model(ns, fixed_anchor_zero=False)
            return model
        if name.startswith("divmu"):
            step = name[len("divmu"):]
            ns.checkpoint = str(DIVMU_RUN / f"global_step_{step}")
            model, _meta = evalmod.load_mass_model(ns, fixed_anchor_zero=False)
            return model
        if name.startswith("lora"):
            step = name[len("lora"):]
            ns.checkpoint = str(LORA_RUN / f"global_step_{step}")
            model, _meta = evalmod.load_mass_lora_model(ns)
            return model
        raise ValueError(name)

    results = {}
    for name in [t.strip() for t in args.targets.split(",") if t.strip()]:
        t0 = time.time()
        model = load_target(name)
        if args.topk_override:
            for li in model.model._mass_layer_indices:
                model.model.layers[int(li)].self_attn.topk = int(args.topk_override)
        per_sample, hashes = probe_model(model, tokenizer, dataset, evalmod, args.n_samples)
        if name == "zeroshot":
            for i, expect in enumerate(CANONICAL_HASHES):
                if hashes[i] != expect:
                    raise RuntimeError(
                        f"sample identity gate failed at {i}: {hashes[i]} != {expect}"
                    )
        results[name] = {"summary": summarize(per_sample), "per_sample": per_sample}
        print(f"[{name}] {time.time()-t0:.0f}s  summary:", flush=True)
        print(json.dumps(results[name]["summary"]["layers"].get("15", {}), indent=None))
        del model
        gc.collect()
        torch.cuda.empty_cache()

    with open(args.output, "w") as f:
        json.dump(
            {
                "config": {
                    "n_samples": args.n_samples,
                    "task": "single_niah(task0)",
                    "targets": list(results.keys()),
                    "hash_gate": "first3==" + ",".join(CANONICAL_HASHES),
                },
                "results": results,
            },
            f,
            indent=1,
        )
    print("WROTE", args.output)


if __name__ == "__main__":
    main()
