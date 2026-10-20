#!/usr/bin/env python3
"""lm-eval full RULER (13 subtasks, multi-length) for OLMo-200M HF checkpoints
(olmo3 dense / olmo_hils / olmo_mass), mirroring lmeval_ruler_200m/hils200m_step15000_fullruler.json.

Usage:
  CUDA_VISIBLE_DEVICES=0 python eval/run_ruler_lmeval_200m.py \
    --hf-dir .../checkpoints/global_step_15000/hf_ckpt \
    --output .../<name>_step15000_fullruler.json \
    [--lengths 4096 8192] [--limit 50] [--batch-size 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RULER_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue",
    "ruler_cwe", "ruler_fwe", "ruler_vt",
    "ruler_qa_squad", "ruler_qa_hotpot",
]


def register_custom(model_type: str) -> None:
    from transformers import AutoConfig, AutoModelForCausalLM

    if model_type == "olmo3":
        return
    if model_type == "olmo_hils":
        from models.FlashHiLS.modeling_olmo_hils import HiLSForCausalLM
        from models.FlashHiLS.configuration_hils import HiLSConfig

        class Cfg(HiLSConfig):
            pass

        Cfg.model_type = model_type
        HiLSForCausalLM.config_class = Cfg
        cls = HiLSForCausalLM
    elif model_type == "olmo_mass":
        from models.MASSIndex.modeling_olmo_mass import MASSForCausalLM
        from models.MASSIndex.configuration_mass import OLMoMASSConfig

        Cfg = OLMoMASSConfig
        if Cfg.model_type != model_type:
            class _Cfg(Cfg):
                pass

            _Cfg.model_type = model_type
            Cfg = _Cfg
        cls = MASSForCausalLM
    else:
        raise ValueError(f"unsupported model_type: {model_type}")

    try:
        cls.config_class = Cfg
    except Exception:
        pass
    try:
        AutoConfig.register(Cfg.model_type, Cfg)
    except ValueError as e:
        print(f"[register] AutoConfig {Cfg.model_type}: {e}", flush=True)
    try:
        AutoModelForCausalLM.register(Cfg, cls)
    except ValueError as e:
        raise RuntimeError(f"AutoModelForCausalLM.register failed for {Cfg.model_type}: {e}")
    print(f"[register] {Cfg.model_type} -> {cls.__name__}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192])
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--tasks", nargs="+", default=RULER_TASKS)
    args = ap.parse_args()

    model_type = json.load(open(Path(args.hf_dir) / "config.json"))["model_type"]
    register_custom(model_type)

    import lm_eval
    from lm_eval.tasks import TaskManager

    tm = TaskManager(metadata={
        "max_seq_lengths": [int(x) for x in args.lengths],
        "pretrained": args.hf_dir,
        "tokenizer": args.hf_dir,
    })
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=dict(pretrained=args.hf_dir, dtype="bfloat16", trust_remote_code=True),
        tasks=list(args.tasks),
        limit=args.limit,
        batch_size=args.batch_size,
        task_manager=tm,
        random_seed=42,
        numpy_random_seed=42,
        torch_random_seed=42,
        fewshot_random_seed=42,
    )

    compact_results = {}
    for task in args.tasks:
        entry = {"name": task, "alias": task}
        ns = results.get("n-samples", {}).get(task, {})
        entry["sample_len"] = ns.get("effective", args.limit)
        for k, v in results["results"].get(task, {}).items():
            if isinstance(v, (int, float, str)):
                entry[k] = v
        compact_results[task] = entry

    out = {
        "config": {
            "hf_dir": args.hf_dir,
            "max_seq_lengths": [int(x) for x in args.lengths],
            "limit": args.limit,
        },
        "results": compact_results,
        "n-samples": results.get("n-samples", {}),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=1)
    print("WROTE", args.output)


if __name__ == "__main__":
    main()
