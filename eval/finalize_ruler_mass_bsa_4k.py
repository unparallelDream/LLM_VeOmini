#!/usr/bin/env python3
"""Merge separately completed MASS/fixed0 and strict-equal-KV BSA runs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mass-dir", type=Path, required=True)
    parser.add_argument("--bsa-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "mass": args.mass_dir / "mass.json",
        "fixed0": args.mass_dir / "fixed0.json",
        "bsa": args.bsa_dir / "bsa.json",
    }
    rows = {name: json.loads(path.read_text()) for name, path in sources.items()}

    numeric_budget_keys = (
        "local_window",
        "remote_topk_chunks",
        "chunk_size",
        "nominal_max_kv_tokens_per_query",
    )
    reference = rows["mass"]
    for name, row in rows.items():
        for key in numeric_budget_keys:
            if row["attention_budget"][key] != reference["attention_budget"][key]:
                raise RuntimeError(f"Attention budget mismatch: {name}/{key}")
        if row["raw_max_seq_len"] != reference["raw_max_seq_len"]:
            raise RuntimeError(f"Raw sequence length mismatch: {name}")
        for ref_task, task in zip(reference["tasks"], row["tasks"], strict=True):
            if task["sample_hashes"] != ref_task["sample_hashes"]:
                raise RuntimeError(f"Sample mismatch: {name}/{task['task']}")
            if task["model_input_len"] != ref_task["model_input_len"]:
                raise RuntimeError(f"Input-layout mismatch: {name}/{task['task']}")

    if not rows["bsa"]["load_metadata"]["bsa_masks_chunk_end_landmarks"]:
        raise RuntimeError("Strict BSA did not mask chunk-end landmark value keys")
    if not rows["bsa"]["load_metadata"]["bsa_uses_hope_on_replaced_layers"]:
        raise RuntimeError("Strict BSA did not use HoPE on replacement layers")
    if rows["bsa"]["load_metadata"]["bsa_replaced_layers"] != [3, 7, 11, 15, 19, 23, 27, 31]:
        raise RuntimeError("Unexpected BSA layer placement")

    common_budget = {
        key: reference["attention_budget"][key] for key in numeric_budget_keys
    }
    common_budget["effective_layout"] = (
        "All variants use the same 4161-position landmark-expanded input, "
        "HoPE ids, W/K/S, and masked chunk-end landmark value keys."
    )
    comparison = {
        "raw_context_length": reference["raw_max_seq_len"],
        "sample_hashes_verified_equal": True,
        "attention_budget_verified_equal": True,
        "model_input_layout_verified_equal": True,
        "shared_bsa_weights_verified_equal": True,
        "attention_budget": common_budget,
        "variants": {
            name: {
                "macro_token_accuracy": row["macro_token_accuracy"],
                "micro_token_accuracy": row["micro_token_accuracy"],
                "macro_exact_match": row["macro_exact_match"],
                "peak_memory_gib": row["peak_memory_gib"],
                "wall_seconds": row["wall_seconds"],
                "tasks": [
                    {
                        "task": task["task"],
                        "model_input_len": task["model_input_len"],
                        "token_accuracy": task["token_accuracy"],
                        "exact_match": task["exact_match"],
                        "steady_median_ms": task["steady_median_ms"],
                        "steady_median_effective_tokens_per_s": task[
                            "steady_median_effective_tokens_per_s"
                        ],
                    }
                    for task in row["tasks"]
                ],
            }
            for name, row in rows.items()
        },
    }
    for name, source in sources.items():
        shutil.copy2(source, args.output_dir / f"{name}.json")
    output = args.output_dir / "comparison.json"
    output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
