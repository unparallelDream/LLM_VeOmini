"""Convert an official tied-embedding Qwen3 checkpoint to a frozen MASS base.

The conversion is intentionally strict: every target tensor must be copied,
derived deterministically, or be one of the newly initialized MASS factors.
The standalone landmark embedding is initialized from Qwen's EOS embedding so
an anchor-only run never freezes a random structural parameter.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

import models  # noqa: F401 - register qwen_mass
from models.MASSIndex.configuration_mass import MASSConfig
from models.MASSIndex.modeling_qwen_mass import MASSForCausalLM


TOKENIZER_ASSETS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "generation_config.json",
    "README.md",
    "LICENSE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    source_weights = args.source / "model.safetensors"
    if not source_weights.is_file():
        raise FileNotFoundError(source_weights)
    source = load_file(str(source_weights), device="cpu")
    source_embed = source.get("model.embed_tokens.weight")
    if source_embed is None:
        raise KeyError("Official Qwen checkpoint has no model.embed_tokens.weight")

    config = MASSConfig.from_pretrained(str(args.config))
    if not getattr(config, "enable_external_lmk_embed", False):
        raise ValueError("Conversion requires enable_external_lmk_embed=true")
    if getattr(config, "enable_lmk_q_proj", False):
        raise ValueError("Anchor-only conversion refuses a new lmk_q_proj")
    if source_embed.shape[0] != config.vocab_size:
        raise ValueError(
            f"Source vocab rows {source_embed.shape[0]} != config vocab_size {config.vocab_size}"
        )

    torch.manual_seed(args.seed)
    model = MASSForCausalLM(config)
    target = model.state_dict()
    converted: dict[str, torch.Tensor] = {}
    copied: list[str] = []
    derived: list[str] = []
    initialized: list[str] = []

    for name, target_tensor in target.items():
        source_tensor = source.get(name)
        if source_tensor is not None and source_tensor.shape == target_tensor.shape:
            converted[name] = source_tensor.contiguous()
            copied.append(name)
            continue

        if name == "model.embed_tokens.weight":
            if target_tensor.shape[1:] != source_embed.shape[1:] or target_tensor.shape[0] < source_embed.shape[0]:
                raise ValueError(
                    f"Cannot pad embedding {tuple(source_embed.shape)} -> {tuple(target_tensor.shape)}"
                )
            padded = source_embed.new_empty(target_tensor.shape)
            padded[: source_embed.shape[0]].copy_(source_embed)
            padded[source_embed.shape[0] :].copy_(source_embed[config.eos_token_id])
            converted[name] = padded.contiguous()
            derived.append(name)
            continue

        if name == "lm_head.weight":
            if config.tie_word_embeddings:
                # Tied snapshot: lm_head shares embed_tokens (same shape, incl. the
                # landmark padding rows).  Clone so safetensors sees distinct buffers.
                tied_embed = converted.get("model.embed_tokens.weight")
                if tied_embed is None or tied_embed.shape != target_tensor.shape:
                    raise ValueError(
                        f"Tied lm_head shape {tuple(target_tensor.shape)} != converted embedding"
                    )
                converted[name] = tied_embed.clone().contiguous()
                derived.append(name)
                continue
            if target_tensor.shape != source_embed.shape:
                raise ValueError(
                    f"LM head shape {tuple(target_tensor.shape)} != source embedding {tuple(source_embed.shape)}"
                )
            converted[name] = source_embed.clone().contiguous()
            derived.append(name)
            continue

        if name == "model.lmk_embed":
            converted[name] = source_embed[config.eos_token_id].clone().contiguous()
            derived.append(name)
            continue

        if ".mass_anchor_generator." in name:
            converted[name] = target_tensor.to(dtype=source_embed.dtype).contiguous()
            initialized.append(name)
            continue

        if source_tensor is not None:
            raise ValueError(
                f"Shape mismatch for {name}: source={tuple(source_tensor.shape)}, target={tuple(target_tensor.shape)}"
            )
        raise KeyError(f"Unaccounted target tensor missing from official Qwen checkpoint: {name}")

    unused_source = sorted(set(source).difference(target).difference({"lm_head.weight"}))
    if unused_source:
        raise KeyError(f"Official Qwen tensors not represented in MASS target: {unused_source}")

    save_file(converted, str(args.output / "model.safetensors"))
    shutil.copy2(args.config, args.output / "config.json")
    for asset in TOKENIZER_ASSETS:
        source_asset = args.source / asset
        if source_asset.is_file():
            shutil.copy2(source_asset, args.output / asset)

    mass_layers = sorted(
        {
            int(name.split(".layers.", 1)[1].split(".", 1)[0])
            for name in initialized
        }
    )
    trainable_count = sum(converted[name].numel() for name in initialized)
    report = {
        "source": str(args.source.resolve()),
        "config": str(args.config.resolve()),
        "seed": args.seed,
        "tied_word_embeddings": bool(config.tie_word_embeddings),
        "copied_tensor_count": len(copied),
        "derived_tensors": derived,
        "initialized_anchor_tensors": {
            name: list(converted[name].shape) for name in initialized
        },
        "mass_layers": mass_layers,
        "trainable_anchor_parameter_count": trainable_count,
        "lmk_embed_initialization": f"source embedding row eos_token_id={config.eos_token_id}",
    }
    with (args.output / "conversion_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")

    reloaded = load_file(str(args.output / "model.safetensors"), device="cpu")
    if set(reloaded) != set(target):
        raise RuntimeError("Saved checkpoint tensor inventory changed after reload")
    for name, tensor in reloaded.items():
        if tensor.shape != target[name].shape:
            raise RuntimeError(f"Reloaded shape mismatch for {name}")
    if not torch.equal(reloaded["model.lmk_embed"], source_embed[config.eos_token_id]):
        raise RuntimeError("Landmark embedding is not an exact EOS embedding copy")

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
