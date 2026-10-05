#!/usr/bin/env python3
"""Structured width pruning of an Olmo-3 checkpoint (1025-7B -> ~140M/~200M).

Adapted from utils/prune_qwen3_width.py (Minitron recipe) for the Olmo3
architecture:

  * layer count (32, incl. the 3:1 SWA/full ``layer_types`` pattern), head_dim
    (128), vocab (100,278) and tokenizer stay untouched; only hidden_size /
    num heads / intermediate_size shrink;
  * Olmo3 is MHA (num_key_value_heads == num_attention_heads) and untied:
    teacher group size is 1 and ``lm_head.weight`` is sliced with the same
    residual-channel selection as the embedding;
  * rope_scaling (YaRN 8x), sliding_window and eos/pad ids are carried over
    verbatim so the student inherits the teacher's long-context geometry;
  * sliding_window/layer_types require transformers>=4.57 (Olmo3ForCausalLM).

Self-verifying: reloads the student, compares calibration CE against the
teacher and asserts the parameter count before writing model.safetensors.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, Olmo3ForCausalLM

TOKENIZER_ASSETS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.jinja",
    "README.md",
    "LICENSE",
)

WIDTH_FIELDS = ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="HF dir with model.safetensors + config.json")
    parser.add_argument("--output", type=Path, required=True)
    for field in WIDTH_FIELDS:
        parser.add_argument(f"--{field.replace('_', '-')}", type=int, dest=field, help=f"override {field}")
    parser.add_argument("--strategy", choices=("importance", "uniform"), default="importance")
    parser.add_argument(
        "--calib-data",
        type=Path,
        default=Path("/9950backfile/zhangem/data/dolma3_1025_3B_packed/000_00000.parquet.data"),
        help="flat uint32 .parquet.data in the Olmo3 vocab",
    )
    parser.add_argument("--num-calib-seqs", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--num-eval-seqs", type=int, default=4)
    parser.add_argument("--eval-len", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_target(args: argparse.Namespace, source_config) -> dict[str, int]:
    target = {f: getattr(args, f) for f in WIDTH_FIELDS}
    missing = [f for f in WIDTH_FIELDS if target[f] is None]
    if missing:
        raise SystemExit(f"Missing required --flags: {missing}")
    if target["num_attention_heads"] != target["num_key_value_heads"]:
        raise SystemExit("Olmo3 pruner keeps MHA (num_attention_heads == num_key_value_heads)")
    head_dim = source_config.hidden_size // source_config.num_attention_heads
    if target["num_attention_heads"] * head_dim != target["hidden_size"]:
        raise SystemExit(
            f"hidden_size must equal heads*head_dim ({target['num_attention_heads']}*{head_dim})"
        )
    if target["hidden_size"] >= source_config.hidden_size:
        raise SystemExit("target hidden_size must be smaller than the teacher")
    return target


def load_source(source_dir: Path) -> tuple[dict[str, torch.Tensor], AutoConfig]:
    weights_path = source_dir / "model.safetensors"
    if weights_path.is_file():
        source = load_file(str(weights_path), device="cpu")
    else:
        index_path = source_dir / "model.safetensors.index.json"
        shards = sorted(source_dir.glob("model-*-of-*.safetensors"))
        if not (index_path.is_file() and shards):
            raise FileNotFoundError(weights_path)
        source = {}
        for shard in shards:
            source.update(load_file(str(shard), device="cpu"))
    config = AutoConfig.from_pretrained(str(source_dir))
    if "lm_head.weight" not in source:
        raise SystemExit("expected untied lm_head.weight in the Olmo3 checkpoint")
    return source, config


def sample_tokens(data_path: Path, start_frac: float, seq_len: int, rng: np.random.Generator) -> torch.Tensor:
    tokens = np.memmap(data_path, dtype="<u4", mode="r")
    span = len(tokens) - seq_len - 1
    start = int(start_frac * span)
    start = int(np.clip(start, 0, span))
    batch = np.asarray(tokens[start : start + seq_len], dtype=np.int64)
    return torch.from_numpy(batch)[None, :]


@torch.no_grad()
def collect_importance(model, batches: list[torch.Tensor], device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layers = model.model.layers
    num_layers = len(layers)
    config = model.config
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads
    inter = config.intermediate_size

    head_sq = torch.zeros(num_layers, num_heads, dtype=torch.float64)
    neuron_sum = torch.zeros(num_layers, inter, dtype=torch.float64)
    dim_sum = torch.zeros(num_layers, config.hidden_size, dtype=torch.float64)
    token_count = 0

    def o_proj_pre_hook(idx):
        def hook(_module, inputs):
            x = inputs[0]
            x = x.reshape(x.shape[0], x.shape[1], num_heads, head_dim)
            head_sq[idx] += x.float().pow(2).sum(dim=(0, 1, 3)).double().cpu()
        return hook

    def down_proj_pre_hook(idx):
        def hook(_module, inputs):
            neuron_sum[idx] += inputs[0].float().abs().sum(dim=(0, 1)).double().cpu()
        return hook

    def layer_post_hook(idx):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            dim_sum[idx] += hidden.float().abs().sum(dim=(0, 1)).double().cpu()
        return hook

    handles = []
    for idx, layer in enumerate(layers):
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(o_proj_pre_hook(idx)))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(down_proj_pre_hook(idx)))
        handles.append(layer.register_forward_hook(layer_post_hook(idx)))
    try:
        for batch in batches:
            _ = model(input_ids=batch.to(device))
            token_count += batch.numel()
    finally:
        for handle in handles:
            handle.remove()

    head_imp = (head_sq / token_count).sqrt()
    neuron_imp = neuron_sum / token_count
    dim_imp = dim_sum / token_count
    return head_imp, neuron_imp, dim_imp


def select_indices(
    head_imp: torch.Tensor,
    neuron_imp: torch.Tensor,
    dim_imp: torch.Tensor,
    target: dict[str, int],
    strategy: str,
) -> dict[str, list[int]]:
    num_heads = head_imp.shape[1]

    if strategy == "uniform":
        dim_scores = neuron_scores = head_scores = None
    else:
        dim_scores = dim_imp.sum(dim=0)
        neuron_scores = neuron_imp.sum(dim=0)
        head_scores = head_imp.sum(dim=0)

    def topk(scores: torch.Tensor | None, k: int, total: int) -> list[int]:
        if scores is None:
            return sorted({int(v) for v in np.round(np.linspace(0, total - 1, k)).tolist()})
        return sorted(torch.topk(scores, k).indices.tolist())

    sel_dims = topk(dim_scores, target["hidden_size"], dim_imp.shape[1])
    sel_neurons = topk(neuron_scores, target["intermediate_size"], neuron_imp.shape[1])
    # MHA teacher: group size 1, so selected heads are simultaneously the
    # surviving q and kv heads; query->KV mapping stays the identity.
    sel_heads = topk(head_scores, target["num_attention_heads"], num_heads)
    return {
        "selected_hidden_dims": sel_dims,
        "selected_neurons": sel_neurons,
        "selected_kv_groups": sel_heads,
        "selected_q_heads": sel_heads,
    }


def build_student_state_dict(
    source: dict[str, torch.Tensor], sel: dict[str, list[int]], target: dict[str, int], head_dim: int
) -> dict[str, torch.Tensor]:
    sel_dims = torch.tensor(sel["selected_hidden_dims"], dtype=torch.long)
    sel_neurons = torch.tensor(sel["selected_neurons"], dtype=torch.long)
    q_rows = torch.cat(
        [torch.arange(h * head_dim, (h + 1) * head_dim) for h in sel["selected_q_heads"]]
    )
    kv_rows = torch.cat(
        [torch.arange(g * head_dim, (g + 1) * head_dim) for g in sel["selected_kv_groups"]]
    )

    layer_re = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
    student: dict[str, torch.Tensor] = {}

    def slice_weight(name: str, tensor: torch.Tensor) -> torch.Tensor:
        if name in ("model.embed_tokens.weight", "lm_head.weight"):
            return tensor[:, sel_dims]
        match = layer_re.match(name)
        if match is None:
            if name == "model.norm.weight":
                return tensor[sel_dims]
            raise KeyError(f"Unaccounted tensor in source checkpoint: {name}")
        rest = match.group(2)
        if rest in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "self_attn.q_norm.weight",  # OLMo3: hidden-wide QK norms
            "self_attn.k_norm.weight",
        ):
            return tensor[sel_dims]
        if rest == "self_attn.q_proj.weight":  # (num_heads*hd, hidden)
            return tensor[q_rows][:, sel_dims]
        if rest in ("self_attn.k_proj.weight", "self_attn.v_proj.weight"):  # (MHA: same rows)
            return tensor[kv_rows][:, sel_dims]
        if rest == "self_attn.o_proj.weight":  # (hidden, num_heads*hd)
            return tensor[sel_dims][:, q_rows]
        if rest in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):  # (inter, hidden)
            return tensor[sel_neurons][:, sel_dims]
        if rest == "mlp.down_proj.weight":  # (hidden, inter)
            return tensor[sel_dims][:, sel_neurons]
        raise KeyError(f"Unaccounted per-layer tensor: {name}")

    for name, tensor in source.items():
        student[name] = slice_weight(name, tensor).contiguous()
    return student


def expected_shape(name: str, target: dict[str, int], head_dim: int, vocab: int) -> tuple[int, ...]:
    hidden = target["hidden_size"]
    q_dim = target["num_attention_heads"] * head_dim
    inter = target["intermediate_size"]
    layer_re = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return (vocab, hidden)
    match = layer_re.match(name)
    if match is None:
        if name == "model.norm.weight":
            return (hidden,)
        raise KeyError(name)
    rest = match.group(2)
    table = {
        "input_layernorm.weight": (hidden,),
        "post_attention_layernorm.weight": (hidden,),
        "post_feedforward_layernorm.weight": (hidden,),
        "self_attn.q_norm.weight": (hidden,),
        "self_attn.k_norm.weight": (hidden,),
        "self_attn.q_proj.weight": (q_dim, hidden),
        "self_attn.k_proj.weight": (q_dim, hidden),
        "self_attn.v_proj.weight": (q_dim, hidden),
        "self_attn.o_proj.weight": (hidden, q_dim),
        "mlp.gate_proj.weight": (inter, hidden),
        "mlp.up_proj.weight": (inter, hidden),
        "mlp.down_proj.weight": (hidden, inter),
    }
    if rest not in table:
        raise KeyError(name)
    return table[rest]


@torch.no_grad()
def mean_ce_loss(model, batches: list[torch.Tensor], device: str) -> float:
    total_loss = 0.0
    total_tokens = 0
    for batch in batches:
        input_ids = batch.to(device)
        logits = model(input_ids=input_ids).logits[:, :-1, :]
        targets = input_ids[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1), reduction="sum"
        )
        total_loss += loss.item()
        total_tokens += targets.numel()
    return total_loss / total_tokens


def analytic_param_count(target: dict[str, int], num_layers: int, head_dim: int, vocab: int) -> int:
    hidden = target["hidden_size"]
    q_dim = target["num_attention_heads"] * head_dim
    inter = target["intermediate_size"]
    per_layer = 4 * hidden * q_dim + 3 * hidden * inter + 4 * hidden  # post_attn/post_ffn/q_norm/k_norm 均为 hidden 宽
    embed_and_head = 2 * vocab * hidden  # untied
    return num_layers * per_layer + embed_and_head + hidden


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    source, config = load_source(args.source)
    target = resolve_target(args, config)
    head_dim = config.hidden_size // config.num_attention_heads
    print(f"[prune-olmo3] teacher: {config.hidden_size}h/{config.num_hidden_layers}L/"
          f"{config.num_attention_heads}heads/inter{config.intermediate_size}/vocab{config.vocab_size}")
    print(f"[prune-olmo3] student target: {target} head_dim={head_dim}")

    spans = rng.uniform(0.05, 0.95, size=args.num_calib_seqs + args.num_eval_seqs)
    calib_batches = [sample_tokens(args.calib_data, float(f), args.seq_len, rng) for f in spans[: args.num_calib_seqs]]
    eval_batches = [sample_tokens(args.calib_data, float(f), args.eval_len, rng) for f in spans[args.num_calib_seqs :]]

    teacher = Olmo3ForCausalLM.from_pretrained(str(args.source), torch_dtype=torch.bfloat16).to(args.device)
    teacher.eval()
    teacher_ce = mean_ce_loss(teacher, eval_batches, args.device)
    print(f"[prune-olmo3] teacher CE on eval seqs: {teacher_ce:.4f}", flush=True)
    head_imp, neuron_imp, dim_imp = collect_importance(teacher, calib_batches, args.device)
    del teacher
    torch.cuda.empty_cache()

    sel = select_indices(head_imp, neuron_imp, dim_imp, target, args.strategy)
    student_state = build_student_state_dict(source, sel, target, head_dim)

    expected = analytic_param_count(target, config.num_hidden_layers, head_dim, config.vocab_size)
    actual = sum(t.numel() for t in student_state.values())
    print(f"[prune-olmo3] params: analytic={expected:,} actual={actual:,}")
    if actual != expected:
        raise SystemExit("parameter count mismatch")

    args.output.mkdir(parents=True, exist_ok=True)
    student_config = config.to_dict()
    for field in WIDTH_FIELDS:
        student_config[field] = target[field]
    student_config["architectures"] = ["Olmo3ForCausalLM"]
    student_config_path = args.output / "config.json"
    student_config_path.write_text(json.dumps(student_config, indent=2))

    for name, tensor in student_state.items():
        want = expected_shape(name, target, head_dim, config.vocab_size)
        if tuple(tensor.shape) != want:
            raise SystemExit(f"shape mismatch for {name}: {tuple(tensor.shape)} != {want}")

    save_file(student_state, str(args.output / "model.safetensors"))
    for asset in TOKENIZER_ASSETS:
        src = args.source / asset
        if src.is_file():
            shutil.copy2(src, args.output / asset)

    student = Olmo3ForCausalLM.from_pretrained(str(args.output), torch_dtype=torch.bfloat16).to(args.device)
    student.eval()
    student_ce = mean_ce_loss(student, eval_batches, args.device)
    print(f"[prune-olmo3] student CE on eval seqs: {student_ce:.4f} (teacher {teacher_ce:.4f})")

    (args.output / "prune_summary.json").write_text(
        json.dumps(
            {
                "source": str(args.source),
                "target": target,
                "strategy": args.strategy,
                "head_dim": head_dim,
                "selected": {k: v for k, v in sel.items()},
                "teacher_ce": teacher_ce,
                "student_ce": student_ce,
                "params": actual,
                "calib_data": str(args.calib_data),
            },
            indent=1,
        )
    )
    print(f"[prune-olmo3] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
