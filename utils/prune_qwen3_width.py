"""Structured width pruning of a Qwen3 checkpoint (e.g. 0.6B -> 90M/145M/200M).

The layer count, head_dim, vocab and tokenizer are kept untouched; only
hidden_size / num_attention_heads / num_key_value_heads / intermediate_size
shrink.  Selection of the surviving channels follows the Minitron recipe:

  * attention heads ranked by the activation energy arriving at ``o_proj``
  * MLP neurons ranked by the mean |activation| entering ``down_proj``
  * residual channels ranked by the aggregated hidden-state magnitude across
    all layer boundaries (the same selection is applied to the embedding
    columns and every RMSNorm)

GQA consistency is preserved: heads are selected per KV group (each group owns
its 2 query heads in the official Qwen3 head-major layout), so the student
keeps a valid query->KV mapping and an untouched RoPE basis.

The script is self-verifying: after slicing it reloads the student, compares
calibration loss against the teacher, and asserts the produced parameter count
against the analytic expectation before writing ``model.safetensors``.
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
from transformers import AutoConfig, Qwen3ForCausalLM


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

PRESETS: dict[str, dict[str, int]] = {
    # 28 layers, vocab 151936 and head_dim 128 fixed -> ~90M / ~144M / ~200M total (tied)
    "90m": dict(hidden_size=320, intermediate_size=1024, num_attention_heads=4, num_key_value_heads=2),
    "145m": dict(hidden_size=512, intermediate_size=1024, num_attention_heads=4, num_key_value_heads=2),
    "200m": dict(hidden_size=640, intermediate_size=1408, num_attention_heads=4, num_key_value_heads=2),
}

WIDTH_FIELDS = ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="HF dir with model.safetensors + config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), help="target size preset (90m/145m/200m)")
    for field in WIDTH_FIELDS:
        parser.add_argument(f"--{field.replace('_', '-')}", type=int, dest=field, help=f"override {field}")
    parser.add_argument("--strategy", choices=("importance", "uniform"), default="importance")
    parser.add_argument("--calib-data", type=Path, default=Path("/9950backfile/zhangem/data/fineweb_qwen3_data/000_00000.parquet.data"))
    parser.add_argument("--num-calib-seqs", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--num-eval-seqs", type=int, default=4)
    parser.add_argument("--eval-len", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_target(args: argparse.Namespace, source_config) -> dict[str, int]:
    target = dict(PRESETS[args.preset]) if args.preset else {}
    for field in WIDTH_FIELDS:
        override = getattr(args, field)
        if override is not None:
            target[field] = override
    missing = [f for f in WIDTH_FIELDS if f not in target]
    if missing:
        raise SystemExit(f"No preset/--flag given for: {missing}")
    if target["num_attention_heads"] % target["num_key_value_heads"] != 0:
        raise SystemExit("num_attention_heads must be divisible by num_key_value_heads")
    head_dim = getattr(source_config, "head_dim", None) or source_config.hidden_size // source_config.num_attention_heads
    if target["num_attention_heads"] * head_dim < target["hidden_size"] // 2:
        raise SystemExit("attention query dim is suspiciously small relative to hidden_size")
    return target


def load_source(source_dir: Path) -> tuple[dict[str, torch.Tensor], AutoConfig]:
    weights_path = source_dir / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    source = load_file(str(weights_path), device="cpu")
    config = AutoConfig.from_pretrained(str(source_dir))
    if not config.tie_word_embeddings:
        raise SystemExit("Source config must use tie_word_embeddings=true (official Qwen3 does)")
    if getattr(config, "sliding_window", None):
        raise SystemExit("Source uses sliding window; this pruner assumes full attention")
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
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // num_heads
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
    teacher_num_kv_heads: int,
    strategy: str,
) -> dict[str, list[int]]:
    num_heads = head_imp.shape[1]
    # Teacher-side GQA grouping: each teacher KV head owns num_heads//teacher_kv query
    # heads (2 for Qwen3-0.6B).  We keep `target kv` whole teacher groups so the
    # student inherits a valid query->KV mapping.
    group_size = num_heads // teacher_num_kv_heads
    num_teacher_groups = teacher_num_kv_heads

    if strategy == "uniform":
        dim_scores = None
        neuron_scores = None
        head_scores = None
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
    if head_scores is None:
        sel_groups = sorted(set(int(v) for v in np.round(np.linspace(0, num_teacher_groups - 1, target["num_key_value_heads"])).tolist()))
    else:
        group_scores = head_scores.reshape(num_teacher_groups, group_size).sum(dim=1)
        sel_groups = sorted(torch.topk(group_scores, target["num_key_value_heads"]).indices.tolist())
    sel_q_heads = sorted({g * group_size + j for g in sel_groups for j in range(group_size)})
    return {
        "selected_hidden_dims": sel_dims,
        "selected_neurons": sel_neurons,
        "selected_kv_groups": sel_groups,
        "selected_q_heads": sel_q_heads,
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
        if rest in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            return tensor[sel_dims]
        if rest in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"):
            return tensor  # head_dim unchanged
        if rest == "self_attn.q_proj.weight":  # (num_heads*hd, hidden)
            return tensor[q_rows][:, sel_dims]
        if rest in ("self_attn.k_proj.weight", "self_attn.v_proj.weight"):  # (num_kv*hd, hidden)
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


def expected_shape(name: str, source_tensor: torch.Tensor, target: dict[str, int], head_dim: int, source_config) -> tuple[int, ...]:
    hidden = target["hidden_size"]
    q_dim = target["num_attention_heads"] * head_dim
    kv_dim = target["num_key_value_heads"] * head_dim
    inter = target["intermediate_size"]
    layer_re = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return (source_tensor.shape[0], hidden)
    match = layer_re.match(name)
    if match is None:
        if name == "model.norm.weight":
            return (hidden,)
        raise KeyError(name)
    rest = match.group(2)
    table = {
        "input_layernorm.weight": (hidden,),
        "post_attention_layernorm.weight": (hidden,),
        "self_attn.q_norm.weight": (head_dim,),
        "self_attn.k_norm.weight": (head_dim,),
        "self_attn.q_proj.weight": (q_dim, hidden),
        "self_attn.k_proj.weight": (kv_dim, hidden),
        "self_attn.v_proj.weight": (kv_dim, hidden),
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


def analytic_param_count(config, target: dict[str, int], include_lm_head_copy: bool) -> tuple[int, int]:
    """Return (tied parameter count, stored tensor count incl. duplicated lm_head)."""
    hidden = target["hidden_size"]
    head_dim = config.head_dim
    q_dim = target["num_attention_heads"] * head_dim
    kv_dim = target["num_key_value_heads"] * head_dim
    inter = target["intermediate_size"]
    per_layer = (
        hidden * q_dim  # q_proj
        + 2 * hidden * kv_dim  # k/v_proj
        + q_dim * hidden  # o_proj
        + 2 * head_dim  # q/k norm
        + 3 * hidden * inter  # mlp
        + 2 * hidden  # layernorms
    )
    vocab = config.vocab_size
    tied_total = vocab * hidden + config.num_hidden_layers * per_layer + hidden
    stored = tied_total + (vocab * hidden if include_lm_head_copy else 0)
    return tied_total, stored


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory: {args.output}")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    source, source_config = load_source(args.source)
    target = resolve_target(args, source_config)
    head_dim = source_config.head_dim

    print(f"[prune] loading teacher from {args.source}")
    teacher = Qwen3ForCausalLM.from_pretrained(
        str(args.source), torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    teacher.to(args.device).eval()

    print(
        f"[prune] calibrating on {args.num_calib_seqs} x {args.seq_len} tokens "
        f"from {args.calib_data} (strategy={args.strategy})"
    )
    calib_batches = [
        sample_tokens(args.calib_data, (i + 0.25) / args.num_calib_seqs, args.seq_len, rng)
        for i in range(args.num_calib_seqs)
    ]
    if args.strategy == "importance":
        head_imp, neuron_imp, dim_imp = collect_importance(teacher, calib_batches, args.device)
    else:
        num_layers = source_config.num_hidden_layers
        head_imp = torch.zeros(num_layers, source_config.num_attention_heads)
        neuron_imp = torch.zeros(num_layers, source_config.intermediate_size)
        dim_imp = torch.zeros(num_layers, source_config.hidden_size)
    sel = select_indices(head_imp, neuron_imp, dim_imp, target, source_config.num_key_value_heads, args.strategy)
    print(
        "[prune] selections:",
        json.dumps(
            {
                "q_heads": sel["selected_q_heads"],
                "kv_groups": sel["selected_kv_groups"],
                "hidden_dims_head10": sel["selected_hidden_dims"][:10],
                "neurons_head10": sel["selected_neurons"][:10],
            }
        ),
    )

    student_sd = build_student_state_dict(source, sel, target, head_dim)
    for name, tensor in student_sd.items():
        expected = expected_shape(name, source[name], target, head_dim, source_config)
        if tuple(tensor.shape) != expected:
            raise RuntimeError(f"Shape mismatch for {name}: got {tuple(tensor.shape)}, expected {expected}")

    produced = sum(t.numel() for t in student_sd.values())
    tied_total, expected_stored = analytic_param_count(
        source_config, target, include_lm_head_copy="lm_head.weight" in source
    )
    print(
        f"[prune] student parameters (tied): {tied_total:,} | "
        f"stored tensors: {produced:,} (expectation {expected_stored:,})"
    )
    if produced != expected_stored:
        raise RuntimeError("Produced tensor inventory does not match the analytic count")

    student_config = AutoConfig.from_pretrained(str(args.source))
    for field, value in target.items():
        setattr(student_config, field, value)

    eval_batches = [
        sample_tokens(args.calib_data, 0.93 + 0.01 * i, args.eval_len, rng)
        for i in range(args.num_eval_seqs)
    ]
    teacher_loss = mean_ce_loss(teacher, eval_batches, args.device)

    student = Qwen3ForCausalLM(student_config)
    missing, unexpected = student.load_state_dict(student_sd, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]  # tied: filled by embed
    if missing or unexpected:
        raise RuntimeError(f"Student load mismatch: missing={missing} unexpected={unexpected}")
    student.tie_weights()
    student.to(dtype=torch.bfloat16, device=args.device).eval()
    student_loss = mean_ce_loss(student, eval_batches, args.device)
    print(f"[prune] eval CE loss: teacher={teacher_loss:.4f} student={student_loss:.4f}")

    args.output.mkdir(parents=True, exist_ok=True)
    save_file(student_sd, str(args.output / "model.safetensors"))
    student_config.to_json_file(str(args.output / "config.json"))
    for asset in TOKENIZER_ASSETS:
        asset_path = args.source / asset
        if asset_path.is_file():
            shutil.copy2(asset_path, args.output / asset)

    report = {
        "source": str(args.source.resolve()),
        "strategy": args.strategy,
        "target": target,
        "calibration": {
            "data": str(args.calib_data.resolve()),
            "num_seqs": args.num_calib_seqs,
            "seq_len": args.seq_len,
        },
        "selected_q_heads": sel["selected_q_heads"],
        "selected_kv_groups": sel["selected_kv_groups"],
        "selected_hidden_dims": sel["selected_hidden_dims"],
        "selected_neurons": sel["selected_neurons"],
        "student_param_count_tied": tied_total,
        "student_param_count_stored": produced,
        "analytic_param_count_stored": expected_stored,
        "eval_ce_loss_teacher": teacher_loss,
        "eval_ce_loss_student": student_loss,
    }
    with (args.output / "prune_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"[prune] wrote student to {args.output}")


if __name__ == "__main__":
    main()
