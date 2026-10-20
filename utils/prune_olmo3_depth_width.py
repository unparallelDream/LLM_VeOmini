#!/usr/bin/env python3
"""Depth+width pruning of an Olmo-3 checkpoint (1025-7B -> paper mid-scale ~1.4B).

Extends utils/prune_olmo3_width.py (the 200M recipe, Minitron-style importance
width pruning) toward the HiLS paper's second scale ("1.4B" in paper Table 13 /
"1B" in configs/hils_attention/config_hils_attn_8KA2K_HoPE_1B_prop3p1_qcal_r128.json):

    layers 22 (teacher 32), hidden 2048, intermediate 5632,
    32 Q heads / 4 KV heads with head_dim 64 in the paper.

OLMo3-7B is MHA with head_dim 128, so the equivalent OLMo3 student keeps the
teacher's head_dim (RoPE/QK-norm/kernel compatible, same choice as the 200M
prune) and expresses the paper's 4 KV groups as GQA:

    16 Q heads x 128 (= hidden 2048), 4 KV heads x 128 (= kv_dim 512),
    inter 5632, 22 layers, vocab 100,278, untied lm_head  =>  1,402,838,528 params.

Pruning steps (all importance estimated on the same 12x8192 calibration
sequences as the 200M run):
  * depth  : per-block cosine similarity between input/output hidden states
             (Minitron block importance); keep the top-22, first/last always
             kept; ``layer_types`` rebuilt positionally as the 3:1
             sliding/full interleave (SWA layers use plain RoPE, full layers
             use YaRN, so the interleave pattern is semantic);
  * width  : hidden dims / FFN neurons / heads by activation importance
             (identical hooks to prune_olmo3_width.py);
  * GQA    : the 4 highest-importance surviving heads also provide K/V.

Self-verifying: reloads the student, compares calibration CE against the
teacher and asserts the analytic parameter count before writing safetensors.
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
    parser.add_argument("--source", type=Path, required=True, help="HF dir with Olmo-3-1025-7B checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    for field in WIDTH_FIELDS:
        parser.add_argument(f"--{field.replace('_', '-')}", type=int, dest=field, help=f"override {field}")
    parser.add_argument("--num-hidden-layers", type=int, dest="num_hidden_layers", help="target layer count")
    parser.add_argument("--depth-strategy", choices=("importance", "uniform"), default="importance")
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
    if args.num_hidden_layers is None:
        raise SystemExit("Missing required --num-hidden-layers")
    target["num_hidden_layers"] = args.num_hidden_layers
    head_dim = source_config.hidden_size // source_config.num_attention_heads
    if target["num_attention_heads"] * head_dim != target["hidden_size"]:
        raise SystemExit(
            f"hidden_size must equal heads*head_dim ({target['num_attention_heads']}*{head_dim})"
        )
    if target["num_attention_heads"] % target["num_key_value_heads"] != 0:
        raise SystemExit("num_attention_heads must be divisible by num_key_value_heads (GQA groups)")
    if target["num_key_value_heads"] >= target["num_attention_heads"]:
        raise SystemExit("target must be GQA (num_key_value_heads < num_attention_heads)")
    if target["hidden_size"] >= source_config.hidden_size:
        raise SystemExit("target hidden_size must be smaller than the teacher")
    if target["num_hidden_layers"] >= source_config.num_hidden_layers:
        raise SystemExit("target layer count must be smaller than the teacher")
    return target


def rebuild_layer_types(num_layers: int) -> list[str]:
    """Positional 3:1 sliding/full interleave, the OLMo-3 pattern.

    SWA layers run plain RoPE while full layers run the YaRN-scaled rope, so
    the pattern is semantic and is re-derived from position after depth
    pruning regardless of which teacher layers survive.
    """
    return ["full_attention" if (i % 4 == 3) else "sliding_attention" for i in range(num_layers)]


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
def collect_depth_importance(model, batches: list[torch.Tensor], device: str) -> torch.Tensor:
    """Minitron block importance: 1 - cos(input, output) per decoder layer."""
    layers = model.model.layers
    cos_sum = torch.zeros(len(layers), dtype=torch.float64)

    def make_hooks(idx):
        pre = {}

        def pre_hook(_module, args, kwargs):
            pre["x"] = args[0].detach()

        def post_hook(_module, _args, output):
            y = output[0] if isinstance(output, tuple) else output
            x, y = pre["x"].float().flatten(), y.float().flatten()
            cos_sum[idx] += torch.nn.functional.cosine_similarity(x, y, dim=0).double().cpu()

        return pre_hook, post_hook

    handles = []
    for idx, layer in enumerate(layers):
        pre_hook, post_hook = make_hooks(idx)
        handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(layer.register_forward_hook(post_hook))
    token_count = 0
    try:
        for batch in batches:
            _ = model(input_ids=batch.to(device))
            token_count += 1
    finally:
        for handle in handles:
            handle.remove()
    return 1.0 - cos_sum / token_count


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


def select_layers(depth_imp: torch.Tensor | None, num_layers: int) -> list[int]:
    total = depth_imp.shape[0] if depth_imp is not None else num_layers
    if depth_imp is None:
        return sorted({int(v) for v in np.round(np.linspace(0, total - 1, num_layers)).tolist()})
    # First/last layers are always kept (embedding transitions), the rest by importance.
    scores = depth_imp.clone()
    scores[0] = scores[-1] = float("inf")
    keep = torch.topk(scores, num_layers).indices.tolist()
    return sorted(keep)


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
    sel_heads = topk(head_scores, target["num_attention_heads"], num_heads)
    if strategy == "uniform":
        sel_kv = sel_heads[: target["num_key_value_heads"]]
    else:
        # GQA: the highest-importance surviving heads double as KV providers.
        rank = {int(h): r for r, h in enumerate(torch.topk(head_scores, num_heads).indices.tolist())}
        sel_kv = sorted(sel_heads, key=lambda h: rank[h])[: target["num_key_value_heads"]]
    return {
        "selected_hidden_dims": sel_dims,
        "selected_neurons": sel_neurons,
        "selected_q_heads": sel_heads,
        "selected_kv_groups": sel_kv,
    }


def build_student_state_dict(
    source: dict[str, torch.Tensor],
    keep_layers: list[int],
    sel: dict[str, list[int]],
    head_dim: int,
) -> dict[str, torch.Tensor]:
    sel_dims = torch.tensor(sel["selected_hidden_dims"], dtype=torch.long)
    sel_neurons = torch.tensor(sel["selected_neurons"], dtype=torch.long)
    q_rows = torch.cat(
        [torch.arange(h * head_dim, (h + 1) * head_dim) for h in sel["selected_q_heads"]]
    )
    kv_rows = torch.cat(
        [torch.arange(g * head_dim, (g + 1) * head_dim) for g in sel["selected_kv_groups"]]
    )
    remap = {src: dst for dst, src in enumerate(keep_layers)}

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
        src_layer = int(match.group(1))
        if src_layer not in remap:
            return tensor[:0]  # marker for dropped layers; filtered below
        rest = match.group(2)
        if rest in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ):
            return tensor[sel_dims]
        if rest == "self_attn.q_norm.weight":  # (num_heads*head_dim,) - q-dim wide
            return tensor[q_rows]
        if rest == "self_attn.k_norm.weight":  # (num_kv_heads*head_dim,) - kv-dim wide
            return tensor[kv_rows]
        if rest == "self_attn.q_proj.weight":  # (num_heads*hd, hidden)
            return tensor[q_rows][:, sel_dims]
        if rest in ("self_attn.k_proj.weight", "self_attn.v_proj.weight"):  # GQA rows
            return tensor[kv_rows][:, sel_dims]
        if rest == "self_attn.o_proj.weight":  # (hidden, num_heads*hd)
            return tensor[sel_dims][:, q_rows]
        if rest in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):  # (inter, hidden)
            return tensor[sel_neurons][:, sel_dims]
        if rest == "mlp.down_proj.weight":  # (hidden, inter)
            return tensor[sel_dims][:, sel_neurons]
        raise KeyError(f"Unaccounted per-layer tensor: {name}")

    for name, tensor in source.items():
        sliced = slice_weight(name, tensor)
        if sliced.numel() == 0 and name.startswith("model.layers."):
            continue  # dropped layer
        dst = layer_re.sub(lambda m: f"model.layers.{remap.get(int(m.group(1)), 0)}.{m.group(2)}", name)
        student[dst] = sliced.contiguous()
    return student


def expected_shape(name: str, target: dict[str, int], head_dim: int, vocab: int) -> tuple[int, ...]:
    hidden = target["hidden_size"]
    q_dim = target["num_attention_heads"] * head_dim
    kv_dim = target["num_key_value_heads"] * head_dim
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
        "self_attn.q_norm.weight": (q_dim,),
        "self_attn.k_norm.weight": (kv_dim,),
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


def analytic_param_count(target: dict[str, int], num_layers: int, head_dim: int, vocab: int, tied: bool = False) -> int:
    hidden = target["hidden_size"]
    q_dim = target["num_attention_heads"] * head_dim
    kv_dim = target["num_key_value_heads"] * head_dim
    inter = target["intermediate_size"]
    # 2 post-norms (hidden) + q_norm (q_dim) + k_norm (kv_dim)
    per_layer = 2 * hidden * q_dim + 2 * hidden * kv_dim + 3 * hidden * inter + 2 * hidden + q_dim + kv_dim
    embed_and_head = (2 if not tied else 1) * vocab * hidden
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
    depth_imp = (
        collect_depth_importance(teacher, calib_batches, args.device)
        if args.depth_strategy == "importance"
        else None
    )
    del teacher
    torch.cuda.empty_cache()

    if depth_imp is not None:
        print(f"[prune-olmo3] block importance (1-cos): {[f'{v:.4f}' for v in depth_imp.tolist()]}")

    keep_layers = select_layers(depth_imp, target["num_hidden_layers"])
    print(f"[prune-olmo3] kept teacher layers: {keep_layers}")

    sel = select_indices(head_imp, neuron_imp, dim_imp, target, args.strategy)
    print(
        f"[prune-olmo3] q_heads={sel['selected_q_heads']} kv_groups={sel['selected_kv_groups']}"
    )
    student_state = build_student_state_dict(source, keep_layers, sel, head_dim)

    expected = analytic_param_count(
        target, target["num_hidden_layers"], head_dim, config.vocab_size,
        tied=bool(getattr(config, "tie_word_embeddings", False)),
    )
    actual = sum(t.numel() for t in student_state.values())
    print(f"[prune-olmo3] params: analytic={expected:,} actual={actual:,}")
    if actual != expected:
        raise SystemExit("parameter count mismatch")

    args.output.mkdir(parents=True, exist_ok=True)
    student_config = config.to_dict()
    for field in WIDTH_FIELDS:
        student_config[field] = target[field]
    student_config["num_hidden_layers"] = target["num_hidden_layers"]
    student_config["layer_types"] = rebuild_layer_types(target["num_hidden_layers"])
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
                "depth_strategy": args.depth_strategy,
                "head_dim": head_dim,
                "kept_teacher_layers": keep_layers,
                "layer_types": student_config["layer_types"],
                "selected": {k: v for k, v in sel.items()},
                "block_importance": None if depth_imp is None else depth_imp.tolist(),
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
