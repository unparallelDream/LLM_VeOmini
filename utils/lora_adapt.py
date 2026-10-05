"""Self-contained LoRA injection for frozen-backbone MASS adaptation.

Wraps selected ``nn.Linear`` projections with a parallel low-rank branch
(``out = base(x) + (alpha/r) * B(A(x))``).  ``A`` uses Kaiming init and ``B``
starts at zero, so an injected model is bit-identical to the original at
step 0 — required to not disturb pretrained weights.

Typical use (env-gated in tasks/pretrain_mass_with_ruler.py):

    inject_lora(
        model, rank=16, alpha=32,
        targets="self_attn.k_proj,self_attn.v_proj",
        only_layers=model.model._mass_layer_indices,
    )

Injected parameters are named ``<orig>.lora_A.weight`` / ``<orig>.lora_B.weight``
so ``--train.freeze_pattern`` can keep them trainable next to the MASS anchors,
and trainable-only checkpoints pick them up automatically.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Frozen base Linear + trainable low-rank parallel branch."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"lora rank must be > 0, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        # Trainable branch in FP32 on the base weight's device: injection
        # happens after the model is already on its init device, so the
        # default CPU device would break the first forward.
        device = base.weight.device
        self.lora_A = nn.Linear(base.in_features, rank, bias=False).to(device)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False).to(device)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.scale = alpha / rank

    # The surrounding attention code accesses ``o_proj.weight.dtype`` /
    # ``q_proj.weight.device`` etc. directly, so proxy the base Linear's
    # public attributes through the wrapper.
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        out = self.base(x)
        delta = self.lora_B(self.lora_A(x.to(self.lora_A.weight.dtype)))
        return out + (delta * self.scale).to(dtype)


def inject_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    targets: str,
    only_layers=None,
):
    """Replace matching Linears in-place with LoRALinear wrappers.

    Args:
        model: the foundation model (anything with named modules).
        rank / alpha: LoRA hyper-parameters.
        targets: comma-separated substrings matched against module names,
            e.g. ``"self_attn.k_proj,self_attn.v_proj"``.
        only_layers: optional iterable of decoder layer indices; when given,
            only ``layers.{i}.`` for i in it are adapted (e.g. the seven
            MASS/HiLS retrieval layers).
    Returns: (wrapped_count, trainable_param_count)
    """
    keys = [t.strip() for t in targets.split(",") if t.strip()]
    if not keys:
        raise ValueError("empty lora target list")
    layer_ids = None if only_layers is None else {int(i) for i in only_layers}

    wrapped = 0
    trainable = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(k in name for k in keys):
            continue
        if ".base" in name:  # already wrapped
            continue
        if layer_ids is not None:
            parts = name.split(".")
            try:
                if "layers" in parts and int(parts[parts.index("layers") + 1]) not in layer_ids:
                    continue
            except (ValueError, IndexError):
                continue
        parent = model
        for attr in name.split(".")[:-1]:
            parent = getattr(parent, attr)
        leaf = name.split(".")[-1]
        setattr(parent, leaf, LoRALinear(module, rank, alpha))
        wrapped += 1
    for pname, p in model.named_parameters():
        if ".lora_" in pname and p.requires_grad:
            trainable += p.numel()
    if wrapped == 0:
        raise RuntimeError(f"no module matched lora targets {keys}")
    return wrapped, trainable
