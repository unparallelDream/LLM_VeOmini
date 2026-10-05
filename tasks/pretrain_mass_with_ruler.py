"""Opt-in MASS training entry point; leaves the HiLS task unchanged."""

import os

from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer, ModelState, OptimizerState

# Register the independent qwen_mass model/config before the shared trainer
# parses the JSON configuration.
import models.MASSIndex.modeling_registry  # noqa: F401
import pretrain_with_ruler


if os.environ.get("MASS_TRAINABLE_ONLY_CHECKPOINT", "0") == "1":
    _full_model_state_dict = ModelState.state_dict
    _full_optimizer_state_dict = OptimizerState.state_dict

    def _trainable_model_state_dict(self):
        state = _full_model_state_dict(self)
        trainable = {name for name, param in self.model.named_parameters() if param.requires_grad}
        filtered = {name: tensor for name, tensor in state.items() if name in trainable}
        missing = sorted(trainable.difference(filtered))
        if missing:
            raise RuntimeError(f"trainable parameters missing from checkpoint state: {missing}")
        if not filtered:
            raise RuntimeError("trainable-only checkpoint would be empty")
        if not hasattr(self, "_mass_checkpoint_summary_logged"):
            print(
                f"[mass-checkpoint] saving {len(filtered)} trainable tensors only: "
                f"{sorted(filtered)}",
                flush=True,
            )
            self._mass_checkpoint_summary_logged = True
        return filtered

    def _load_trainable_model_state_dict(self, state_dict):
        return set_model_state_dict(
            model=self.model,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )

    def _checkpoint_optimizer_state_dict(self):
        # DCP initializes empty Adam state with a synthetic zero-gradient step.
        # Mark that internal step so the real-gradient fail-fast hook ignores it.
        self.optimizer._mass_initializing_checkpoint_state = True
        try:
            return _full_optimizer_state_dict(self)
        finally:
            self.optimizer._mass_initializing_checkpoint_state = False

    def _create_small_storage_writer(cls, checkpoint_dir):
        thread_count = max(1, int(os.environ.get("MASS_DCP_THREAD_COUNT", "1")))
        return FileSystemWriter(
            checkpoint_dir,
            thread_count=thread_count,
            single_file_per_rank=True,
            sync_files=True,
        )

    ModelState.state_dict = _trainable_model_state_dict
    ModelState.load_state_dict = _load_trainable_model_state_dict
    OptimizerState.state_dict = _checkpoint_optimizer_state_dict
    DistributedCheckpointer._create_storage_writer = classmethod(_create_small_storage_writer)


if os.environ.get("MASS_LORA_ADAPT", "0") == "1":
    # Anchor + LoRA adaptation: wrap build_foundation_model so the built model
    # gets LoRA branches (utils/lora_adapt.py) injected right after loading.
    # Zero-init keeps step-0 outputs bit-identical to the frozen base; the
    # injected params stay trainable via the freeze_pattern's lora exemption.
    from utils.lora_adapt import inject_lora

    _lora_rank = int(os.environ.get("MASS_LORA_RANK", "16"))
    _lora_alpha = float(os.environ.get("MASS_LORA_ALPHA", "32"))
    _lora_targets = os.environ.get(
        "MASS_LORA_TARGETS", "self_attn.k_proj,self_attn.v_proj"
    )
    _lora_scope = os.environ.get("MASS_LORA_SCOPE", "retrieval")  # retrieval|all

    _orig_build = pretrain_with_ruler.build_foundation_model

    def _build_with_lora(*args, **kwargs):
        model = _orig_build(*args, **kwargs)
        only_layers = None
        if _lora_scope == "retrieval":
            only_layers = list(model.model._mass_layer_indices)
        elif _lora_scope != "all":
            raise ValueError(f"MASS_LORA_SCOPE must be retrieval|all, got {_lora_scope!r}")
        wrapped, trainable = inject_lora(
            model, rank=_lora_rank, alpha=_lora_alpha,
            targets=_lora_targets, only_layers=only_layers,
        )
        print(
            f"[mass-lora] injected rank={_lora_rank} alpha={_lora_alpha} "
            f"scope={_lora_scope} targets={_lora_targets}: "
            f"{wrapped} linears, {trainable} trainable lora params",
            flush=True,
        )
        return model

    pretrain_with_ruler.build_foundation_model = _build_with_lora


if __name__ == "__main__":
    pretrain_with_ruler.main()
