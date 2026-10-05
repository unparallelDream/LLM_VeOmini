"""Small, recoverable DCP checkpoints for frozen-backbone training.

Only parameters with ``requires_grad=True`` are materialized in the model state.
The optimizer already contains only those parameters.  Checkpoints are first
written to an ``.incomplete`` directory and atomically published only after DCP
metadata and every rank's extra state are present.
"""

import json
import os
from typing import Any, Dict, Optional

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.utils import logging


logger = logging.get_logger(__name__)
_GLOBAL_STEP_PREFIX = "global_step_"
_COMPLETED_FILE = "COMPLETED.json"


class TrainableModelState(Stateful):
    """DCP stateful wrapper that excludes every frozen model parameter."""

    def __init__(self, model):
        self.model = model
        self.trainable_names = tuple(name for name, param in model.named_parameters() if param.requires_grad)
        if not self.trainable_names:
            raise ValueError("Trainable-only checkpoint requested, but the model has no trainable parameters.")
        self.save_options = StateDictOptions(ignore_frozen_params=True, strict=True)
        # torch's load path still reports frozen parameters as missing, even
        # when ignore_frozen_params=True, so strictness is enforced manually
        # for the trainable subset below.
        self.load_options = StateDictOptions(ignore_frozen_params=True, strict=False)

    @property
    def tensor_count(self) -> int:
        return len(self.trainable_names)

    def state_dict(self):
        return get_model_state_dict(model=self.model, options=self.save_options)

    def load_state_dict(self, state_dict):
        incompatible = set_model_state_dict(
            model=self.model,
            model_state_dict=state_dict,
            options=self.load_options,
        )
        missing_trainable = [name for name in incompatible.missing_keys if name in self.trainable_names]
        if missing_trainable or incompatible.unexpected_keys:
            raise RuntimeError(
                "Trainable-only model checkpoint mismatch: "
                f"missing_trainable={missing_trainable}, unexpected={incompatible.unexpected_keys}"
            )


class TrainableOptimizerState(Stateful):
    """Optimizer wrapper without VeOmni EP/global-parallel-state dependencies."""

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.options = StateDictOptions(ignore_frozen_params=True, strict=True)

    def state_dict(self):
        # DCP materializes a missing optimizer state by issuing an internal
        # optimizer.step() before loading the saved tensors.  MASS installs a
        # first-real-step gradient assertion, so mark this bookkeeping step and
        # let the assertion run later, after the first actual backward pass.
        marker = "_mass_initializing_checkpoint_state"
        marker_was_present = hasattr(self.optimizer, marker)
        previous_marker = getattr(self.optimizer, marker, None)
        setattr(self.optimizer, marker, True)
        try:
            return get_optimizer_state_dict(
                model=self.model,
                optimizers=self.optimizer,
                options=self.options,
            )
        finally:
            if marker_was_present:
                setattr(self.optimizer, marker, previous_marker)
            else:
                delattr(self.optimizer, marker)

    def load_state_dict(self, state_dict):
        set_optimizer_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            optim_state_dict=state_dict,
            options=self.options,
        )


class TrainableDistributedCheckpointer(DistributedCheckpointer):
    """DCP checkpointer for a frozen base model plus trainable adapter/anchor parameters."""

    @classmethod
    def _create_storage_writer(cls, checkpoint_dir: str) -> FileSystemWriter:
        # A single writer per rank avoids the 32 concurrent, ~1 GiB NFS streams
        # produced by the full 7B checkpoint path.
        return FileSystemWriter(
            checkpoint_dir,
            thread_count=1,
            single_file_per_rank=True,
            sync_files=True,
        )

    @classmethod
    def save(
        cls,
        path: str,
        state: Dict[str, Any],
        save_async: bool = False,
        global_steps: Optional[int] = None,
        storage_writer: Optional[FileSystemWriter] = None,
    ) -> None:
        if save_async:
            raise ValueError("Atomic trainable-only checkpoints currently require synchronous saving.")
        if "model" not in state:
            raise ValueError("Model must be provided to save a trainable-only checkpoint.")

        checkpoint_name = f"{_GLOBAL_STEP_PREFIX}{global_steps}" if global_steps is not None else "checkpoint"
        checkpoint_dir = os.path.join(path, checkpoint_name)
        incomplete_dir = f"{checkpoint_dir}.incomplete"

        if dist.get_rank() == 0:
            if os.path.exists(checkpoint_dir):
                raise FileExistsError(f"Refusing to overwrite completed checkpoint: {checkpoint_dir}")
            if os.path.exists(incomplete_dir):
                raise FileExistsError(f"Remove or inspect stale incomplete checkpoint first: {incomplete_dir}")
        dist.barrier()

        cls._create_checkpoint_dir(incomplete_dir)
        cls._save_extra_state(checkpoint_dir=incomplete_dir, state=state)

        model_state = TrainableModelState(state["model"])
        save_state = {"model": model_state}
        if "optimizer" in state:
            save_state["optimizer"] = TrainableOptimizerState(
                model=state["model"], optimizer=state["optimizer"]
            )

        if storage_writer is None:
            storage_writer = cls._create_storage_writer(incomplete_dir)

        logger.info_rank0(
            f"Saving trainable-only checkpoint with {model_state.tensor_count} trainable parameter tensors "
            f"to {incomplete_dir}"
        )
        cls.execute_save(save_state=save_state, storage_writer=storage_writer, save_async=False)

        if dist.get_rank() == 0:
            metadata_path = os.path.join(incomplete_dir, ".metadata")
            missing_extra = [
                rank
                for rank in range(dist.get_world_size())
                if not os.path.isfile(os.path.join(incomplete_dir, "extra_state", f"extra_state_rank_{rank}.pt"))
            ]
            if not os.path.isfile(metadata_path) or missing_extra:
                raise RuntimeError(
                    f"Checkpoint validation failed: metadata={os.path.isfile(metadata_path)}, "
                    f"missing_extra_state_ranks={missing_extra}"
                )
            marker_path = os.path.join(incomplete_dir, _COMPLETED_FILE)
            with open(marker_path, "w", encoding="utf-8") as marker:
                json.dump(
                    {
                        "format": "trainable-only-dcp-v1",
                        "global_step": global_steps,
                        "world_size": dist.get_world_size(),
                        "trainable_parameter_tensors": model_state.tensor_count,
                    },
                    marker,
                    sort_keys=True,
                )
                marker.write("\n")
                marker.flush()
                os.fsync(marker.fileno())
            os.rename(incomplete_dir, checkpoint_dir)
        dist.barrier()
        logger.info_rank0(f"Saved trainable-only checkpoint to {checkpoint_dir}")

    @classmethod
    def load(
        cls,
        path: str,
        state: Dict[str, Any],
        process_group=None,
        storage_reader: Optional[FileSystemReader] = None,
    ) -> Dict[str, Any]:
        if state is None or "model" not in state:
            raise ValueError("Model must be provided to load a trainable-only checkpoint.")
        if not os.path.isfile(os.path.join(path, _COMPLETED_FILE)):
            raise RuntimeError(f"Checkpoint is incomplete or not trainable-only format: {path}")

        load_state = {"model": TrainableModelState(state["model"])}
        if "optimizer" in state:
            load_state["optimizer"] = TrainableOptimizerState(
                model=state["model"], optimizer=state["optimizer"]
            )
        if storage_reader is None:
            storage_reader = cls._create_storage_reader(path)

        dcp.load(state_dict=load_state, storage_reader=storage_reader, process_group=process_group)
        cls._load_extra_state(checkpoint_dir=path, state=state)
        logger.info_rank0(f"Loaded trainable-only checkpoint from {path}")
        return state
