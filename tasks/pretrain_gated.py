"""Minimal VeOmni pretrain task for the gated_mla_hybrid model family.

Adapted from tasks/pretrain_with_ruler.py with the RULER synthesis and the MASS
hooks removed: this is the standard plaintext pretraining path (tokenize -> chunk
to max_seq_len -> CE loss). Use it for the gated_mla / swa_gated_mla /
hils_gated_mla / kda_gated_mla baselines ported from LLM_Standardization.

KDA configs additionally require flash-linear-attention; FLA_TILELANG is forced
to 0 below because this environment's tilelang pin serves the HiLS kernels and
fla's tilelang backend does not compile against it.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from functools import partial
from typing import Any, Dict, List

os.environ.setdefault("FLA_TILELANG", "0")

import torch
import torch.distributed as dist
import wandb
from tqdm import trange
from utils.training_utils import format_freeze_summary, freeze_parameters

from torch.distributed.checkpoint import FileSystemWriter
from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.data import (
    build_dataloader,
    build_dataset,
)
from veomni.data.data_transform import process_pretrain_example
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, parse_args, save_args
from utils.arguments import ExtendedTrainingArguments
from veomni.utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
    is_nccl_backend,
    synchronize,
)
from veomni.utils.dist_utils import all_reduce
from veomni.utils.loss_utils import count_loss_token, mean_global_loss

import models


logger = helper.create_logger(__name__)
INDEX_TENSOR_KEYS = {"input_ids", "labels", "position_ids", "cache_position", "token_type_ids"}


def _create_small_storage_writer(cls, checkpoint_dir):
    # NFS-safe DCP writes: default thread_count=16 per rank bursts ~32 concurrent
    # write streams (EIO risk); keep serial unless explicitly overridden.
    thread_count = max(1, int(os.environ.get("DCP_WRITE_THREADS", "1")))
    return FileSystemWriter(
        checkpoint_dir,
        thread_count=thread_count,
        single_file_per_rank=True,
        sync_files=True,
    )


DistributedCheckpointer._create_storage_writer = classmethod(_create_small_storage_writer)


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "ExtendedTrainingArguments" = field(default_factory=ExtendedTrainingArguments)


def normalize_model_inputs(batch: Dict[str, Any]) -> Dict[str, Any]:
    normalized_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and key in INDEX_TENSOR_KEYS and value.dtype != torch.long:
            normalized_batch[key] = value.to(dtype=torch.long)
        else:
            normalized_batch[key] = value
    return normalized_batch


def _save_with_retry(save_fn, checkpoint_root, global_step, max_retries=3, base_delay_sec=60.0):
    """Collective DCP save with exponential-backoff retry for transient storage faults.

    An NFS EIO during DCP writes raises CheckpointException on every rank
    (DCP gathers write results before returning), so all ranks execute the
    same retry schedule and stay in sync.
    """
    import shutil

    for attempt in range(max_retries + 1):
        try:
            save_fn()
            return
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)) or attempt == max_retries:
                raise
            delay = base_delay_sec * (2**attempt)
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            if rank == 0:
                logger.warning_rank0(
                    f"[ckpt-retry] DCP save failed at global_step_{global_step} "
                    f"(attempt {attempt + 1}/{max_retries + 1}): {type(exc).__name__}: {exc}. "
                    f"Retrying in {delay:.0f}s."
                )
                incomplete = os.path.join(checkpoint_root, f"global_step_{global_step}.incomplete")
                if os.path.isdir(incomplete):
                    shutil.rmtree(incomplete, ignore_errors=True)
            dist.barrier()
            time.sleep(delay)


def main():
    nccl_timeout = os.getenv("NCCL_TIMEOUT", None)
    pg_nccl_timeout = None
    if nccl_timeout is not None and is_nccl_backend():
        pg_nccl_timeout = timedelta(seconds=int(nccl_timeout))
    dist.init_process_group(backend=get_dist_comm_backend(), timeout=pg_nccl_timeout)

    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)
    transform = partial(
        process_pretrain_example,
        tokenizer=tokenizer,
        max_seq_len=args.data.max_seq_len,
        text_keys=args.data.text_keys,
    )

    train_dataset = build_dataset(
        dataset_name=args.data.dataset_name,
        transform=transform,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        **asdict(args.data),
    )
    dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
    if args.data.datasets_type == "mapping":
        dataset_length = dataset_length / args.train.data_parallel_size
    args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)

    train_dataloader = build_dataloader(
        dataloader_type=args.data.dataloader_type,
        dataset=train_dataset,
        micro_batch_size=args.train.micro_batch_size,
        global_batch_size=args.train.global_batch_size,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        max_seq_len=args.data.max_seq_len,
        train_steps=args.train.train_steps,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        bsz_warmup_ratio=args.train.bsz_warmup_ratio,
        bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
        dyn_bsz_margin=args.train.dyn_bsz_margin,
        dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
        num_workers=args.data.num_workers,
        drop_last=args.data.drop_last,
        pin_memory=args.data.pin_memory,
        prefetch_factor=args.data.prefetch_factor,
    )

    logger.info_rank0("Prepare model")
    # Precision layout: pure-bf16 keeps every tensor BF16 (fla kernels are
    # bf16-tuned; the fp32 bwd variant fails on this env's triton 3.2).
    # Otherwise build FP32 masters and let veomni cast for compute: fsdp2 uses
    # fully_shard MixedPrecisionPolicy (needs world_size > 1 in this veomni),
    # ddp uses DDP native MixedPrecision -- both feed BF16 to custom kernels.
    pure_bf16 = (
        args.train.data_parallel_mode == "ddp"
        and args.train.enable_mixed_precision
        and os.environ.get("GATED_PURE_BF16", "0") == "1"
    )
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="bfloat16" if pure_bf16 else "float32",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
    )
    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    get_optimizer_pre_hook = getattr(model, "get_optimizer_pre_hook", None)
    parallel_mixed_precision = args.train.enable_mixed_precision and not pure_bf16
    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=parallel_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
        broadcast_model_weights_from_rank0=getattr(args.train, "broadcast_model_weights_from_rank0", False),
    )
    raw_model = getattr(model, "module", model)

    freeze_pattern = getattr(args.train, "freeze_pattern", None)
    if freeze_pattern is not None:
        freeze_summary = freeze_parameters(model, freeze_pattern)
        logger.info_rank0(
            f"Applied freeze_pattern before optimizer build (pattern={freeze_pattern!r}):\n"
            f"{format_freeze_summary(freeze_summary)}"
        )
        if freeze_summary.frozen == 0 and freeze_summary.trainable == freeze_summary.total:
            logger.warning_rank0(
                "freeze_pattern matched zero parameters; check shell quoting "
                "(pattern must not include literal quote characters)."
            )

    if pure_bf16:
        logger.info_rank0(
            "GATED_PURE_BF16=1: pure-BF16 training (no FP32 master). Same rationale as "
            "MASS_PURE_BF16: fla chunk kernels are bf16-tuned; the fp32 backward variant "
            "fails to compile on this env's triton 3.2."
        )
    elif args.train.enable_mixed_precision and args.train.data_parallel_mode == "ddp":
        logger.info_rank0(
            "DDP mixed precision: FP32 master params, BF16 compute via DDP native "
            "MixedPrecision (param_dtype=bf16, reduce_dtype=fp32) -- custom kernels see BF16."
        )

    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
        no_decay_modules=args.train.no_decay_modules,
        no_decay_params=args.train.no_decay_params,
    )
    if get_optimizer_pre_hook is not None:
        optimizer_pre_hook = get_optimizer_pre_hook(model, model_config, args.train.data_parallel_mode)
        optimizer.register_step_pre_hook(optimizer_pre_hook)

    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=args.train.train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                settings=wandb.Settings(console="off"),
                config={**vars(args.model), **vars(args.data), **vars(args.train)},
            )

        # save model_assets before training
        model_assets = [model_config, tokenizer]
        save_model_assets(args.train.model_assets_dir, model_assets)

    if args.train.profile_this_rank:
        profiler = helper.create_profiler(
            start_step=args.train.profile_start_step,
            end_step=args.train.profile_end_step,
            trace_dir=args.train.profile_trace_dir,
            record_shapes=args.train.profile_record_shapes,
            profile_memory=args.train.profile_profile_memory,
            with_stack=args.train.profile_with_stack,
            global_rank=args.train.global_rank,
        )
        profiler.start()

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
        enable_multisource=args.data.enable_multisource,
        dataloader=train_dataloader,
        data_path=args.data.train_path,
    )

    if args.train.load_checkpoint_path:
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}
        Checkpointer.load(args.train.load_checkpoint_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // args.train.train_steps
        start_step = global_step % args.train.train_steps
        if getattr(args.train, "load_lr_scheduler_state", True):
            lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        if getattr(args.train, "load_dataloader_state", True):
            train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        if getattr(args.train, "load_environ_meter_state", True):
            environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        if getattr(args.train, "load_rng_state", True):
            torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if start_step == 0:
            iter(train_dataloader)

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

        if freeze_pattern is not None:
            freeze_summary = freeze_parameters(model, freeze_pattern)
            logger.info_rank0(
                "Re-applied freeze_pattern after checkpoint load:\n" f"{format_freeze_summary(freeze_summary)}"
            )

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, args.train.train_steps):
            global_step += 1

            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            synchronize()
            start_time = time.time()

            micro_batches_token_num = count_loss_token(micro_batches)
            for micro_batch in micro_batches:
                environ_meter.add(micro_batch)
                micro_batch_token_num = count_loss_token(micro_batch)
                if args.data.enable_multisource:
                    micro_batch.pop("ds_idx", None)
                    micro_batch.pop("cur_token_num", None)
                    micro_batch.pop("source_name", None)

                micro_batch = {
                    k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                }
                micro_batch = normalize_model_inputs(micro_batch)
                with model_fwd_context:
                    model_outputs = model(**micro_batch, use_cache=False)

                loss_terms = model_outputs.loss
                loss, loss_dict = mean_global_loss(loss_terms, micro_batch_token_num, micro_batches_token_num)

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                del model_outputs
                del micro_batch

            if args.train.data_parallel_mode == "ddp":
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    (param for param in model.parameters() if param.requires_grad),
                    args.train.max_grad_norm,
                )
            else:
                grad_norm = veomni_clip_grad_norm(model, args.train.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            total_loss, grad_norm = all_reduce(
                (total_loss, grad_norm),
                group=get_parallel_state().fsdp_group,
            )
            synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            data_loader_tqdm.set_postfix_str(
                f"loss: {total_loss:.4f}, grad_norm: {grad_norm:.4f}, lr: {lr:.2e}",
                refresh=False,
            )
            data_loader_tqdm.update()

            if args.train.global_rank == 0:
                if args.train.use_wandb:
                    train_metrics.update(
                        {
                            "training/loss": total_loss,
                            "training/grad_norm": grad_norm,
                            "training/lr": lr,
                        }
                    )
                    wandb.log(train_metrics, step=global_step)

            if args.train.profile_this_rank and global_step <= args.train.profile_end_step:
                profiler.step()
                if global_step == args.train.profile_end_step:
                    profiler.stop()

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                _save_with_retry(
                    lambda: Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step),
                    args.train.save_checkpoint_path,
                    global_step,
                )
                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            _save_with_retry(
                lambda: Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step),
                args.train.save_checkpoint_path,
                global_step,
            )
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

    synchronize()
    del optimizer, lr_scheduler
    helper.empty_cache()
    if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=save_checkpoint_path,
            ckpt_manager=args.train.ckpt_manager,
        )
        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
