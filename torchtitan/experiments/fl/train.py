# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan MosaicML training entry point.

This script provides an example of how to train a TorchTitan model using the
MosaicML streaming dataloader integration.

This script is a lightweight wrapper around the main `torchtitan.train.Trainer`.
It demonstrates how to:
1.  Define a custom job configuration (`MosaicJobConfig`) that includes
    settings for MosaicML's streaming dataloader and tokenizer.
2.  Dynamically modify a model's `TrainSpec` to use the Mosaic dataloader.
3.  Use the `ConfigManager` to parse a TOML configuration file and launch
    the training job.

To run this script, you can use a command like:
`torchrun --nproc_per_node=2 experiments/fl/train.py --config-path experiments/fl/configs/fl_job.toml`
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from torchtitan.experiments.fl.configs import MosaicConfigManager
from torchtitan.experiments.fl.ft_override import configure_desloc
from torchtitan.experiments.fl.ft_utils import ensure_torchft_init_sync
from torchtitan.experiments.fl.s3_checkpoint import (
    get_s3_checkpoint_wrapper_factory,
    S3CheckpointWrapper,
    setup_s3_checkpointing,
)
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer


def _format_param_count(count: int) -> str:
    """Return a human-friendly string for parameter counts."""
    if count >= 1_000_000:
        return f"{count:,} ({count / 1_000_000:.3f}M)"
    if count >= 1_000:
        return f"{count:,} ({count / 1_000:.3f}K)"
    return f"{count:,}"


def _log_model_summary(trainer: Trainer) -> None:
    """Emit a concise summary of the trainer's model parameters."""
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    model_parts = getattr(trainer, "model_parts", None)
    if not model_parts:
        logger.info("Model summary skipped: trainer has no model parts.")
        return

    seen_params: set[int] = set()
    dtype_counts: dict[str, int] = {}
    device_counts: dict[str, int] = {}
    part_module_counts: dict[str, dict[str, int]] = {}
    bias_samples: list[str] = []

    total_params = 0
    trainable_params = 0
    bias_tensors = 0
    trainable_bias_tensors = 0

    for part_idx, part in enumerate(model_parts):
        part_label = f"part{part_idx}"
        part_module_counts.setdefault(part_label, {"__total__": 0})
        for name, param in part.named_parameters():
            param_id = id(param)
            if param_id in seen_params:
                continue
            seen_params.add(param_id)

            param_count = param.numel()
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count

            dtype_key = str(param.dtype)
            dtype_counts[dtype_key] = dtype_counts.get(dtype_key, 0) + param_count

            device_key = str(param.device)
            device_counts[device_key] = device_counts.get(device_key, 0) + param_count

            part_module_counts[part_label]["__total__"] += param_count
            top_level = name.split(".", 1)[0] if "." in name else name
            if top_level:
                part_module_counts[part_label][top_level] = (
                    part_module_counts[part_label].get(top_level, 0) + param_count
                )

            if name.endswith("bias"):
                bias_tensors += 1
                if param.requires_grad:
                    trainable_bias_tensors += 1
                qualified_name = f"{part_label}.{name}"
                if len(bias_samples) < 5:
                    bias_samples.append(qualified_name)

    if not total_params:
        logger.info("Model summary: no parameters found.")
        return

    frozen_params = total_params - trainable_params
    active_fraction = trainable_params / total_params * 100.0

    logger.info(
        "Model summary: %d unique parameter tensors | total=%s | trainable=%s | frozen=%s | active=%.2f%%",
        len(seen_params),
        _format_param_count(total_params),
        _format_param_count(trainable_params),
        _format_param_count(frozen_params),
        active_fraction,
    )

    if bias_tensors:
        logger.info(
            "Bias tensors present: %d total (%d trainable). Samples: %s",
            bias_tensors,
            trainable_bias_tensors,
            ", ".join(bias_samples),
        )
    else:
        logger.info("Bias tensors present: none detected.")

    dtype_summary = ", ".join(
        f"{dtype}: {_format_param_count(count)}"
        for dtype, count in sorted(dtype_counts.items(), key=lambda item: item[0])
    )
    logger.info("Parameter dtype distribution: %s", dtype_summary or "n/a")

    device_summary = ", ".join(
        f"{device}: {_format_param_count(count)}"
        for device, count in sorted(device_counts.items(), key=lambda item: item[0])
    )
    logger.info("Parameter device placement: %s", device_summary or "n/a")

    breakdown_lines: list[str] = []
    for part_label in sorted(part_module_counts):
        module_counts = part_module_counts[part_label]
        total = module_counts.get("__total__", 0)
        breakdown_lines.append(f"{part_label}: total={_format_param_count(total)}")
        top_entries = sorted(
            (
                (name, count)
                for name, count in module_counts.items()
                if name != "__total__"
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:8]
        for module_name, count in top_entries:
            breakdown_lines.append(f"  - {module_name}: {_format_param_count(count)}")

    if breakdown_lines:
        logger.info("Model parameter breakdown:\n%s", "\n".join(breakdown_lines))

    def _render_module(
        module: nn.Module, indent: int = 0, name: str | None = None
    ) -> list[str]:
        indent_str = "  " * indent
        module_name = module.__class__.__name__
        header = (
            f"{indent_str}{module_name}("
            if name is None
            else f"{indent_str}({name}): {module_name}("
        )
        lines = [header]

        for param_name, param in module.named_parameters(recurse=False):
            lines.append(
                f"{'  ' * (indent + 1)}({param_name}): Parameters({param.shape})"
            )

        if isinstance(module, (nn.Sequential, nn.ModuleList)):
            lines.extend(_render_container(module, indent))
        else:
            for child_name, child in module.named_children():
                lines.extend(_render_module(child, indent + 1, child_name))

        lines.append(f"{indent_str})")
        return lines

    def _render_container(container: nn.Module, indent: int) -> list[str]:
        lines: list[str] = []
        groups: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        for idx, child in enumerate(container):
            child_lines = _render_module(child, indent + 2, None)
            key = tuple(child_lines)
            if current is not None and current["key"] == key:
                current["end"] = idx
                current["count"] += 1
            else:
                if current is not None:
                    groups.append(current)
                current = {
                    "start": idx,
                    "end": idx,
                    "count": 1,
                    "lines": child_lines,
                    "key": key,
                }
        if current is not None:
            groups.append(current)

        for group in groups:
            start = group["start"]
            end = group["end"]
            count = group["count"]
            child_lines = group["lines"]

            if count == 1:
                index_label = str(start)
                count_prefix = ""
            else:
                index_label = f"{start}-{end}"
                count_prefix = f"{count} x "

            first_line_body = child_lines[0].lstrip()
            lines.append(
                f"{'  ' * (indent + 1)}({index_label}): {count_prefix}{first_line_body}"
            )
            lines.extend(child_lines[1:])

        return lines

    for idx, part in enumerate(model_parts):
        structure_lines = _render_module(part)
        structure_text = "\n".join(structure_lines)
        if len(model_parts) == 1:
            logger.info("%s", structure_text)
        else:
            logger.info("Model part %d structure:\n%s", idx, structure_text)


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """The main entry point for the Mosaic training script.

    This function parses the job configuration, sets up the Mosaic-enabled
    TrainSpec, and launches the TorchTitan trainer.
    """
    init_logger()

    # Use a ConfigManager to parse the TOML configuration file into our
    # custom MosaicJobConfig dataclass.
    config_manager = MosaicConfigManager()
    job_config = config_manager.parse_args()

    # Apply RUN_UUID from environment if provided
    run_uuid = os.getenv("RUN_UUID")
    if run_uuid:
        job_config.s3_checkpoint.run_uuid = run_uuid
        if not job_config.s3_checkpoint.remote_checkpoint_folder:
            job_config.s3_checkpoint.remote_checkpoint_folder = f"torchtitan/{run_uuid}"
        # Update dump folder to include run_uuid
        job_config.job.dump_folder = f"./outputs/{run_uuid}"
        logger.info(f"Using RUN_UUID: {run_uuid}")

    # Apply RESUME_FROM_RUN_STEP from environment if provided
    # Format: "{run_uuid}/step-{N}" (e.g., "16M-baseline-20251011-122516/step-10")
    resume_from_run_step = os.getenv("RESUME_FROM_RUN_STEP")
    if resume_from_run_step:
        job_config.s3_checkpoint.resume_from_run_step = resume_from_run_step  # type: ignore[attr-defined]
        logger.info(f"Will resume training from run step: {resume_from_run_step}")

    # Launch the trainer
    trainer: Trainer | None = None
    s3_manager: S3CheckpointWrapper | None = None
    download_manager: S3CheckpointWrapper | None = None

    try:
        with configure_desloc(job_config):
            trainer = Trainer(job_config)
            ensure_torchft_init_sync(trainer)
            _log_model_summary(trainer)

            checkpointer = trainer.checkpointer
            ft_manager = getattr(checkpointer, "ft_manager", None)
            ft_mode = bool(getattr(ft_manager, "enabled", False))
            if ft_mode:
                checkpointer.enable = False

            if ft_mode:
                is_checkpoint_writer = True
            elif ft_manager is not None:
                is_checkpoint_writer = ft_manager.participating_rank() == 0
                if torch.distributed.is_initialized():
                    is_checkpoint_writer = (
                        is_checkpoint_writer and torch.distributed.get_rank() == 0
                    )
            elif torch.distributed.is_initialized():
                is_checkpoint_writer = torch.distributed.get_rank() == 0
            else:
                is_checkpoint_writer = True

            s3_checkpointing_active = (
                job_config.s3_checkpoint.enable
                and bool(job_config.s3_checkpoint.bucket)
                and job_config.s3_checkpoint.prefix
                is not None  # Empty string "" is valid!
            )

        wrapper_factory = (
            get_s3_checkpoint_wrapper_factory(job_config)
            if s3_checkpointing_active
            else None
        )

        if s3_checkpointing_active and wrapper_factory is not None:
            if is_checkpoint_writer:
                logger.info(
                    "[S3 DEBUG] Creating S3 manager as checkpoint writer (with install=True)"
                )
                s3_manager = wrapper_factory(
                    checkpointer,
                    enable_uploads=True,
                )
                s3_manager.attach_to_trainer(trainer)
                download_manager = s3_manager
                checkpointer = trainer.checkpointer
                logger.info(
                    f"[S3 DEBUG] s3_manager={s3_manager}, download_manager={download_manager}"
                )
            elif job_config.s3_checkpoint.download_on_start:
                logger.info(
                    "[S3 DEBUG] Creating download-only S3 manager (with install=False)"
                )
                download_manager = wrapper_factory(
                    checkpointer,
                    enable_uploads=False,
                )
                logger.info(f"[S3 DEBUG] download_manager={download_manager}")

        # Override WandB run name to include rank if save_for_all_ranks is enabled
        if job_config.metrics.save_for_all_ranks and job_config.metrics.enable_wandb:
            try:
                import wandb  # noqa: PLC0415

                if wandb.run is not None:
                    if torch.distributed.is_initialized():
                        local_rank = torch.distributed.get_rank()
                        world_size = torch.distributed.get_world_size()
                    else:
                        local_rank = 0
                        world_size = 1

                    replica_identifier: int | str | None = None
                    if ft_mode and ft_manager is not None:
                        replica_identifier = getattr(ft_manager, "replica_id", None)
                    if replica_identifier in (None, "", -1):
                        replica_identifier = getattr(
                            job_config.fault_tolerance, "replica_id", None
                        )
                    if replica_identifier in (None, "", -1):
                        for env_var in (
                            "TORCHFT_REPLICA_ID",
                            "FAULT_TOLERANCE_REPLICA_ID",
                            "FT_REPLICA_ID",
                            "REPLICA_ID",
                        ):
                            env_value = os.getenv(env_var)
                            if env_value:
                                try:
                                    replica_identifier = int(env_value)
                                except ValueError:
                                    replica_identifier = env_value
                                break
                    replica_index: int | None
                    try:
                        replica_index = (
                            int(replica_identifier)
                            if replica_identifier not in (None, "", -1)
                            else None
                        )
                    except (TypeError, ValueError):
                        replica_index = None

                    if replica_index is not None:
                        global_worker_id = replica_index * world_size + local_rank
                        replica_suffix = f"rep{replica_index}"
                    elif replica_identifier not in (None, "", -1):
                        global_worker_id = f"{replica_identifier}-rank{local_rank}"
                        replica_suffix = f"rep{replica_identifier}"
                    else:
                        replica_identifier = os.getpid()
                        global_worker_id = f"pid{replica_identifier}-rank{local_rank}"
                        replica_suffix = f"rep{replica_identifier}"

                    suffix = f"{replica_suffix}-rank{local_rank}"

                    original_name = wandb.run.name or "torchtitan"
                    if f"-worker{global_worker_id}" in original_name:
                        new_name = original_name
                    else:
                        new_name = f"{original_name}-worker{global_worker_id}-{suffix}"
                        wandb.run.name = new_name
                        wandb.run.save()
                        logger.info(
                            "Updated WandB run name from '%s' to '%s' (global worker id %s)",
                            original_name,
                            new_name,
                            global_worker_id,
                        )
            except ImportError:
                logger.warning("wandb not available, skipping run name update")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to update WandB run name: {e}")

        if job_config.checkpoint.create_seed_checkpoint:
            assert (
                int(os.environ["WORLD_SIZE"]) == 1
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                job_config.checkpoint.enable
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        else:
            logger.info(
                f"[S3 DEBUG] S3 setup: active={s3_checkpointing_active}, "
                f"is_checkpoint_writer={is_checkpoint_writer}, "
                f"download_on_start={job_config.s3_checkpoint.download_on_start}, "
                f"bucket={job_config.s3_checkpoint.bucket}, "
                f"prefix={job_config.s3_checkpoint.prefix}"
            )

            if s3_checkpointing_active:
                if is_checkpoint_writer:
                    logger.info(
                        "[S3 DEBUG] Creating S3 manager as checkpoint writer (with install=True)"
                    )
                    s3_manager = setup_s3_checkpointing(checkpointer, job_config)
                    if s3_manager is not None:
                        trainer.checkpointer = s3_manager  # type: ignore[assignment]
                        download_manager = s3_manager
                        checkpointer = trainer.checkpointer
                    logger.info(
                        f"[S3 DEBUG] s3_manager={s3_manager}, download_manager={download_manager}"
                    )
                elif job_config.s3_checkpoint.download_on_start:
                    logger.info(
                        "[S3 DEBUG] Creating download-only S3 manager (with install=False)"
                    )
                    download_manager = setup_s3_checkpointing(
                        checkpointer, job_config, install=False
                    )
                    logger.info(f"[S3 DEBUG] download_manager={download_manager}")

            # Override WandB run name to include rank if save_for_all_ranks is enabled
            if (
                job_config.metrics.save_for_all_ranks
                and job_config.metrics.enable_wandb
            ):
                try:
                    import wandb  # noqa: PLC0415

                    if wandb.run is not None:
                        if torch.distributed.is_initialized():
                            local_rank = torch.distributed.get_rank()
                            world_size = torch.distributed.get_world_size()
                        else:
                            local_rank = 0
                            world_size = 1

                        replica_identifier: int | str | None = None
                        if ft_mode and ft_manager is not None:
                            replica_identifier = getattr(ft_manager, "replica_id", None)
                        if replica_identifier in (None, "", -1):
                            replica_identifier = getattr(
                                job_config.fault_tolerance, "replica_id", None
                            )
                        if replica_identifier in (None, "", -1):
                            for env_var in (
                                "TORCHFT_REPLICA_ID",
                                "FAULT_TOLERANCE_REPLICA_ID",
                                "FT_REPLICA_ID",
                                "REPLICA_ID",
                            ):
                                env_value = os.getenv(env_var)
                                if env_value:
                                    try:
                                        replica_identifier = int(env_value)
                                    except ValueError:
                                        replica_identifier = env_value
                                    break
                        replica_index: int | None
                        try:
                            replica_index = (
                                int(replica_identifier)
                                if replica_identifier not in (None, "", -1)
                                else None
                            )
                        except (TypeError, ValueError):
                            replica_index = None

                        if replica_index is not None:
                            global_worker_id = replica_index * world_size + local_rank
                            replica_suffix = f"rep{replica_index}"
                        elif replica_identifier not in (None, "", -1):
                            global_worker_id = f"{replica_identifier}-rank{local_rank}"
                            replica_suffix = f"rep{replica_identifier}"
                        else:
                            replica_identifier = os.getpid()
                            global_worker_id = (
                                f"pid{replica_identifier}-rank{local_rank}"
                            )
                            replica_suffix = f"rep{replica_identifier}"

                        suffix = f"{replica_suffix}-rank{local_rank}"

                        original_name = wandb.run.name or "torchtitan"
                        if f"-worker{global_worker_id}" in original_name:
                            new_name = original_name
                        else:
                            new_name = (
                                f"{original_name}-worker{global_worker_id}-{suffix}"
                            )
                            wandb.run.name = new_name
                            wandb.run.save()
                            logger.info(
                                "Updated WandB run name from '%s' to '%s' (global worker id %s)",
                                original_name,
                                new_name,
                                global_worker_id,
                            )
                except ImportError:
                    logger.warning("wandb not available, skipping run name update")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to update WandB run name: {e}")

            if job_config.checkpoint.create_seed_checkpoint:
                assert (
                    int(os.environ["WORLD_SIZE"]) == 1
                ), "Must create seed checkpoint using a single device, to disable sharding."
                assert (
                    job_config.checkpoint.enable
                ), "Must enable checkpointing when creating a seed checkpoint."
                trainer.checkpointer.save(curr_step=0, last_step=True)
                logger.info("Created seed checkpoint")
            else:
                logger.info(
                    f"[S3 DEBUG] download_manager={download_manager}, s3_checkpointing_active={s3_checkpointing_active}"
                )
                if download_manager:
                    logger.info(
                        "[S3 DEBUG] Calling download_manager.download_if_needed()"
                    )
                    download_manager.download_if_needed()  # type: ignore[attr-defined]
                    logger.info("[S3 DEBUG] download_if_needed() completed")
                else:
                    logger.warning(
                        "[S3 DEBUG] download_manager is None! S3 download will not occur."
                    )
                if s3_checkpointing_active and torch.distributed.is_initialized():
                    torch.distributed.barrier()
                trainer.train()
    finally:
        for manager in {m for m in (s3_manager, download_manager) if m is not None}:
            manager.close()
        if trainer:
            trainer.close()
        # In some cases, the process group is not destroyed automatically,
        # so we need to do it manually.
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            logger.info("Process group destroyed")


if __name__ == "__main__":
    main()
