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

import torch

from torchtitan.experiments.fl.components import build_metrics_processor
from torchtitan.experiments.fl.configs import MosaicConfigManager
from torchtitan.experiments.fl.dataloader.dataloader import build_mosaic_dataloader
from torchtitan.experiments.fl.dataloader.tokenizer import build_mosaic_tokenizer
from torchtitan.experiments.fl.ft_override import configure_desloc
from torchtitan.experiments.fl.models.utils import ensure_mosaic_spec
from torchtitan.experiments.fl.s3_checkpoint import (
    get_s3_checkpoint_wrapper_factory,
    S3CheckpointWrapper,
)
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer


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

    # If the user has requested a specific mosaic spec (e.g. "mosaic_llama3"),
    # it will have been registered already and get_train_spec will find it.
    # Otherwise, we are in the generic case, where we take a standard model
    # and wrap it with mosaic components.
    if not job_config.model.name.startswith("mosaic_"):
        mosaic_spec_name = ensure_mosaic_spec(
            job_config.model.name,
            dataloader_fn=build_mosaic_dataloader,
            tokenizer_fn=build_mosaic_tokenizer,
            metrics_processor_fn=build_metrics_processor,
        )
        job_config.model.name = mosaic_spec_name

    # Launch the trainer
    trainer: Trainer | None = None
    s3_manager: S3CheckpointManager | None = None
    download_manager: S3CheckpointManager | None = None

    s3_manager: S3CheckpointWrapper | None = None
    download_manager: S3CheckpointWrapper | None = None

    try:
        with configure_desloc(job_config):
            trainer = Trainer(job_config)

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
