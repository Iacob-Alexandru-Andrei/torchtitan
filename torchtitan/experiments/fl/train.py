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
from dataclasses import replace
from typing import cast

import torch

from torchtitan.config import ConfigManager
from torchtitan.experiments.fl.components import build_metrics_processor
from torchtitan.experiments.fl.configs.config import MosaicJobConfig
from torchtitan.experiments.fl.dataloader.dataloader import build_mosaic_dataloader
from torchtitan.experiments.fl.dataloader.tokenizer import build_mosaic_tokenizer
from torchtitan.experiments.fl.s3_checkpoint import (
    S3CheckpointManager,
    setup_s3_checkpointing,
)
from torchtitan.protocols.train_spec import (
    get_train_spec,
    register_train_spec,
    TokenizerBuilder,
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
    config_manager = ConfigManager(MosaicJobConfig)
    job_config = config_manager.parse_args()

    # If the user has requested a specific mosaic spec (e.g. "mosaic_llama3"),
    # it will have been registered already and get_train_spec will find it.
    # Otherwise, we are in the generic case, where we take a standard model
    # and wrap it with mosaic components.
    if not job_config.model.name.startswith("mosaic_"):
        # Dynamically get the base TrainSpec for the specified model
        # and modify it to use the Mosaic dataloader and tokenizer.
        base_spec = get_train_spec(job_config.model.name)
        mosaic_spec_name = f"mosaic_{base_spec.name}"

        # Check if the mosaic spec is already registered (e.g., from a previous run)
        try:
            mosaic_spec = get_train_spec(mosaic_spec_name)
            logger.info(f"TrainSpec {mosaic_spec_name} already registered, reusing it")
        except ValueError:
            # Not registered yet, create and register it
            mosaic_spec = replace(
                base_spec,
                build_dataloader_fn=build_mosaic_dataloader,
                build_tokenizer_fn=cast("TokenizerBuilder", build_mosaic_tokenizer),
                build_metrics_processor_fn=build_metrics_processor,
            )
            mosaic_spec.name = mosaic_spec_name
            register_train_spec(mosaic_spec)
            logger.info(f"Registered new TrainSpec: {mosaic_spec_name}")

        # Update the job config to use the mosaic spec
        job_config.model.name = mosaic_spec.name

    # Launch the trainer
    trainer: Trainer | None = None
    s3_manager: S3CheckpointManager | None = None
    download_manager: S3CheckpointManager | None = None
    try:
        trainer = Trainer(job_config)
        s3_manager = setup_s3_checkpointing(trainer.checkpointer, job_config)
        if s3_manager is not None:
            trainer.checkpointer = s3_manager  # type: ignore[assignment]

        checkpointer = trainer.checkpointer
        ft_manager = getattr(checkpointer, "ft_manager", None)
        if ft_manager is not None:
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
            and bool(job_config.s3_checkpoint.prefix)
        )

        if s3_checkpointing_active:
            if is_checkpoint_writer:
                s3_manager = setup_s3_checkpointing(checkpointer, job_config)
                download_manager = s3_manager
            elif job_config.s3_checkpoint.download_on_start:
                download_manager = setup_s3_checkpointing(
                    checkpointer, job_config, install=False
                )

        # Override WandB run name to include rank if save_for_all_ranks is enabled
        if job_config.metrics.save_for_all_ranks and job_config.metrics.enable_wandb:
            try:
                import wandb  # noqa: PLC0415

                if wandb.run is not None and torch.distributed.is_initialized():
                    rank = torch.distributed.get_rank()
                    original_name = wandb.run.name
                    new_name = f"{original_name}-rank{rank}"
                    wandb.run.name = new_name
                    wandb.run.save()
                    logger.info(
                        f"Updated WandB run name from '{original_name}' to '{new_name}' "
                        f"for rank {rank}"
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
            if download_manager:
                download_manager.download_if_needed(job_config.checkpoint.load_step)
            if s3_checkpointing_active and torch.distributed.is_initialized():
                torch.distributed.barrier()
            trainer.train()
    finally:
        for manager in {
            m for m in (s3_manager, download_manager) if m is not None
        }:
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
