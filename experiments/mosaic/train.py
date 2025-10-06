# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
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
`torchrun --nproc_per_node=2 experiments/mosaic/train.py --config-path experiments/mosaic/configs/mosaic_job.toml`
"""

from __future__ import annotations

import torch

from experiments.mosaic.configs.config import MosaicJobConfig
from experiments.mosaic.models.model_utils import get_mosaic_train_spec
from torchtitan.config import ConfigManager
from torchtitan.protocols.train_spec import get_train_spec, register_train_spec
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer


def main() -> None:
    """
    The main entry point for the Mosaic training script.

    This function parses the job configuration, sets up the Mosaic-enabled
    TrainSpec, and launches the TorchTitan trainer.
    """
    init_logger()

    # Use a ConfigManager to parse the TOML configuration file into our
    # custom MosaicJobConfig dataclass.
    config_manager = ConfigManager(MosaicJobConfig)
    job_config = config_manager.parse_args()

    # Dynamically get the base TrainSpec for the specified model
    # and modify it to use the Mosaic dataloader and tokenizer.
    base_spec = get_train_spec(job_config.model.name)
    mosaic_spec = get_mosaic_train_spec(base_spec)
    # We need to update the name of the spec to avoid conflicts
    mosaic_spec.name = f"{mosaic_spec.name}_mosaic"
    register_train_spec(mosaic_spec)
    # Update the job config to use the new spec
    job_config.model.name = mosaic_spec.name

    # Launch the trainer
    trainer: Trainer | None = None
    try:
        trainer = Trainer(job_config)
        if job_config.checkpoint.create_seed_checkpoint:
            if int(torch.distributed.get_world_size()) != 1:
                raise RuntimeError(
                    "Seed checkpoint creation must run with a single rank."
                )
            if not job_config.checkpoint.enable:
                raise RuntimeError(
                    "Checkpointing must be enabled when creating a seed checkpoint."
                )
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    finally:
        if trainer is not None:
            trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()