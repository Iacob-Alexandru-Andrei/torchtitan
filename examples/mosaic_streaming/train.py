# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This is a simple example of how to use the MosaicML streaming dataloader
integration with TorchTitan. It demonstrates how to register a custom
TrainSpec and use a YAML file to configure the training job.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Sequence

import torch
import yaml

from torchtitan.compat.mosaic.train_spec import register_llama3_mosaic
from torchtitan.config import ConfigManager, JobConfig
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer


def _flatten_yaml_cfg(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> List[str]:
    """Flattens a nested dictionary into a list of command-line arguments."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_yaml_cfg(v, new_key, sep=sep))
        elif isinstance(v, bool):
            # Tyro expects bools as string literals
            items.extend([f"--job.{new_key}", str(v).lower()])
        elif isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            # Tyro can handle lists of strings with comma separation
            items.extend([f"--job.{new_key}", ",".join(v)])
        elif isinstance(v, list):
            # For other list types, we just skip. This can be extended if needed.
            continue
        else:
            items.extend([f"--job.{new_key}", str(v)])
    return items


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train TorchTitan models using MosaicML streaming dataloaders with YAML config."
            "Additional TorchTitan CLI flags can be forwarded after '--'."
        )
    )
    parser.add_argument(
        "--config",
        default="examples/mosaic_streaming/configs/config.yaml",
        help=(
            "Path to a YAML job config file. Defaults to the example config "
            "shipped with this integration."
        ),
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Additional TorchTitan CLI arguments (prefix with '--').",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    register_llama3_mosaic()
    init_logger()

    # Load YAML config
    with open(args.config, "r") as f:
        yaml_cfg = yaml.safe_load(f)

    if "job" not in yaml_cfg:
        raise ValueError("YAML config must have a top-level 'job' key.")

    # Convert the YAML config to a list of command-line arguments
    args_list = _flatten_yaml_cfg(yaml_cfg["job"])

    # Add any extra command-line arguments
    if args.extra:
        args_list.extend(arg for arg in args.extra if arg != "--")

    # Use the ConfigManager to parse the arguments
    config_manager = ConfigManager()
    job_config: JobConfig = config_manager.parse_args(args_list)

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
