"""Launch TorchTitan training with Mosaic streaming dataloaders."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

import torch

from torchtitan.config import ConfigManager
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer

from .train_spec import register_llama3_mosaic, register_mpt_mup_mosaic


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train TorchTitan models using MosaicML streaming dataloaders. "
            "Additional TorchTitan CLI flags can be forwarded after '--'."
        )
    )
    parser.add_argument(
        "--config",
        default="examples/mosaic_photon/configs/mosaic_job.toml",
        help=(
            "Path to a TorchTitan job TOML file. Defaults to the example config "
            "shipped with this integration."
        ),
    )
    parser.add_argument(
        "--mosaic-config",
        default="examples/mosaic_photon/configs/mosaic_dataloader.toml",
        help=(
            "Path to a TOML file containing the Mosaic streaming dataloader "
            "configuration."
        ),
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Additional TorchTitan CLI arguments (prefix with '--').",
    )
    return parser.parse_args(argv)


def _load_mosaic_config(path: str | None) -> dict[str, Any]:
    if not path:
        raise ValueError("A Mosaic dataloader config path must be provided.")

    cfg_path = Path(path)
    with cfg_path.open("rb") as handle:
        data = tomllib.load(handle)

    try:
        mosaic_cfg = data["mosaic_dataloader"]
    except KeyError as exc:  # pragma: no cover - user config error
        raise ValueError(
            "Expected 'mosaic_dataloader' table in Mosaic config file."
        ) from exc

    if not isinstance(mosaic_cfg, dict):
        raise TypeError("mosaic_dataloader entry must be a TOML table.")

    return mosaic_cfg


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    register_llama3_mosaic()
    try:
        register_mpt_mup_mosaic()
    except RuntimeError as exc:
        logger.warning("Skipping mpt_mup_mosaic registration: %s", exc)

    init_logger()
    config_manager = ConfigManager()

    forwarded_args: list[str] = []
    if args.config:
        forwarded_args.append(f"--job.config-file={args.config}")
    if args.extra:
        forwarded_args.extend(arg for arg in args.extra if arg != "--")

    job_config = config_manager.parse_args(forwarded_args)

    mosaic_dataloader_cfg = _load_mosaic_config(args.mosaic_config)
    setattr(job_config.training, "mosaic_dataloader", mosaic_dataloader_cfg)

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


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
