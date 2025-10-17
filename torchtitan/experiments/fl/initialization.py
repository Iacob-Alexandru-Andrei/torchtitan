# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared initialization utilities for FL experiments.

This module provides helpers to pre-generate deterministic seed checkpoints that
can be reused by both DDP and TorchFT training launches, ensuring identical
parameter initialization across execution modes.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from datetime import datetime, UTC
from pathlib import Path
from typing import TYPE_CHECKING

import torch.distributed as dist

from torchtitan.experiments.fl.configs import load_mosaic_job_config, MosaicJobConfig
from torchtitan.experiments.fl.ft_override import configure_desloc
from torchtitan.experiments.fl.ft_utils import ensure_torchft_init_sync
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer

if TYPE_CHECKING:
    from collections.abc import Sequence

_ENV_LOCAL_RANK = "LOCAL_RANK"
_ENV_RANK = "RANK"
_ENV_WORLD_SIZE = "WORLD_SIZE"
_ENV_MASTER_ADDR = "MASTER_ADDR"
_ENV_MASTER_PORT = "MASTER_PORT"


def apply_runtime_overrides(job_config: MosaicJobConfig) -> None:
    """Apply environment-based overrides to the job configuration.

    Mirrors the logic used in the main training entry point so auxiliary tooling
    (such as the seed initializer) produces identical dump folders and S3 paths.
    """
    run_uuid = os.getenv("RUN_UUID")
    if run_uuid:
        job_config.s3_checkpoint.run_uuid = run_uuid
        if not job_config.s3_checkpoint.remote_checkpoint_folder:
            job_config.s3_checkpoint.remote_checkpoint_folder = f"torchtitan/{run_uuid}"
        job_config.job.dump_folder = f"./outputs/{run_uuid}"
        if _is_global_leader():
            logger.info("Using RUN_UUID: %s", run_uuid)

    resume_from_run_step = os.getenv("RESUME_FROM_RUN_STEP")
    if resume_from_run_step:
        job_config.s3_checkpoint.resume_from_run_step = resume_from_run_step  # type: ignore[assignment]
        if _is_global_leader():
            logger.info("Will resume training from run step: %s", resume_from_run_step)


def _ensure_rank_env_initialized() -> None:
    """Populate torch.distributed environment variables for single-process runs."""
    os.environ.setdefault(_ENV_LOCAL_RANK, "0")
    os.environ.setdefault(_ENV_RANK, "0")
    os.environ.setdefault(_ENV_WORLD_SIZE, "1")
    os.environ.setdefault(_ENV_MASTER_ADDR, "127.0.0.1")
    os.environ.setdefault(_ENV_MASTER_PORT, "29500")


def _current_rank() -> int:
    raw_rank = os.environ.get(_ENV_RANK)
    if raw_rank is None:
        return 0
    try:
        return int(raw_rank)
    except ValueError:
        return 0


def _dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_global_leader() -> bool:
    if _dist_initialized():
        return dist.get_rank() == 0
    return _current_rank() == 0


def resolve_seed_dir(job_config: MosaicJobConfig, dump_folder: str | Path | None = None) -> Path:
    """Return the directory that should contain the seed checkpoint."""
    base = Path(dump_folder or job_config.job.dump_folder).resolve()
    return base / job_config.initialization.seed_subdir


def resolve_seed_step_path(job_config: MosaicJobConfig, dump_folder: str | Path | None = None) -> Path:
    """Return the expected step folder for the seed checkpoint."""
    return resolve_seed_dir(job_config, dump_folder) / job_config.initialization.checkpoint_step


def _write_seed_metadata(seed_dir: Path, job_config: MosaicJobConfig, *, targets: Sequence[Path]) -> None:
    """Write a metadata file describing the generated seed checkpoint."""
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": job_config.model.name,
        "flavor": job_config.model.flavor,
        "seed": job_config.training.seed,
        "dtype": job_config.training.dtype,
        "targets": [str(path) for path in targets],
    }
    metadata_path = seed_dir / job_config.initialization.metadata_filename
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _sanitize_seed_job_config(
    job_config: MosaicJobConfig,
    dump_folder: Path,
) -> MosaicJobConfig:
    """Create a copy of the job config tailored for seed checkpoint generation."""
    seed_config = copy.deepcopy(job_config)
    seed_config.job.dump_folder = str(dump_folder)
    seed_config.checkpoint.enable = True
    seed_config.checkpoint.folder = job_config.initialization.seed_subdir
    seed_config.checkpoint.create_seed_checkpoint = True
    seed_config.checkpoint.initial_load_path = None
    seed_config.checkpoint.initial_load_model_only = True
    seed_config.checkpoint.initial_load_in_hf = False
    seed_config.checkpoint.load_only = False
    seed_config.initialization.enable = False

    # Disable fault tolerance to avoid requiring TorchFT lighthouse services during seed generation.
    seed_config.fault_tolerance.enable = False
    seed_config.fault_tolerance.replica_id = 0
    seed_config.fault_tolerance.group_size = 1
    seed_config.fault_tolerance.min_replica_size = 1

    # Disable remote checkpointing and external logging during seed creation.
    seed_config.s3_checkpoint.enable = False
    seed_config.s3_checkpoint.download_on_start = False
    seed_config.metrics.enable_wandb = False
    seed_config.metrics.enable_tensorboard = False
    seed_config.metrics.save_for_all_ranks = False
    seed_config.initialization.build_dataloader = False

    desloc_cfg = getattr(seed_config.optimizer, "desloc", None)
    if desloc_cfg is not None:
        desloc_cfg.enabled = False

    return seed_config


def _cleanup_distributed_if_needed() -> None:
    """Tear down the default process group when the initializer exits."""
    if _dist_initialized():
        dist.destroy_process_group()


def create_seed_checkpoint(
    job_config: MosaicJobConfig,
    dump_folder: Path,
    *,
    force: bool = False,
) -> Path:
    """Create (or reuse) the seed checkpoint inside ``dump_folder``.

    Args:
        job_config: Typed Mosaic job configuration.
        dump_folder: Destination dump folder for the seed checkpoint.
        force: When True, delete any existing seed checkpoint before creating a new one.

    Returns:
        Path to the concrete checkpoint folder (typically ``step-0``).
    """
    seed_step_path = resolve_seed_step_path(job_config, dump_folder)
    seed_dir = seed_step_path.parent

    if seed_step_path.exists():
        if force:
            if _is_global_leader():
                shutil.rmtree(seed_dir, ignore_errors=True)
        else:
            if _is_global_leader():
                logger.info("Seed checkpoint already exists at %s", seed_step_path)
            return seed_step_path

    # Ensure the parent directories exist before trainer initialization.
    dump_folder.mkdir(parents=True, exist_ok=True)

    seed_config = _sanitize_seed_job_config(job_config, dump_folder)

    trainer: Trainer | None = None
    _ensure_rank_env_initialized()

    try:
        with configure_desloc(seed_config):
            trainer = Trainer(seed_config)
            ensure_torchft_init_sync(trainer)
            should_save = not _dist_initialized() or _is_global_leader()
            if should_save:
                trainer.checkpointer.save(curr_step=0, last_step=True)
                if _is_global_leader():
                    logger.info("Seed checkpoint written to %s", seed_step_path)
            if _dist_initialized():
                dist.barrier()
                if not should_save and not seed_step_path.exists():
                    msg = f"Seed checkpoint was not materialized at {seed_step_path}"
                    raise RuntimeError(msg)
    finally:
        if trainer is not None:
            trainer.close()
        _cleanup_distributed_if_needed()

    return seed_step_path


def replicate_seed_checkpoint(
    job_config: MosaicJobConfig,
    source_seed_dir: Path,
    target_dump_folder: Path,
    *,
    force: bool = False,
) -> Path:
    """Copy an existing seed checkpoint into another dump folder."""
    target_seed_dir = resolve_seed_dir(job_config, target_dump_folder)
    target_step_path = target_seed_dir / job_config.initialization.checkpoint_step

    if target_step_path.exists():
        if force:
            shutil.rmtree(target_seed_dir, ignore_errors=True)
        else:
            logger.info("Seed checkpoint already present at %s", target_step_path)
            return target_step_path

    target_seed_dir.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(source_seed_dir, target_seed_dir, dirs_exist_ok=False)
    logger.info("Replicated seed checkpoint from %s to %s", source_seed_dir, target_seed_dir)
    return target_step_path


def initialize_seed_checkpoints(
    job_config: MosaicJobConfig,
    targets: Sequence[Path],
    *,
    force: bool = False,
) -> dict[str, Path]:
    """Generate a seed checkpoint and propagate it to all target dump folders."""
    if not job_config.initialization.enable:
        if _is_global_leader():
            logger.info("Initialization disabled; skipping seed checkpoint generation.")
        return {}

    if not targets:
        msg = "At least one target dump folder must be provided."
        raise ValueError(msg)

    canonical_target = targets[0]
    canonical_step_path = create_seed_checkpoint(job_config, canonical_target, force=force)
    if _is_global_leader():
        for target in targets[1:]:
            replicate_seed_checkpoint(job_config, canonical_step_path.parent, target, force=force)
        # Store metadata next to the canonical checkpoint describing all resolved targets.
        _write_seed_metadata(canonical_step_path.parent, job_config, targets=targets)

    return {str(target): resolve_seed_step_path(job_config, target) for target in targets}


def configure_initial_checkpoint(job_config: MosaicJobConfig, *, require_existing: bool = True) -> Path | None:
    """Configure the job to load a pre-generated seed checkpoint."""
    if not job_config.initialization.enable:
        return None

    seed_step_path = resolve_seed_step_path(job_config)
    if not seed_step_path.exists():
        if require_existing:
            msg = (
                f"Expected seed checkpoint at {seed_step_path}. "
                "Run `python -m torchtitan.experiments.fl.initialization` "
                "before launching training to generate it."
            )
            raise FileNotFoundError(msg)
        return None

    if not job_config.checkpoint.enable:
        logger.info(
            "Enabling checkpointing so the seed checkpoint at %s can be loaded.",
            seed_step_path,
        )
        job_config.checkpoint.enable = True

    job_config.checkpoint.initial_load_path = str(seed_step_path)
    job_config.checkpoint.initial_load_model_only = True
    job_config.checkpoint.create_seed_checkpoint = False

    return seed_step_path


def _parse_cli(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Generate shared seed checkpoints for FL experiments.")
    parser.add_argument(
        "--init-target",
        dest="targets",
        action="append",
        default=[],
        help=(
            "Override the target dump folder used for the seed checkpoint. "
            "May be specified multiple times to replicate the checkpoint into "
            "additional directories (e.g. one for DDP, one for TorchFT). "
            "Defaults to the dump folder defined in the config."
        ),
    )
    parser.add_argument(
        "--init-force",
        dest="force",
        action="store_true",
        help="Recreate the seed checkpoint even if it already exists.",
    )
    return parser.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the seed initialization helper."""
    known_args, config_args = _parse_cli(argv)
    job_config = load_mosaic_job_config(config_args)

    init_logger()
    apply_runtime_overrides(job_config)

    targets = (
        [Path(target).resolve() for target in known_args.targets]
        if known_args.targets
        else [Path(job_config.job.dump_folder).resolve()]
    )

    results = initialize_seed_checkpoints(job_config, targets, force=known_args.force)

    if results and _is_global_leader():
        for dump_folder, step_path in results.items():
            logger.info("Seed checkpoint ready at %s (dump folder %s)", step_path, dump_folder)
    elif _is_global_leader():
        logger.info("No seed checkpoint created.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
