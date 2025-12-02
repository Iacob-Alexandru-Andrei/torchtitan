# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flux dataloader builders with worker/prefetch support for the FL stack."""

from __future__ import annotations

import os

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.experiments.fl.dataloader.flux_parallel import (
    FluxParallelAwareDataloader,
)
from torchtitan.experiments.flux.dataset.flux_dataset import (
    FluxDataset,
    FluxValidationDataset,
)
from torchtitan.experiments.flux.dataset.tokenizer import (
    build_flux_tokenizer,
    FluxTokenizer,
)
from torchtitan.tools.logging import logger


def _resolve_loader_kwargs(dp_world_size: int) -> tuple[int, int | None]:
    """Choose dataloader worker/prefetch settings with env overrides."""
    env_workers = os.getenv("FLUX_DATALOADER_WORKERS")
    env_prefetch = os.getenv("FLUX_DATALOADER_PREFETCH")
    cpu_count = os.cpu_count() or 8

    if env_workers is not None:
        num_workers = max(0, int(env_workers))
    else:
        num_workers = min(8, max(2, cpu_count // max(1, dp_world_size)))

    if num_workers > 0:
        if env_prefetch is not None:
            prefetch_factor = max(2, int(env_prefetch))
        else:
            prefetch_factor = 2
    else:
        prefetch_factor = None

    return num_workers, prefetch_factor


def build_fl_flux_dataloader(
    dp_world_size: int,
    dp_rank: int,
    job_config: JobConfig,
    # This parameter is not used, keep it for compatibility
    tokenizer: FluxTokenizer | None,
    infinite: bool = True,
) -> FluxParallelAwareDataloader:
    """Build a Flux dataloader with worker support for FL training."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size

    t5_tokenizer, clip_tokenizer = build_flux_tokenizer(job_config)

    ds = FluxDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        t5_tokenizer=t5_tokenizer,
        clip_tokenizer=clip_tokenizer,
        job_config=job_config,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    num_workers, prefetch_factor = _resolve_loader_kwargs(dp_world_size)
    logger.info(
        "[FluxDataloader] workers=%s prefetch=%s dp_rank=%s dp_world_size=%s",
        num_workers,
        prefetch_factor,
        dp_rank,
        dp_world_size,
    )

    return FluxParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        persistent_workers=True,
    )


def build_fl_flux_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    job_config: JobConfig,
    # This parameter is not used, keep it for compatibility
    tokenizer: BaseTokenizer | None,
    generate_timestamps: bool = True,
    infinite: bool = False,
) -> FluxParallelAwareDataloader:
    """Build a Flux validation dataloader with worker support for FL validation."""
    dataset_name = job_config.validation.dataset
    dataset_path = job_config.validation.dataset_path
    batch_size = job_config.validation.local_batch_size

    t5_tokenizer, clip_tokenizer = build_flux_tokenizer(job_config)

    ds = FluxValidationDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        t5_tokenizer=t5_tokenizer,
        clip_tokenizer=clip_tokenizer,
        job_config=job_config,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        generate_timesteps=generate_timestamps,
        infinite=infinite,
    )

    num_workers, prefetch_factor = _resolve_loader_kwargs(dp_world_size)
    logger.info(
        "[FluxValDataloader] workers=%s prefetch=%s dp_rank=%s dp_world_size=%s",
        num_workers,
        prefetch_factor,
        dp_rank,
        dp_world_size,
    )

    return FluxParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        persistent_workers=True,
    )
