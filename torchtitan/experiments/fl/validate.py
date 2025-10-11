# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validation helpers for Mosaic streaming experiments."""

from __future__ import annotations

from collections.abc import Generator

from torch.distributed.pipelining.schedules import _PipelineSchedule

from torchtitan.components.loss import LossFunction
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.components.validate import Validator
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.fl.configs.config import MosaicJobConfig
from torchtitan.experiments.fl.dataloader.dataloader import (
    build_mosaic_validation_dataloader,
)


class MosaicValidator(Validator):
    """Validator variant that swaps in the Mosaic streaming dataloader."""

    def __init__(
        self,
        job_config: MosaicJobConfig,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        parallel_dims: ParallelDims,
        loss_fn: LossFunction,
        validation_context: Generator[None, None, None],
        maybe_enable_amp: Generator[None, None, None],
        metrics_processor: MetricsProcessor,
        pp_schedule: _PipelineSchedule | None = None,
        pp_has_first_stage: bool | None = None,
        pp_has_last_stage: bool | None = None,
    ) -> None:
        super().__init__(
            job_config=job_config,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            tokenizer=tokenizer,
            parallel_dims=parallel_dims,
            loss_fn=loss_fn,
            validation_context=validation_context,
            maybe_enable_amp=maybe_enable_amp,
            metrics_processor=metrics_processor,
            pp_schedule=pp_schedule,
            pp_has_first_stage=pp_has_first_stage,
            pp_has_last_stage=pp_has_last_stage,
        )
        self.validation_dataloader = build_mosaic_validation_dataloader(
            job_config=job_config,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            tokenizer=tokenizer,
        )


def build_mosaic_validator(
    job_config: MosaicJobConfig,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    parallel_dims: ParallelDims,
    loss_fn: LossFunction,
    validation_context: Generator[None, None, None],
    maybe_enable_amp: Generator[None, None, None],
    metrics_processor: MetricsProcessor,
    pp_schedule: _PipelineSchedule | None = None,
    pp_has_first_stage: bool | None = None,
    pp_has_last_stage: bool | None = None,
) -> MosaicValidator:
    """Build a validator that uses Mosaic streaming for the validation split."""

    return MosaicValidator(
        job_config=job_config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        parallel_dims=parallel_dims,
        loss_fn=loss_fn,
        validation_context=validation_context,
        maybe_enable_amp=maybe_enable_amp,
        metrics_processor=metrics_processor,
        pp_schedule=pp_schedule,
        pp_has_first_stage=pp_has_first_stage,
        pp_has_last_stage=pp_has_last_stage,
    )


__all__ = ["build_mosaic_validator", "MosaicValidator"]
