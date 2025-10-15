# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validation helpers for Mosaic streaming experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from torchtitan.components.validate import Validator
from torchtitan.experiments.fl.dataloader.dataloader import (
    build_mosaic_validation_dataloader,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from torch.distributed.pipelining.schedules import _PipelineSchedule

    from torchtitan.components.loss import LossFunction
    from torchtitan.components.metrics import MetricsProcessor
    from torchtitan.components.tokenizer import BaseTokenizer
    from torchtitan.distributed import ParallelDims
    from torchtitan.experiments.fl.configs.config import MosaicJobConfig


@dataclass(frozen=True)
class MosaicValidatorRequest:
    """Configuration required to construct a :class:`MosaicValidator`."""

    job_config: MosaicJobConfig
    dp_world_size: int
    dp_rank: int
    tokenizer: BaseTokenizer
    parallel_dims: ParallelDims
    loss_fn: LossFunction
    validation_context: Generator[None, None, None]
    maybe_enable_amp: Generator[None, None, None]
    metrics_processor: MetricsProcessor
    pp_schedule: _PipelineSchedule | None = None
    pp_has_first_stage: bool | None = None
    pp_has_last_stage: bool | None = None


class MosaicValidator(Validator):
    """Validator variant that swaps in the Mosaic streaming dataloader."""

    def __init__(self, request: MosaicValidatorRequest) -> None:
        super().__init__(**vars(request))
        self.validation_dataloader = build_mosaic_validation_dataloader(
            job_config=request.job_config,
            dp_world_size=request.dp_world_size,
            dp_rank=request.dp_rank,
            tokenizer=request.tokenizer,
        )


def build_mosaic_validator(**kwargs: Any) -> MosaicValidator:
    """Build a validator that uses Mosaic streaming for the validation split."""
    request = MosaicValidatorRequest(**kwargs)
    return MosaicValidator(request)


__all__ = ["MosaicValidator", "MosaicValidatorRequest", "build_mosaic_validator"]
