# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build FL experiment components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchtitan.experiments.fl.metrics import FLMetricsProcessor

if TYPE_CHECKING:
    from torchtitan.config import JobConfig
    from torchtitan.distributed import ParallelDims
    from torchtitan.protocols import BaseModelArgs


def build_metrics_processor(
    job_config: JobConfig,
    parallel_dims: ParallelDims,
    model_args: BaseModelArgs | None = None,  # noqa: ARG001
    tag: str | None = None,
) -> FLMetricsProcessor:
    """Create a metrics processor for the FL experiment."""
    return FLMetricsProcessor(job_config, parallel_dims, tag)
