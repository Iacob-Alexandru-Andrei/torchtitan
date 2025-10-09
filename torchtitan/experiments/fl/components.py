from __future__ import annotations

from typing import TYPE_CHECKING

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.fl.metrics import FLMetricsProcessor

if TYPE_CHECKING:
    from torchtitan.protocols import BaseModelArgs


def build_metrics_processor(
    job_config: JobConfig,
    parallel_dims: ParallelDims,
    model_args: "BaseModelArgs | None" = None,
    tag: str | None = None,
) -> FLMetricsProcessor:
    """Create a metrics processor for the FL experiment."""
    return FLMetricsProcessor(job_config, parallel_dims, tag)