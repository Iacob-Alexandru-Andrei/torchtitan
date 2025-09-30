"""Composer callback that logs quasi-hyperbolic optimizer parameters."""

from __future__ import annotations

from composer.core import Callback, State
from composer.loggers import Logger
from llmfoundry.callbacks import callbacks

__all__ = ["QuasiHyperbolicParameterMonitor"]


@callbacks.register(name="quasi_hyperbolic_parameter_monitor")
class QuasiHyperbolicParameterMonitor(Callback):
    """Log the ``vs`` tuple stored on parameter groups."""

    def batch_end(self, state: State, logger: Logger) -> None:  # noqa: PLR6301
        assert state.optimizers is not None, "optimizers must be defined"
        step = state.timestamp.batch.value
        for optimizer in state.optimizers:
            for group_idx, group in enumerate(optimizer.param_groups):
                vs = group.get("vs")
                if vs is None:
                    continue
                for v_idx, value in enumerate(vs):
                    logger.log_metrics({f"v{v_idx}-group{group_idx}": value}, step)
