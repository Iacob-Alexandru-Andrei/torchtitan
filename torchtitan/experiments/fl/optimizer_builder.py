# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Builder for optimizers used in FL experiments."""

from __future__ import annotations

from typing import Any

import torch

from torchtitan.components.ft import FTManager
from torchtitan.components.optimizer import (
    OptimizersContainer,
    build_optimizers,
    register_optimizer_class,
    register_optimizer_container,
)
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.fl.configs.optimizers import (
    DesLocConfig,
    MosaicOptimizerConfig,
)
from torchtitan.experiments.fl.desloc import DesLocFTOptimizersContainer
from torchtitan.experiments.fl.optimizers import (
    ADOPT,
    AggMoAdamW,
    AggMoAdopt,
    DecoupledAdamW,
    QHAdamW,
    QHADOPT,
)

_REGISTRATION_DONE = False


def _ensure_optimizer_registration() -> None:
    global _REGISTRATION_DONE
    if _REGISTRATION_DONE:
        return

    register_optimizer_class("ADOPT", ADOPT, exist_ok=True)
    register_optimizer_class("QHADOPT", QHADOPT, exist_ok=True)
    register_optimizer_class("QHAdamW", QHAdamW, exist_ok=True)
    register_optimizer_class("DecoupledAdamW", DecoupledAdamW, exist_ok=True)
    register_optimizer_class("AggMoAdopt", AggMoAdopt, exist_ok=True)
    register_optimizer_class("AggMoAdamW", AggMoAdamW, exist_ok=True)

    def _desloc_enabled(optimizer_config: MosaicOptimizerConfig, context: Any) -> bool:
        desloc = getattr(optimizer_config, "desloc", None)
        return bool(getattr(desloc, "enabled", False))

    def _build_desloc_container(
        optimizer_config: MosaicOptimizerConfig, context: Any
    ) -> OptimizersContainer:
        desloc_cfg = optimizer_config.desloc
        if isinstance(desloc_cfg, dict):  # pragma: no cover - defensive conversion
            desloc_cfg = DesLocConfig(**desloc_cfg)
            optimizer_config.desloc = desloc_cfg

        if optimizer_config.early_step_in_backward:
            raise NotImplementedError(
                "DES-LOC does not support optimizers in backward. "
                "Disable early_step_in_backward to enable DES-LOC."
            )

        ft_manager = context.ft_manager
        if ft_manager is None or not ft_manager.enabled:
            msg = (
                "DES-LOC requires TorchFT to be enabled. Set fault_tolerance.enable to true."
            )
            raise ValueError(msg)

        return DesLocFTOptimizersContainer(
            context.model_parts,
            context.optimizer_cls,
            context.optimizer_kwargs,
            ft_manager.manager,
            desloc_cfg,
            use_ft_optimizer=ft_manager.use_async_quorum,
            param_groups=context.param_groups,
        )

    register_optimizer_container(
        _desloc_enabled, _build_desloc_container, priority=100
    )

    _REGISTRATION_DONE = True


def _normalize_mosaic_optimizer_config(
    optimizer_config: MosaicOptimizerConfig | dict[str, Any]
) -> tuple[MosaicOptimizerConfig, dict[str, Any]]:
    if isinstance(optimizer_config, dict):
        config = MosaicOptimizerConfig(**optimizer_config)
    else:
        config = optimizer_config

    if isinstance(config.desloc, dict):
        config.desloc = DesLocConfig(**config.desloc)

    extra_kwargs: dict[str, Any] = {}
    name = config.name

    if name in {"AggMoAdopt", "AggMoAdamW"}:
        extra_kwargs["betas"] = config.get_betas_tuple()
    if name in {"QHADOPT", "QHAdamW", "AggMoAdopt", "AggMoAdamW"}:
        extra_kwargs["vs"] = config.vs
    if name in {"ADOPT", "QHADOPT", "AggMoAdopt"}:
        extra_kwargs["clip_lambda"] = None
    if name in {"DecoupledAdamW", "AggMoAdopt", "AggMoAdamW"}:
        extra_kwargs["decouple"] = config.decouple

    return config, extra_kwargs


def build_mosaic_optimizers(
    model_parts: list[torch.nn.Module],
    optimizer_config: MosaicOptimizerConfig | dict[str, Any],
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
    param_groups: list[dict[str, Any]] | None = None,
) -> OptimizersContainer:
    """Build optimizers via the shared ``torchtitan.components.optimizer`` path."""

    _ensure_optimizer_registration()
    normalized_config, extra_kwargs = _normalize_mosaic_optimizer_config(optimizer_config)

    return build_optimizers(
        model_parts=model_parts,
        optimizer_config=normalized_config,
        parallel_dims=parallel_dims,
        ft_manager=ft_manager,
        param_groups=param_groups,
        extra_optimizer_kwargs=extra_kwargs,
    )
