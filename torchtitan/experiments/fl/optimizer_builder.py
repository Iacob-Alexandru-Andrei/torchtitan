# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Builder for optimizers used in FL experiments."""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer

from torchtitan.components.ft import FTManager
from torchtitan.components.optimizer import (
    FTOptimizersContainer,
    OptimizersContainer,
    OptimizersInBackwardContainer,
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

_BASE_OPTIMIZER_CLASSES: dict[str, type[Optimizer]] = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
}

_MOSAIC_OPTIMIZER_CLASSES: dict[str, type[Optimizer]] = {
    "ADOPT": ADOPT,
    "QHADOPT": QHADOPT,
    "QHAdamW": QHAdamW,
    "DecoupledAdamW": DecoupledAdamW,
    "AggMoAdopt": AggMoAdopt,
    "AggMoAdamW": AggMoAdamW,
}

_ALL_OPTIMIZER_CLASSES: dict[str, type[Optimizer]] = {
    **_BASE_OPTIMIZER_CLASSES,
    **_MOSAIC_OPTIMIZER_CLASSES,
}


def _resolve_optimizer_class(name: str) -> type[Optimizer]:
    try:
        return _ALL_OPTIMIZER_CLASSES[name]
    except KeyError as exc:  # pragma: no cover - validated in configuration tests
        raise NotImplementedError(f"Optimizer {name!r} is not registered for FL experiments.") from exc


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

    optim_in_bwd = optimizer_config.early_step_in_backward
    if optim_in_bwd:
        if parallel_dims.ep_enabled:
            msg = "Optimizers in backward is not supported with Expert Parallel."
            raise NotImplementedError(msg)
        if parallel_dims.pp_enabled:
            msg = "Optimizers in backward is not supported with Pipeline Parallel."
            raise NotImplementedError(msg)
        if ft_manager and ft_manager.enabled:
            msg = "TorchFT is not supported with optimizers in backward."
            raise NotImplementedError(msg)

    if desloc_enabled and not (ft_manager and ft_manager.enabled):
        msg = "DES-LOC requires TorchFT to be enabled. Set fault_tolerance.enable to true."
        raise ValueError(msg)

    name = optimizer_config.name
    lr = optimizer_config.lr
    beta1 = optimizer_config.beta1
    beta2 = optimizer_config.beta2
    eps = optimizer_config.eps
    weight_decay = optimizer_config.weight_decay

    optim_implementation = optimizer_config.implementation
    assert optim_implementation in ["fused", "foreach", "for-loop"]

    fused = optim_implementation == "fused"
    foreach = optim_implementation == "foreach"

    optimizer_classes = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "ADOPT": ADOPT,
        "QHADOPT": QHADOPT,
        "QHAdamW": QHAdamW,
        "DecoupledAdamW": DecoupledAdamW,
        "AggMoAdopt": AggMoAdopt,
        "AggMoAdamW": AggMoAdamW,
    }
    if name not in optimizer_classes:
        msg = f"Optimizer {name} not added."
        raise NotImplementedError(msg)
    optimizer_cls = optimizer_classes[name]

    # For AggMo optimizers, use get_betas_tuple() to construct proper betas
    betas = (
        optimizer_config.get_betas_tuple()
        if name in ["AggMoAdopt", "AggMoAdamW"]
        else (beta1, beta2)
    )

def _build_optimizer_kwargs(
    config: MosaicOptimizerConfig, extra_kwargs: dict[str, Any]
) -> dict[str, Any]:
    optim_implementation = config.implementation
    assert optim_implementation in {"fused", "foreach", "for-loop"}

    optimizer_kwargs: dict[str, Any] = {
        "lr": config.lr,
        "betas": (config.beta1, config.beta2),
        "eps": config.eps,
        "weight_decay": config.weight_decay,
        "fused": optim_implementation == "fused",
        "foreach": optim_implementation == "foreach",
    }
    optimizer_kwargs.update(extra_kwargs)
    return optimizer_kwargs


def _build_desloc_container(
    *,
    model_parts: list[torch.nn.Module],
    optimizer_cls: type[Optimizer],
    optimizer_kwargs: dict[str, Any],
    desloc_cfg: DesLocConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager,
    param_groups: list[dict[str, Any]] | None,
) -> OptimizersContainer:
    if parallel_dims.ep_enabled:
        raise NotImplementedError("DES-LOC is not supported with Expert Parallel.")
    if parallel_dims.pp_enabled:
        raise NotImplementedError("DES-LOC is not supported with Pipeline Parallel.")

    return DesLocFTOptimizersContainer(
        model_parts,
        optimizer_cls,
        optimizer_kwargs,
        ft_manager.manager,
        desloc_cfg,
        use_ft_optimizer=ft_manager.use_async_quorum,
        param_groups=param_groups,
    )


def _build_optimizer_container(
    *,
    model_parts: list[torch.nn.Module],
    optimizer_cls: type[Optimizer],
    optimizer_kwargs: dict[str, Any],
    config: MosaicOptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None,
    param_groups: list[dict[str, Any]] | None,
) -> OptimizersContainer:
    optim_in_bwd = config.early_step_in_backward

    if optim_in_bwd:
        if parallel_dims.ep_enabled:
            raise NotImplementedError(
                "Optimizers in backward is not supported with Expert Parallel."
            )
        if parallel_dims.pp_enabled:
            raise NotImplementedError(
                "Optimizers in backward is not supported with Pipeline Parallel."
            )
        if ft_manager and ft_manager.enabled:
            raise NotImplementedError(
                "TorchFT is not supported with optimizers in backward."
            )

    desloc_cfg = config.desloc
    if desloc_cfg.enabled:
        if optim_in_bwd:
            raise NotImplementedError(
                "DES-LOC does not support optimizers in backward. Disable early_step_in_backward."
            )
        if ft_manager is None or not ft_manager.enabled:
            msg = (
                "DES-LOC requires TorchFT to be enabled. Set fault_tolerance.enable to true."
            )
            raise ValueError(msg)
        # If the configuration was loaded from a file or passed as a dictionary,
        # desloc_cfg may still be a dict. Convert it to ensure type consistency.
        if isinstance(desloc_cfg, dict):  # pragma: no cover - defensive conversion
            desloc_cfg = DesLocConfig(**desloc_cfg)
            config.desloc = desloc_cfg
        return _build_desloc_container(
            model_parts=model_parts,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            desloc_cfg=desloc_cfg,
            parallel_dims=parallel_dims,
            ft_manager=ft_manager,
            param_groups=param_groups,
        )

    if desloc_enabled and ft_manager and desloc_config is not None:
        return DesLocFTOptimizersContainer(
            model_parts,
            optimizer_cls,
            optimizer_kwargs,
            ft_manager.manager,
            desloc_config,
            use_ft_optimizer=ft_manager.use_async_quorum,
            param_groups=param_groups,
    if optim_in_bwd:
        return OptimizersInBackwardContainer(
            model_parts, optimizer_cls, optimizer_kwargs
        )

    if ft_manager and ft_manager.enabled:
        return FTOptimizersContainer(
            model_parts,
            optimizer_cls,
            optimizer_kwargs,
            ft_manager.manager,
            use_ft_optimizer=ft_manager.use_async_quorum,
            param_groups=param_groups,
        )

    return OptimizersContainer(
        model_parts,
        optimizer_cls,
        optimizer_kwargs,
        param_groups=param_groups,
    )


def build_mosaic_optimizers(
    model_parts: list[torch.nn.Module],
    optimizer_config: MosaicOptimizerConfig | dict[str, Any],
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
    param_groups: list[dict[str, Any]] | None = None,
) -> OptimizersContainer:
    """Build optimizers for Mosaic jobs without modifying core TorchTitan components."""

    normalized_config, extra_kwargs = _normalize_mosaic_optimizer_config(
        optimizer_config
    )
    optimizer_cls = _resolve_optimizer_class(normalized_config.name)
    optimizer_kwargs = _build_optimizer_kwargs(normalized_config, extra_kwargs)

    return _build_optimizer_container(
        model_parts=model_parts,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        config=normalized_config,
        parallel_dims=parallel_dims,
        ft_manager=ft_manager,
        param_groups=param_groups,
    )
