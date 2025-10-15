# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Builder for optimizers used in FL experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING

import torch

from torchtitan.components.optimizer import (
    FTOptimizersContainer,
    OptimizersContainer,
    OptimizersInBackwardContainer,
)
from torchtitan.experiments.fl.configs.optimizers import (
    DesLocConfig,
    MosaicOptimizerConfig,
)
from torchtitan.experiments.fl.desloc import (
    DesLocFTOptimizersConfig,
    DesLocFTOptimizersContainer,
)
from torchtitan.experiments.fl.optimizers import (
    ADOPT,
    AggMoAdamW,
    AggMoAdopt,
    DecoupledAdamW,
    QHAdamW,
    QHADOPT,
)

try:  # pragma: no cover - optional dependency for non-MuP models
    from torchtitan.experiments.fl.models.llama3_mup.model.mup_model import (
        SupportsMuPOptimizerOverrides,
    )
except ImportError:  # pragma: no cover - MuP model not available in some builds
    SupportsMuPOptimizerOverrides = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from torchtitan.components.ft import FTManager
    from torchtitan.distributed import ParallelDims


@dataclass(frozen=True)
class OptimizerContainerRequest:
    """Input payload for building a TorchTitan optimizer container."""

    model_parts: list[torch.nn.Module]
    optimizer_cls: type[Optimizer]
    optimizer_kwargs: dict[str, Any]
    config: MosaicOptimizerConfig
    parallel_dims: ParallelDims
    ft_manager: FTManager | None
    param_groups: list[dict[str, Any]] | None


@dataclass(frozen=True)
class DeslocContainerRequest:
    """Request data for constructing a DES-LOC-enabled optimizer container."""

    base: OptimizerContainerRequest
    desloc_cfg: DesLocConfig


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
        msg = f"Optimizer {name!r} is not registered for FL experiments."
        raise NotImplementedError(msg) from exc


def _normalize_mosaic_optimizer_config(
    optimizer_config: MosaicOptimizerConfig | dict[str, Any],
) -> tuple[MosaicOptimizerConfig, dict[str, Any]]:
    config = (
        MosaicOptimizerConfig(**optimizer_config)
        if isinstance(optimizer_config, dict)
        else optimizer_config
    )

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


def _apply_mup_overrides(
    model_parts: list[torch.nn.Module],
    config: MosaicOptimizerConfig,
    param_groups: list[dict[str, Any]] | None,
) -> tuple[MosaicOptimizerConfig, list[dict[str, Any]] | None]:
    """Inject MuP-aware overrides from the first model part that provides them."""
    if param_groups is not None:
        return config, param_groups

    protocol = SupportsMuPOptimizerOverrides
    if protocol is None:
        return config, None

    for part in model_parts:
        if isinstance(part, protocol):
            overrides = part.build_mup_optimizer_overrides(
                lr=config.lr,
                eps=config.eps,
                weight_decay=config.weight_decay,
            )
            if overrides is None:
                continue
            updated_config = (
                replace(config, **overrides.config_updates)
                if overrides.config_updates
                else config
            )
            return updated_config, overrides.param_groups

    return config, None


def _build_desloc_container(request: DeslocContainerRequest) -> OptimizersContainer:
    """Instantiate an optimizer container with DES-LOC synchronization enabled."""
    parallel_dims = request.base.parallel_dims
    ft_manager = request.base.ft_manager
    assert ft_manager is not None  # defensive: enforced by caller

    if parallel_dims.ep_enabled:
        msg = "DES-LOC is not supported with Expert Parallel."
        raise NotImplementedError(msg)
    if parallel_dims.pp_enabled:
        msg = "DES-LOC is not supported with Pipeline Parallel."
        raise NotImplementedError(msg)

    desloc_config = DesLocFTOptimizersConfig(
        model_parts=request.base.model_parts,
        optimizer_cls=request.base.optimizer_cls,
        optimizer_kwargs=request.base.optimizer_kwargs,
        ft_manager=ft_manager.manager,
        desloc_config=request.desloc_cfg,
        use_ft_optimizer=ft_manager.use_async_quorum,
        param_groups=request.base.param_groups,
    )
    return DesLocFTOptimizersContainer(desloc_config)


def _validate_optim_in_backward(request: OptimizerContainerRequest) -> None:
    """Validate the configuration for optimizers that step during backward."""
    if not request.config.early_step_in_backward:
        return

    parallel_dims = request.parallel_dims
    if parallel_dims.ep_enabled:
        msg = "Optimizers in backward is not supported with Expert Parallel."
        raise NotImplementedError(msg)
    if parallel_dims.pp_enabled:
        msg = "Optimizers in backward is not supported with Pipeline Parallel."
        raise NotImplementedError(msg)
    ft_manager = request.ft_manager
    if ft_manager and ft_manager.enabled:
        msg = "TorchFT is not supported with optimizers in backward."
        raise NotImplementedError(msg)


def _build_optimizer_container(
    request: OptimizerContainerRequest,
) -> OptimizersContainer:
    """Construct the appropriate optimizer container for the given request."""
    _validate_optim_in_backward(request)

    config = request.config
    desloc_cfg = config.desloc

    if desloc_cfg.enabled:
        if config.early_step_in_backward:
            msg = (
                "DES-LOC does not support optimizers in backward. "
                "Disable early_step_in_backward."
            )
            raise NotImplementedError(msg)

        ft_manager = request.ft_manager
        if ft_manager is None or not ft_manager.enabled:
            msg = (
                "DES-LOC requires TorchFT to be enabled. "
                "Set fault_tolerance.enable to true."
            )
            raise ValueError(msg)

        if isinstance(desloc_cfg, dict):  # pragma: no cover - defensive conversion
            desloc_cfg = DesLocConfig(**desloc_cfg)
            config.desloc = desloc_cfg

        return _build_desloc_container(
            DeslocContainerRequest(base=request, desloc_cfg=desloc_cfg)
        )

    if config.early_step_in_backward:
        return OptimizersInBackwardContainer(
            request.model_parts, request.optimizer_cls, request.optimizer_kwargs
        )

    ft_manager = request.ft_manager
    if ft_manager and ft_manager.enabled:
        return FTOptimizersContainer(
            request.model_parts,
            request.optimizer_cls,
            request.optimizer_kwargs,
            ft_manager.manager,
            use_ft_optimizer=ft_manager.use_async_quorum,
            param_groups=request.param_groups,
        )

    return OptimizersContainer(
        request.model_parts,
        request.optimizer_cls,
        request.optimizer_kwargs,
        param_groups=request.param_groups,
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
    normalized_config, param_groups = _apply_mup_overrides(
        model_parts,
        normalized_config,
        param_groups,
    )
    optimizer_cls = _resolve_optimizer_class(normalized_config.name)
    optimizer_kwargs = _build_optimizer_kwargs(normalized_config, extra_kwargs)

    return _build_optimizer_container(
        OptimizerContainerRequest(
            model_parts=model_parts,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            config=normalized_config,
            parallel_dims=parallel_dims,
            ft_manager=ft_manager,
            param_groups=param_groups,
        )
    )
