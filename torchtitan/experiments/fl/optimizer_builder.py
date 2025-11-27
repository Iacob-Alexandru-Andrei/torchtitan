# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Builder for optimizers used in FL experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any, TYPE_CHECKING

import torch

from torchtitan.components.optimizer import (
    build_optimizers,
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
    GaLore,
    Muon,
    QHAdamW,
    QHADOPT,
    Scion,
    ScionAggMo,
    ScionLight,
    QHScion,
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
    "Scion": Scion,
    "ScionLight": ScionLight,
    "ScionQH": QHScion,
    "ScionAggMo": ScionAggMo,
    "GaLore": GaLore,
    "Muon": Muon,
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
    config = MosaicOptimizerConfig(**optimizer_config) if isinstance(optimizer_config, dict) else optimizer_config

    if isinstance(config.desloc, dict):
        config.desloc = DesLocConfig(**config.desloc)

    extra_kwargs: dict[str, Any] = {}
    name = config.name

    if name in {"AggMoAdopt", "AggMoAdamW"}:
        extra_kwargs["betas"] = config.get_betas_tuple()
    if name in {"QHADOPT", "QHAdamW", "AggMoAdopt", "AggMoAdamW"}:
        extra_kwargs["vs"] = config.vs
    if name in {"DecoupledAdamW", "AggMoAdopt", "AggMoAdamW"}:
        extra_kwargs["decouple"] = config.decouple
    if name in {"Scion", "ScionLight", "ScionQH", "ScionAggMo"}:
        extra_kwargs.update(
            {
                "norm": config.norm,
                "norm_kwargs": config.norm_kwargs or {},
                "unconstrained": config.unconstrained,
                "zeropower_coeffs": config.resolved_zeropower_coefficients(),
            }
        )
        if name in {"Scion", "ScionLight", "ScionQH"}:
            extra_kwargs["betas"] = config.get_betas_tuple()
        if name == "ScionQH":
            if config.scion_v is not None:
                vs_tuple = (float(config.scion_v),)
            else:
                vs_raw = config.vs
                if vs_raw is None:
                    vs_source: tuple[float, ...] = ()
                elif isinstance(vs_raw, tuple):
                    vs_source = vs_raw
                else:
                    vs_source = tuple(vs_raw)
                vs_tuple = tuple(float(v) for v in vs_source) if len(vs_source) > 0 else (1.0,)
            extra_kwargs["vs"] = vs_tuple
        if name == "ScionAggMo":
            extra_kwargs["betas"] = config.scion_momentums
            extra_kwargs["weights"] = config.scion_weights
    if name == "Muon":
        extra_kwargs.update(
            {
                "norm": config.norm,
                "norm_kwargs": config.norm_kwargs or {},
                "zeropower_coeffs": config.resolved_zeropower_coefficients(),
                "betas": (config.beta1,),
                "nesterov": config.muon_nesterov,
            }
        )

    return config, extra_kwargs


def _build_optimizer_kwargs(config: MosaicOptimizerConfig, extra_kwargs: dict[str, Any]) -> dict[str, Any]:
    if config.name in {"Scion", "ScionLight", "ScionQH", "ScionAggMo"}:
        kwargs: dict[str, Any] = {"lr": config.lr}
        kwargs.update(extra_kwargs)
        return kwargs
    if config.name == "Muon":
        kwargs = {
            "lr": config.lr,
            "weight_decay": config.weight_decay,
        }
        kwargs.update(extra_kwargs)
        return kwargs
    if config.name == "GaLore":
        kwargs: dict[str, Any] = {
            "lr": config.lr,
            "betas": (config.beta1, config.beta2),
            "eps": config.eps,
            "weight_decay": config.weight_decay,
            "v1": config.galore_v1,
            "rank": config.galore_rank,
            "update_proj_gap": config.galore_update_proj_gap,
            "scale": config.galore_scale,
            "proj_type": config.galore_proj_type,
            "dim": config.galore_dim,
            "rotate_moments_on_refresh": config.galore_rotate_moments_on_refresh,
        }
        kwargs.update(extra_kwargs)
        return kwargs

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


def _compute_galore_rank_overrides(
    model_parts: list[torch.nn.Module],
    config: MosaicOptimizerConfig,
) -> dict[torch.nn.Parameter, int]:
    """Cache regex-based GaLore rank overrides keyed by parameter."""
    overrides: dict[torch.nn.Parameter, int] = {}
    patterns = config.galore_param_regexes or []
    if not patterns:
        return overrides

    compiled: list[tuple[re.Pattern[str], int]] = []
    for spec in patterns:
        pattern = spec.get("param_str_match")
        rank = spec.get("rank")
        if not pattern or not isinstance(rank, int):
            continue
        compiled.append((re.compile(pattern), rank))

    if not compiled:
        return overrides

    for model in model_parts:
        for name, param in model.named_parameters():
            for regex, rank in compiled:
                if regex.search(name):
                    overrides[param] = rank
                    break
    return overrides


def _base_galore_group_from_config(
    params: list[torch.nn.Parameter],
    config: MosaicOptimizerConfig,
) -> dict[str, Any]:
    return {
        "params": params,
        "lr": config.lr,
        "betas": (config.beta1, config.beta2),
        "eps": config.eps,
        "weight_decay": config.weight_decay,
        "rank": config.galore_rank,
        "update_proj_gap": config.galore_update_proj_gap,
        "scale": config.galore_scale,
        "proj_type": config.galore_proj_type,
        "dim": config.galore_dim,
        "v1": config.galore_v1,
        "rotate_moments_on_refresh": config.galore_rotate_moments_on_refresh,
    }



def _build_galore_param_groups(
    model_parts: list[torch.nn.Module],
    config: MosaicOptimizerConfig,
) -> list[dict[str, Any]] | None:
    """Construct per-parameter GaLore param groups from regex specs."""

    if not config.param_groups:
        return None

    if len(model_parts) != 1:
        msg = "optimizer.param_groups with regex matching is supported only for a single model part."
        raise ValueError(msg)

    named_parameters = dict(model_parts[0].named_parameters())
    remaining = dict(named_parameters)
    param_groups: list[dict[str, Any]] = []

    default_rank = config.galore_rank

    for spec in config.param_groups:
        pattern = spec.get("param_str_match")
        if not pattern:
            continue
        compiled = re.compile(pattern)
        matched = [(name, param) for name, param in list(remaining.items()) if compiled.search(name)]
        if not matched:
            continue

        params = [param for _, param in matched]
        for name, _ in matched:
            remaining.pop(name, None)

        group_rank = spec.get("rank", default_rank)
        param_groups.append(
            {
                "params": params,
                "lr": spec.get("lr", config.lr),
                "betas": spec.get("betas", (config.beta1, config.beta2)),
                "eps": spec.get("eps", config.eps),
                "weight_decay": spec.get("weight_decay", config.weight_decay),
                "rank": group_rank,
                "update_proj_gap": spec.get("update_proj_gap", config.galore_update_proj_gap),
                "scale": spec.get("scale", config.galore_scale),
                "proj_type": spec.get("proj_type", config.galore_proj_type),
                "dim": spec.get("dim", config.galore_dim),
                "v1": spec.get("v1", config.galore_v1),
                "rotate_moments_on_refresh": spec.get(
                    "rotate_moments_on_refresh",
                    config.galore_rotate_moments_on_refresh,
                ),
            }
        )

    if remaining:
        # Assign any leftover parameters to a default group.
        params = list(remaining.values())
        param_groups.append(_base_galore_group_from_config(params, config))

    return param_groups


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
                # scion_hidden_scale=config.scion_hidden_scale,
                # scion_output_scale=config.scion_output_scale,
                # scion_hidden_norm=config.scion_hidden_norm,
                # scion_output_norm=config.scion_output_norm,
                # scion_hidden_norm_kwargs=config.scion_hidden_norm_kwargs,
                # scion_output_norm_kwargs=config.scion_output_norm_kwargs,
            )
            if overrides is None:
                continue
            updated_config = replace(config, **overrides.config_updates) if overrides.config_updates else config
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

    outer_optimizer = request.desloc_cfg.normalized_outer_optimizer()
    desloc_config = DesLocFTOptimizersConfig(
        model_parts=request.base.model_parts,
        optimizer_cls=request.base.optimizer_cls,
        optimizer_kwargs=request.base.optimizer_kwargs,
        ft_manager=ft_manager.manager,
        desloc_config=request.desloc_cfg,
        use_ft_optimizer=ft_manager.use_async_quorum,
        param_groups=request.base.param_groups,
        outer_optimizer=outer_optimizer,
        streaming=request.desloc_cfg.resolved_streaming() if hasattr(request.desloc_cfg, "resolved_streaming") else None,
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
            msg = "DES-LOC does not support optimizers in backward. Disable early_step_in_backward."
            raise NotImplementedError(msg)

        ft_manager = request.ft_manager
        if ft_manager is None or not ft_manager.enabled:
            msg = "DES-LOC requires TorchFT to be enabled. Set fault_tolerance.enable to true."
            raise ValueError(msg)

        if isinstance(desloc_cfg, dict):  # pragma: no cover - defensive conversion
            desloc_cfg = DesLocConfig(**desloc_cfg)
            config.desloc = desloc_cfg

        return _build_desloc_container(DeslocContainerRequest(base=request, desloc_cfg=desloc_cfg))

    if config.early_step_in_backward:
        return OptimizersInBackwardContainer(request.model_parts, request.optimizer_cls, request.optimizer_kwargs)

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
    normalized_config, extra_kwargs = _normalize_mosaic_optimizer_config(optimizer_config)
    normalized_config, param_groups = _apply_mup_overrides(
        model_parts,
        normalized_config,
        param_groups,
    )

    if normalized_config.builder == "default":
        if normalized_config.desloc.enabled:
            msg = "DES-LOC is only supported when optimizer.builder is set to 'mosaic'."
            raise ValueError(msg)
        if normalized_config.name in _MOSAIC_OPTIMIZER_CLASSES:
            msg = f"Optimizer {normalized_config.name!r} requires optimizer.builder='mosaic'."
            raise ValueError(msg)
        return build_optimizers(
            model_parts=model_parts,
            optimizer_config=normalized_config,
            parallel_dims=parallel_dims,
            ft_manager=ft_manager,
            param_groups=param_groups,
        )

    optimizer_cls = _resolve_optimizer_class(normalized_config.name)

    effective_extra_kwargs = extra_kwargs
    if normalized_config.name == "GaLore":
        rank_overrides = _compute_galore_rank_overrides(model_parts, normalized_config)

        if param_groups is None:
            built_groups = _build_galore_param_groups(model_parts, normalized_config)
            if built_groups is not None:
                param_groups = built_groups

        effective_extra_kwargs = dict(extra_kwargs)
        effective_extra_kwargs["rank_overrides"] = rank_overrides

    optimizer_kwargs = _build_optimizer_kwargs(normalized_config, effective_extra_kwargs)

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
