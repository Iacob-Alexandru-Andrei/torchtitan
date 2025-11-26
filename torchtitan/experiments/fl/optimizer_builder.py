# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Builder for optimizers used in FL experiments."""

from __future__ import annotations

from fnmatch import fnmatch
from collections.abc import MutableMapping
from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer

from torchtitan.components.ft import has_torchft
from torchtitan.components.optimizer import (
    build_optimizers,
    FTOptimizersContainer,
    OptimizersContainer,
    OptimizersInBackwardContainer,
)
from torchtitan.experiments.fl.configs.optimizers import (
    CompositeOptimizerSpec,
    DesLocConfig,
    MosaicOptimizerConfig,
)
from torchtitan.experiments.fl.desloc import (
    DesLocController,
    DesLocControllerConfig,
    DesLocFTOptimizersConfig,
    DesLocFTOptimizersContainer,
    StreamingDesLocController,
)
from torchtitan.experiments.fl.optimizers import (
    ADOPT,
    AggMoAdamW,
    AggMoAdopt,
    DecoupledAdamW,
    AggMoMuon,
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


@dataclass(frozen=True)
class _MuPContext:
    """MuP-specific parameter grouping and scaling metadata."""

    buckets: dict[str, list[torch.nn.Parameter]]
    label_by_param: dict[torch.nn.Parameter, str]
    width_lr_scaling: float
    depth_lr_scaling: float
    adjusted_eps: float | None


def _normalize_composite_specs(
    composite: list[CompositeOptimizerSpec] | tuple[CompositeOptimizerSpec, ...] | None,
) -> list[CompositeOptimizerSpec] | None:
    """Normalize composite optimizer specs to dataclass instances."""
    if composite is None:
        return None

    normalized: list[CompositeOptimizerSpec] = []
    for entry in composite:
        if isinstance(entry, CompositeOptimizerSpec):
            normalized.append(entry)
        elif isinstance(entry, dict):  # pragma: no cover - defensive conversion
            normalized.append(CompositeOptimizerSpec(**entry))
        else:  # pragma: no cover - config validation should prevent this
            msg = (
                "optimizer.composite entries must be CompositeOptimizerSpec or mappings; "
                f"received {type(entry)!r}."
            )
            raise TypeError(msg)
    return normalized


def _collect_trainable_params(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    """Return trainable parameters with their qualified names."""
    return [
        (name, param)
        for name, param in model.named_parameters(remove_duplicate=True)
        if param.requires_grad
    ]


def _build_mup_context(
    model: torch.nn.Module,
    base_eps: float,
) -> _MuPContext | None:
    """Extract MuP bucket/grouping metadata when available on the model."""
    if not (
        hasattr(model, "_bucketize_parameters")
        and hasattr(model, "_iter_trainable_params")
    ):
        return None

    try:
        param_entries = model._iter_trainable_params()  # type: ignore[attr-defined]
        buckets = model._bucketize_parameters(param_entries)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - defensive fallback
        return None

    if hasattr(model, "_validate_bucket_counts"):
        model._validate_bucket_counts(len(param_entries), buckets)  # type: ignore[attr-defined]

    label_by_param: dict[torch.nn.Parameter, str] = {}
    for label, params in buckets.items():
        for param in params:
            label_by_param[param] = label

    width_lr_scaling = 1.0
    depth_lr_scaling = 1.0
    adjusted_eps: float | None = None

    mup_cfg = getattr(model, "mup_config", None)
    lr_scaling_enabled = bool(
        mup_cfg
        and getattr(mup_cfg, "mup_enabled", False)
        and not getattr(mup_cfg, "mup_disable_hidden_lr_scaling", False)
    )

    if lr_scaling_enabled and hasattr(model, "_compute_lr_scaling"):
        width_lr_scaling, depth_lr_scaling = model._compute_lr_scaling()  # type: ignore[attr-defined]

    if lr_scaling_enabled and hasattr(model, "_resolve_optimizer_eps"):
        adjusted_eps = model._resolve_optimizer_eps(  # type: ignore[attr-defined]
            base_eps,
            width_lr_scaling=width_lr_scaling,
        )

    return _MuPContext(
        buckets=buckets,
        label_by_param=label_by_param,
        width_lr_scaling=width_lr_scaling,
        depth_lr_scaling=depth_lr_scaling,
        adjusted_eps=adjusted_eps,
    )


def _resolve_group_hparams(
    label: str,
    config: MosaicOptimizerConfig,
    mup_ctx: _MuPContext,
) -> tuple[float, float]:
    """Return (lr, weight_decay) for a MuP bucket label."""
    if label == "emb":
        return config.lr, config.weight_decay
    if label == "hidden_ln":
        return config.lr * mup_ctx.depth_lr_scaling, 0.0
    if label == "decay_lr":
        lr = config.lr * mup_ctx.width_lr_scaling * mup_ctx.depth_lr_scaling
        weight_decay = config.weight_decay / mup_ctx.width_lr_scaling
        return lr, weight_decay
    if label == "hidden_bias":
        return config.lr * mup_ctx.depth_lr_scaling, 0.0
    if label == "no_decay":
        return config.lr, 0.0
    return config.lr, config.weight_decay


class _CompositeOptimizerStateProxy(MutableMapping):
    """Expose underlying optimizer states through a unified mapping."""

    def __init__(self, optimizers: list[Optimizer]) -> None:
        self._optimizers = optimizers
        self._param_owner: dict[nn.Parameter, Optimizer] = {}
        self.refresh()

    def refresh(self) -> None:
        self._param_owner.clear()
        for optimizer in self._optimizers:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if isinstance(param, nn.Parameter):
                        self._param_owner[param] = optimizer

    def _get_owner(self, param: nn.Parameter) -> Optimizer:
        owner = self._param_owner.get(param)
        if owner is None:
            self.refresh()
            owner = self._param_owner.get(param)
        if owner is None:
            raise KeyError(param)
        return owner

    def __getitem__(self, key: nn.Parameter) -> dict[str, Any]:
        owner = self._get_owner(key)
        return owner.state[key]

    def __setitem__(self, key: nn.Parameter, value: dict[str, Any]) -> None:
        owner = self._get_owner(key)
        owner.state[key] = value

    def setdefault(
        self,
        key: nn.Parameter,
        default: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        owner = self._get_owner(key)
        if default is None:
            default = {}
        return owner.state.setdefault(key, default)

    def __delitem__(self, key: nn.Parameter) -> None:
        owner = self._get_owner(key)
        del owner.state[key]

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, nn.Parameter):
            return False
        try:
            owner = self._get_owner(key)
        except KeyError:
            return False
        return key in owner.state

    def __iter__(self):
        seen: set[nn.Parameter] = set()
        for optimizer in self._optimizers:
            for param in optimizer.state:
                if param not in seen:
                    seen.add(param)
                    yield param

    def __len__(self) -> int:
        return sum(1 for _ in self.__iter__())

    def clear(self) -> None:
        for optimizer in self._optimizers:
            optimizer.state.clear()

    @property
    def default_factory(self):
        return dict


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
    "AggMoMuon": AggMoMuon,
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

_MUP_LABEL_ORDER: tuple[str, ...] = ("emb", "hidden_ln", "decay_lr", "hidden_bias", "no_decay")


def _resolve_optimizer_class(name: str) -> type[Optimizer]:
    try:
        return _ALL_OPTIMIZER_CLASSES[name]
    except KeyError as exc:  # pragma: no cover - validated in configuration tests
        msg = f"Optimizer {name!r} is not registered for FL experiments."
        raise NotImplementedError(msg) from exc


class CompositeOptimizersContainer(Optimizer, Stateful):
    """Container for heterogeneous optimizers over a single model part."""

    def __init__(self, model_parts: list[torch.nn.Module], optimizers: list[Optimizer]) -> None:
        if len(model_parts) != 1:
            msg = "Composite optimizers currently support exactly one model part."
            raise NotImplementedError(msg)

        self.model_parts = model_parts
        self.optimizers = optimizers

        combined_param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]

        # Defaults are unused at the container level; delegate to inner opts.
        Optimizer.__init__(self, combined_param_groups, {})
        self._refresh_views()

    def _refresh_views(self) -> None:
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]
        if not hasattr(self, "_state_proxy"):
            self._state_proxy = _CompositeOptimizerStateProxy(self.optimizers)
        else:
            self._state_proxy.refresh()
        self.state = self._state_proxy

    def __iter__(self):
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    @Optimizer.profile_hook_step
    def step(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        options = StateDictOptions(flatten_optimizer_state_dict=True)
        return {
            k: v
            for optimizer in self.optimizers
            for k, v in get_optimizer_state_dict(
                self.model_parts[0],
                optimizer,
                options=options,
            ).items()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        options = StateDictOptions(flatten_optimizer_state_dict=True)
        for optimizer in self.optimizers:
            set_optimizer_state_dict(
                self.model_parts[0],
                optimizer,
                optim_state_dict=state_dict,
                options=options,
            )
        self._refresh_views()


class CompositeFTOptimizersContainer(CompositeOptimizersContainer):
    """Composite optimizer container with TorchFT integration."""

    def __init__(
        self,
        model_parts: list[torch.nn.Module],
        optimizers: list[Optimizer],
        ft_manager: Any,
        use_ft_optimizer: bool = True,
    ) -> None:
        if not has_torchft:
            msg = "TorchFT is required for CompositeFTOptimizersContainer."
            raise ImportError(msg)

        super().__init__(model_parts, optimizers)

        options = StateDictOptions(flatten_optimizer_state_dict=True)
        _ = {
            k: v
            for optimizer in self.optimizers
            for k, v in get_optimizer_state_dict(
                self.model_parts[0],
                optimizer,
                options=options,
            ).items()
        }

        import torchft as ft  # imported lazily to mirror FTOptimizersContainer

        self.cache_state_dict: dict[str, Any] = {}
        self._ft_optimizer = ft.Optimizer(ft_manager, self)
        self._use_ft_optimizer: bool = use_ft_optimizer

    def init_cache_state_dict(self) -> None:
        self.cache_state_dict = super().state_dict()

    def state_dict(self) -> dict[str, Any]:
        return self.cache_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.cache_state_dict = {}
        super().load_state_dict(state_dict)
        self.init_cache_state_dict()

    @Optimizer.profile_hook_step
    def step(self, *args, **kwargs) -> None:
        if self._use_ft_optimizer:
            self._use_ft_optimizer = False
            self._ft_optimizer.step(*args, **kwargs)
            self._use_ft_optimizer = True
        else:
            super().step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        if self._use_ft_optimizer:
            self._use_ft_optimizer = False
            self._ft_optimizer.zero_grad(*args, **kwargs)
            self._use_ft_optimizer = True
        else:
            super().zero_grad(*args, **kwargs)


class CompositeDesLocFTOptimizersContainer(CompositeFTOptimizersContainer):
    """Composite optimizer container augmented with DES-LOC synchronization."""

    def __init__(
        self,
        model_parts: list[torch.nn.Module],
        optimizers: list[Optimizer],
        *,
        desloc_config: DesLocConfig,
        ft_manager: Any,
        use_ft_optimizer: bool,
        outer_optimizer: Any = None,
        streaming: Any = None,
    ) -> None:
        if desloc_config.param_sync_every <= 0:
            msg = "desloc.param_sync_every must be a positive integer."
            raise ValueError(msg)

        streaming_cfg = streaming or desloc_config.resolved_streaming()
        super().__init__(model_parts, optimizers, ft_manager, use_ft_optimizer=use_ft_optimizer)

        backup_device = desloc_config.resolved_backup_device()
        optimizer_sync = desloc_config.normalized_optimizer_sync()
        outer_optimizer_spec = outer_optimizer or desloc_config.normalized_outer_optimizer()

        self._desloc_controllers: list[DesLocController | StreamingDesLocController] = []
        for idx, optimizer in enumerate(self.optimizers):
            controller_config = DesLocControllerConfig(
                manager=ft_manager,
                model=self.model_parts[0],
                optimizer=optimizer,
                param_sync_every=desloc_config.param_sync_every,
                optimizer_sync_every=optimizer_sync,
                backup_device=backup_device,
                pin_memory=desloc_config.pin_memory,
                name_prefix=f"desloc_{idx}",
                quorum_timeout_seconds=desloc_config.quorum_timeout_seconds,
                outer_optimizer=outer_optimizer_spec,
                log_outer_metrics=desloc_config.log_outer_metrics,
                metrics_logger=None,
                checkpoint_outer_optimizer=desloc_config.checkpoint_outer_optimizer,
                disable_optimizer_state_sync=desloc_config.disable_optimizer_state_sync,
            )
            if streaming_cfg is not None:
                controller = StreamingDesLocController(controller_config, streaming_cfg)
            else:
                controller = DesLocController(controller_config)
            self._desloc_controllers.append(controller)

    def close_desloc(self) -> None:
        """Detach DES-LOC hooks from wrapped optimizers."""
        for controller in self._desloc_controllers:
            controller.close()
        self._desloc_controllers.clear()

    def set_desloc_metrics_logger(self, logger_fn: Any | None) -> None:
        for controller in self._desloc_controllers:
            controller.set_metrics_logger(logger_fn)


def _normalize_mosaic_optimizer_config(
    optimizer_config: MosaicOptimizerConfig | dict[str, Any],
) -> tuple[MosaicOptimizerConfig, dict[str, Any]]:
    config = MosaicOptimizerConfig(**optimizer_config) if isinstance(optimizer_config, dict) else optimizer_config

    if isinstance(config.desloc, dict):
        config.desloc = DesLocConfig(**config.desloc)
    config.composite = _normalize_composite_specs(config.composite)

    extra_kwargs: dict[str, Any] = {}
    name = config.name

    if name in {"AggMoAdopt", "AggMoAdamW"}:
        extra_kwargs["betas"] = config.get_betas_tuple()
    if name in {"QHADOPT", "QHAdamW", "AggMoAdopt", "AggMoAdamW"}:
        extra_kwargs["vs"] = config.vs
    if name == "AggMoMuon":
        betas = config.betas if config.betas is not None else tuple([config.beta1] * len(config.vs))
        extra_kwargs["betas"] = betas
        extra_kwargs["vs"] = config.vs
        extra_kwargs["nesterov"] = config.muon_nesterov
        extra_kwargs["ns_coefficients"] = config.resolved_zeropower_coefficients()
        extra_kwargs["adjust_lr_fn"] = config.adjust_lr_fn
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
                "momentum": config.beta1,
                "nesterov": config.muon_nesterov,
                "ns_coefficients": config.resolved_zeropower_coefficients(),
                "adjust_lr_fn": config.adjust_lr_fn,
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
            "eps": config.eps,
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
        }
        kwargs.update(extra_kwargs)
        return kwargs
    if config.name == "AggMoMuon":
        kwargs = {
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "eps": config.eps,
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


def _build_param_groups_for_spec(
    spec: CompositeOptimizerSpec,
    params: list[torch.nn.Parameter],
    spec_config: MosaicOptimizerConfig,
    mup_ctx: _MuPContext | None,
) -> list[dict[str, Any]]:
    """Construct parameter groups for a composite optimizer shard."""
    if not params:
        return []

    if mup_ctx is None:
        return [
            {"params": params, "lr": spec_config.lr, "weight_decay": spec_config.weight_decay}
        ]

    grouped: dict[str, list[torch.nn.Parameter]] = {}
    fallback: list[torch.nn.Parameter] = []

    for param in params:
        label = mup_ctx.label_by_param.get(param)
        if label is None:
            fallback.append(param)
        else:
            grouped.setdefault(label, []).append(param)

    param_groups: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for label in _MUP_LABEL_ORDER:
        if label not in grouped:
            continue
        group_params = grouped[label]
        if not group_params:
            continue
        lr, weight_decay = _resolve_group_hparams(label, spec_config, mup_ctx)
        param_groups.append({"params": group_params, "lr": lr, "weight_decay": weight_decay})
        seen_labels.add(label)

    remaining_labels = sorted(set(grouped) - seen_labels)
    for label in remaining_labels:
        group_params = grouped[label]
        if not group_params:
            continue
        lr, weight_decay = _resolve_group_hparams(label, spec_config, mup_ctx)
        param_groups.append({"params": group_params, "lr": lr, "weight_decay": weight_decay})

    if fallback:
        param_groups.append(
            {"params": fallback, "lr": spec_config.lr, "weight_decay": spec_config.weight_decay}
        )

    return param_groups


def _build_composite_optimizers(
    model_parts: list[torch.nn.Module],
    config: MosaicOptimizerConfig,
    specs: list[CompositeOptimizerSpec],
    parallel_dims: ParallelDims,
    ft_manager: "FTManager | None",
    param_groups: list[dict[str, Any]] | None = None,
) -> OptimizersContainer:
    """Build heterogeneous optimizers when optimizer.composite is set."""
    if param_groups is not None:
        msg = "Custom param_groups are not supported alongside optimizer.composite."
        raise ValueError(msg)

    if config.builder == "default":
        msg = "optimizer.builder must be 'mosaic' when optimizer.composite is provided."
        raise ValueError(msg)

    if config.early_step_in_backward:
        if parallel_dims.ep_enabled:
            msg = "Optimizers in backward are not supported with optimizer.composite and Expert Parallel."
            raise NotImplementedError(msg)
        if parallel_dims.pp_enabled:
            msg = "Optimizers in backward are not supported with optimizer.composite and Pipeline Parallel."
            raise NotImplementedError(msg)
        if ft_manager and ft_manager.enabled:
            msg = "TorchFT is not supported with optimizers in backward."
            raise NotImplementedError(msg)

    if len(model_parts) != 1:
        msg = "optimizer.composite currently supports a single model part."
        raise NotImplementedError(msg)

    model = model_parts[0]
    param_entries = _collect_trainable_params(model)
    if not param_entries:
        msg = "No trainable parameters found for composite optimizer construction."
        raise ValueError(msg)

    name_by_param = {param: name for name, param in param_entries}
    all_params = [param for _, param in param_entries]

    specs = list(specs)
    default_specs = [spec for spec in specs if spec.default]
    if len(default_specs) > 1:
        msg = "Only one optimizer.composite entry may set default=True."
        raise ValueError(msg)
    default_spec = default_specs[0] if default_specs else None
    non_default_specs = [spec for spec in specs if not spec.default]

    mup_ctx = _build_mup_context(model, config.eps)
    mup_eps_scale: float | None = None
    if mup_ctx and mup_ctx.adjusted_eps is not None and config.eps != 0.0:
        mup_eps_scale = mup_ctx.adjusted_eps / config.eps
    assigned: set[torch.nn.Parameter] = set()
    assignments: list[tuple[CompositeOptimizerSpec, list[torch.nn.Parameter]]] = []

    def _select_params_for_spec(spec: CompositeOptimizerSpec) -> list[torch.nn.Parameter]:
        selected: list[torch.nn.Parameter] = []
        seen: set[torch.nn.Parameter] = set()

        if spec.labels:
            if mup_ctx is None:
                msg = (
                    f"Composite optimizer spec {spec.name!r} uses MuP labels but MuP grouping "
                    "metadata is unavailable for this model."
                )
                raise ValueError(msg)
            for label in spec.labels:
                for param in mup_ctx.buckets.get(label, []):
                    if param in seen:
                        continue
                    if param in assigned:
                        pname = name_by_param.get(param, "<unknown>")
                        msg = f"Parameter '{pname}' assigned to multiple composite specs."
                        raise ValueError(msg)
                    selected.append(param)
                    seen.add(param)

        if spec.patterns:
            for pattern in spec.patterns:
                for name, param in param_entries:
                    if fnmatch(name, pattern):
                        if param in seen:
                            continue
                        if param in assigned:
                            msg = f"Parameter '{name}' assigned to multiple composite specs."
                            raise ValueError(msg)
                        selected.append(param)
                        seen.add(param)

        return selected

    for spec in non_default_specs:
        selected = _select_params_for_spec(spec)
        if not selected:
            msg = f"Composite optimizer spec {spec.name!r} did not match any parameters."
            raise ValueError(msg)
        assignments.append((spec, selected))
        assigned.update(selected)

    if default_spec is not None:
        if default_spec.labels or default_spec.patterns:
            msg = "Default composite optimizer entries may not set labels or patterns."
            raise ValueError(msg)
        remaining = [param for param in all_params if param not in assigned]
        if remaining:
            assignments.append((default_spec, remaining))
            assigned.update(remaining)

    if len(assigned) != len(all_params):
        unassigned = [name_by_param[param] for param in all_params if param not in assigned]
        msg = (
            "optimizer.composite configuration left parameters unassigned: "
            + ", ".join(sorted(unassigned))
        )
        raise ValueError(msg)

    optimizers: list[Optimizer] = []
    for spec, params in assignments:
        spec_config = replace(config, name=spec.name, composite=None)
        if spec.config_overrides:
            try:
                spec_config = replace(spec_config, **spec.config_overrides)
            except TypeError as exc:
                msg = f"Invalid config_overrides for composite spec {spec.name!r}: {exc}"
                raise ValueError(msg) from exc
        if mup_ctx and mup_ctx.adjusted_eps is not None:
            if mup_eps_scale is None:
                spec_eps = mup_ctx.adjusted_eps
            else:
                spec_eps = spec_config.eps * mup_eps_scale
            spec_config = replace(spec_config, eps=spec_eps)

        spec_config, extra_kwargs = _normalize_mosaic_optimizer_config(spec_config)
        optimizer_cls = _resolve_optimizer_class(spec_config.name)
        optimizer_kwargs = _build_optimizer_kwargs(spec_config, extra_kwargs)

        param_groups_for_spec = _build_param_groups_for_spec(
            spec,
            params,
            spec_config,
            mup_ctx,
        )
        if not param_groups_for_spec:
            msg = f"Composite optimizer spec {spec.name!r} produced no parameter groups."
            raise ValueError(msg)

        optimizers.append(optimizer_cls(param_groups_for_spec, **optimizer_kwargs))

    if config.desloc.enabled:
        desloc_cfg = config.desloc
        if config.early_step_in_backward:
            msg = "DES-LOC does not support optimizers in backward. Disable early_step_in_backward."
            raise NotImplementedError(msg)
        if ft_manager is None or not ft_manager.enabled:
            msg = "DES-LOC requires TorchFT to be enabled. Set fault_tolerance.enable to true."
            raise ValueError(msg)
        if parallel_dims.ep_enabled:
            msg = "DES-LOC is not supported with Expert Parallel."
            raise NotImplementedError(msg)
        if parallel_dims.pp_enabled:
            msg = "DES-LOC is not supported with Pipeline Parallel."
            raise NotImplementedError(msg)

        outer_optimizer = desloc_cfg.normalized_outer_optimizer()
        streaming_cfg = (
            desloc_cfg.resolved_streaming() if hasattr(desloc_cfg, "resolved_streaming") else None
        )
        return CompositeDesLocFTOptimizersContainer(
            model_parts,
            optimizers,
            desloc_config=desloc_cfg,
            ft_manager=ft_manager.manager,
            use_ft_optimizer=ft_manager.use_async_quorum,
            outer_optimizer=outer_optimizer,
            streaming=streaming_cfg,
        )

    if ft_manager and ft_manager.enabled:
        return CompositeFTOptimizersContainer(
            model_parts,
            optimizers,
            ft_manager.manager,
            use_ft_optimizer=ft_manager.use_async_quorum,
        )

    return CompositeOptimizersContainer(model_parts, optimizers)


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

    if normalized_config.composite:
        return _build_composite_optimizers(
            model_parts=model_parts,
            config=normalized_config,
            specs=normalized_config.composite,
            parallel_dims=parallel_dims,
            ft_manager=ft_manager,
            param_groups=param_groups,
        )

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
