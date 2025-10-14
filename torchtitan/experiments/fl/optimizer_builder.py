# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Builder for optimizers used in FL experiments."""

from typing import Any

import torch

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
from torchtitan.experiments.fl.optimizers import (
    ADOPT,
    AggMoAdamW,
    AggMoAdopt,
    DecoupledAdamW,
    QHAdamW,
    QHADOPT,
)


def build_mosaic_optimizers(  # noqa: C901, PLR0912
    model_parts: list[torch.nn.Module],
    optimizer_config: MosaicOptimizerConfig | dict[str, Any],
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
    param_groups: list[dict[str, Any]] | None = None,
) -> OptimizersContainer:
    """Builds optimizers based on the provided configuration.

    Args:
        model_parts: List of model parts to optimize.
        optimizer_config: Configuration for the optimizer (or dict to be converted).
        parallel_dims: Parallel dimensions information.
        ft_manager: Optional FTManager for using TorchFT.
        param_groups: Optional parameter groups for the optimizer.

    """
    # Convert dict to MosaicOptimizerConfig if needed
    if isinstance(optimizer_config, dict):
        optimizer_config = MosaicOptimizerConfig(**optimizer_config)

    desloc_config = getattr(optimizer_config, "desloc", None)
    if isinstance(desloc_config, dict):
        desloc_config = DesLocConfig(**desloc_config)

    desloc_enabled = bool(getattr(desloc_config, "enabled", False))

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

    optimizer_kwargs: dict[str, Any] = {
        "lr": lr,
        "betas": betas,
        "eps": eps,
        "weight_decay": weight_decay,
        "fused": fused,
        "foreach": foreach,
    }

    if name in ["QHADOPT", "QHAdamW", "AggMoAdopt", "AggMoAdamW"]:
        optimizer_kwargs["vs"] = optimizer_config.vs
    if name in ["ADOPT", "QHADOPT", "AggMoAdopt"]:
        optimizer_kwargs["clip_lambda"] = None
    if name in ["DecoupledAdamW", "AggMoAdopt", "AggMoAdamW"]:
        optimizer_kwargs["decouple"] = getattr(optimizer_config, "decouple", True)

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
        model_parts, optimizer_cls, optimizer_kwargs, param_groups=param_groups
    )
