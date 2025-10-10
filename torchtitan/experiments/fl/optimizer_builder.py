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
from torchtitan.experiments.fl.configs.optimizers import MosaicOptimizerConfig
from torchtitan.experiments.fl.optimizers import (
    ADOPT,
    DecoupledAdamW,
    DesLoc,
    QHAdamW,
    QHADOPT,
)


def build_mosaic_optimizers(  # noqa: C901
    model_parts: list[torch.nn.Module],
    optimizer_config: MosaicOptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
    param_groups: list[dict[str, Any]] | None = None,
) -> OptimizersContainer:
    """Builds optimizers based on the provided configuration.

    Args:
        model_parts: List of model parts to optimize.
        optimizer_config: Configuration for the optimizer.
        parallel_dims: Parallel dimensions information.
        ft_manager: Optional FTManager for using TorchFT.
        param_groups: Optional parameter groups for the optimizer.

    """
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

    optimizer_kwargs: dict[str, Any] = {
        "lr": lr,
        "betas": (beta1, beta2),
        "eps": eps,
        "weight_decay": weight_decay,
        "fused": fused,
        "foreach": foreach,
    }

    optimizer_classes = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "ADOPT": ADOPT,
        "QHADOPT": QHADOPT,
        "QHAdamW": QHAdamW,
        "DecoupledAdamW": DecoupledAdamW,
        "DesLoc": torch.optim.AdamW,  # DesLoc wraps an existing optimizer
    }
    if name not in optimizer_classes:
        msg = f"Optimizer {name} not added."
        raise NotImplementedError(msg)
    optimizer_cls = optimizer_classes[name]

    if name in ["QHADOPT", "QHAdamW"]:
        optimizer_kwargs["vs"] = optimizer_config.vs
    if name in ["ADOPT", "QHADOPT"]:
        optimizer_kwargs["clip_lambda"] = None
    if name == "DecoupledAdamW":
        optimizer_kwargs["decouple"] = getattr(optimizer_config, "decouple", True)

    if optim_in_bwd:
        return OptimizersInBackwardContainer(
            model_parts, optimizer_cls, optimizer_kwargs
        )

    container_class = (
        FTOptimizersContainer
        if ft_manager and ft_manager.enabled
        else OptimizersContainer
    )
    optimizers = container_class(
        model_parts, optimizer_cls, optimizer_kwargs, param_groups=param_groups
    )

    if name == "DesLoc":
        if not (ft_manager and ft_manager.enabled):
            raise ValueError("DesLoc requires TorchFT to be enabled.")
        # Attach the DesLoc instance to the container to prevent it from being
        # garbage collected and ensure its hooks remain active.
        optimizers._desloc_instance = DesLoc(
            manager=ft_manager.manager,
            model=torch.nn.Sequential(*model_parts),
            optimizer=optimizers,
            param_sync_every=optimizer_config.param_sync_every,
            optimizer_sync_every=optimizer_config.optimizer_sync_every,
        )

    return optimizers
