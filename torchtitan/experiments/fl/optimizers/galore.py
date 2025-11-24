# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""GaLore optimizer family for FL experiments."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import Tensor
from torch.optim import AdamW

log = logging.getLogger(__name__)

__all__ = ["GaLore", "GaLoreProjector", "classify_low_rank_parameters", ]

GALORE_MAX_SUPPORT_DIM = 2
_HIGH_WEIGHT_DECAY_WARNING = 1e-1

STD_PROJ = "std"
RIGHT_PROJ = "right"
LEFT_PROJ = "left"
FULL_PROJ = "full"
REV_STD_PROJ = "reverse_std"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _orthogonal_matrix(weights: Tensor, rank: int, proj_type: str) -> Tensor | list[Tensor]:
    matrix = weights.data
    original_dtype = matrix.dtype
    original_device = matrix.device
    matrix = matrix.float()

    u_matrix, _, vh_matrix = torch.linalg.svd(matrix, full_matrices=False)
    if proj_type == RIGHT_PROJ:
        result = vh_matrix[:rank, :]
    elif proj_type == LEFT_PROJ:
        result = u_matrix[:, :rank]
    elif proj_type == FULL_PROJ:
        return [
            u_matrix[:, :rank].to(device=original_device, dtype=original_dtype),
            vh_matrix[:rank, :].to(device=original_device, dtype=original_dtype),
        ]
    else:
        raise ValueError(f"Unknown projection type {proj_type!r}.")

    return result.to(device=original_device, dtype=original_dtype)


@dataclass
class GaLoreProjector:
    """Project tensors to/from their low-rank representation."""

    rank: int
    update_proj_gap: int = 200
    scale: float = 1.0
    proj_type: str = STD_PROJ

    def __post_init__(self) -> None:
        _require(self.rank > 0, "rank must be positive")
        _require(self.update_proj_gap > 0, "update_proj_gap must be positive")
        self._orthogonal: Tensor | list[Tensor] | None = None

    def _maybe_refresh(self, weights: Tensor, iteration: Tensor) -> None:
        if self._orthogonal is None or (iteration % self.update_proj_gap).item() == 0:
            self._orthogonal = _orthogonal_matrix(weights, self.rank, self._proj_choice(weights))

    def _proj_choice(self, tensor: Tensor) -> str:
        if self.proj_type in {STD_PROJ, REV_STD_PROJ}:
            if tensor.shape[0] >= tensor.shape[1]:
                return RIGHT_PROJ if self.proj_type == STD_PROJ else LEFT_PROJ
            return LEFT_PROJ if self.proj_type == STD_PROJ else RIGHT_PROJ
        return self.proj_type

    def project(self, full_rank_grad: Tensor, iteration: Tensor) -> Tensor:
        if full_rank_grad.ndim > GALORE_MAX_SUPPORT_DIM:
            raise NotImplementedError("GaLore currently supports tensors up to rank 2.")

        proj_type = self._proj_choice(full_rank_grad)
        self._maybe_refresh(full_rank_grad, iteration)
        orthogonal = self._orthogonal
        if orthogonal is None:
            raise RuntimeError("Projection matrix not initialised.")

        if proj_type == RIGHT_PROJ:
            assert isinstance(orthogonal, Tensor)
            return full_rank_grad @ orthogonal.T.to(full_rank_grad.device)
        if proj_type == LEFT_PROJ:
            assert isinstance(orthogonal, Tensor)
            return orthogonal.T.to(full_rank_grad.device) @ full_rank_grad
        if proj_type == FULL_PROJ:
            assert isinstance(orthogonal, list)
            a_matrix, b_matrix = orthogonal
            return a_matrix.T.to(full_rank_grad.device) @ full_rank_grad @ b_matrix.T.to(full_rank_grad.device)
        raise ValueError(f"Unsupported projection type {proj_type!r}")

    def project_back(self, low_rank_grad: Tensor) -> Tensor:
        orthogonal = self._orthogonal
        if orthogonal is None:
            return low_rank_grad * self.scale

        if isinstance(orthogonal, Tensor):
            matrix = orthogonal.to(low_rank_grad.device)
            if matrix.shape[0] == low_rank_grad.shape[-1]:
                return (low_rank_grad @ matrix) * self.scale
            return (matrix @ low_rank_grad) * self.scale
        a_matrix, b_matrix = orthogonal
        return (a_matrix.to(low_rank_grad.device) @ low_rank_grad @ b_matrix.to(low_rank_grad.device)) * self.scale


class GaLore(AdamW):
    """GaLore optimiser with optional quasi-hyperbolic momentum."""

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 1e-5,
        *,
        v1: float = 0.0,
        rank: int | None = None,
        update_proj_gap: int = 200,
        scale: float = 1.0,
        proj_type: str = STD_PROJ,
        dim: int = 2,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if weight_decay >= _HIGH_WEIGHT_DECAY_WARNING:
            log.warning(
                "High weight_decay=%s for GaLore. Model weights are multiplied by %.6f every step.",
                weight_decay,
                1.0 - weight_decay,
            )
        if not 0.0 <= v1 <= 1.0:
            raise ValueError(f"Invalid quasi-hyperbolic parameter v1={v1}")

        super().__init__(params=params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.v1 = v1
        self._defaults = {
            "rank": rank,
            "update_proj_gap": update_proj_gap,
            "scale": scale,
            "proj_type": proj_type,
            "dim": dim,
        }
        for group in self.param_groups:
            group.setdefault("rank", rank)
            group.setdefault("update_proj_gap", update_proj_gap)
            group.setdefault("scale", scale)
            group.setdefault("proj_type", proj_type)
            group.setdefault("dim", dim)
            group["initial_lr"] = group["lr"]

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        loss = None 
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            rank = group.get("rank")
            use_low_rank = rank is not None
            dim = group.get("dim", GALORE_MAX_SUPPORT_DIM)
            if use_low_rank and dim > GALORE_MAX_SUPPORT_DIM:
                raise NotImplementedError("GaLore supports tensors up to 2 dimensions.")

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("GaLore does not support sparse gradients.")

                state = self.state[param]
                if "step" not in state:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=param.device)

                projector = state.get("projector")
                if use_low_rank and projector is None:
                    projector = GaLoreProjector(
                        rank=rank,
                        update_proj_gap=group["update_proj_gap"],
                        scale=group["scale"],
                        proj_type=group["proj_type"],
                    )
                    state["projector"] = projector

                if projector is not None:
                    grad = projector.project(grad, state["step"])

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(
                        grad,
                        memory_format=torch.preserve_format,
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        grad,
                        memory_format=torch.preserve_format,
                    )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"].add_(1)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                step_count = state["step"]
                beta1_t = step_count.new_tensor(beta1)
                beta2_t = step_count.new_tensor(beta2)
                bias_correction1 = 1 - torch.pow(beta1_t, step_count)
                bias_correction2 = 1 - torch.pow(beta2_t, step_count)
                denom = exp_avg_sq.sqrt() / bias_correction2.sqrt() + eps

                if self.v1 == 0.0:
                    step_tensor = (exp_avg / bias_correction1) / denom
                else:
                    blended = (1 - self.v1) * grad + self.v1 * (exp_avg / bias_correction1)
                    step_tensor = blended / denom

                if projector is not None:
                    step_tensor = projector.project_back(step_tensor)

                param.add_(step_tensor, alpha=-lr)
                if weight_decay > 0.0:
                    param.add_(param, alpha=-lr * weight_decay)
        return loss


def classify_low_rank_parameters(
    parameter_names: list[str],
    optimizer_config: dict | None = None,
) -> dict[str, int]:
    """Classify parameter names as low-rank based on config patterns."""

    if not optimizer_config:
        return {}
    param_groups = optimizer_config.get("param_groups")
    regex_overrides = optimizer_config.get("galore_param_regexes") or []
    default_rank = optimizer_config.get("galore_rank")

    low_rank: dict[str, int] = {}
    remaining = set(parameter_names)

    if param_groups:
        for group in param_groups:
            pattern = group.get("param_str_match")
            rank = group.get("rank", default_rank)
            if not pattern or not isinstance(rank, int):
                continue
            compiled = re.compile(pattern)
            for name in list(remaining):
                if compiled.search(name):
                    low_rank[name] = rank
                    remaining.remove(name)

    for override in regex_overrides:
        pattern = override.get("param_str_match")
        rank = override.get("rank")
        if not pattern or not isinstance(rank, int):
            continue
        compiled = re.compile(pattern)
        for name in list(remaining):
            if compiled.search(name):
                low_rank[name] = rank
                remaining.remove(name)

    if default_rank is not None:
        for name in remaining:
            low_rank[name] = default_rank
    return low_rank
