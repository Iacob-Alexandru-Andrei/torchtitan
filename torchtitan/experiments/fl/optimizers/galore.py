# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""GaLore optimizer family for FL experiments."""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Callable, Iterable

import torch
from torch import Tensor
from torch.optim import AdamW

from ._decoupled_decay import _compute_decay_factor
from ._metric_utils import prepare_metrics_for_reduction, reduce_metrics_across_ranks

log = logging.getLogger(__name__)

__all__ = ["GaLore", "classify_low_rank_parameters", ]

GALORE_MAX_SUPPORT_DIM = 2
_HIGH_WEIGHT_DECAY_WARNING = 1e-1

STD_PROJ = "std"
RIGHT_PROJ = "right"
LEFT_PROJ = "left"
FULL_PROJ = "full"
REV_STD_PROJ = "reverse_std"


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


def _resolve_proj_choice(proj_type: str, tensor: Tensor) -> str:
    if proj_type in {STD_PROJ, REV_STD_PROJ}:
        if tensor.shape[0] >= tensor.shape[1]:
            return RIGHT_PROJ if proj_type == STD_PROJ else LEFT_PROJ
        return LEFT_PROJ if proj_type == STD_PROJ else RIGHT_PROJ
    return proj_type


def _maybe_refresh_projector(state: dict[str, Any], weights: Tensor, iteration: Tensor) -> None:
    meta = state.setdefault(
        "projector_meta",
        {
            "rank": None,
            "update_proj_gap": None,
            "scale": None,
            "proj_type": None,
            "resolved_proj_type": None,
        },
    )
    rank = meta["rank"]
    update_proj_gap = meta["update_proj_gap"]
    proj_type = meta.get("proj_type") or meta.get("resolved_proj_type") or STD_PROJ
    resolved_proj_type = _resolve_proj_choice(proj_type, weights)
    meta["resolved_proj_type"] = resolved_proj_type
    if rank is None or update_proj_gap is None:
        return

    orthogonal = state.get("projector_basis")
    if orthogonal is None or (iteration % update_proj_gap).item() == 0:
        state["projector_basis"] = _orthogonal_matrix(weights, rank, resolved_proj_type)


def _project(
    state: dict[str, Any],
    full_rank_grad: Tensor,
    iteration: Tensor,
) -> Tensor:
    if full_rank_grad.ndim > GALORE_MAX_SUPPORT_DIM:
        raise NotImplementedError("GaLore currently supports tensors up to rank 2.")

    meta = state.get("projector_meta", {})
    proj_type = meta.get("proj_type") or meta.get("resolved_proj_type") or STD_PROJ
    proj_type = _resolve_proj_choice(proj_type, full_rank_grad)
    meta["resolved_proj_type"] = proj_type
    state["projector_meta"] = meta
    _maybe_refresh_projector(state, full_rank_grad, iteration)
    orthogonal = state.get("projector_basis")
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


def _project_back(state: dict[str, Any], low_rank_grad: Tensor) -> Tensor:
    orthogonal = state.get("projector_basis")
    scale = state.get("projector_meta", {}).get("scale", 1.0)
    if orthogonal is None:
        return low_rank_grad * scale

    if isinstance(orthogonal, Tensor):
        matrix = orthogonal.to(low_rank_grad.device)
        if matrix.shape[0] == low_rank_grad.shape[-1]:
            return (low_rank_grad @ matrix) * scale
        return (matrix @ low_rank_grad) * scale
    a_matrix, b_matrix = orthogonal
    return (a_matrix.to(low_rank_grad.device) @ low_rank_grad @ b_matrix.to(low_rank_grad.device)) * scale


class GaLore(AdamW):
    """GaLore optimiser with optional quasi-hyperbolic momentum."""

    metric_functions: dict[str, Callable[[Tensor, dict[str, Any], Tensor], Tensor]] = {
        "l2_norm/exp_avg": (
            lambda _param, optim_state, _step_tensor: torch.linalg.vector_norm(
                optim_state["exp_avg"],
            )
        ),
        "l2_norm/exp_avg_sq": (
            lambda _param, optim_state, _step_tensor: torch.linalg.vector_norm(
                optim_state["exp_avg_sq"],
            )
        ),
        "min/exp_avg_sq": lambda _param, optim_state, _step_tensor: torch.min(
            optim_state["exp_avg_sq"],
        ),
        "max/exp_avg_sq": lambda _param, optim_state, _step_tensor: torch.max(
            optim_state["exp_avg_sq"],
        ),
        "l2_norm/param": (
            lambda param, _optim_state, _step_tensor: torch.linalg.vector_norm(
                param.data,
            )
        ),
        "l2_norm/update": (
            lambda _param, _optim_state, step_tensor: torch.linalg.vector_norm(
                step_tensor,
            )
        ),
        "l2_norm/grad": (
            lambda param, _optim_state, _step_tensor: torch.linalg.vector_norm(
                param.grad,
            )
        ),
    }

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

                if use_low_rank:
                    state.setdefault(
                        "projector_meta",
                        {
                            "rank": rank,
                            "update_proj_gap": group["update_proj_gap"],
                            "scale": group["scale"],
                            "proj_type": group["proj_type"],
                        },
                    )
                    meta = state["projector_meta"]
                    meta["rank"] = rank
                    meta["update_proj_gap"] = group["update_proj_gap"]
                    meta["scale"] = group["scale"]
                    meta["proj_type"] = group["proj_type"]
                    grad = _project(state, grad, state["step"])

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

                if use_low_rank:
                    step_tensor = _project_back(state, step_tensor)

                param.add_(step_tensor, alpha=-lr)
                if weight_decay > 0.0:
                    param.add_(param, alpha=-lr * weight_decay)
        return loss

    @staticmethod
    def pre_reduce_metrics(optimizer_metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Prepare metrics for distributed reduction."""
        return prepare_metrics_for_reduction(optimizer_metrics)

    @staticmethod
    def dist_reduce_metrics(optimizer_metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Reduce metrics across ranks."""
        return reduce_metrics_across_ranks(optimizer_metrics)

    def _projector_eigenvalues(
        self,
        optim_state: dict[str, Any],
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        basis = optim_state.get("projector_basis")
        if basis is None:
            return None, None

        if isinstance(basis, Tensor):
            proj_matrix = basis @ basis.T
        elif isinstance(basis, list):
            left_basis, _ = basis
            proj_matrix = left_basis @ left_basis.T
        else:
            return None, None

        proj_matrix = proj_matrix.to(device=device)
        eigenvalues = torch.linalg.eigvalsh(proj_matrix).real
        return eigenvalues, torch.prod(eigenvalues)

    def report_per_parameter_metrics(
        self,
        param: torch.Tensor,
        name: str,
        optimizer_metrics: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Report per-parameter metrics including GaLore projection stats."""
        lr = self.param_groups[0]["lr"]
        eps = self.param_groups[0]["eps"]
        weight_decay = self.param_groups[0]["weight_decay"]
        initial_lr = self.param_groups[0]["initial_lr"]

        beta1, beta2 = self.param_groups[0]["betas"]
        if param in self.state:
            param_optim_state = self.state[param]
            step_state = param_optim_state["step"]
            if "max/optimizer_step" not in optimizer_metrics:
                if isinstance(step_state, torch.Tensor):
                    step_tensor = step_state.detach().clone()
                    if step_tensor.device != param.device:
                        step_tensor = step_tensor.to(param.device)
                else:
                    step_tensor = torch.tensor(float(step_state), device=param.device)
                optimizer_metrics["max/optimizer_step"] = step_tensor

            step = param_optim_state["step"].item()
            grad = param.grad
            meta = param_optim_state.get("projector_meta")
            use_low_rank = meta is not None and meta.get("rank") is not None
            if grad is not None and use_low_rank:
                grad = _project(param_optim_state, grad, param_optim_state["step"])

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            denom = param_optim_state["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2) + eps
            step_size = lr
            step_tensor = param_optim_state["exp_avg"] / bias_correction1
            if self.v1 > 0.0 and grad is not None:
                step_tensor = (1.0 - self.v1) * grad + self.v1 * step_tensor
            step_tensor = step_tensor / denom
            if use_low_rank:
                step_tensor = _project_back(param_optim_state, step_tensor)
            step_tensor = step_tensor.mul(step_size)

            if weight_decay != 0:
                decay_factor = _compute_decay_factor(lr, initial_lr)
                scaling_factor = (decay_factor * weight_decay) / (1 - decay_factor * weight_decay)
                step_tensor.mul_(1 + scaling_factor).add_(param, alpha=scaling_factor)

            for metric in self.metric_functions:
                optimizer_metrics[f"{metric}/{name}"] = self.metric_functions[metric](
                    param,
                    param_optim_state,
                    step_tensor,
                )

            if use_low_rank:
                eigenvalues, eig_product = self._projector_eigenvalues(param_optim_state, param.device)
                if eigenvalues is not None:
                    for idx, eig in enumerate(eigenvalues):
                        optimizer_metrics[f"mean/projection_eigenvalue_{idx}/{name}"] = eig
                if eig_product is not None:
                    optimizer_metrics[f"mean/projection_eigenvalue_product/{name}"] = eig_product

        return optimizer_metrics


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
