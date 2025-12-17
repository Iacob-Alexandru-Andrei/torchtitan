# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""GaLore optimizer family for FL experiments."""

from __future__ import annotations

import inspect
import logging
import math
import re
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple

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
PROJ_TO_CODE: dict[str, int] = {
    STD_PROJ: 0,
    REV_STD_PROJ: 1,
    LEFT_PROJ: 2,
    RIGHT_PROJ: 3,
    FULL_PROJ: 4,
}
CODE_TO_PROJ: dict[int, str] = {code: name for name, code in PROJ_TO_CODE.items()}

ProjectionBasis = Tensor | list[Tensor]
_OPTIONAL_PROJECTOR_META_KEYS: tuple[str, ...] = ("full_rank_shape",)
_RUNTIME_PROJECTOR_STATE_KEYS: tuple[str, ...] = (
    "_bootstrap_projector",
    "_placeholder_projector",
    "_bootstrap_identity_logged",
    "_placeholder_identity_logged",
)


def _strip_optional_projector_metadata(state: Any) -> None:
    """Remove optional projector metadata fields from serialized state."""

    if not isinstance(state, dict):
        return

    for entry in state.values():
        if not isinstance(entry, dict):
            continue
        for key in _RUNTIME_PROJECTOR_STATE_KEYS:
            entry.pop(key, None)
        meta = entry.get("projector_meta")
        if not isinstance(meta, dict):
            continue
        for key in _OPTIONAL_PROJECTOR_META_KEYS:
            meta.pop(key, None)


def _strip_all_projector_state(state: Any) -> None:
    """Remove all projector-related state for checkpoint compatibility.

    This enables GaLore checkpoints to be loaded by AdamW and vice versa.
    The projector state will be rebuilt by ``_repair_projector_states``
    after checkpoint loading.
    """
    if not isinstance(state, dict):
        return

    for entry in state.values():
        if not isinstance(entry, dict):
            continue
        # Remove runtime projector keys
        for key in _RUNTIME_PROJECTOR_STATE_KEYS:
            entry.pop(key, None)
        # Remove projector metadata entirely
        entry.pop("projector_meta", None)
        # Remove projector basis - will be recomputed on first step
        entry.pop("projector_basis", None)
        # Remove error feedback - not needed for checkpoint compatibility
        entry.pop("error_feedback", None)


class _RotationContext(NamedTuple):
    beta1: float
    beta2: float


def _apply_axis_transform(tensor: Tensor, matrix: Tensor, axis: int) -> Tensor:
    if axis == -1:
        original_shape = tensor.shape
        reshaped = tensor.reshape(-1, original_shape[-1])
        rotated = reshaped @ matrix
        return rotated.reshape(original_shape)
    if axis == 0:
        original_shape = tensor.shape
        reshaped = tensor.reshape(original_shape[0], -1)
        rotated = matrix @ reshaped
        return rotated.reshape(original_shape)
    raise ValueError(f"Unsupported axis {axis} for GaLore moment rotation.")


def _rotate_moments_to_new_basis(
    state: dict[str, Any],
    *,
    old_basis: Tensor,
    new_basis: Tensor,
    proj_type: str,
    rotation_context: _RotationContext,
) -> None:
    if proj_type not in {LEFT_PROJ, RIGHT_PROJ}:
        return

    exp_avg: Tensor | None = state.get("exp_avg")
    exp_avg_sq: Tensor | None = state.get("exp_avg_sq")
    step_tensor: Tensor | None = state.get("step")
    if exp_avg is None or exp_avg_sq is None or step_tensor is None:
        return

    step_value = int(step_tensor.item())
    if step_value <= 0:
        return

    beta1_corr = 1.0 - rotation_context.beta1**step_value
    beta2_corr = 1.0 - rotation_context.beta2**step_value
    if beta1_corr <= 0.0 or beta2_corr <= 0.0:
        return

    device = exp_avg.device
    dtype = exp_avg.dtype
    old_basis_tensor = old_basis.to(device=device, dtype=dtype)
    new_basis_tensor = new_basis.to(device=device, dtype=dtype)

    old_columns = old_basis_tensor.T if proj_type == RIGHT_PROJ else old_basis_tensor
    new_columns = new_basis_tensor.T if proj_type == RIGHT_PROJ else new_basis_tensor
    transform = new_columns.transpose(-1, -2) @ old_columns
    coeff_matrix = transform.T if proj_type == RIGHT_PROJ else transform
    coeff_matrix = coeff_matrix.to(device=device, dtype=dtype)
    var_matrix = coeff_matrix.pow(2)
    axis = -1 if proj_type == RIGHT_PROJ else 0

    m_hat_old = exp_avg / beta1_corr
    v_hat_old = exp_avg_sq / beta2_corr
    var_hat_old = torch.clamp(v_hat_old - m_hat_old.pow(2), min=0.0)

    rotated_exp_avg = _apply_axis_transform(exp_avg, coeff_matrix, axis)
    m_hat_rot = _apply_axis_transform(m_hat_old, coeff_matrix, axis)
    var_hat_rot = _apply_axis_transform(var_hat_old, var_matrix, axis)
    v_hat_rot = torch.abs(var_hat_rot + m_hat_rot.pow(2))

    state["exp_avg"] = rotated_exp_avg
    state["exp_avg_sq"] = v_hat_rot * beta2_corr
    print("Finished rortating GaLore moment tensors to new basis.")


def _infer_projector_rank(
    orthogonal: Tensor | list[Tensor] | None,
    resolved_proj_type: str | None,
) -> int | None:
    if orthogonal is None:
        return None
    if isinstance(orthogonal, Tensor):
        if resolved_proj_type == LEFT_PROJ:
            return orthogonal.shape[1]
        return orthogonal.shape[0]
    if isinstance(orthogonal, list) and orthogonal:
        left_matrix = orthogonal[0]
        if isinstance(left_matrix, Tensor):
            return left_matrix.shape[1]
    return None


def _orthogonal_matrix(weights: Tensor, rank: int, proj_type: str) -> ProjectionBasis:
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


def _basis_similarity(old_basis: ProjectionBasis, new_basis: ProjectionBasis, proj_type: str) -> dict[str, float]:
    """Compute principal-angle similarity between projector bases.

    Similarity is mean(sigma^2) where sigma are singular values of the rotation
    between the two subspaces. Returns an empty dict if inputs are incompatible.
    """

    def _sim(a: Tensor, b: Tensor, left_space: bool) -> float:
        # For left bases (m x r), use Q_new^T Q_old; for right bases (r x n), use Q_new Q_old^T.
        rot = a.transpose(-2, -1) @ b if left_space else a @ b.transpose(-2, -1)
        sigma = torch.linalg.svdvals(rot)
        return float((sigma.pow(2)).mean().item())

    if proj_type == FULL_PROJ:
        if not (isinstance(old_basis, list) and isinstance(new_basis, list)):
            return {}
        if len(old_basis) != 2 or len(new_basis) != 2:
            return {}
        new_left, new_right = new_basis
        old_left, old_right = old_basis
        return {
            "left": _sim(new_left, old_left, left_space=True),
            "right": _sim(new_right, old_right, left_space=False),
        }

    if isinstance(old_basis, Tensor) and isinstance(new_basis, Tensor):
        is_left = proj_type == LEFT_PROJ
        return {"single": _sim(new_basis, old_basis, left_space=is_left)}

    return {}


def _resolve_proj_choice(proj_type: str, tensor: Tensor) -> str:
    if proj_type in {STD_PROJ, REV_STD_PROJ}:
        if tensor.shape[0] >= tensor.shape[1]:
            return RIGHT_PROJ if proj_type == STD_PROJ else LEFT_PROJ
        return LEFT_PROJ if proj_type == STD_PROJ else RIGHT_PROJ
    return proj_type


def _proj_name_from_value(value: Any, default: str = STD_PROJ) -> str:
    if isinstance(value, str) and value in PROJ_TO_CODE:
        return value
    if isinstance(value, int) and value in CODE_TO_PROJ:
        return CODE_TO_PROJ[value]
    return default


def _canonicalize_projection_tensor(tensor: Tensor) -> Tensor:
    """Ensure tensors have at least 2 dims when building projectors."""

    if tensor.ndim >= 2:
        return tensor
    if tensor.ndim == 1:
        return tensor.reshape(-1, 1)
    return tensor.reshape(1, 1)


def _maybe_refresh_projector(
    state: dict[str, Any],
    weights: Tensor,
    iteration: Tensor,
    rotation_context: _RotationContext | None = None,
) -> None:
    weights = _canonicalize_projection_tensor(weights)
    meta = state.setdefault(
        "projector_meta",
        {
            "rank": None,
            "update_proj_gap": None,
            "scale": None,
            "proj_type": PROJ_TO_CODE[STD_PROJ],
            "resolved_proj_type": PROJ_TO_CODE[STD_PROJ],
        },
    )
    rank = meta["rank"]
    update_proj_gap = meta["update_proj_gap"]
    proj_type = _proj_name_from_value(meta.get("proj_type", STD_PROJ))
    resolved_proj_type = _resolve_proj_choice(proj_type, weights)
    meta["proj_type"] = PROJ_TO_CODE[proj_type]
    meta["resolved_proj_type"] = PROJ_TO_CODE[resolved_proj_type]
    if rank is None or update_proj_gap is None:
        return

    orthogonal = state.get("projector_basis")
    should_refresh = orthogonal is None or (iteration % update_proj_gap).item() == 0
    if not should_refresh:
        return
    print("Refreshing the new-basis")
    new_basis = _orthogonal_matrix(weights, rank, resolved_proj_type)
    if orthogonal is not None:
        similarity = _basis_similarity(orthogonal, new_basis, resolved_proj_type)
        if similarity:
            # Store similarity in state for metric logging
            if "single" in similarity:
                state["basis_similarity"] = torch.tensor(
                    similarity["single"], device=weights.device, dtype=torch.float32
                )
            elif "left" in similarity and "right" in similarity:
                # For FULL_PROJ, store average of left and right similarities
                avg_sim = (similarity["left"] + similarity["right"]) / 2.0
                state["basis_similarity"] = torch.tensor(
                    avg_sim, device=weights.device, dtype=torch.float32
                )
            log.info("GaLore basis similarity (%s): %s", resolved_proj_type, similarity)
    if (
        rotation_context is not None
        and orthogonal is not None
        and isinstance(orthogonal, Tensor)
        and isinstance(new_basis, Tensor)
    ):
        print("Rotating GaLore moment tensors to new basis.")
        _rotate_moments_to_new_basis(
            state,
            old_basis=orthogonal,
            new_basis=new_basis,
            proj_type=resolved_proj_type,
            rotation_context=rotation_context,
        )
    state["projector_basis"] = new_basis


def _project(
    state: dict[str, Any],
    full_rank_grad: Tensor,
    iteration: Tensor,
    rotation_context: _RotationContext | None = None,
) -> Tensor:
    if full_rank_grad.ndim > GALORE_MAX_SUPPORT_DIM:
        raise NotImplementedError("GaLore currently supports tensors up to rank 2.")

    original_shape = tuple(full_rank_grad.shape)
    full_rank_grad = _canonicalize_projection_tensor(full_rank_grad)
    meta = state.get("projector_meta", {})
    proj_type_name = _proj_name_from_value(meta.get("proj_type", STD_PROJ))
    proj_type = _resolve_proj_choice(proj_type_name, full_rank_grad)
    meta["proj_type"] = PROJ_TO_CODE[proj_type_name]
    meta["resolved_proj_type"] = PROJ_TO_CODE[proj_type]
    if original_shape:
        meta["full_rank_shape"] = torch.tensor(
            list(original_shape),
            device=full_rank_grad.device,
            dtype=torch.int64,
        )
    state["projector_meta"] = meta
    _maybe_refresh_projector(state, full_rank_grad, iteration, rotation_context)
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
    meta = state.get("projector_meta", {})
    scale = meta.get("scale", 1.0)
    if orthogonal is None:
        return low_rank_grad * scale

    if isinstance(orthogonal, Tensor):
        matrix = orthogonal.to(low_rank_grad.device)
        if matrix.shape[0] == low_rank_grad.shape[-1]:
            restored = low_rank_grad @ matrix
        else:
            restored = matrix @ low_rank_grad
    else:
        a_matrix, b_matrix = orthogonal
        restored = a_matrix.to(low_rank_grad.device) @ low_rank_grad @ b_matrix.to(low_rank_grad.device)

    restored = restored * scale
    shape_tensor = meta.get("full_rank_shape")
    if isinstance(shape_tensor, torch.Tensor):
        desired_shape = tuple(int(x) for x in shape_tensor.tolist())
        if desired_shape and tuple(restored.shape) != desired_shape:
            restored = restored.reshape(desired_shape)
    return restored


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
        "l2_norm/error_feedback": (
            lambda _param, optim_state, _step_tensor: torch.linalg.vector_norm(
                optim_state["error_feedback"],
            )
            if "error_feedback" in optim_state
            else torch.tensor(0.0, device=_param.device)
        ),
        "mean/basis_similarity": (
            lambda _param, optim_state, _step_tensor: optim_state["basis_similarity"]
            if "basis_similarity" in optim_state
            else torch.tensor(float("nan"), device=_param.device)
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
        rank_overrides: dict[Tensor | int, int] | None = None,
        rotate_moments_on_refresh: bool = False,
        use_error_feedback: bool = False,
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
        self.use_error_feedback = use_error_feedback
        self._defaults = {
            "rank": rank,
            "update_proj_gap": update_proj_gap,
            "scale": scale,
            "proj_type": proj_type,
            "dim": dim,
            "rotate_moments_on_refresh": rotate_moments_on_refresh,
            "use_error_feedback": use_error_feedback,
        }
        overrides: dict[int, int] = {}
        if rank_overrides:
            for key, override_rank in rank_overrides.items():
                param_id = key if isinstance(key, int) else id(key)
                overrides[param_id] = override_rank
        self._param_rank_overrides = overrides
        for group in self.param_groups:
            group.setdefault("rank", rank)
            group.setdefault("update_proj_gap", update_proj_gap)
            group.setdefault("scale", scale)
            group.setdefault("proj_type", proj_type)
            group.setdefault("dim", dim)
            group.setdefault("rotate_moments_on_refresh", rotate_moments_on_refresh)
            group.setdefault("use_error_feedback", use_error_feedback)
            group["initial_lr"] = group["lr"]

        self.register_load_state_dict_post_hook(
            lambda optimizer: optimizer._repair_projector_states()  # type: ignore[attr-defined]
        )

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        serialized = super().state_dict()
        # Strip all projector state for checkpoint compatibility with AdamW
        _strip_all_projector_state(serialized.get("state"))
        return serialized

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
            base_rank = group.get("rank")
            dim = group.get("dim", GALORE_MAX_SUPPORT_DIM)
            rotation_context = None
            if group.get("rotate_moments_on_refresh", False):
                rotation_context = _RotationContext(beta1=beta1, beta2=beta2)

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("GaLore does not support sparse gradients.")

                rank = self._resolve_rank_for_param(param, base_rank)
                use_low_rank = rank is not None
                if use_low_rank and dim > GALORE_MAX_SUPPORT_DIM:
                    raise NotImplementedError("GaLore supports tensors up to 2 dimensions.")
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

                    # PowerSGD-style error feedback: add accumulated error before projection
                    use_ef = group.get("use_error_feedback", False)
                    if use_ef:
                        error_feedback = state.get("error_feedback")
                        if error_feedback is not None:
                            grad = grad + error_feedback

                    # Project gradient to low-rank subspace
                    low_rank_grad = _project(state, grad, state["step"], rotation_context)

                    # Compute reconstruction error for error feedback
                    if use_ef:
                        reconstructed = _project_back(state, low_rank_grad)
                        # Reshape reconstructed if needed to match grad shape
                        if reconstructed.shape != grad.shape:
                            reconstructed = reconstructed.reshape(grad.shape)
                        new_error = grad - reconstructed
                        state["error_feedback"] = new_error

                    grad = low_rank_grad

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

    def _repair_projector_states(self) -> None:
        """Ensure projector metadata matches the configured rank after load."""

        for group in self.param_groups:
            base_rank = group.get("rank")
            update_proj_gap = group.get("update_proj_gap")
            scale = group.get("scale")
            proj_type_name = _proj_name_from_value(group.get("proj_type", STD_PROJ))
            for param in group["params"]:
                desired_rank = self._resolve_rank_for_param(param, base_rank)
                if desired_rank is None or param not in self.state:
                    continue
                state = self.state[param]
                meta = state.setdefault(
                    "projector_meta",
                    {
                        "rank": desired_rank,
                        "update_proj_gap": update_proj_gap,
                        "scale": scale,
                        "proj_type": PROJ_TO_CODE[proj_type_name],
                        "resolved_proj_type": PROJ_TO_CODE[
                            _resolve_proj_choice(proj_type_name, param)
                        ],
                    },
                )
                meta["rank"] = desired_rank
                meta["update_proj_gap"] = update_proj_gap
                meta["scale"] = scale
                meta_proj_type = _proj_name_from_value(meta.get("proj_type", proj_type_name))
                meta["proj_type"] = PROJ_TO_CODE[meta_proj_type]
                resolved_proj_type = _resolve_proj_choice(meta_proj_type, param)
                meta["resolved_proj_type"] = PROJ_TO_CODE[resolved_proj_type]

                current_rank = _infer_projector_rank(
                    state.get("projector_basis"), resolved_proj_type
                )
                if current_rank is None or current_rank == desired_rank:
                    continue

                log.warning(
                    "Resetting projector basis for %s: checkpoint rank %s != configured rank %s.",
                    getattr(param, "_base_name", "<unnamed_param>"),
                    current_rank,
                    desired_rank,
                )
                state.pop("projector_basis", None)

    def _resolve_rank_for_param(self, param: Tensor, fallback: int | None) -> int | None:
        override = self._param_rank_overrides.get(id(param))
        if override is not None:
            return override
        return fallback


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
