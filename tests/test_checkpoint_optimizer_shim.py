from unittest.mock import Mock

import torch

from torchtitan.components.checkpoint import (
    _OptimizerStateLoadShim,
    _optimizer_requires_projector_basis,
)
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.experiments.fl.optimizers.galore_global import GaLoreGlobal


def _sample_state() -> dict[str, object]:
    return {
        "optimizer.param_groups.0.initial_projector": "identity",
        "optimizer.param_groups.0.initial_projector_mode": 1,
        "optimizer.param_groups.0.lr": 0.016,
        "optimizer.state.layers.0.projector_meta.rank": 4,
        "optimizer.state.layers.0.projector_basis": "dummy",
        "optimizer.state.layers.0.exp_avg": 1.23,
    }


def test_optimizer_state_load_shim_passthrough_by_default() -> None:
    mock_optimizer = Mock(spec=OptimizersContainer)
    sample = _sample_state()
    mock_optimizer.state_dict.return_value = sample

    shim = _OptimizerStateLoadShim(mock_optimizer)
    assert shim.state_dict() is sample

    payload = {"optimizer.param_groups.0.lr": 0.02}
    shim.load_state_dict(payload)
    mock_optimizer.load_state_dict.assert_called_once_with(payload)


def test_optimizer_state_load_shim_filters_and_restores_projector_state() -> None:
    mock_optimizer = Mock(spec=OptimizersContainer)
    mock_optimizer.state_dict.return_value = _sample_state()

    shim = _OptimizerStateLoadShim(mock_optimizer, drop_projector_state=True)
    filtered = shim.state_dict()

    assert "optimizer.param_groups.0.lr" in filtered
    assert "optimizer.state.layers.0.exp_avg" in filtered
    assert all("initial_projector" not in key for key in filtered)
    assert all("projector_meta" not in key for key in filtered)
    assert all("projector_basis" not in key for key in filtered)

    shim.load_state_dict({"optimizer.param_groups.0.lr": 0.02})
    mock_optimizer.load_state_dict.assert_called_once()
    patched_payload = mock_optimizer.load_state_dict.call_args.args[0]
    assert patched_payload["optimizer.param_groups.0.initial_projector"] == "identity"
    assert patched_payload["optimizer.param_groups.0.initial_projector_mode"] == 1
    assert patched_payload["optimizer.state.layers.0.projector_meta.rank"] == 4
    assert patched_payload["optimizer.state.layers.0.projector_basis"] == "dummy"


def test_optimizer_state_load_shim_drops_select_tokens() -> None:
    mock_optimizer = Mock(spec=OptimizersContainer)
    mock_optimizer.state_dict.return_value = _sample_state()

    shim = _OptimizerStateLoadShim(
        mock_optimizer,
        drop_projector_tokens=("initial_projector_mode",),
    )
    filtered = shim.state_dict()

    assert "optimizer.param_groups.0.initial_projector" in filtered
    assert "optimizer.param_groups.0.initial_projector_mode" not in filtered

    shim.load_state_dict({"optimizer.param_groups.0.lr": 0.02})
    mock_optimizer.load_state_dict.assert_called_once()
    patched_payload = mock_optimizer.load_state_dict.call_args.args[0]
    assert patched_payload["optimizer.param_groups.0.initial_projector_mode"] == 1


def test_optimizer_requires_projector_basis_detects_galore_global() -> None:
    param = torch.nn.Parameter(torch.zeros(2, 2))
    galore = GaLoreGlobal([param], rank=1)
    mock_container = Mock(spec=OptimizersContainer)
    mock_container.optimizers = [galore]

    assert _optimizer_requires_projector_basis(mock_container) is True


def test_optimizer_requires_projector_basis_defaults_to_false() -> None:
    param = torch.nn.Parameter(torch.zeros(2, 2))
    adamw = torch.optim.AdamW([param])
    mock_container = Mock(spec=OptimizersContainer)
    mock_container.optimizers = [adamw]

    assert _optimizer_requires_projector_basis(mock_container) is False


def test_optimizer_state_load_shim_warms_projector_states() -> None:
    param = torch.nn.Parameter(torch.zeros(2, 2))

    class DummyGaLoreGlobal(GaLoreGlobal):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.repair_calls = 0

        def _repair_projector_states(self) -> None:  # type: ignore[override]
            self.repair_calls += 1

    dummy = DummyGaLoreGlobal([param], rank=1)
    mock_container = Mock(spec=OptimizersContainer)
    mock_container.optimizers = [dummy]

    shim = _OptimizerStateLoadShim(
        mock_container,
        requires_projector_state=True,
    )
    shim.maybe_warm_projector_state()
    assert dummy.repair_calls == 1
