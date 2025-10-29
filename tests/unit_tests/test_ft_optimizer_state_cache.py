import torch
import pytest
from types import SimpleNamespace

from torchtitan.components import optimizer as optimizer_mod


@pytest.mark.skipif(
    not getattr(optimizer_mod, "has_torchft", False),
    reason="TorchFT is not installed; FT optimizer cache not exercised.",
)
def test_ft_optimizer_state_cache_refresh(monkeypatch):
    class StubFTOptimizer:
        def __init__(self, manager, optim_container):
            self.manager = manager
            self.optim_container = optim_container

        def step(self, *args, **kwargs):
            # When use_ft_optimizer=False this is never invoked.
            pass

        def zero_grad(self, *args, **kwargs):
            pass

    monkeypatch.setattr(optimizer_mod.ft, "Optimizer", StubFTOptimizer)

    model = torch.nn.Linear(4, 4)
    container = optimizer_mod.FTOptimizersContainer(
        [model],
        torch.optim.AdamW,
        {
            "lr": 0.01,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0,
            "fused": False,
            "foreach": False,
        },
        SimpleNamespace(),
        use_ft_optimizer=False,
    )

    state_init = container.state_dict()
    exp_key = next(k for k in state_init if k.endswith("exp_avg"))
    exp_init = state_init[exp_key].clone()

    inputs = torch.randn(2, 4)
    (model(inputs).sum()).backward()
    container.step()
    container.zero_grad()

    state_after_step = container.state_dict()
    exp_after = state_after_step[exp_key]
    assert not torch.allclose(exp_after, exp_init)
    assert container._cache_dirty is False

    # Mutate the underlying optimizer state and ensure cache invalidation works.
    inner_optimizer = container.optimizers[0]
    param = next(iter(inner_optimizer.state))
    inner_optimizer.state[param]["exp_avg"].zero_()
    container.mark_state_dirty()

    state_after_mutation = container.state_dict()
    exp_mutated = state_after_mutation[exp_key]
    assert torch.allclose(exp_mutated, torch.zeros_like(exp_mutated))
    assert container._cache_dirty is False
