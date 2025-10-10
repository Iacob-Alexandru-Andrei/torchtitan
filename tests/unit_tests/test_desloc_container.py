import gc
import types
import weakref

import pytest
import torch.nn as nn
import torch.optim as optim

from tests.utils.module_loading import load_module_from_path


_DESLOC_MODULE, _cleanup_desloc_module = load_module_from_path(
    "torchtitan.experiments.fl.desloc", "torchtitan/experiments/fl/desloc.py"
)

DesLocFTOptimizersContainer = _DESLOC_MODULE.DesLocFTOptimizersContainer
get_desloc_activator = _DESLOC_MODULE.get_desloc_activator
register_desloc_activator = _DESLOC_MODULE.register_desloc_activator


@pytest.fixture(scope="module", autouse=True)
def _cleanup_desloc_import(request):
    request.addfinalizer(_cleanup_desloc_module)


class _FakeFtOptimizer:
    def __init__(self, manager, container):
        self.manager = manager
        self.container = container

    def step(self, *args, **kwargs):
        return None

    def zero_grad(self, *args, **kwargs):
        return None


@pytest.fixture(autouse=True)
def _patch_ft(monkeypatch):
    """Patch TorchFT dependencies with lightweight stubs for the tests."""

    import torchtitan.components.optimizer as optimizer_module

    monkeypatch.setattr(
        optimizer_module,
        "ft",
        types.SimpleNamespace(Optimizer=_FakeFtOptimizer),
        raising=False,
    )
    yield


class _FakeHandle:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _build_test_container(builder) -> tuple[DesLocFTOptimizersContainer, _FakeHandle]:
    previous = get_desloc_activator()
    register_desloc_activator(builder)
    model = nn.Linear(2, 2)
    container = DesLocFTOptimizersContainer(
        [model],
        optim.SGD,
        {"lr": 0.1},
        ft_manager=object(),
        use_ft_optimizer=False,
    )
    register_desloc_activator(previous)
    return container, builder.handles[-1]


def test_desloc_hooks_closed_when_container_closed():
    handles: list[_FakeHandle] = []

    def builder(model_parts, optimizers):
        handle = _FakeHandle()
        handles.append(handle)
        return handle

    builder.handles = handles

    container, handle = _build_test_container(builder)
    assert not handle.closed

    container.close()
    assert handle.closed

    # Idempotent close
    container.close()


def test_desloc_hooks_closed_on_garbage_collection():
    handles: list[_FakeHandle] = []

    def builder(model_parts, optimizers):
        handle = _FakeHandle()
        handles.append(handle)
        return handle

    builder.handles = handles

    previous = get_desloc_activator()
    register_desloc_activator(builder)
    model = nn.Linear(2, 2)
    container = DesLocFTOptimizersContainer(
        [model],
        optim.SGD,
        {"lr": 0.1},
        ft_manager=object(),
        use_ft_optimizer=False,
    )
    handle = handles[-1]
    ref = weakref.ref(container)

    del container
    gc.collect()

    assert ref() is None
    assert handle.closed
    register_desloc_activator(previous)


def test_reinitialising_optimizer_detaches_previous_desloc_handles():
    handles: list[_FakeHandle] = []

    def builder(model_parts, optimizers):
        handle = _FakeHandle()
        handles.append(handle)
        return handle

    builder.handles = handles

    previous = get_desloc_activator()
    register_desloc_activator(builder)
    model = nn.Linear(2, 2)
    container = DesLocFTOptimizersContainer(
        [model],
        optim.SGD,
        {"lr": 0.1},
        ft_manager=object(),
        use_ft_optimizer=False,
    )
    first_handle = handles[-1]

    # Reinitialise and ensure the previous handle has been torn down.
    del container
    gc.collect()
    model2 = nn.Linear(2, 2)
    new_container = DesLocFTOptimizersContainer(
        [model2],
        optim.SGD,
        {"lr": 0.1},
        ft_manager=object(),
        use_ft_optimizer=False,
    )
    gc.collect()
    assert first_handle.closed

    # Clean up
    new_container.close()
    register_desloc_activator(previous)
