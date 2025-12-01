# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for DES-LOC optimizer sync interval resolution."""

from __future__ import annotations

import pytest

from torchtitan.experiments.fl.desloc import DesLocController, StreamingDesLocController


def _stub_controller(raw_spec: int | list[int] | dict[str, int] | None) -> DesLocController:
    """Build a DesLocController instance with just the sync config primed."""
    controller = DesLocController.__new__(DesLocController)
    controller._param_fragment = type("ParamFragment", (), {"sync_every": 8})()
    controller._raw_optimizer_sync_config = raw_spec
    return controller


def _stub_streaming_controller(
    raw_spec: int | list[int] | dict[str, int] | None,
) -> StreamingDesLocController:
    """Build a StreamingDesLocController instance with just the sync config primed."""
    controller = StreamingDesLocController.__new__(StreamingDesLocController)
    controller._fragment_stride = 8
    controller._raw_optimizer_sync_config = raw_spec
    return controller


def test_desloc_broadcasts_first_and_second_moment_intervals() -> None:
    """Two-element lists should broadcast to first/second moment optimizer states."""
    controller = _stub_controller([32, 64])
    state_keys = ["exp_avg", "exp_avg_sq", "momentum_buffer"]

    intervals = controller._resolve_optimizer_sync_intervals(state_keys)

    assert intervals == [32, 64, 32]


def test_desloc_dict_aliases_momentum_buffer_to_exp_avg() -> None:
    """exp_avg mapping should apply to Muon momentum buffers via aliasing."""
    controller = _stub_controller({"exp_avg": 7, "exp_avg_sq": 11})
    state_keys = ["momentum_buffer", "exp_avg_sq"]

    intervals = controller._resolve_optimizer_sync_intervals(state_keys)

    assert intervals == [7, 11]


def test_streaming_ignores_unused_second_moment_interval() -> None:
    """Streaming controller should accept first/second moment lists when only the first exists."""
    controller = _stub_streaming_controller([5, 9])
    state_keys = ["momentum_buffer"]

    intervals = controller._resolve_optimizer_sync_intervals(state_keys)

    assert intervals == [5]


def test_broadcasting_rejects_unknown_state_keys() -> None:
    """Lists that cannot be broadcast onto discovered states should raise an error."""
    controller = _stub_controller([1, 2])

    with pytest.raises(ValueError):
        controller._resolve_optimizer_sync_intervals(["custom_state"])
