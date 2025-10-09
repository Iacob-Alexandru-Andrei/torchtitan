#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test script to verify ActivationMonitor customization works correctly."""

import torch
from torchtitan.experiments.fl.metrics import ActivationMonitor


def test_default_metrics():
    """Test that default metrics are correctly set."""
    monitor = ActivationMonitor(interval=1)

    expected_metrics = {
        "activations/average/max/full_model_input",
        "activations/average/max/full_model_output",
        "activations/average/median/full_model_input",
        "activations/average/median/full_model_output",
        "activations/l2_norm/full_model_input",
        "activations/l2_norm/full_model_output",
        "activations/max/full_model_input",
        "activations/max/full_model_output",
    }

    assert (
        monitor.enabled_metrics == expected_metrics
    ), f"Default metrics mismatch!\nExpected: {expected_metrics}\nGot: {monitor.enabled_metrics}"

    # Test that default metrics are enabled
    for metric in expected_metrics:
        assert monitor._is_metric_enabled(
            metric
        ), f"Metric {metric} should be enabled by default"

    # Test that non-default metrics are disabled
    disabled_metrics = [
        "activations/skewness/max/full_model_input",
        "activations/kurtosis/median/full_model_output",
        "activations/average/min/full_model_input",
    ]
    for metric in disabled_metrics:
        assert not monitor._is_metric_enabled(
            metric
        ), f"Metric {metric} should be disabled by default"

    print("✓ Default metrics test passed")


def test_custom_metrics():
    """Test that custom metrics can be set."""
    custom_metrics = {
        "activations/l2_norm/full_model_input",
        "activations/skewness/max/full_model_input",
    }

    monitor = ActivationMonitor(interval=1, enabled_metrics=custom_metrics)

    assert (
        monitor.enabled_metrics == custom_metrics
    ), f"Custom metrics mismatch!\nExpected: {custom_metrics}\nGot: {monitor.enabled_metrics}"

    # Test that custom metrics are enabled
    for metric in custom_metrics:
        assert monitor._is_metric_enabled(metric), f"Metric {metric} should be enabled"

    # Test that non-custom metrics are disabled
    assert not monitor._is_metric_enabled(
        "activations/max/full_model_input"
    ), "Non-custom metric should be disabled"

    print("✓ Custom metrics test passed")


def test_metric_collection():
    """Test that only enabled metrics are collected."""
    custom_metrics = {
        "activations/l2_norm/full_model_input",
        "activations/max/full_model_input",
    }

    monitor = ActivationMonitor(interval=1, enabled_metrics=custom_metrics)
    monitor._collect_this_step = True

    # Simulate metric collection
    test_tensor = torch.randn(10, 20)
    monitor._add_metrics("_input", test_tensor)

    # Check that only enabled metrics were collected
    assert (
        "activations/l2_norm/full_model_input" in monitor._metrics
    ), "L2 norm should be collected"
    assert (
        "activations/max/full_model_input" in monitor._metrics
    ), "Max should be collected"

    # Check that disabled metrics were not collected
    assert (
        "activations/average/full_model_input" not in monitor._metrics
    ), "Average should not be collected (not enabled)"
    assert (
        "activations/skewness/full_model_input" not in monitor._metrics
    ), "Skewness should not be collected (not enabled)"

    print("✓ Metric collection test passed")


def test_empty_metrics():
    """Test that empty metrics set works."""
    monitor = ActivationMonitor(interval=1, enabled_metrics=set())

    assert len(monitor.enabled_metrics) == 0, "Metrics should be empty"
    assert not monitor._is_metric_enabled(
        "activations/l2_norm/full_model_input"
    ), "No metrics should be enabled"

    monitor._collect_this_step = True
    test_tensor = torch.randn(10, 20)
    monitor._add_metrics("_input", test_tensor)

    # No metrics should be collected
    assert (
        len(monitor._metrics) == 0
    ), "No metrics should be collected with empty enabled set"

    print("✓ Empty metrics test passed")


if __name__ == "__main__":
    print("Running ActivationMonitor customization tests...\n")
    test_default_metrics()
    test_custom_metrics()
    test_metric_collection()
    test_empty_metrics()
    print("\n✓ All tests passed!")
