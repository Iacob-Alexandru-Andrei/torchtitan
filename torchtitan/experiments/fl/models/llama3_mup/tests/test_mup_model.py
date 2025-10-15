# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests covering the MuP-enabled LLaMA3 model variant."""

from dataclasses import replace as dataclass_replace
import importlib
import unittest

import pytest

_schedule_module = pytest.importorskip(
    "torch.distributed.pipelining.schedules",
    reason="MuP tests require torch.distributed.pipelining",
)
if not hasattr(_schedule_module, "ScheduleDualPipeV"):
    pytest.skip("MuP tests require ScheduleDualPipeV support")

import torch
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.fl.configs.optimizers import DesLocConfig, MosaicOptimizerConfig
from torchtitan.experiments.fl.optimizer_builder import build_mosaic_optimizers

try:  # pragma: no cover - optional dependencies may be unavailable in CI
    from torchtitan.experiments.fl.models.llama3_mup.model.mup_args import (
        TransformerModelArgs,
    )
    from torchtitan.experiments.fl.models.llama3_mup.model.mup_model import Transformer
except ImportError as exc:  # pragma: no cover - skip when PyTorch lacks pipeline support
    pytest.skip(f"MuP tests require pipeline schedules: {exc}")


class TestMuPLlamaModel(unittest.TestCase):
    """Validate core behaviours of the MuP LLaMA transformer."""

    def setUp(self) -> None:
        """Instantiate a transformer with MuP configuration for reuse."""
        self.mup_config = {
            "mup_enabled": True,
            "mup_width_multiplier": 2.0,
            "mup_input_alpha": 2.0,
            "mup_output_alpha": 2.0,
            "completep_depth_alpha_enabled": True,
            "completep_depth_multiplier": 2.0,
            "completep_depth_alpha_exp": 0.5,
        }
        self.init_config = {"init_std": 0.02, "emb_init_std": 0.01}
        self.model_args = TransformerModelArgs(
            dim=128,
            n_layers=2,
            n_heads=4,
            vocab_size=1000,
            max_seq_len=256,
            use_embedding_norm=True,
            use_peri_norm=True,
            mup_config=self.mup_config,
            init_config=self.init_config,
        )
        self.model = Transformer(self.model_args)

    def _get_expected_mup_eps(self, base_eps: float) -> float:
        """Calculate the expected epsilon after MuP scaling."""

        expected_eps = base_eps * (1 / self.mup_config["mup_width_multiplier"])
        expected_eps *= self.mup_config["completep_depth_multiplier"] ** (
            -1.0 * self.mup_config["completep_depth_alpha_exp"]
        )
        return expected_eps

    def test_model_initialization(self) -> None:
        """Ensure peri-norm layers and embedding norms are constructed."""
        assert isinstance(self.model, Transformer)
        # Check if peri_norm layers are created
        for layer in self.model.layers.values():
            assert hasattr(layer, "post_attn_norm")
            assert hasattr(layer, "post_ffn_norm")
        # Check if embedding norm is created
        assert self.model.tok_embeddings.norm is not None

    def test_forward_pass(self) -> None:
        """Verify that forward pass returns logits of the expected shape."""
        input_ids = torch.randint(0, self.model_args.vocab_size, (2, 128))
        output = self.model.forward(input_ids)
        assert output.shape == (2, 128, self.model_args.vocab_size)

    def test_weight_initialization(self) -> None:
        """Ensure weight initialization completes without raising errors."""
        # A simple check to ensure no errors during init
        self.model.init_weights()

    def test_optimizer_overrides_build_param_groups(self) -> None:
        """MuP override hook should return parameter groups and epsilon scaling."""

        base_eps = 1e-8
        overrides = self.model.build_mup_optimizer_overrides(
            lr=0.01,
            eps=base_eps,
            weight_decay=0.1,
        )

        assert overrides is not None
        assert overrides.param_groups is not None
        assert len(overrides.param_groups) > 1
        self.assertIn("eps", overrides.config_updates)

        expected_eps = self._get_expected_mup_eps(base_eps)
        self.assertAlmostEqual(overrides.config_updates["eps"], expected_eps, places=12)

    def test_optimizer_overrides_disabled_when_hidden_scaling_off(self) -> None:
        """The override protocol should opt-out when hidden scaling is disabled."""

        disabled_args = dataclass_replace(
            self.model_args,
            mup_config={**self.mup_config, "mup_disable_hidden_lr_scaling": True},
        )
        disabled_args.__post_init__()
        model = Transformer(disabled_args)
        overrides = model.build_mup_optimizer_overrides(
            lr=0.01,
            eps=1e-8,
            weight_decay=0.1,
        )
        assert overrides is None

    def test_mosaic_builder_integrates_mup_overrides(self) -> None:
        """build_mosaic_optimizers should consume the MuP override protocol."""

        config = MosaicOptimizerConfig(
            name="AdamW",
            lr=0.01,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=0.1,
            implementation="for-loop",
        )
        dims = ParallelDims(1, -1, 1, 1, 1, 1, 1, world_size=1)

        container = build_mosaic_optimizers([self.model], config, dims)
        optimizer = next(iter(container))

        # Original config remains untouched and the optimizer picks up MuP epsilon scaling.
        assert config.eps == 1e-8
        assert len(optimizer.param_groups) > 1

        expected_eps = self._get_expected_mup_eps(1e-8)
        self.assertAlmostEqual(optimizer.defaults["eps"], expected_eps, places=12)

    def test_mosaic_builder_desloc_requires_ft(self) -> None:
        """DES-LOC validation should still trigger when overrides are present."""

        config = MosaicOptimizerConfig(
            name="AdamW",
            lr=0.01,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=0.1,
            implementation="for-loop",
            desloc=DesLocConfig(enabled=True),
        )
        dims = ParallelDims(1, -1, 1, 1, 1, 1, 1, world_size=1)

        with self.assertRaisesRegex(ValueError, "DES-LOC requires TorchFT"):
            build_mosaic_optimizers([self.model], config, dims)


if __name__ == "__main__":
    unittest.main()
