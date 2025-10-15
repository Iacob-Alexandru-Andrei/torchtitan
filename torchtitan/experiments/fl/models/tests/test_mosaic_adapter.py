# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for MosaicTrainSpecAdapter behaviour."""

from __future__ import annotations

import unittest

from tests.unit_tests import test_mosaic_spec_utils as _mosaic_spec_utils  # noqa: F401

from torchtitan.experiments.fl.models.mosaic_adapter import MosaicTrainSpecAdapter
from torchtitan.experiments.fl.models.utils import MosaicSpecOverrides
from torchtitan.protocols.train_spec import (
    TrainSpec,
    get_train_spec,
    unregister_train_spec,
)


def _dummy_builder(*_args, **_kwargs):  # noqa: ANN001, ANN002 - test helper
    return None


class MosaicTrainSpecAdapterTest(unittest.TestCase):
    """Validate deterministic registration through the adapter."""

    def tearDown(self) -> None:  # noqa: D401
        """Clean up any adapter-registered specs from the global registry."""

        unregister_train_spec("test_mosaic_llama3_adapter")

    def test_build_uses_mosaic_name_by_default(self) -> None:
        """Adapter derives a stable mosaic-prefixed spec name when omitted."""

        adapter = MosaicTrainSpecAdapter("llama3")
        spec = adapter.build()
        self.assertIsInstance(spec, TrainSpec)
        self.assertEqual(spec.name, "mosaic_llama3")

    def test_register_applies_builder_overrides(self) -> None:
        """Adapter registration wires provided builders deterministically."""

        overrides = MosaicSpecOverrides(
            dataloader=_dummy_builder,
            tokenizer=_dummy_builder,
            metrics_processor=_dummy_builder,
            optimizers=_dummy_builder,
            validator=_dummy_builder,
        )
        adapter = MosaicTrainSpecAdapter(
            "llama3",
            spec_name="test_mosaic_llama3_adapter",
            overrides=overrides,
        )

        registered_spec = adapter.register()
        try:
            fetched_spec = get_train_spec("test_mosaic_llama3_adapter")
            self.assertEqual(fetched_spec.name, "test_mosaic_llama3_adapter")
            self.assertIs(fetched_spec.build_dataloader_fn, _dummy_builder)
            self.assertIs(fetched_spec.build_tokenizer_fn, _dummy_builder)
            self.assertIs(fetched_spec.build_metrics_processor_fn, _dummy_builder)
            self.assertIs(fetched_spec.build_optimizers_fn, _dummy_builder)
            self.assertIs(fetched_spec.build_validator_fn, _dummy_builder)
        finally:
            unregister_train_spec(registered_spec.name)


if __name__ == "__main__":
    unittest.main()
