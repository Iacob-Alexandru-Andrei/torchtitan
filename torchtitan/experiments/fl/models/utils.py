# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Utilities for working with Mosaic-enabled training specs."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import cast

from torchtitan.experiments.fl.components import build_metrics_processor
from torchtitan.experiments.fl.dataloader.dataloader import build_mosaic_dataloader
from torchtitan.experiments.fl.dataloader.tokenizer import build_mosaic_tokenizer
from torchtitan.protocols.train_spec import (
    DataLoaderBuilder,
    MetricsProcessorBuilder,
    OptimizersBuilder,
    TokenizerBuilder,
    TrainSpec,
    ValidatorBuilder,
)


PostTransform = Callable[[TrainSpec, TrainSpec], TrainSpec]
"""Callable applied after constructing the Mosaic spec."""


@dataclass(frozen=True)
class MosaicSpecOverrides:
    """Optional overrides for Mosaic-enabled train specs."""

    dataloader: DataLoaderBuilder | None = None
    tokenizer: TokenizerBuilder | None = None
    metrics_processor: MetricsProcessorBuilder | None = None
    optimizers: OptimizersBuilder | None = None
    validator: ValidatorBuilder | None = None
    post_transform: PostTransform | None = None


def build_mosaic_spec(
    base_spec: TrainSpec,
    *,
    spec_name: str,
    overrides: MosaicSpecOverrides | None = None,
) -> TrainSpec:
    overrides = overrides or MosaicSpecOverrides()

    dataloader_builder = overrides.dataloader or partial(
        build_mosaic_dataloader,
    )
    tokenizer_builder = overrides.tokenizer or cast(
        "TokenizerBuilder", build_mosaic_tokenizer
    )
    metrics_builder = overrides.metrics_processor or build_metrics_processor

    replace_kwargs: dict[str, object] = {
        "name": spec_name,
        "build_dataloader_fn": dataloader_builder,
        "build_tokenizer_fn": tokenizer_builder,
        "build_metrics_processor_fn": metrics_builder,
    }

    if overrides.optimizers is not None:
        replace_kwargs["build_optimizers_fn"] = overrides.optimizers
    if overrides.validator is not None:
        replace_kwargs["build_validator_fn"] = overrides.validator

    mosaic_spec = replace(base_spec, **replace_kwargs)

    if overrides.post_transform is not None:
        mosaic_spec = overrides.post_transform(base_spec, mosaic_spec)
        if mosaic_spec.name != spec_name:
            mosaic_spec = replace(mosaic_spec, name=spec_name)

    return mosaic_spec


def ensure_mosaic_spec(
    base_spec_name: str,
    *,
    spec_name: str | None = None,
    overrides: MosaicSpecOverrides | None = None,
) -> str:
    """Deprecated wrapper around :class:`MosaicTrainSpecAdapter` registration."""
    warnings.warn(
        "ensure_mosaic_spec is deprecated; use MosaicTrainSpecAdapter instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    from .mosaic_adapter import MosaicTrainSpecAdapter

    adapter = MosaicTrainSpecAdapter(
        base_spec_name,
        spec_name=spec_name,
        overrides=overrides,
    )
    return adapter.register().name


__all__ = [
    "MosaicSpecOverrides",
    "PostTransform",
    "build_mosaic_spec",
    "ensure_mosaic_spec",
]
