"""Utilities for working with Mosaic-enabled training specs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
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
    get_train_spec,
    register_train_spec,
)
from torchtitan.tools.logging import logger


PostTransform = Callable[[TrainSpec, TrainSpec], TrainSpec]


def ensure_mosaic_spec(
    base_spec_name: str,
    *,
    spec_name: str | None = None,
    dataloader_fn: DataLoaderBuilder | None = None,
    tokenizer_fn: TokenizerBuilder | None = None,
    metrics_processor_fn: MetricsProcessorBuilder | None = None,
    optimizers_fn: OptimizersBuilder | None = None,
    validator_fn: ValidatorBuilder | None = None,
    post_transform: PostTransform | None = None,
) -> str:
    """Ensure that a Mosaic-wrapped train spec is registered.

    Args:
        base_spec_name: Name of the base train spec to wrap.
        spec_name: Optional name for the Mosaic spec (defaults to ``mosaic_<base>``).
        dataloader_fn: Optional dataloader builder override. Defaults to
            :func:`build_mosaic_dataloader`.
        tokenizer_fn: Optional tokenizer builder override. Defaults to
            :func:`build_mosaic_tokenizer`.
        metrics_processor_fn: Optional metrics processor builder override.
            Defaults to :func:`build_metrics_processor`.
        optimizers_fn: Optional optimizers builder override.
        validator_fn: Optional validator builder override.
        post_transform: Optional callable that receives the original base spec and the
            Mosaic spec produced by the helper and returns the final spec to register.

    Returns:
        The name of the Mosaic-enabled train spec.
    """

    base_spec = get_train_spec(base_spec_name)
    mosaic_spec_name = spec_name or f"mosaic_{base_spec.name}"

    try:
        get_train_spec(mosaic_spec_name)
    except ValueError:
        dataloader_builder = dataloader_fn or build_mosaic_dataloader
        tokenizer_builder = tokenizer_fn or cast("TokenizerBuilder", build_mosaic_tokenizer)
        metrics_builder = metrics_processor_fn or build_metrics_processor

        replace_kwargs: dict[str, object] = {
            "name": mosaic_spec_name,
            "build_dataloader_fn": dataloader_builder,
            "build_tokenizer_fn": tokenizer_builder,
            "build_metrics_processor_fn": metrics_builder,
        }

        if optimizers_fn is not None:
            replace_kwargs["build_optimizers_fn"] = optimizers_fn
        if validator_fn is not None:
            replace_kwargs["build_validator_fn"] = validator_fn

        mosaic_spec = replace(base_spec, **replace_kwargs)

        if post_transform is not None:
            mosaic_spec = post_transform(base_spec, mosaic_spec)
            if mosaic_spec.name != mosaic_spec_name:
                mosaic_spec = replace(mosaic_spec, name=mosaic_spec_name)

        register_train_spec(mosaic_spec)
        logger.info(f"Registered new TrainSpec: {mosaic_spec_name}")
    else:
        logger.info(f"TrainSpec {mosaic_spec_name} already registered, reusing it")

    return mosaic_spec_name


__all__ = ["ensure_mosaic_spec"]

