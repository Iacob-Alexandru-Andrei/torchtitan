# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers for integrating MosaicML streaming dataloaders and optimizers."""

from __future__ import annotations

import copy
import os
from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

import torch

try:  # pragma: no cover - Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older versions
    import tomli as tomllib

from torch.optim.optimizer import Optimizer

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger

__all__ = [
    "HasGetParamGroups",
    "build_mosaic_dataloader",
    "build_optimizer",
    "resolve_mosaic_tokenizer",
]


def _import_llmfoundry_build_dataloader():
    try:
        from llmfoundry.data.dataloader import build_dataloader as llm_build
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "llm-foundry is required to use Mosaic streaming dataloaders. "
            "Install it with `pip install mosaicml-llm-foundry`."
        ) from exc
    return llm_build


def _import_llmfoundry_optimizer_utils():
    try:
        from llmfoundry import registry
        from llmfoundry.utils.builders import _extract_param_groups
        from llmfoundry.utils.registry_utils import construct_from_registry
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "llm-foundry is required to build optimizers with custom parameter groups."
        ) from exc
    return registry, _extract_param_groups, construct_from_registry


def _import_hf_auto_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "transformers is required to build Mosaic streaming dataloaders."
        ) from exc
    return AutoTokenizer


def _load_toml(path: str) -> dict[str, Any]:
    with open(os.path.expanduser(path), "rb") as handle:
        return tomllib.load(handle)


class ComposerDataSpecLoader(BaseDataLoader):
    """Adapter that exposes a Composer ``DataSpec`` as a TorchTitan dataloader."""

    def __init__(self, data_spec: Any) -> None:
        self._data_spec = data_spec

    def __iter__(self):  # pragma: no cover - iterator is runtime exercised
        dataloader = getattr(self._data_spec, "dataloader", self._data_spec)
        batch_transform = getattr(self._data_spec, "batch_transforms", lambda x: x)
        for batch in dataloader:
            yield batch_transform(batch)

    def state_dict(self) -> dict[str, Any]:
        dataloader = getattr(self._data_spec, "dataloader", None)
        if hasattr(dataloader, "state_dict"):
            return dataloader.state_dict()
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        dataloader = getattr(self._data_spec, "dataloader", None)
        if hasattr(dataloader, "load_state_dict"):
            dataloader.load_state_dict(state_dict)


def resolve_mosaic_tokenizer(job_config: JobConfig, tokenizer: Any | None) -> Any:
    """Resolve a ``transformers`` tokenizer for Mosaic streaming datasets."""

    mosaic_cfg = getattr(job_config, "mosaic", None)
    if mosaic_cfg is None:
        raise ValueError("JobConfig is missing mosaic configuration")

    if tokenizer is not None:
        if hasattr(tokenizer, "hf_tokenizer"):
            return tokenizer.hf_tokenizer
        if hasattr(tokenizer, "tokenizer"):
            return tokenizer.tokenizer
        if hasattr(tokenizer, "tokenizer_path") and mosaic_cfg.tokenizer_name is None:
            tokenizer_path = tokenizer.tokenizer_path
        else:
            tokenizer_path = None
    else:
        tokenizer_path = None

    tokenizer_path = (
        mosaic_cfg.tokenizer_name
        or tokenizer_path
        or job_config.model.hf_assets_path
    )

    tokenizer_kwargs = mosaic_cfg.tokenizer_kwargs or {}
    auto_tokenizer = _import_hf_auto_tokenizer()
    logger.info("Loading Mosaic tokenizer from %s", tokenizer_path)
    return auto_tokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)


def build_mosaic_dataloader(
    *,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: Any | None,
    job_config: JobConfig,
) -> BaseDataLoader:
    """Build a TorchTitan dataloader backed by Mosaic streaming datasets."""

    mosaic_cfg = getattr(job_config, "mosaic", None)
    if mosaic_cfg is None or not mosaic_cfg.enable:
        raise ValueError("Mosaic dataloader requested but mosaic.enable is False")

    config: dict[str, Any] = {}
    if mosaic_cfg.dataloader_config_path:
        file_cfg = _load_toml(mosaic_cfg.dataloader_config_path)
        if isinstance(file_cfg, dict):
            config_section = file_cfg.get("train_dataloader")
            if isinstance(config_section, dict):
                config.update(config_section)
            else:
                config.update(file_cfg)

    if mosaic_cfg.dataloader:
        config.update(mosaic_cfg.dataloader)

    if "name" not in config:
        raise ValueError(
            "Mosaic dataloader configuration must include a 'name' entry registered in llm-foundry"
        )

    config.setdefault("dp_world_size", dp_world_size)
    config.setdefault("dp_rank", dp_rank)

    device_batch_size = (
        mosaic_cfg.device_batch_size or job_config.training.local_batch_size
    )

    hf_tokenizer = resolve_mosaic_tokenizer(job_config, tokenizer)
    llm_build = _import_llmfoundry_build_dataloader()
    data_spec = llm_build(copy.deepcopy(config), hf_tokenizer, device_batch_size)
    return ComposerDataSpecLoader(data_spec)


@runtime_checkable
class HasGetParamGroups(Protocol):
    """Protocol for models that can provide their optimizer parameter groups."""

    def get_optimizer_param_groups(
        self,
        optimizer_config: dict[str, Any],
    ) -> tuple[Iterable[torch.Tensor] | Iterable[dict[str, Any]], dict[str, Any]]:
        """Return the optimizer parameter groups and any mutated optimizer config."""


def build_optimizer(
    model: torch.nn.Module,
    name: str,
    optimizer_config: dict[str, Any],
) -> Optimizer:
    """Build an llm-foundry optimizer respecting custom parameter groups."""

    if not isinstance(model, HasGetParamGroups):
        from llmfoundry.utils.builders import build_optimizer as llmfoundry_build_optimizer

        return llmfoundry_build_optimizer(
            model=model,
            name=name,
            optimizer_config=optimizer_config,
        )

    registry, extract_param_groups, construct_from_registry = (
        _import_llmfoundry_optimizer_utils()
    )

    # Ensure disable_grad and other transformations are applied consistently.
    extract_param_groups(model, optimizer_config)
    params, new_optim_config = model.get_optimizer_param_groups(optimizer_config)
    kwargs = {**new_optim_config}

    if "params" in kwargs:
        raise ValueError(
            "The `params` will be automatically extracted from the model and optimizer config. "
            "Please remove it from the optimizer config kwargs."
        )

    kwargs["params"] = params
    return construct_from_registry(
        name=name,
        registry=registry.optimizers,
        partial_function=True,
        pre_validation_function=Optimizer,
        post_validation_function=None,
        kwargs=kwargs,
    )
