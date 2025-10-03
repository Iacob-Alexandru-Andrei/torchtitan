# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Adapters for using Mosaic streaming dataloaders with TorchTitan."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterator

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger


class DataSpecDataLoader(BaseDataLoader):
    """Wrap a Composer :class:`DataSpec` so it matches TorchTitan's API."""

    def __init__(self, data_spec: Any) -> None:
        self._data_spec = data_spec

    @property
    def data_spec(self) -> Any:
        """
        Returns:
            Any: The wrapped Composer :class:`DataSpec` instance, which contains the underlying dataloader and related configuration.

        Purpose:
            Provides direct access to the underlying :class:`DataSpec` object for advanced usage or integration with Composer/MosaicML APIs.
        """
        return self._data_spec

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data_spec.dataloader)

    def state_dict(self) -> dict[str, Any]:
        dataloader = getattr(self._data_spec, "dataloader", None)
        if hasattr(dataloader, "state_dict"):
            return dataloader.state_dict()
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return

        dataloader = getattr(self._data_spec, "dataloader", None)
        if hasattr(dataloader, "load_state_dict"):
            dataloader.load_state_dict(state_dict)
        else:
            logger.warning(
                "Composer DataSpec dataloader does not support checkpoint loading; "
                "ignoring provided state with keys %s.",
                list(state_dict.keys()),
            )


def _maybe_extract_hf_tokenizer(tokenizer: BaseTokenizer | Any | None) -> Any:
    """
    Attempt to extract a HuggingFace tokenizer from the provided tokenizer object.

    Parameters
    ----------
    tokenizer : BaseTokenizer or Any or None
        The tokenizer to inspect. May be a HuggingFace tokenizer, a wrapper around one,
        or None.

    Returns
    -------
    PreTrainedTokenizerBase or None
        Returns the HuggingFace tokenizer instance if found, otherwise None.
        If the input is None or does not wrap a HuggingFace tokenizer, returns None.

    Notes
    -----
    This function attempts to lazily import HuggingFace's `PreTrainedTokenizerBase`
    to avoid a hard dependency on `transformers`. If `transformers` is not installed,
    always returns None.
    """
    if tokenizer is None:
        return None

    try:  # Import lazily so TorchTitan does not hard depend on transformers.
        from transformers import PreTrainedTokenizerBase  # type: ignore
    except Exception:  # pragma: no cover - transformers may be absent
        PreTrainedTokenizerBase = ()  # type: ignore[assignment]

    if isinstance(tokenizer, PreTrainedTokenizerBase):
        return tokenizer

    inner_tokenizer = getattr(tokenizer, "tokenizer", None)
    if isinstance(inner_tokenizer, PreTrainedTokenizerBase):
        return inner_tokenizer

    return None


def build_mosaic_dataloader(
    *,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer | None,
    job_config: JobConfig,
) -> DataSpecDataLoader:
    """Build a dataloader using MosaicML's ``build_dataloader`` helper."""

    mosaic_cfg = getattr(job_config.training, "mosaic_dataloader", None)
    if mosaic_cfg is None:
        raise ValueError(
            "job_config.training.mosaic_dataloader must be set before constructing "
            "the Trainer. Use the Mosaic example CLI to attach the configuration "
            "prior to instantiating TorchTitan's Trainer."
        )

    try:
        from llmfoundry.data.dataloader import (  # type: ignore[import]
            build_dataloader as mosaic_build_dataloader,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "llm-foundry is required to build Mosaic dataloaders. Please install "
            "llm-foundry and composer to enable this integration."
        ) from exc

    cfg = deepcopy(mosaic_cfg)
    cfg.setdefault("dp_world_size", dp_world_size)
    cfg.setdefault("dp_rank", dp_rank)

    hf_tokenizer = _maybe_extract_hf_tokenizer(tokenizer)
    if hf_tokenizer is None and tokenizer is not None:
        logger.debug(
            "TorchTitan could not extract a HuggingFace tokenizer from %s; "
            "falling back to passing None to Mosaic's build_dataloader.",
            type(tokenizer).__name__,
        )

    data_spec = mosaic_build_dataloader(
        cfg=cfg,
        tokenizer=hf_tokenizer,
        device_batch_size=job_config.training.local_batch_size,
    )

    return DataSpecDataLoader(data_spec)


__all__ = ["DataSpecDataLoader", "build_mosaic_dataloader"]
