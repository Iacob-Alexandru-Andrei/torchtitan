# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mosaic streaming dataloader integration for TorchTitan FL experiments."""

from __future__ import annotations

import ast
import inspect
import json
import os
from contextlib import suppress
import posixpath
import shutil
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtitan.experiments.fl.metrics import (
    PureUnigramCrossEntropy,
    UnigramMetricHandle,
    UnigramMetricManager,
    get_or_create_unigram_manager,
)

try:
    from llmfoundry.data.text_data import StreamingTextDataset
    from streaming import Stream, StreamingDataset
except ImportError as exc:  # pragma: no cover - optional dependency
    msg = (
        "llm-foundry and streaming are required to build Mosaic dataloaders. "
        "Please install llm-foundry, mosaicml-streaming, and composer to enable this integration."
    )
    raise RuntimeError(msg) from exc

try:
    from streaming.base.util import clean_stale_shared_memory
except ImportError:  # pragma: no cover - optional dependency
    clean_stale_shared_memory = None  # type: ignore[assignment]

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.experiments.fl.s3_checkpoint import (
    create_remote_up_down,
    download_file_from_s3,
)
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from torchtitan.components.tokenizer import BaseTokenizer
    from torchtitan.experiments.fl.configs.config import (
        MosaicJobConfig,
        UnigramMetricConfig,
    )
    from torchtitan.experiments.fl.s3_checkpoint import RemoteUploaderDownloader

_SHM_CLEANED: bool = False


@dataclass(frozen=True)
class MosaicRuntimeConfig:
    """Runtime settings for the Mosaic dataloader workers."""

    num_workers: int
    prefetch_factor: int | None
    pin_memory: bool
    persistent_workers: bool
    drop_last: bool
    batch_size: int


@dataclass(frozen=True)
class NormalizedMosaicConfig:
    """Normalized configuration payload passed between dataloader helpers."""

    dataset_config: dict[str, Any]
    runtime: MosaicRuntimeConfig


@dataclass(frozen=True)
class StreamExtractionResult:
    """Container describing the resolved Mosaic streams and metadata."""

    streams: list[Stream] | None
    dataset_config: dict[str, Any]
    sampling_group_indices: list[list[int]] | None
    dataset_root_remote: str | None
    dataset_split_remote: str | None


@dataclass(frozen=True)
class StreamAssignment:
    """Stream subset assigned to the current rank after sampling-group selection."""

    streams: list[Stream] | None
    group_index: int | None
    dataset_root_remote: str | None
    dataset_split_remote: str | None


@dataclass(frozen=True)
class DatasetFactoryConfig:
    """Keyword arguments used to instantiate the StatefulStreamingTextDataset."""

    kwargs: dict[str, Any]


@dataclass(frozen=True)
class UnigramSetupResult:
    """Result of configuring unigram metrics for a stream subset."""

    collate_fn: Callable
    group_key: str | None = None
    handle: UnigramMetricHandle | None = None


@dataclass(frozen=True)
class ParallelDataLoaderRequest:
    """Parameters required to construct a :class:`MosaicParallelAwareDataloader`."""

    dp_rank: int
    dp_world_size: int
    runtime: MosaicRuntimeConfig
    collate_fn: Callable | None = None
    group_key: str | None = None
    unigram_handle: UnigramMetricHandle | None = None


@dataclass(frozen=True)
class DataloaderBuildRequest:
    """Input parameters required to build a Mosaic dataloader."""

    job_config: MosaicJobConfig
    tokenizer: BaseTokenizer
    dp_world_size: int
    dp_rank: int
    split: str
    default_drop_last: bool


@dataclass(frozen=True)
class UnigramMetricContext:
    """Aggregated context used to build unigram metrics."""

    streams: list[Stream] | None
    default_split: str
    tokenizer: BaseTokenizer
    config: UnigramMetricConfig
    group_key: str
    dataset_root_remote: str | None
    dataset_split_remote: str | None


def _is_uri(path: str | None) -> bool:
    return bool(path and "://" in path)


def _join_remote_path(root: str | None, path: str | None) -> str | None:
    if path is None or _is_uri(path):
        return path
    if root is None:
        return path
    if _is_uri(root):
        return f"{root.rstrip('/')}/{path.lstrip('/')}"
    return posixpath.join(root, path)


def _join_local_path(root: str | None, path: str | None) -> str | None:
    """Join ``root`` and ``path`` using :class:`pathlib.Path` semantics."""
    if path is None or root is None:
        return path

    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)

    return str(Path(root) / candidate)


def _flatten_stream_configs(streams_cfg: Any) -> dict[str, dict[str, Any]]:
    """Flatten nested stream configurations into a simple mapping."""
    flattened: dict[str, dict[str, Any]] = {}

    def _collect(config: Any, parent_key: str | None = None) -> None:
        if isinstance(config, Mapping):
            if "remote" in config or "local" in config:
                flattened_key = (
                    config.get("name") or parent_key or f"stream_{len(flattened)}"
                )
                flattened[flattened_key] = dict(config)
                flattened[flattened_key].setdefault("name", flattened_key)
                return

            for key, value in config.items():
                if key == "client_streams":
                    _collect(value)
                elif isinstance(value, Mapping) and (
                    "remote" in value or "local" in value
                ):
                    flattened[key] = dict(value)
                    flattened[key].setdefault("name", key)
                else:
                    _collect(value, key)
        elif isinstance(config, (list, tuple)):
            for item in config:
                _collect(item)

    _collect(streams_cfg)
    return flattened


def _normalize_sampling_groups(config: Any) -> list[dict[str, Any]]:
    """Normalize sampling group definitions into a list of mappings."""
    if not config:
        return []

    normalized: list[dict[str, Any]] = []
    if isinstance(config, Mapping):
        iterator = config.items()
    elif isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
        iterator = enumerate(config)
    else:
        msg = "sampling_groups must be a mapping or a sequence of mappings"
        raise TypeError(msg)

    for key, value in iterator:
        if value is None:
            continue
        if not isinstance(value, Mapping):
            msg = "Each sampling group entry must be a mapping"
            raise TypeError(msg)
        group = deepcopy(dict(value))
        default_name = str(key) if not isinstance(key, int) else f"group_{key}"
        group["name"] = str(group.get("name") or default_name)
        normalized.append(group)
    return normalized


def _collect_group_stream_entries(group: Mapping[str, Any]) -> list[Any]:
    """Extract the raw stream entries referenced by a sampling group."""
    raw_streams = group.get("streams") or group.get("client_streams")
    if raw_streams is None:
        msg = f"Sampling group '{group.get('name')}' must define a 'streams' section"
        raise ValueError(msg)

    if isinstance(raw_streams, Mapping):
        flattened = _flatten_stream_configs(raw_streams)
        return list(flattened.values())
    if isinstance(raw_streams, Sequence) and not isinstance(raw_streams, (str, bytes)):
        return list(raw_streams)
    if isinstance(raw_streams, str):
        return [raw_streams]
    msg = "Sampling group stream entries must be mappings, sequences, or strings"
    raise TypeError(msg)


def _normalize_sampling_mode(mode_raw: Any) -> str:
    """Normalize the sampling group mode into a lowercase string."""
    return str(mode_raw or "grouped").lower()


def _should_concat_sampling_groups(mode: str, sampling_groups_cfg: Any) -> bool:
    """Return ``True`` when sampling groups should be concatenated."""
    return bool(sampling_groups_cfg) and mode in {
        "concat",
        "concatenate",
        "merged",
        "merge",
        "combined",
        "unified",
        "all",
    }


def _resolve_group_candidate(
    entry: Any,
    *,
    flattened: dict[str, dict[str, Any]],
    group: Mapping[str, Any],
    entry_index: int,
) -> dict[str, Any]:
    """Resolve a sampling group entry into a concrete stream mapping."""
    if isinstance(entry, str):
        if entry not in flattened:
            msg = (
                f"Sampling group '{group.get('name')}' references unknown stream '{entry}'"
            )
            raise KeyError(msg)
        candidate = deepcopy(flattened[entry])
        candidate.setdefault("name", entry)
        return candidate

    if isinstance(entry, Mapping):
        entry_dict = dict(entry)
        name = entry_dict.get("name")
        base = entry_dict.get("base")
        base_cfg: dict[str, Any] = {}
        if base and base in flattened:
            base_cfg = deepcopy(flattened[base])
        elif name and name in flattened:
            base_cfg = deepcopy(flattened[name])
        candidate = {
            **base_cfg,
            **{k: v for k, v in entry_dict.items() if k != "base"},
        }
        if name:
            candidate.setdefault("name", name)
        else:
            candidate.setdefault("name", f"{group.get('name')}_stream_{entry_index}")
        return candidate

    msg = "Sampling group stream entries must be mappings, sequences, or strings"
    raise TypeError(msg)


def _aggregate_sampling_groups(
    flattened: dict[str, dict[str, Any]],
    sampling_groups_cfg: Any,
    mode_raw: Any,
) -> dict[str, dict[str, Any]]:
    """Concatenate sampling groups into a unified stream catalog."""
    sampling_groups = _normalize_sampling_groups(sampling_groups_cfg)
    aggregated: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for group in sampling_groups:
        entries = _collect_group_stream_entries(group)
        for index, entry in enumerate(entries):
            candidate = _resolve_group_candidate(
                entry,
                flattened=flattened,
                group=group,
                entry_index=index,
            )
            candidate.setdefault("name", f"stream_{len(aggregated)}")
            identifier = (
                candidate.get("name"),
                candidate.get("local"),
                candidate.get("remote"),
                candidate.get("split"),
            )
            if identifier in seen:
                continue
            seen.add(identifier)
            aggregated.append(candidate)

    if aggregated:
        logger.info(
            "Concatenating %s sampling group(s) into a unified stream list (%s entries total).",
            len(sampling_groups),
            len(aggregated),
        )
        return {cfg["name"]: cfg for cfg in aggregated}

    logger.warning(
        "sampling_groups_mode was set to '%s' but no streams were resolved; falling back to the original stream catalog.",
        mode_raw,
    )
    return flattened


def _resolve_group_stream_names(
    flattened: dict[str, dict[str, Any]], sampling_groups_cfg: Any
) -> list[list[str]] | None:
    """Resolve stream names referenced by sampling groups."""
    sampling_groups = _normalize_sampling_groups(sampling_groups_cfg)
    group_stream_names: list[list[str]] = []

    for group in sampling_groups:
        resolved_names: list[str] = []
        entries = _collect_group_stream_entries(group)
        for entry in entries:
            candidate_name: str | None = None
            if isinstance(entry, str):
                candidate_name = entry
            elif isinstance(entry, Mapping):
                entry_dict = dict(entry)
                candidate_name = entry_dict.get("name") or entry_dict.get("base")
                if not candidate_name:
                    candidate_name = group.get("name")

            if candidate_name is None:
                continue
            if candidate_name not in flattened:
                logger.warning(
                    "Sampling group '%s' references unknown stream '%s'.",
                    group.get("name"),
                    candidate_name,
                )
                continue
            resolved_names.append(candidate_name)

        if resolved_names:
            group_stream_names.append(resolved_names)

    return group_stream_names or None


def _materialize_streams(
    flattened: dict[str, dict[str, Any]],
    *,
    root_remote: str | None,
    root_local: str | None,
) -> tuple[list[Stream], list[str]]:
    """Instantiate :class:`Stream` objects from flattened configuration maps."""
    streams: list[Stream] = []
    stream_names: list[str] = []

    for name, stream_cfg in flattened.items():
        stream_kwargs = {
            key: value for key, value in dict(stream_cfg).items() if value is not None
        }
        stream_kwargs.pop("name", None)

        if "remote" in stream_kwargs:
            stream_kwargs["remote"] = _join_remote_path(
                root_remote, stream_kwargs["remote"]
            )
        elif root_remote is not None:
            logger.warning(
                "Stream %s is missing a remote path; root_remote was provided.",
                name,
            )

        if "local" in stream_kwargs:
            stream_kwargs["local"] = _join_local_path(
                root_local, stream_kwargs["local"]
            )
        elif root_local is not None:
            logger.warning(
                "Stream %s is missing a local path; root_local was provided.",
                name,
            )

        streams.append(Stream(**stream_kwargs))
        stream_names.append(name)

    logger.info("Built %d streams for Mosaic dataloader", len(streams))
    return streams, stream_names


def _compute_group_indices(
    group_stream_names: list[list[str]] | None, stream_names: list[str]
) -> list[list[int]] | None:
    """Map group stream names to indices in the instantiated stream list."""
    if not group_stream_names:
        return None

    name_to_index = {stream_name: idx for idx, stream_name in enumerate(stream_names)}
    group_indices: list[list[int]] = []
    for names in group_stream_names:
        indices = [name_to_index[name] for name in names if name in name_to_index]
        if indices:
            group_indices.append(indices)

    if not group_indices:
        return None

    logger.info(
        "Resolved %d sampling group(s) for grouped Mosaic streams.",
        len(group_indices),
    )
    for idx, indices in enumerate(group_indices):
        selected = [stream_names[i] for i in indices]
        logger.info("Group %d contains streams: %s", idx, selected)
    return group_indices


def _resolve_unigram_remote_path(
    remote_uri: str,
    *,
    root_remote: str | None,
    split: str,
) -> tuple[str, str] | None:
    """Return the S3 bucket and key for a unigram statistics file."""
    parsed = urlparse(remote_uri)
    if parsed.scheme != "s3":
        logger.warning(
            "Unigram metric download skipped for %s (unsupported scheme '%s').",
            remote_uri,
            parsed.scheme or "unknown",
        )
        return None

    bucket = parsed.netloc
    remote_path = parsed.path.lstrip("/")
    root_prefix = ""

    if root_remote:
        root_parsed = urlparse(root_remote)
        if root_parsed.scheme not in ("", "s3"):
            logger.warning(
                "Unigram metric download skipped for %s (unsupported root scheme '%s').",
                remote_uri,
                root_parsed.scheme,
            )
            return None
        if root_parsed.netloc:
            bucket = root_parsed.netloc
        root_prefix = root_parsed.path.lstrip("/")
        if root_prefix and remote_path.startswith(root_prefix):
            remote_path = remote_path[len(root_prefix) :].lstrip("/")

    first_segment = remote_path.split("/", 1)[0] if remote_path else ""
    split_component = split if split and first_segment != split else ""

    remote_key_parts = [part for part in (root_prefix, remote_path) if part]
    if split_component:
        remote_key_parts.append(split_component)
    remote_key_parts.append("1_gram.json")

    remote_key = posixpath.join(*remote_key_parts)
    return bucket, remote_key


def _create_remote_unigram_client(
    bucket: str, config: UnigramMetricConfig
) -> RemoteUploaderDownloader:
    """Create the remote uploader/downloader used for unigram metrics."""
    remote_up_down = create_remote_up_down(
        bucket_name=bucket,
        prefix="",
        num_attempts=config.num_attempts,
        client_config=config.client_config,
    )
    remote_up_down._run_name = (  # pyright: ignore[reportAttributeAccessIssue]
        "unigram_metrics"
    )
    return remote_up_down


def _select_dataset_config(
    dataset_cfg: Mapping[str, Any] | None, split: str
) -> dict[str, Any]:
    if not dataset_cfg:
        return {}

    cfg = deepcopy(dict(dataset_cfg))
    if "common" in cfg or split in cfg:
        merged: dict[str, Any] = {}
        merged.update(cfg.pop("common", {}) or {})
        merged.update(cfg.pop(split, {}) or {})
        return merged
    return cfg


def _resolve_unigram_cache_path(
    stream: Stream,
    *,
    root_remote: str | None,
    dataset_split: str | None,
    default_split: str,
    config: UnigramMetricConfig,
) -> tuple[Path, Path]:
    """Determine the cache and split-specific unigram file locations."""
    local_root = getattr(stream, "local", None)
    stream_split = getattr(stream, "split", None) or dataset_split or default_split

    if not local_root:
        message = f"Stream '{getattr(stream, 'name', 'unknown')}' is missing a local path."
        raise RuntimeError(message)

    local_root_path = Path(local_root)
    cache_path = local_root_path / "1_gram.json"
    split_dir = local_root_path / stream_split if stream_split else local_root_path
    split_path = split_dir / "1_gram.json"

    if split_path.exists():
        return cache_path, split_path
    if cache_path.exists():
        return cache_path, split_path

    downloaded = _maybe_download_unigram_file(
        getattr(stream, "remote", None),
        root_remote,
        stream_split or "",
        cache_path,
        config,
    )
    if not downloaded and not cache_path.exists():
        message = (
            "Unigram frequency file not found for stream "
            f"'{getattr(stream, 'name', 'unknown')}' at {cache_path}"
        )
        raise RuntimeError(message)

    return cache_path, split_path


def _materialize_split_cache(cache_path: Path, split_path: Path) -> None:
    """Ensure split-specific unigram files reuse the global cache when possible."""
    if split_path == cache_path:
        return

    try:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        if not split_path.exists():
            try:
                os.link(cache_path, split_path)
            except OSError:
                shutil.copy2(cache_path, split_path)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "Unable to materialize split-specific unigram cache at %s: %s",
            split_path,
            exc,
        )


def _extract_streams(dataset_cfg: dict[str, Any]) -> StreamExtractionResult:
    root_remote = dataset_cfg.pop("root_remote", None)
    root_local = dataset_cfg.pop("root_local", None)
    streams_cfg = dataset_cfg.pop("streams", None)
    sampling_groups_cfg = dataset_cfg.pop("sampling_groups", None)
    sampling_groups_cfg = dataset_cfg.pop("stream_groups", sampling_groups_cfg)
    sampling_groups_mode_raw = dataset_cfg.pop("sampling_groups_mode", None)
    dataset_split = dataset_cfg.pop("split", None)

    if not streams_cfg:
        return StreamExtractionResult(
            streams=None,
            dataset_config=dataset_cfg,
            sampling_group_indices=None,
            dataset_root_remote=root_remote,
            dataset_split_remote=dataset_split,
        )

    flattened = _flatten_stream_configs(streams_cfg)
    sampling_mode = _normalize_sampling_mode(sampling_groups_mode_raw)
    group_stream_names = None

    if _should_concat_sampling_groups(sampling_mode, sampling_groups_cfg):
        flattened = _aggregate_sampling_groups(
            flattened, sampling_groups_cfg, sampling_groups_mode_raw
        )
    elif sampling_groups_cfg:
        group_stream_names = _resolve_group_stream_names(flattened, sampling_groups_cfg)

    streams, stream_names = _materialize_streams(
        flattened, root_remote=root_remote, root_local=root_local
    )
    group_indices = _compute_group_indices(group_stream_names, stream_names)
    return StreamExtractionResult(
        streams=streams or None,
        dataset_config=dataset_cfg,
        sampling_group_indices=group_indices,
        dataset_root_remote=root_remote,
        dataset_split_remote=dataset_split,
    )


def _maybe_download_unigram_file(
    remote_uri: str | None,
    root_remote: str | None,
    split: str,
    destination: Path,
    config: UnigramMetricConfig,
) -> bool:
    if not remote_uri or not config.download_missing:
        return False

    resolved = _resolve_unigram_remote_path(
        remote_uri, root_remote=root_remote, split=split
    )
    if resolved is None:
        return False

    bucket, remote_key = resolved
    remote_up_down = _create_remote_unigram_client(bucket, config)

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        download_file_from_s3(remote_up_down, remote_key, destination)
    except FileNotFoundError as exc:
        logger.warning(
            "Unigram frequency file %s not found in remote location %s: %s",
            remote_key,
            remote_uri,
            exc,
        )
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to download unigram frequency file %s from %s: %s",
            remote_key,
            remote_uri,
            exc,
        )
        return False
    else:
        logger.info("Downloaded unigram frequencies to %s", destination)
        return True
    finally:
        with suppress(Exception):
            remote_up_down.close()


def _load_stream_unigram_counts(
    stream: Stream,
    *,
    root_remote: str | None,
    dataset_split: str | None,
    default_split: str,
    config: UnigramMetricConfig,
) -> Counter:
    cache_path, split_path = _resolve_unigram_cache_path(
        stream,
        root_remote=root_remote,
        dataset_split=dataset_split,
        default_split=default_split,
        config=config,
    )

    unigram_path = split_path if split_path.exists() else cache_path
    if unigram_path is cache_path:
        _materialize_split_cache(cache_path, split_path)

    try:
        with unigram_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        msg = f"Unable to open unigram file {unigram_path}"
        raise RuntimeError(msg) from exc

    counts: Counter = Counter()
    for key, value in payload.items():
        try:
            token_id = int(key)
        except (ValueError, TypeError):
            token_id = int(ast.literal_eval(key))

        freq = int(value[0]) if isinstance(value, (list, tuple)) else int(value)

        counts[token_id] += freq

    return counts


def _normalize_mosaic_dataloader_config(
    job_config: MosaicJobConfig,
    *,
    split: str,
    default_drop_last: bool,
) -> NormalizedMosaicConfig:
    """Normalize high-level Mosaic dataloader configuration into typed payloads."""
    mosaic_cfg = job_config.mosaic_dataloader
    if not mosaic_cfg:
        msg = "mosaic_dataloader config must be set."
        raise ValueError(msg)

    cfg = deepcopy(mosaic_cfg)
    dataset_cfg_raw = cfg.dataset
    dataset_cfg = _select_dataset_config(dataset_cfg_raw, split)

    num_workers = cfg.num_workers
    prefetch_factor = cfg.prefetch_factor
    pin_memory = cfg.pin_memory
    persistent_workers = cfg.persistent_workers
    drop_last = cfg.drop_last if cfg.drop_last is not None else default_drop_last

    batch_size = (
        job_config.validation.local_batch_size
        if split == "val"
        else job_config.training.local_batch_size
    )

    runtime = MosaicRuntimeConfig(
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
        batch_size=batch_size,
    )

    return NormalizedMosaicConfig(dataset_config=dataset_cfg, runtime=runtime)


def _select_stream_subset(
    extraction: StreamExtractionResult,
    *,
    dp_rank: int,
    dp_world_size: int,
) -> StreamAssignment:
    """Select the stream subset for the current rank based on sampling groups."""
    streams = extraction.streams
    sampling_group_indices = extraction.sampling_group_indices

    if streams is None:
        return StreamAssignment(
            streams=None,
            group_index=None,
            dataset_root_remote=extraction.dataset_root_remote,
            dataset_split_remote=extraction.dataset_split_remote,
        )

    group_index: int | None = None
    stream_subset: list[Stream] | None

    if sampling_group_indices:
        group_count = len(sampling_group_indices)
        if group_count:
            group_index = dp_rank % group_count
            selected_indices = sampling_group_indices[group_index]
            stream_subset = [streams[idx] for idx in selected_indices]
            if len(stream_subset) == len(streams):
                logger.info(
                    "Mosaic group %d yielded all streams for dp_rank=%d; deferring to rank-based split.",
                    group_index,
                    dp_rank,
                )
                group_count = dp_world_size if dp_world_size > 0 else len(streams)
                group_index = dp_rank % group_count
                stream_subset = [streams[group_index % len(streams)]]
        else:
            stream_subset = streams

        if not stream_subset:
            msg = f"No streams resolved for Mosaic sampling group {group_index} (dp_rank={dp_rank})."
            raise ValueError(msg)
        logger.info(
            "Assigning Mosaic sampling group %s (dp_rank=%d) with %d stream(s): %s",
            "global" if group_index is None else group_index,
            dp_rank,
            len(stream_subset),
            [getattr(stream, "local", None) for stream in stream_subset],
        )
    else:
        stream_subset = streams

    return StreamAssignment(
        streams=stream_subset,
        group_index=group_index,
        dataset_root_remote=extraction.dataset_root_remote,
        dataset_split_remote=extraction.dataset_split_remote,
    )


def _prepare_dataset_kwargs(
    dataset_cfg: dict[str, Any],
    *,
    dataset_split_remote: str | None,
) -> DatasetFactoryConfig:
    """Prepare keyword arguments for the streaming dataset factory."""
    valid_params = {
        *inspect.signature(StreamingTextDataset).parameters,
        *inspect.signature(StreamingDataset).parameters,
    }
    dataset_kwargs = {k: v for k, v in dataset_cfg.items() if k in valid_params}

    subset_num_samples = dataset_cfg.get("subset_num_samples")
    if subset_num_samples is not None:
        dataset_kwargs["epoch_size"] = subset_num_samples

    if dataset_split_remote is not None:
        dataset_kwargs["split"] = dataset_split_remote

    return DatasetFactoryConfig(kwargs=dataset_kwargs)


def _create_streaming_dataset(
    *,
    assignment: StreamAssignment,
    tokenizer: BaseTokenizer,
    dataset_config: DatasetFactoryConfig,
    batch_size: int,
    split: str,
) -> StatefulStreamingTextDataset:
    """Instantiate the stateful streaming dataset for the resolved stream subset."""
    hf_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    logger.info(
        "Building StreamingTextDataset (%s split) with config: %s",
        split,
        dataset_config.kwargs,
    )
    return StatefulStreamingTextDataset(
        tokenizer=hf_tokenizer,
        streams=assignment.streams,
        batch_size=batch_size,
        **dataset_config.kwargs,
    )


def _setup_unigram_metric(
    assignment: StreamAssignment,
    *,
    job_config: MosaicJobConfig,
    split: str,
    tokenizer: BaseTokenizer,
    manager: UnigramMetricManager | None = None,
) -> UnigramSetupResult:
    """Build and register unigram metrics for the current stream subset."""
    collate_fn: Callable = titan_collate_fn
    if not job_config.unigram_metric.enable:
        return UnigramSetupResult(collate_fn=collate_fn, group_key=None)

    unigram_manager = manager or get_or_create_unigram_manager(job_config)

    group_label = (
        f"group_{assignment.group_index}"
        if assignment.group_index is not None and assignment.streams is not None
        else "global"
    )
    unigram_group_key = f"{split}/{group_label}"

    try:
        context = UnigramMetricContext(
            streams=assignment.streams,
            default_split=split,
            tokenizer=tokenizer,
            config=job_config.unigram_metric,
            group_key=unigram_group_key,
            dataset_root_remote=assignment.dataset_root_remote,
            dataset_split_remote=assignment.dataset_split_remote,
        )
        unigram_metric = _build_unigram_metric_for_group(context)
    except Exception as exc:
        if job_config.unigram_metric.allow_failures:
            logger.warning(
                "Unable to construct unigram metric for %s: %s",
                unigram_group_key,
                exc,
            )
            return UnigramSetupResult(collate_fn=collate_fn, group_key=None)
        msg = f"Unable to construct unigram metric for {unigram_group_key}: {exc}"
        raise RuntimeError(msg) from exc

    if unigram_metric is not None:
        handle = unigram_manager.register(unigram_metric, unigram_group_key)
        return UnigramSetupResult(
            collate_fn=collate_fn, group_key=unigram_group_key, handle=handle
        )

    return UnigramSetupResult(collate_fn=collate_fn, group_key=None, handle=None)


def _build_unigram_metric_for_group(
    context: UnigramMetricContext,
) -> PureUnigramCrossEntropy | None:
    if not context.config.enable or not context.streams:
        return None

    _ = context.tokenizer  # Tokenizer is unused; kept for API compatibility.

    aggregate_counts: Counter = Counter()
    for stream in context.streams:
        counts = _load_stream_unigram_counts(
            stream,
            root_remote=context.dataset_root_remote,
            dataset_split=context.dataset_split_remote,
            default_split=context.default_split,
            config=context.config,
        )
        aggregate_counts.update(counts)

    if not aggregate_counts:
        message = (
            f"No unigram counts collected for group '{context.group_key}'. "
            "Ensure 1_gram.json files are available for the configured streams."
        )
        raise RuntimeError(message)

    max_token_id = max(aggregate_counts)
    if max_token_id < 0:
        message = f"Invalid token ids encountered for group '{context.group_key}'."
        raise RuntimeError(message)

    probabilities = torch.zeros(max_token_id + 1, dtype=torch.float32)

    for token_id, count in aggregate_counts.items():
        probabilities[token_id] = float(count)

    total = probabilities.sum().item()
    if total <= 0:
        message = (
            f"Aggregate unigram counts sum to zero for group '{context.group_key}'."
        )
        raise RuntimeError(message)

    probabilities /= total

    logger.info(
        "Constructed unigram probabilities for %s (total tokens=%d).",
        context.group_key,
        int(total),
    )
    return PureUnigramCrossEntropy(
        probabilities,
        ignore_index=context.config.ignore_index,
    )


class StatefulStreamingTextDataset(StreamingTextDataset):
    """Stateful wrapper that keeps track of yielded samples.

    This makes the dataset compatible with dataloaders like TorchTitan's
    ``StatefulDataLoader`` that do not pass arguments to the dataset's
    :meth:`state_dict` method.

    Args:
        *args: Positional arguments to pass to :class:`StreamingTextDataset`.
        **kwargs: Keyword arguments to pass to :class:`StreamingTextDataset`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._num_samples_yielded = 0

    def __getitem__(self, idx: int) -> dict[str, list[int]] | torch.Tensor:
        """Increment the internal counter each time an item is fetched."""
        self._num_samples_yielded += 1
        return super().__getitem__(idx)

    def state_dict(
        self, num_samples: int | None = None, *, from_beginning: bool = True
    ) -> dict[str, Any]:
        """Saves the dataset's state.

        If `num_samples` is not provided by the caller, it defaults to using
        the internal `_num_samples_yielded` counter. This makes it compatible
        with the StatefulDataLoader.
        """
        effective_num_samples = (
            num_samples if num_samples is not None else self._num_samples_yielded
        )
        parent_state = super().state_dict(
            num_samples=effective_num_samples, from_beginning=from_beginning
        )
        parent_state["_num_samples_yielded"] = self._num_samples_yielded
        return parent_state

    def load_state_dict(self, obj: dict[str, Any]) -> None:
        """Restores the dataset's state from a state dictionary."""
        self._num_samples_yielded = obj.pop("_num_samples_yielded", 0)
        super().load_state_dict(obj)


class MosaicParallelAwareDataloader(StatefulDataLoader, BaseDataLoader):
    """Dataloader for Mosaic StreamingTextDataset with distributed data parallelism support.

    This dataloader inherits from torchdata's StatefulDataLoader to provide
    full checkpointing support with prefetching capabilities for streaming datasets,
    following the same pattern as the standard ParallelAwareDataloader.

    Args:
        dataset: The StreamingTextDataset instance to iterate over.
        dp_rank: Data parallelism rank for this dataloader.
        dp_world_size: The world size of the data parallelism.
        batch_size: The batch size to use for each iteration.
        collate_fn: Optional function to collate samples in a batch.
        num_workers: Number of worker processes for data loading.
        prefetch_factor: Number of batches to prefetch per worker.
        pin_memory: Whether to pin memory for faster GPU transfer.
        persistent_workers: Whether to keep workers alive between epochs.
        drop_last: Whether to drop the last incomplete batch.
    """

    dp_rank: int
    dp_world_size: int
    batch_size: int

    def __init__(
        self, dataset: StatefulStreamingTextDataset, request: ParallelDataLoaderRequest
    ) -> None:
        runtime = request.runtime
        self.dp_world_size = request.dp_world_size
        self.dp_rank = request.dp_rank
        self.batch_size = runtime.batch_size
        super().__init__(
            dataset,
            batch_size=runtime.batch_size,
            collate_fn=request.collate_fn,
            num_workers=runtime.num_workers,
            prefetch_factor=runtime.prefetch_factor,
            pin_memory=runtime.pin_memory,
            persistent_workers=runtime.persistent_workers,
            drop_last=runtime.drop_last,
        )
        self._rank_id = f"dp_rank_{request.dp_rank}"
        self.unigram_metric_handle = request.unigram_handle
        self.unigram_metric_group = request.group_key

    def state_dict(self) -> dict[str, Any]:
        """Save dataloader state for checkpointing."""
        return {
            self._rank_id: deepcopy(super().state_dict()),
            "world_size": self.dp_world_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load dataloader state from checkpoint."""
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(
                f"DataLoader state is empty for dp rank {self.dp_rank}, expected key {self._rank_id}"
            )
            return

        assert (
            self.dp_world_size == state_dict["world_size"]
        ), "dp_degree is inconsistent before and after checkpoint, dataloader resharding is not supported yet."
        loader_state = state_dict[self._rank_id]
        super().load_state_dict(deepcopy(loader_state))


def titan_collate_fn(batch: list[Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Collate samples from :class:`StreamingTextDataset` for TorchTitan.

    Args:
        batch: A list of samples from the dataset.

    Returns:
        A tuple where the first element is an `input_dict` and the second is a
        tensor of corresponding labels.
    """
    if isinstance(batch[0], dict):
        input_ids_list = [item["input_ids"] for item in batch]
        if not isinstance(input_ids_list[0], torch.Tensor):
            input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
        else:
            input_ids_tensor = torch.stack(input_ids_list)
    elif isinstance(batch[0], torch.Tensor):
        input_ids_tensor = torch.stack(batch)
    else:
        msg = f"Unsupported batch item type from dataset: {type(batch[0])}"
        raise TypeError(msg)

    model_inputs = input_ids_tensor[:, :-1].contiguous()
    labels = input_ids_tensor[:, 1:].contiguous()
    input_dict = {"input": model_inputs}
    return input_dict, labels


def _build_mosaic_dataloader(
    request: DataloaderBuildRequest,
) -> MosaicParallelAwareDataloader:
    normalized = _normalize_mosaic_dataloader_config(
        request.job_config,
        split=request.split,
        default_drop_last=request.default_drop_last,
    )
    mosaic_cfg = request.job_config.mosaic_dataloader
    if not mosaic_cfg.dataset:
        msg = "mosaic_dataloader config must define a dataset section."
        raise ValueError(msg)

    dataset_cfg_raw = deepcopy(mosaic_cfg.dataset)
    dataset_cfg = _select_dataset_config(dataset_cfg_raw, request.split)

    # Extract dataloader-specific config
    num_workers = mosaic_cfg.num_workers
    prefetch_factor = mosaic_cfg.prefetch_factor
    pin_memory = mosaic_cfg.pin_memory
    persistent_workers = mosaic_cfg.persistent_workers
    drop_last = (
        mosaic_cfg.drop_last
        if mosaic_cfg.drop_last is not None
        else request.default_drop_last
    )

    # Allow per-split overrides
    split_overrides = mosaic_cfg.get_split_overrides(request.split)
    if split_overrides:
        num_workers = split_overrides.get("num_workers", num_workers)
        prefetch_factor = split_overrides.get("prefetch_factor", prefetch_factor)
        pin_memory = split_overrides.get("pin_memory", pin_memory)
        persistent_workers = split_overrides.get(
            "persistent_workers", persistent_workers
        )
        drop_last = split_overrides.get("drop_last", drop_last)

    stream_extraction_result = _extract_streams(dataset_cfg)

    dataset_cfg = stream_extraction_result.dataset_config

    # Filter dataset config to only include valid StreamingTextDataset parameters
    valid_params = {
        *inspect.signature(StreamingTextDataset).parameters,
        *inspect.signature(StreamingDataset).parameters,
    }
    dataset_config_filtered = {
        k: v for k, v in dataset_cfg.items() if k in valid_params
    }

    # Resolve optional subset configuration
    subset_num_samples = dataset_cfg.pop("subset_num_samples", None)
    if subset_num_samples is not None:
        dataset_config_filtered["epoch_size"] = subset_num_samples

    extraction = _extract_streams(normalized.dataset_config)
    assignment = _select_stream_subset(
        extraction,
        dp_rank=request.dp_rank,
        dp_world_size=request.dp_world_size,
    )

    dataset_factory_config = _prepare_dataset_kwargs(
        extraction.dataset_config,
        dataset_split_remote=assignment.dataset_split_remote,
    )

    unigram_manager = get_or_create_unigram_manager(request.job_config)

    unigram_setup = _setup_unigram_metric(
        assignment,
        job_config=request.job_config,
        split=request.split,
        tokenizer=request.tokenizer,
        manager=unigram_manager,
    )

    text_dataset = _create_streaming_dataset(
        assignment=assignment,
        tokenizer=request.tokenizer,
        dataset_config=dataset_factory_config,
        batch_size=normalized.runtime.batch_size,
        split=request.split,
    )

    loader_request = ParallelDataLoaderRequest(
        dp_rank=request.dp_rank,
        dp_world_size=request.dp_world_size,
        runtime=normalized.runtime,
        collate_fn=unigram_setup.collate_fn,
        group_key=unigram_setup.group_key,
        unigram_handle=unigram_setup.handle,
    )
    return MosaicParallelAwareDataloader(text_dataset, loader_request)


def build_mosaic_dataloader(
    *,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: MosaicJobConfig,
) -> MosaicParallelAwareDataloader:
    """Build a Mosaic dataloader for the training split."""
    request = DataloaderBuildRequest(
        job_config=job_config,
        tokenizer=tokenizer,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        split="train",
        default_drop_last=True,
    )
    return _build_mosaic_dataloader(request)


def build_mosaic_validation_dataloader(
    *,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: MosaicJobConfig,
    infinite: bool = False,  # noqa: ARG001 - kept for compatibility
) -> MosaicParallelAwareDataloader:
    """Build a Mosaic dataloader for the validation split.

    Parameters
    ----------
    dp_world_size : int
        Data parallel world size.
    dp_rank : int
        Data parallel rank.
    tokenizer : BaseTokenizer
        Tokenizer instance.
    job_config : MosaicJobConfig
        Job configuration.
    infinite : bool, optional
        Unused parameter kept for compatibility with previous versions of the API.
        It may be removed in a future release; downstream callers should not rely
        on it.
    """
    request = DataloaderBuildRequest(
        job_config=job_config,
        tokenizer=tokenizer,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        split="val",
        default_drop_last=False,
    )
    return _build_mosaic_dataloader(request)


__all__ = [
    "MosaicParallelAwareDataloader",
    "build_mosaic_dataloader",
    "build_mosaic_validation_dataloader",
]
