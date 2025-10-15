# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Utilities for resolving Mosaic streaming datasets."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from torchtitan.tools.logging import logger

try:
    from streaming import Stream
except ImportError as exc:  # pragma: no cover - optional dependency
    msg = (
        "llm-foundry and streaming are required to build Mosaic dataloaders. "
        "Please install llm-foundry, mosaicml-streaming, and composer to enable this integration."
    )
    raise RuntimeError(msg) from exc


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


def _is_uri(path: str | None) -> bool:
    return bool(path and "://" in path)


def _join_remote_path(root: str | None, path: str | None) -> str | None:
    if path is None or _is_uri(path):
        return path
    if root is None:
        return path
    if _is_uri(root):
        normalized_root = root.rstrip("/") + "/"
        normalized_path = path.lstrip("/")
        return urljoin(normalized_root, normalized_path)
    return "/".join(part.strip("/") for part in (root, path) if part)


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

    def _collect_sequence(items: Sequence[Any], key_prefix: str | None = None) -> None:
        for index, item in enumerate(items):
            child_key = f"{key_prefix}_{index}" if key_prefix else None
            _collect(item, child_key)

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
                    if isinstance(value, Sequence) and not isinstance(
                        value, (str, bytes)
                    ):
                        _collect_sequence(value, key)
                    else:
                        _collect(value, key)
                else:
                    _collect(value, key)
        elif isinstance(config, (list, tuple)):
            _collect_sequence(config)

    _collect(streams_cfg)
    return flattened


def _normalize_sampling_groups(config: Any) -> list[dict[str, Any]]:
    """Normalize sampling group definitions into a list of mappings."""
    if not config:
        return []

    normalized: list[dict[str, Any]] = []
    if isinstance(config, Mapping):
        iterator: Iterable[tuple[Any, Any]] = config.items()
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
        group = dict(value)
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
            msg = f"Sampling group '{group.get('name')}' references unknown stream '{entry}'"
            raise KeyError(msg)
        candidate = dict(flattened[entry])
        candidate.setdefault("name", entry)
        return candidate

    if isinstance(entry, Mapping):
        entry_dict = dict(entry)
        name = entry_dict.get("name")
        base = entry_dict.get("base")
        base_cfg: dict[str, Any] = {}
        if base and base in flattened:
            base_cfg = dict(flattened[base])
        elif name and name in flattened:
            base_cfg = dict(flattened[name])
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
        "Resolved %d sampling group(s) for grouped Mosaic streams.", len(group_indices)
    )
    for idx, indices in enumerate(group_indices):
        selected = [stream_names[i] for i in indices]
        logger.info("Group %d contains streams: %s", idx, selected)
    return group_indices


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
    stream_subset: list[Stream]

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


__all__ = [
    "StreamAssignment",
    "StreamExtractionResult",
    "_extract_streams",
    "_join_local_path",
    "_join_remote_path",
    "_select_stream_subset",
]
