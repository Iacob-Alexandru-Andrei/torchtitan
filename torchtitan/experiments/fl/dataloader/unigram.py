"""Helpers for configuring unigram cross-entropy metrics."""

from __future__ import annotations

import json
import os
import posixpath
import shutil
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import torch

from torchtitan.experiments.fl.s3_checkpoint import create_remote_up_down, download_file_from_s3
from torchtitan.tools.logging import logger

try:
    from streaming import Stream
except ImportError as exc:  # pragma: no cover - optional dependency
    msg = (
        "llm-foundry and streaming are required to build Mosaic dataloaders. "
        "Please install llm-foundry, mosaicml-streaming, and composer to enable this integration."
    )
    raise RuntimeError(msg) from exc

if TYPE_CHECKING:
    from collections.abc import Callable

    from torchtitan.components.tokenizer import BaseTokenizer
    from torchtitan.experiments.fl.configs.config import MosaicJobConfig, UnigramMetricConfig
    from torchtitan.experiments.fl.s3_checkpoint import RemoteUploaderDownloader

from torchtitan.experiments.fl.metrics import (
    PureUnigramCrossEntropy,
    UnigramMetricHandle,
    UnigramMetricManager,
    get_or_create_unigram_manager,
)

from .streams import StreamAssignment


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


@dataclass(frozen=True)
class UnigramSetupResult:
    """Result of configuring unigram metrics for a stream subset."""

    collate_fn: Callable
    group_key: str | None
    handle: UnigramMetricHandle | None = None

    @property
    def metric(self) -> PureUnigramCrossEntropy | None:
        """Expose the underlying metric for compatibility with legacy call sites."""

        return self.handle.metric if self.handle is not None else None


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
    remote_up_down._run_name = "unigram_metrics"  # pyright: ignore[reportAttributeAccessIssue]
    return remote_up_down


def _maybe_download_unigram_file(
    remote_uri: str | None,
    root_remote: str | None,
    split: str,
    destination: Path,
    config: UnigramMetricConfig,
) -> bool:
    if not remote_uri or not config.download_missing:
        return False

    resolved = _resolve_unigram_remote_path(remote_uri, root_remote=root_remote, split=split)
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
        if split_path.exists():
            return
        split_path.parent.mkdir(parents=True, exist_ok=True)
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
            try:
                parsed_key = json.loads(key)
            except (json.JSONDecodeError, TypeError) as exc:
                msg = f"Unigram file contains non-numeric token identifier: {key!r}"
                raise ValueError(msg) from exc

            if isinstance(parsed_key, float):
                if not parsed_key.is_integer():
                    msg = f"Unigram file contains non-integral token identifier: {parsed_key!r}"
                    raise ValueError(msg)
                token_id = int(parsed_key)
            elif isinstance(parsed_key, int):
                token_id = parsed_key
            else:
                msg = f"Unigram file contains unsupported token identifier type: {type(parsed_key)!r}"
                raise ValueError(msg)

        freq = int(value[0]) if isinstance(value, (list, tuple)) else int(value)
        counts[token_id] += freq

    return counts


def _build_unigram_metric_for_group(
    context: UnigramMetricContext,
) -> PureUnigramCrossEntropy | None:
    if not context.config.enable or not context.streams:
        return None

    # The `tokenizer` attribute is currently unused in this function, but is retained in the context
    # for API compatibility with other components that may expect it, and for potential future use.
    # Do not remove without verifying downstream dependencies.
    _ = context.tokenizer

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
        message = f"Aggregate unigram counts sum to zero for group '{context.group_key}'."
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


def setup_unigram_metric(
    assignment: StreamAssignment,
    *,
    job_config: MosaicJobConfig,
    split: str,
    tokenizer: BaseTokenizer,
    collate_fn: Callable,
    manager: UnigramMetricManager | None = None,
) -> UnigramSetupResult:
    """Build and return unigram metric wiring for the current stream subset."""

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
    except Exception as exc:  # noqa: BLE001
        if job_config.unigram_metric.allow_failures:
            logger.warning(
                "Unable to construct unigram metric for %s: %s",
                unigram_group_key,
                exc,
            )
            return UnigramSetupResult(collate_fn=collate_fn, group_key=None)
        msg = f"Unable to construct unigram metric for {unigram_group_key}: {exc}"
        raise RuntimeError(msg) from exc

    if unigram_metric is None:
        return UnigramSetupResult(collate_fn=collate_fn, group_key=None)

    handle = unigram_manager.register(unigram_metric, unigram_group_key)
    return UnigramSetupResult(
        collate_fn=collate_fn,
        group_key=unigram_group_key,
        handle=handle,
    )


__all__ = [
    "UnigramMetricContext",
    "UnigramSetupResult",
    "_build_unigram_metric_for_group",
    "setup_unigram_metric",
]
