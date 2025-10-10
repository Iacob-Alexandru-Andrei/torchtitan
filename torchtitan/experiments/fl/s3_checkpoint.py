"""Utilities for mirroring checkpoints stored on S3 locally.

The real training workflow used for the FL experiments relies on a small helper
that mirrors checkpoints produced by a remote trainer into the node running the
evaluation job.  The original project ships the helper with the job configs,
but it is not part of the public repository used for the exercises.  The
implementation below mirrors the private helper closely enough for the unit
tests bundled with this kata.

The helper centres around :class:`S3CheckpointManager`.  The manager knows how
to discover the latest checkpoint in a remote S3 bucket, keeps track of what is
available locally and downloads new checkpoints on demand.  It supports both
explicitly requested steps (useful for debugging) and the more common
"download the most recent checkpoint" flow used in production.

To keep the implementation light-weight and test friendly, the manager accepts
two callables in the constructor:

``marker_fetch_fn``
    A callable returning the most recent checkpoint step that exists on the
    remote storage.  Returning ``None`` signals that there is no checkpoint yet.

``step_download_fn``
    A callable that is given a step and a destination folder and is responsible
    for downloading the checkpoint contents for that step.

In the production setup both callables are thin wrappers around ``boto3``
helpers.  The abstraction keeps the logic in this module completely decoupled
from the actual storage backend which makes it easy to test.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from torchtitan.tools.logging import logger


MarkerFetcher = Callable[[], int | None]
StepDownloader = Callable[[int, Path], None]


class S3CheckpointManager:
    """Mirror checkpoints stored on S3 into a local folder.

    Parameters
    ----------
    checkpoint_root:
        Local directory where the checkpoints should be materialised.
    marker_fetch_fn:
        Callable returning the latest available remote step. ``None`` means no
        checkpoint is available.
    step_download_fn:
        Callable responsible for copying a particular step locally.
    download_on_start:
        Gate used by the training harness to disable the bootstrap download in
        some configurations.  This flag is respected by :meth:`download_if_needed`.
    marker_filename:
        Name of the file used to cache the last downloaded step locally.  The
        marker is only advisory and is recreated automatically when missing.
    """

    def __init__(
        self,
        *,
        checkpoint_root: str | Path,
        marker_fetch_fn: MarkerFetcher,
        step_download_fn: StepDownloader,
        download_on_start: bool = True,
        marker_filename: str = ".s3_latest_step",
    ) -> None:
        self._root = Path(checkpoint_root)
        self._root.mkdir(parents=True, exist_ok=True)

        self._marker_path = self._root / marker_filename
        self._marker_fetch_fn = marker_fetch_fn
        self._step_download_fn = step_download_fn
        self.download_on_start = download_on_start

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _checkpoint_dir(self, step: int) -> Path:
        return self._root / f"step-{step}"

    def _find_local_latest_step(self) -> int:
        pattern = re.compile(r"step-(\d+)")
        latest = -1
        for child in self._root.iterdir():
            if not child.is_dir():
                continue
            match = pattern.fullmatch(child.name)
            if not match:
                continue
            step = int(match.group(1))
            latest = max(latest, step)
        return latest

    def _read_local_marker_step(self) -> int | None:
        if not self._marker_path.exists():
            return None
        try:
            return int(self._marker_path.read_text().strip())
        except ValueError:
            logger.warning(
                "Local checkpoint marker %s contained unexpected data; ignoring it.",
                self._marker_path,
            )
            return None

    def _write_local_marker_step(self, step: int) -> None:
        try:
            self._marker_path.write_text(str(step))
        except OSError:
            logger.exception(
                "Failed to update local checkpoint marker at %s.",
                self._marker_path,
            )

    def _download_marker(self) -> int | None:
        return self._marker_fetch_fn()

    def _download_step(self, step: int) -> None:
        destination = self._checkpoint_dir(step)
        destination.mkdir(parents=True, exist_ok=True)
        self._step_download_fn(step, destination)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def download_if_needed(self, requested_step: int = -1) -> int | None:
        """Download a checkpoint if it is missing locally.

        The method performs the following steps:

        1. Honour the ``download_on_start`` gate.  When disabled we simply log
           and return immediately.
        2. Consult the remote marker to learn about the latest available step.
        3. Decide which step (if any) should be downloaded based on the
           ``requested_step`` argument and the current local state.
        4. Fetch the checkpoint using ``_download_step`` if we need it.

        Returns
        -------
        int | None
            The step that is now available locally or ``None`` when no action
            was required / possible.
        """

        if not self.download_on_start:
            logger.info(
                "Skipping remote checkpoint download because download_on_start is disabled.",
            )
            return None

        try:
            remote_step = self._download_marker()
        except Exception:
            logger.exception("Failed to determine the latest checkpoint available on S3.")
            return None

        step_to_fetch: int | None = None

        if requested_step != -1:
            step_to_fetch = requested_step
            if self._checkpoint_dir(step_to_fetch).exists():
                logger.info(
                    "Skipping remote download; checkpoint for explicit step %s already exists locally.",
                    step_to_fetch,
                )
                if self._read_local_marker_step() != step_to_fetch:
                    self._write_local_marker_step(step_to_fetch)
                return step_to_fetch
        else:
            if remote_step is None:
                logger.info("Remote checkpoint marker not found; nothing to download.")
                return None

            local_latest_step = self._find_local_latest_step()
            local_marker_step = self._read_local_marker_step()

            if local_latest_step > remote_step:
                logger.info(
                    "Latest local checkpoint (step %s) is newer than the remote marker %s; skipping download.",
                    local_latest_step,
                    remote_step,
                )
                return remote_step

            if (
                local_latest_step == remote_step
                and local_marker_step == remote_step
                and self._checkpoint_dir(remote_step).exists()
            ):
                logger.info(
                    "Skipping remote download; latest remote step %s already exists locally with matching marker.",
                    remote_step,
                )
                return remote_step

            step_to_fetch = remote_step

        if step_to_fetch is None:
            return None

        if self._checkpoint_dir(step_to_fetch).exists():
            logger.info(
                "Checkpoint for step %s already present locally; skipping remote transfer.",
                step_to_fetch,
            )
            if self._read_local_marker_step() != step_to_fetch:
                self._write_local_marker_step(step_to_fetch)
            return step_to_fetch

        try:
            self._download_step(step_to_fetch)
        except Exception:
            logger.exception(
                "Failed to download checkpoint step %s from remote storage.",
                step_to_fetch,
            )
            return None

        self._write_local_marker_step(step_to_fetch)
        logger.info("Downloaded checkpoint step %s from remote storage.", step_to_fetch)
        return step_to_fetch


__all__ = ["S3CheckpointManager"]

