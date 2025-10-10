# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Utilities for syncing TorchTitan checkpoints with S3."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, TYPE_CHECKING

from composer.loggers import RemoteUploaderDownloader

from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torchtitan.components.checkpoint import CheckpointManager

    from .configs.config import MosaicJobConfig, S3CheckpointingConfig

__all__ = [
    "S3CheckpointManager",
    "create_remote_up_down",
    "download_file_from_s3",
    "setup_s3_checkpointing",
    "upload_file_to_s3",
]

MANIFEST_FILENAME = "s3_manifest.json"
LATEST_FILENAME = "s3_latest.txt"


def download_file_from_s3(
    remote_up_down: RemoteUploaderDownloader,
    remote_file_name: str,
    local_file_name: Path | str,
) -> None:
    """Download a file from S3 using the RemoteUploaderDownloader."""
    remote_up_down._check_workers()
    remote_up_down.download_file(
        remote_file_name=remote_file_name,
        destination=str(local_file_name),
        overwrite=True,
    )


def upload_file_to_s3(
    remote_up_down: RemoteUploaderDownloader,
    remote_file_name: str,
    local_file_name: Path,
) -> None:
    """Upload a file to S3 using the RemoteUploaderDownloader."""
    remote_up_down._check_workers()
    remote_up_down.upload_file(
        state=None,
        remote_file_name=remote_file_name,
        file_path=local_file_name,
        overwrite=True,
    )


def create_remote_up_down(  # noqa: PLR0913
    bucket_name: str,
    prefix: str,
    run_uuid: str | None,
    num_attempts: int,
    client_config: dict[str, Any],
    *,
    num_concurrent_uploads: int = 1,
    upload_staging_folder: str | None = None,
    use_procs: bool = True,
) -> RemoteUploaderDownloader:
    """Create a RemoteUploaderDownloader configured for S3."""
    bucket_uri = f"s3://{bucket_name}"
    remote_up_down = RemoteUploaderDownloader(
        bucket_uri=bucket_uri,
        backend_kwargs={
            "bucket": bucket_name,
            "prefix": prefix,
            "region_name": None,
            "endpoint_url": None,
            "aws_access_key_id": None,
            "aws_secret_access_key": None,
            "aws_session_token": None,
            "client_config": client_config,
            "transfer_config": None,
        },
        file_path_format_string="{remote_file_name}",
        num_concurrent_uploads=num_concurrent_uploads,
        upload_staging_folder=upload_staging_folder,
        use_procs=use_procs,
        num_attempts=num_attempts,
    )
    remote_up_down.init(run_name=run_uuid)
    return remote_up_down


class S3CheckpointManager:
    """Synchronise checkpoints produced by a :class:`CheckpointManager` with S3."""

    def __init__(
        self,
        checkpointer: CheckpointManager,
        config: S3CheckpointingConfig,
        job_config: MosaicJobConfig,
    ) -> None:
        self._checkpointer = checkpointer
        self.config = config
        self.job_config = job_config
        self.remote_root = self._resolve_remote_root()
        self.remote_up_down = create_remote_up_down(
            bucket_name=config.bucket,
            prefix=config.prefix,
            run_uuid=config.run_uuid,
            num_attempts=config.num_attempts,
            client_config=config.client_config,
            num_concurrent_uploads=config.num_concurrent_uploads,
            upload_staging_folder=config.upload_staging_folder,
            use_procs=config.use_procs,
        )

        self._pending_steps: deque[tuple[int, Path]] = deque()
        self._uploaded_steps: set[int] = set()
        self._latest_uploaded_step: int | None = None
        self._closed = False

    @property
    def checkpointer(self) -> CheckpointManager:
        return self._checkpointer

    def __getattr__(self, name: str) -> Any:
        if hasattr(self._checkpointer, name):
            return getattr(self._checkpointer, name)
        raise AttributeError(
            f"'{type(self).__name__}' proxy: '{type(self._checkpointer).__name__}' object "
            f"has no attribute '{name}'"
        )

    def __del__(self) -> None:
        self.close()

    def download_if_needed(self, requested_step: int) -> None:
        """Optionally download a checkpoint from S3 before training starts."""
        if not self.config.download_on_start:
            return

        local_latest = self._find_local_latest_step()
        if local_latest != -1:
            return

        step = requested_step
        if step == -1:
            step = self._read_remote_latest_step() or -1
        if step == -1:
            logger.info("No remote checkpoint available for download.")
            return

        try:
            self._download_step(step)
            logger.info("Downloaded checkpoint step %s from S3.", step)
        except Exception:
            logger.exception("Failed to download checkpoint step %s from S3.", step)
            raise

    def close(self) -> None:
        """Flush pending uploads and release remote resources."""
        if self._closed:
            return
        try:
            self._wait_for_staging_with_logging()
            self._process_pending(flush=True)
            if hasattr(self.remote_up_down, "close"):
                try:
                    self.remote_up_down.close()
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to close RemoteUploaderDownloader.")
            self._checkpointer.close()
        finally:
            self._closed = True

    def _wait_for_staging_with_logging(self) -> None:
        try:
            self._checkpointer.maybe_wait_for_staging()
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed while waiting for staged checkpoints before upload."
            )

    def _resolve_remote_root(self) -> str:
        root = self.config.remote_checkpoint_folder or self.job_config.checkpoint.folder
        return root.strip("/")

    def _checkpoint_dir(self, step: int) -> Path:
        return Path(self.checkpointer.folder) / f"step-{step}"

    def _remote_key(self, relative_path: Path) -> str:
        relative_key = relative_path.as_posix()
        if self.remote_root:
            return f"{self.remote_root}/{relative_key}"
        return relative_key

    def save(
        self, curr_step: int, last_step: bool = False
    ) -> None:  # noqa: FBT001, FBT002
        self._checkpointer.save(curr_step, last_step=last_step)
        checkpoint_dir = self._checkpoint_dir(curr_step)
        if checkpoint_dir.exists() or last_step:
            self._pending_steps.append((curr_step, checkpoint_dir))
        if last_step:
            try:
                self._checkpointer.maybe_wait_for_staging()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed while waiting for staged checkpoints before final upload."
                )
            self._process_pending(flush=True)

    def maybe_wait_for_staging(self) -> None:
        self._checkpointer.maybe_wait_for_staging()
        self._process_pending()

    def _process_pending(self, flush: bool = False) -> None:  # noqa: FBT001, FBT002
        pending: deque[tuple[int, Path]] = deque()
        while self._pending_steps:
            step, directory = self._pending_steps.popleft()
            if step in self._uploaded_steps:
                continue
            if not directory.exists():
                if flush:
                    if directory.exists():
                        self._upload_step(step, directory)
                else:
                    pending.append((step, directory))
                continue
            self._upload_step(step, directory)
        self._pending_steps = pending

    def _upload_step(self, step: int, directory: Path) -> None:
        files = sorted(self._iter_checkpoint_files(directory))
        if not files:
            return

        manifest_path = directory / MANIFEST_FILENAME
        manifest_content = [path.relative_to(directory).as_posix() for path in files]
        manifest_path.write_text(json.dumps(manifest_content))

        upload_targets = [*files, manifest_path]
        for file_path in upload_targets:
            relative = file_path.relative_to(Path(self.checkpointer.folder))
            remote_key = self._remote_key(relative)
            upload_file_to_s3(self.remote_up_down, remote_key, file_path)

        self._uploaded_steps.add(step)
        if self._latest_uploaded_step is None or step > self._latest_uploaded_step:
            self._latest_uploaded_step = step
            self._write_latest_marker(step)
        logger.info("Uploaded checkpoint step %s to S3 (%s files).", step, len(files))

    def _iter_checkpoint_files(self, directory: Path) -> Iterable[Path]:
        for path in directory.rglob("*"):
            if path.is_file() and path.name != MANIFEST_FILENAME:
                yield path

    def _write_latest_marker(self, step: int) -> None:
        marker_path = Path(self.checkpointer.folder) / LATEST_FILENAME
        marker_path.write_text(f"{step}\n")
        remote_key = self._remote_key(Path(LATEST_FILENAME))
        upload_file_to_s3(self.remote_up_down, remote_key, marker_path)

    def _find_local_latest_step(self) -> int:
        try:
            return self.checkpointer._find_load_step()
        except Exception:  # noqa: BLE001
            return -1

    def _read_remote_latest_step(self) -> int | None:
        marker_path = Path(self.checkpointer.folder) / LATEST_FILENAME
        try:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            download_file_from_s3(
                self.remote_up_down,
                self._remote_key(Path(LATEST_FILENAME)),
                marker_path,
            )
        except Exception:  # noqa: BLE001
            return None
        try:
            return int(marker_path.read_text().strip())
        except ValueError:
            logger.warning("Invalid latest checkpoint marker downloaded from S3.")
            return None

    def _download_step(self, step: int) -> None:
        checkpoint_dir = self._checkpoint_dir(step)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = checkpoint_dir / MANIFEST_FILENAME
        download_file_from_s3(
            self.remote_up_down,
            self._remote_key(Path(f"step-{step}") / MANIFEST_FILENAME),
            manifest_path,
        )
        manifest_entries = json.loads(manifest_path.read_text())
        for relative in manifest_entries:
            relative_path = Path(relative)
            local_path = checkpoint_dir / relative_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            download_file_from_s3(
                self.remote_up_down,
                self._remote_key(Path(f"step-{step}") / relative_path),
                local_path,
            )
        self._latest_uploaded_step = max(self._latest_uploaded_step or -1, step)
        self._write_latest_marker(step)


def setup_s3_checkpointing(
    checkpointer: CheckpointManager, job_config: MosaicJobConfig
) -> S3CheckpointManager | None:
    """Create an :class:`S3CheckpointManager` if configured."""
    config = job_config.s3_checkpoint
    if not config.enable:
        return None
    if not config.bucket or not config.prefix:
        logger.warning(
            "S3 checkpointing is enabled but bucket or prefix is not provided; skipping."
        )
        return None

    return S3CheckpointManager(checkpointer, config, job_config)
