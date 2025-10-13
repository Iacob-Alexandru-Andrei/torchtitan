# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Utilities for syncing TorchTitan checkpoints with S3."""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Any, TYPE_CHECKING

from composer.loggers import RemoteUploaderDownloader
from composer.loggers.remote_uploader_downloader import _upload_worker

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

# Constants for validation
RESUME_FORMAT_PARTS_COUNT = 2
STEP_PREFIX = "step-"
MAX_FILES_TO_DISPLAY = 20


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
    num_attempts: int,
    client_config: dict[str, Any],
    *,
    num_concurrent_uploads: int = 1,
    upload_staging_folder: str | None = None,
    use_procs: bool = True,
) -> RemoteUploaderDownloader:
    """Create a RemoteUploaderDownloader configured for S3."""
    bucket_uri = f"s3://{bucket_name}"
    return RemoteUploaderDownloader(
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
            num_attempts=config.num_attempts,
            client_config=config.client_config,
            num_concurrent_uploads=config.num_concurrent_uploads,
            upload_staging_folder=config.upload_staging_folder,
            use_procs=config.use_procs,
        )
        # Set the run name for the RemoteUploaderDownloader
        # This is normally set in init() but we're using it standalone
        run_name = (
            config.run_uuid
            or job_config.job.description
            or Path(job_config.job.dump_folder).name
            or "torchtitan-run"
        )
        self.remote_up_down._run_name = str(run_name)

        self._base_folder = Path(checkpointer.folder)
        self._ft_mode = bool(getattr(checkpointer, "ft_manager", None))
        self._ft_folder_path: Path | None = None
        self._ft_relative: Path | None = None
        if self._ft_mode:
            ft_folder_str = checkpointer._ft_folder()
            self._ft_folder_path = Path(ft_folder_str)
            try:
                self._ft_relative = self._ft_folder_path.relative_to(self._base_folder)
            except ValueError:
                self._ft_relative = Path(self._ft_folder_path.name)

        self._pending_steps: deque[tuple[int, Path]] = deque()
        self._uploaded_steps: set[int] = set()
        self._latest_uploaded_step: int | None = None
        self._closed = False

        # Install tracking
        self._missing_directory_steps: set[int] = set()
        self._not_ready_steps: set[int] = set()
        self._installed = False
        self._orig_save = checkpointer.save
        self._orig_maybe_wait = checkpointer.maybe_wait_for_staging

    @property
    def checkpointer(self) -> CheckpointManager:
        """Get the underlying CheckpointManager instance."""
        return self._checkpointer

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying CheckpointManager."""
        if hasattr(self._checkpointer, name):
            return getattr(self._checkpointer, name)
        msg = f"'{type(self).__name__}' proxy: '{type(self._checkpointer).__name__}' object has no attribute '{name}'"
        raise AttributeError(msg)

    def install(self) -> None:
        """Hook into the wrapped :class:`CheckpointManager`."""
        if self._installed:
            return
        self._installed = True

        # Initialize the RemoteUploaderDownloader workers
        self._start_remote_workers()

        self.checkpointer.save = self.save  # type: ignore[assignment]
        self.checkpointer.maybe_wait_for_staging = (  # type: ignore[assignment]
            self.maybe_wait_for_staging
        )

    def _start_remote_workers(self) -> None:
        """Start the RemoteUploaderDownloader background workers."""
        rud = self.remote_up_down

        if rud._worker_flag is not None:
            return  # Already initialized

        rud._worker_flag = rud._finished_cls()

        # Create the enqueue thread
        rud._enqueue_thread_flag = rud._finished_cls()
        rud._enqueue_thread = threading.Thread(target=rud._enqueue_uploads, daemon=True)
        rud._enqueue_thread.start()

        # Start the upload workers
        for _ in range(rud._num_concurrent_uploads):
            worker = rud._proc_class(
                target=_upload_worker,
                kwargs={
                    "file_queue": rud._file_upload_queue,
                    "is_finished": rud._worker_flag,
                    "remote_backend_name": rud.remote_backend_name,
                    "backend_kwargs": rud.backend_kwargs,
                    "num_attempts": rud.num_attempts,
                    "completed_queue": rud._completed_queue,
                    "exception_queue": rud._exception_queue,
                },
                daemon=True,
            )
            worker.start()
            rud._workers.append(worker)

        logger.info("Started %d S3 upload workers", rud._num_concurrent_uploads)

    def __del__(self) -> None:
        """Clean up resources on object destruction."""
        self.close()

    def download_if_needed(self) -> None:
        """Optionally download a checkpoint from S3 before training starts.

        If resume_from_run_step is set, downloads from that specific run/step.
        Otherwise, looks for the latest checkpoint in the current run.
        """
        if not self.config.download_on_start:
            logger.info("S3 download skipped: download_on_start=False")
            return

        if self._ft_mode and self._ft_folder_path is not None:
            base_folder = self._ft_folder_path
            local_latest = self.checkpointer._find_load_step(
                folder=str(self._ft_folder_path)
            )
        else:
            base_folder = self._base_folder
            local_latest = self._find_local_latest_step()
        logger.info(
            "Checking for local checkpoints in: %s (found step: %s)",
            base_folder,
            local_latest if local_latest != -1 else "none",
        )
        if local_latest != -1:
            logger.info(
                "Skipping S3 download: local checkpoint found at step %s", local_latest
            )
            return

        # Determine what to download
        if self.config.resume_from_run_step:
            # Parse format: "{run_uuid}/step-{N}"
            try:
                parts = self.config.resume_from_run_step.split("/")
                if len(parts) != RESUME_FORMAT_PARTS_COUNT or not parts[1].startswith(
                    STEP_PREFIX
                ):
                    self._raise_invalid_resume_format()
                run_uuid = parts[0]
                step_str = parts[1][len(STEP_PREFIX) :]  # Remove "step-" prefix
                step = int(step_str)
                remote_path = f"torchtitan/{run_uuid}"
                relative_suffix = Path(f"step-{step}")
                if self._ft_relative is not None:
                    relative_suffix = self._ft_relative / relative_suffix
                remote_preview = f"{remote_path}/{relative_suffix.as_posix()}"

                prefix_display = (self.config.prefix or "").strip("/")
                components = [comp for comp in (prefix_display, remote_preview) if comp]
                combined_path = "/".join(components)
                logger.info(
                    "Resuming from run step: %s (downloading from: s3://%s/%s)",
                    self.config.resume_from_run_step,
                    self.config.bucket,
                    combined_path,
                )
            except (ValueError, IndexError) as e:
                logger.exception(
                    "Failed to parse resume_from_run_step: %s",
                    self.config.resume_from_run_step,
                )
                self._raise_invalid_resume_format(e)
        else:
            # Look for latest in current run
            remote_path = self.remote_root
            step = self._read_remote_latest_step() or -1

            logger.info(
                "Continuing current run: %s (looking for latest in: s3://%s/%s)",
                self.config.run_uuid,
                self.config.bucket,
                remote_path,
            )

            if step == -1:
                logger.info("No remote checkpoint available for download.")
                return

        try:
            self._download_step(step, remote_path=remote_path)
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
            # Note: RemoteUploaderDownloader cleanup is handled by Composer internally
            self._checkpointer.close()
        finally:
            self._closed = True

    def _wait_for_staging_with_logging(self) -> None:
        try:
            self._orig_maybe_wait()
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed while waiting for staged checkpoints before upload."
            )
        if self._installed:
            self._process_pending(flush=True)
        # Note: RemoteUploaderDownloader cleanup is handled by Composer internally

    def _resolve_remote_root(self) -> str:
        root = self.config.remote_checkpoint_folder or self.job_config.checkpoint.folder
        return root.strip("/")

    def _checkpoint_dir(self, step: int) -> Path:
        if self._ft_mode and self._ft_folder_path is not None:
            return self._ft_folder_path / f"step-{step}"
        return self._base_folder / f"step-{step}"

    def _raise_invalid_resume_format(self, cause: Exception | None = None) -> None:
        """Raise a ValueError for invalid resume_from_run_step format.

        Args:
            cause: Optional exception that caused this error
        """
        msg = (
            f"Invalid resume_from_run_step format: '{self.config.resume_from_run_step}'. "
            "Expected format: '{{run_uuid}}/step-{{N}}' (e.g., '16M-baseline-20251011-122516/step-10')"
        )
        if cause:
            raise ValueError(msg) from cause
        raise ValueError(msg)

    def _remote_key(self, relative_path: Path, remote_root: str | None = None) -> str:
        """Get the remote S3 key.

        Args:
            relative_path: Path relative to the checkpoint folder
            remote_root: Optional override for the remote root. If not provided, uses self.remote_root
        """
        relative_key = relative_path.as_posix()
        root = remote_root if remote_root is not None else self.remote_root
        if root:
            return f"{root}/{relative_key}"
        return relative_key

    def save(self, curr_step: int, *, last_step: bool = False) -> None:
        """Save checkpoint and queue for S3 upload.

        Args:
            curr_step: Current training step
            last_step: Whether this is the final checkpoint
        """
        self._orig_save(curr_step, last_step=last_step)
        checkpoint_dir = self._checkpoint_dir(curr_step)
        if not checkpoint_dir.exists():
            logger.warning(
                "Checkpoint directory %s for step %s does not exist immediately after save; "
                "upload will be retried once it becomes available.",
                checkpoint_dir,
                curr_step,
            )
        self._pending_steps.append((curr_step, checkpoint_dir))
        if last_step:
            try:
                self._orig_maybe_wait()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed while waiting for staged checkpoints before final upload."
                )
            self._process_pending(flush=True)

    def maybe_wait_for_staging(self) -> None:
        """Wait for staged checkpoints and process pending uploads."""
        self._orig_maybe_wait()
        self._process_pending()

    def _process_pending(self, flush: bool = False) -> None:  # noqa: FBT001, FBT002
        pending: deque[tuple[int, Path]] = deque()
        while self._pending_steps:
            step, directory = self._pending_steps.popleft()
            if step in self._uploaded_steps:
                continue
            if not directory.exists():
                if flush:
                    logger.error(
                        "Checkpoint directory %s for step %s does not exist and will not be uploaded during flush.",
                        directory,
                        step,
                    )
                else:
                    if step not in self._missing_directory_steps:
                        logger.warning(
                            "Checkpoint directory %s for step %s is not available yet; retrying staging before upload.",
                            directory,
                            step,
                        )
                        self._missing_directory_steps.add(step)
                    pending.append((step, directory))
                continue
            self._missing_directory_steps.discard(step)
            if not self._is_directory_ready_for_upload(directory):
                if flush:
                    logger.error(
                        "Checkpoint directory %s for step %s is not ready for upload during flush and will be skipped.",
                        directory,
                        step,
                    )
                else:
                    if step not in self._not_ready_steps:
                        logger.info(
                            "Checkpoint directory %s for step %s is still being written; deferring upload.",
                            directory,
                            step,
                        )
                        self._not_ready_steps.add(step)
                    pending.append((step, directory))
                continue
            self._not_ready_steps.discard(step)
            self._upload_step(step, directory)
        self._pending_steps = pending

    def _is_directory_ready_for_upload(self, directory: Path) -> bool:
        try:
            has_entries = any(directory.iterdir())
        except FileNotFoundError:
            return False
        if not has_entries:
            return False
        try:
            has_temp_files = any(directory.rglob("*.tmp"))
        except FileNotFoundError:
            return False
        return not has_temp_files

    def _upload_step(self, step: int, directory: Path) -> None:
        files = sorted(self._iter_checkpoint_files(directory))
        if not files:
            return

        manifest_path = directory / MANIFEST_FILENAME
        manifest_content = [path.relative_to(directory).as_posix() for path in files]
        manifest_path.write_text(json.dumps(manifest_content))

        upload_targets = [*files, manifest_path]
        uploaded_paths = []
        for file_path in upload_targets:
            relative = file_path.relative_to(Path(self.checkpointer.folder))
            remote_key = self._remote_key(relative)
            upload_file_to_s3(self.remote_up_down, remote_key, file_path)
            # Log the full S3 path
            s3_uri = f"s3://{self.config.bucket}/{remote_key}"
            uploaded_paths.append(s3_uri)
            logger.info("Uploaded: %s -> %s", file_path, s3_uri)

        self._uploaded_steps.add(step)
        if self._latest_uploaded_step is None or step > self._latest_uploaded_step:
            self._latest_uploaded_step = step
            self._write_latest_marker(step)
        logger.info("Uploaded checkpoint step %s to S3 (%s files).", step, len(files))
        logger.info("All uploaded files for step %s: %s", step, uploaded_paths)

    def _iter_checkpoint_files(self, directory: Path) -> Iterable[Path]:
        for path in directory.rglob("*"):
            if path.is_file() and path.name != MANIFEST_FILENAME:
                yield path

    def _write_latest_marker(self, step: int) -> None:
        marker_path = Path(self.checkpointer.folder) / LATEST_FILENAME
        marker_path.write_text(f"{step}\n")
        remote_key = self._remote_key(Path(LATEST_FILENAME))
        s3_uri = f"s3://{self.config.bucket}/{remote_key}"
        upload_file_to_s3(self.remote_up_down, remote_key, marker_path)
        logger.info(
            "Uploaded latest marker: %s -> %s (points to step %s)",
            marker_path,
            s3_uri,
            step,
        )

    def _find_local_latest_step(self) -> int:
        try:
            return self.checkpointer._find_load_step()
        except Exception:  # noqa: BLE001
            return -1

    def _read_remote_latest_step(self, remote_root: str | None = None) -> int | None:
        """Read the latest checkpoint step from S3.

        Args:
            remote_root: Optional override for the remote root. If not provided, uses self.remote_root
        """
        marker_path = Path(self.checkpointer.folder) / LATEST_FILENAME
        try:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            download_file_from_s3(
                self.remote_up_down,
                self._remote_key(Path(LATEST_FILENAME), remote_root=remote_root),
                marker_path,
            )
        except Exception:  # noqa: BLE001
            return None
        try:
            return int(marker_path.read_text().strip())
        except ValueError:
            logger.warning("Invalid latest checkpoint marker downloaded from S3.")
            return None

    def _download_step(self, step: int, remote_path: str) -> None:
        """Download a specific checkpoint step from S3.

        Args:
            step: The checkpoint step number to download
            remote_path: The remote S3 path prefix (e.g., "torchtitan/16M-baseline-20251011-122516")
        """
        checkpoint_dir = self._checkpoint_dir(step)
        logger.info("Downloading checkpoint step %s to: %s", step, checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = checkpoint_dir / MANIFEST_FILENAME

        relative_base = Path(f"step-{step}")
        if self._ft_relative is not None:
            relative_base = self._ft_relative / relative_base

        remote_manifest_key = self._remote_key(
            relative_base / MANIFEST_FILENAME, remote_root=remote_path
        )
        logger.info(
            "Downloading manifest from S3: s3://%s/%s/%s",
            self.config.bucket,
            self.config.prefix,
            remote_manifest_key,
        )
        download_file_from_s3(
            self.remote_up_down,
            remote_manifest_key,
            manifest_path,
        )
        manifest_entries = json.loads(manifest_path.read_text())
        logger.info("Manifest contains %d files to download", len(manifest_entries))

        for relative in manifest_entries:
            relative_path = Path(relative)
            local_path = checkpoint_dir / relative_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            download_file_from_s3(
                self.remote_up_down,
                self._remote_key(
                    relative_base / relative_path, remote_root=remote_path
                ),
                local_path,
            )
        logger.info(
            "Successfully downloaded all %d files for step %s",
            len(manifest_entries),
            step,
        )

        # Verify the checkpoint directory structure
        metadata_file = checkpoint_dir / ".metadata"
        if metadata_file.exists():
            logger.info("✓ Checkpoint metadata file exists: %s", metadata_file)
        else:
            logger.error("✗ Checkpoint metadata file MISSING: %s", metadata_file)

        # List all downloaded files for verification
        all_files = list(checkpoint_dir.rglob("*"))
        logger.info(
            "Downloaded checkpoint contains %d total paths (files + directories)",
            len(all_files),
        )
        logger.info("Checkpoint directory structure:")
        for path in sorted(all_files)[:MAX_FILES_TO_DISPLAY]:
            logger.info("  - %s", path.relative_to(checkpoint_dir))
        if len(all_files) > MAX_FILES_TO_DISPLAY:
            logger.info("  ... and %d more", len(all_files) - MAX_FILES_TO_DISPLAY)

        self._latest_uploaded_step = max(self._latest_uploaded_step or -1, step)
        self._write_latest_marker(step)
        logger.info(
            "✓ Checkpoint download complete for step %s. Native checkpointer should now load it.",
            step,
        )


def setup_s3_checkpointing(
    checkpointer: CheckpointManager,
    job_config: MosaicJobConfig,
    *,
    install: bool = True,
) -> S3CheckpointManager | None:
    """Create an :class:`S3CheckpointManager` if configured."""
    config = job_config.s3_checkpoint
    if not config.enable:
        return None
    if not config.bucket or config.prefix is None:
        logger.warning(
            "S3 checkpointing is enabled but bucket or prefix is not provided; skipping."
        )
        return None

    manager = S3CheckpointManager(checkpointer, config, job_config)
    if install:
        manager.install()
    return manager
