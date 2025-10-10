# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import shutil
from typing import Any

from composer.loggers import RemoteUploaderDownloader
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.experiments.fl.configs.config import S3Config
from torchtitan.tools.logging import logger


def create_remote_up_down(
    s3_config: S3Config,
) -> RemoteUploaderDownloader:
    """Create the remote uploader/downloader."""
    bucket_uri = f"s3://{s3_config.bucket_name}"
    remote_up_down = RemoteUploaderDownloader(
        bucket_uri=bucket_uri,
        backend_kwargs={
            "bucket": s3_config.bucket_name,
            "prefix": s3_config.prefix,
            "region_name": None,
            "endpoint_url": None,
            "aws_access_key_id": None,
            "aws_secret_access_key": None,
            "aws_session_token": None,
            "client_config": s3_config.client_config,
            "transfer_config": None,
        },
        file_path_format_string="{remote_file_name}",
        num_concurrent_uploads=s3_config.num_concurrent_uploads,
        upload_staging_folder=None,
        use_procs=True,
        num_attempts=s3_config.num_attempts,
    )
    # Per user instruction, using run_name in init.
    # This assumes the composer version supports this or that this is the intended usage pattern.
    remote_up_down.init(run_name=s3_config.run_uuid)
    return remote_up_down


class S3CheckpointWrapper:
    """
    A wrapper around CheckpointManager to add S3 functionality.
    """

    def __init__(
        self,
        checkpointer: CheckpointManager,
        s3_config: S3Config,
    ):
        self.checkpointer = checkpointer
        self.s3_config = s3_config
        if self.s3_config.enabled:
            self.remote_up_down = create_remote_up_down(s3_config)

    def save(self, curr_step: int, last_step: bool = False) -> None:
        # First, save locally using the original checkpointer
        self.checkpointer.save(curr_step, last_step)

        if not self.s3_config.enabled:
            return

        local_checkpoint_dir = self.checkpointer._create_checkpoint_id(curr_step)
        if not os.path.isdir(local_checkpoint_dir):
            # The checkpointer decided not to save, so we shouldn't either.
            return

        remote_checkpoint_dir = os.path.basename(local_checkpoint_dir)
        logger.info(f"Uploading checkpoint {local_checkpoint_dir} to S3.")

        # Create a manifest of all files in the checkpoint directory
        manifest_files = []
        for root, _, files in os.walk(local_checkpoint_dir):
            for file in files:
                # Do not include the manifest in the manifest
                if file == "manifest.json":
                    continue
                relative_path = os.path.relpath(
                    os.path.join(root, file), local_checkpoint_dir
                )
                manifest_files.append(relative_path)

        manifest_path = os.path.join(local_checkpoint_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest_files, f)

        # Upload the manifest first
        self.remote_up_down.upload_file(
            state=None,
            remote_file_name=os.path.join(remote_checkpoint_dir, "manifest.json"),
            file_path=manifest_path,
            overwrite=True,
        )

        # Upload all files from the manifest
        for file_path in manifest_files:
            local_path = os.path.join(local_checkpoint_dir, file_path)
            remote_path = os.path.join(remote_checkpoint_dir, file_path)
            self.remote_up_down.upload_file(
                state=None,
                remote_file_name=remote_path,
                file_path=local_path,
                overwrite=True,
            )

        logger.info(f"Finished uploading checkpoint {local_checkpoint_dir} to S3.")

    def load(self, step: int = -1) -> bool:
        if not self.s3_config.enabled:
            return self.checkpointer.load(step)

        if step == -1:
            logger.warning(
                "S3 load requires an explicit step. Falling back to local checkpointer."
            )
            return self.checkpointer.load(step)

        remote_checkpoint_dir = os.path.basename(
            self.checkpointer._create_checkpoint_id(step)
        )
        local_checkpoint_dir = self.checkpointer._create_checkpoint_id(step)

        logger.info(f"Downloading checkpoint for step {step} from S3.")

        # Download manifest
        remote_manifest_path = os.path.join(remote_checkpoint_dir, "manifest.json")
        local_manifest_path = os.path.join(local_checkpoint_dir, "manifest.json")
        os.makedirs(local_checkpoint_dir, exist_ok=True)

        try:
            self.remote_up_down.download_file(
                remote_file_name=remote_manifest_path,
                destination=local_manifest_path,
                overwrite=True,
            )
        except Exception as e:
            logger.error(
                f"Failed to download manifest from S3: {e}. "
                "Falling back to local checkpointer."
            )
            shutil.rmtree(local_checkpoint_dir, ignore_errors=True)
            return self.checkpointer.load(step)

        with open(local_manifest_path, "r") as f:
            manifest_files = json.load(f)

        # Download all files from manifest
        for file_path in manifest_files:
            remote_path = os.path.join(remote_checkpoint_dir, file_path)
            local_path = os.path.join(local_checkpoint_dir, file_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.remote_up_down.download_file(
                remote_file_name=remote_path,
                destination=local_path,
                overwrite=True,
            )

        logger.info(f"Finished downloading checkpoint for step {step} from S3.")

        return self.checkpointer.load(step)

    def __getattr__(self, name: str) -> Any:
        """Delegate other attribute access to the wrapped checkpointer."""
        return getattr(self.checkpointer, name)