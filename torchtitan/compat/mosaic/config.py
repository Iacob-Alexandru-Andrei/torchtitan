# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pydantic schemas for MosaicML streaming dataset configuration."""

from __future__ import annotations

from typing import Dict, Optional

from pydantic.dataclasses import dataclass


@dataclass
class Stream:
    """Configuration for a single data stream.

    Attributes:
        local (str): Local path to the stream.
        remote (str): Remote path to the stream.
        proportion (float): The proportion of samples to draw from this stream.
        choose (Optional[int]): The number of samples to draw from this stream.
        download_retry (Optional[int]): Number of download retries.
        download_timeout (Optional[int]): Download timeout in seconds.
        keep_zip (Optional[bool]): Whether to keep the zip file after extraction.
        repeat (Optional[int]): The number of times to repeat the stream.
        validate_hash (Optional[str]): The hash to validate the stream against.
    """

    local: str
    remote: str
    proportion: float
    choose: Optional[int] = None
    download_retry: Optional[int] = 2
    download_timeout: Optional[int] = 60
    keep_zip: Optional[bool] = False
    repeat: Optional[int] = None
    validate_hash: Optional[str] = None


@dataclass
class DatasetSplit:
    """Configuration for a dataset split (e.g., train or val).

    Attributes:
        max_seq_len (int): The maximum sequence length for the model.
        remote (Optional[str]): Remote path to the dataset.
        local (Optional[str]): Local path to the dataset.
        split (Optional[str]): The dataset split to use (e.g., 'train', 'val').
        download_retry (int): Number of download retries.
        download_timeout (int): Download timeout in seconds.
        validate_hash (Optional[str]): The hash to validate the dataset against.
        keep_zip (bool): Whether to keep the zip file after extraction.
        epoch_size (Optional[int]): The number of samples per epoch.
        predownload (Optional[int]): The number of samples to pre-download.
        cache_limit (Optional[str]): The cache limit for the dataset.
        partition_algo (str): The partitioning algorithm to use.
        num_canonical_nodes (Optional[int]): The number of canonical nodes.
        shuffle (bool): Whether to shuffle the dataset.
        shuffle_algo (str): The shuffling algorithm to use.
        shuffle_seed (int): The seed for shuffling.
        shuffle_block_size (Optional[int]): The block size for shuffling.
        sampling_method (str): The sampling method to use.
        sampling_granularity (int): The sampling granularity.
        batching_method (str): The batching method to use.
        streams (Optional[Dict[str, Stream]]): A dictionary of data streams.
    """

    max_seq_len: int
    remote: Optional[str] = None
    local: Optional[str] = None
    split: Optional[str] = None
    download_retry: int = 2
    download_timeout: int = 60
    validate_hash: Optional[str] = None
    keep_zip: bool = False
    epoch_size: Optional[int] = None
    predownload: Optional[int] = None
    cache_limit: Optional[str] = None
    partition_algo: str = "relaxed"
    num_canonical_nodes: Optional[int] = None
    shuffle: bool = True
    shuffle_algo: str = "py1s"
    shuffle_seed: int = 9176
    shuffle_block_size: Optional[int] = None
    sampling_method: str = "balanced"
    sampling_granularity: int = 1
    batching_method: str = "random"
    streams: Optional[Dict[str, Stream]] = None
