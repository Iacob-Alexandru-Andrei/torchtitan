# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from importlib import import_module
from types import ModuleType, SimpleNamespace
from typing import Any

if "torchmetrics" not in sys.modules:
    _torchmetrics = ModuleType("torchmetrics")

    class _Metric:  # pragma: no cover - stub for optional dependency
        pass

    _torchmetrics.Metric = _Metric
    sys.modules["torchmetrics"] = _torchmetrics

if "llmfoundry" not in sys.modules:
    _llmfoundry = ModuleType("llmfoundry")
    _llmfoundry_data = ModuleType("llmfoundry.data")
    _llmfoundry_text = ModuleType("llmfoundry.data.text_data")

    class _StreamingTextDataset:  # pragma: no cover - stub for optional dependency
        pass

    _llmfoundry_text.StreamingTextDataset = _StreamingTextDataset
    _llmfoundry.registry = SimpleNamespace(tokenizers={})
    sys.modules["llmfoundry"] = _llmfoundry
    sys.modules["llmfoundry.data"] = _llmfoundry_data
    sys.modules["llmfoundry.data.text_data"] = _llmfoundry_text

    _llmfoundry_utils = ModuleType("llmfoundry.utils")
    sys.modules["llmfoundry.utils"] = _llmfoundry_utils

    _llmfoundry_registry_utils = ModuleType("llmfoundry.utils.registry_utils")

    def construct_from_registry(
        *_args: Any, **_kwargs: Any
    ) -> Any:  # pragma: no cover - stub
        tokenizer = SimpleNamespace(eos_token="</s>", model_max_length=0)
        return tokenizer

    _llmfoundry_registry_utils.construct_from_registry = construct_from_registry
    sys.modules["llmfoundry.utils.registry_utils"] = _llmfoundry_registry_utils

if "transformers" not in sys.modules:
    _transformers = ModuleType("transformers")

    class _PreTrainedTokenizerBase:  # pragma: no cover - stub
        eos_token = "</s>"
        model_max_length = 0

    class _PreTrainedTokenizerFast(_PreTrainedTokenizerBase):  # pragma: no cover
        pass

    class _AutoTokenizer:  # pragma: no cover - stub
        @staticmethod
        def from_pretrained(*_args: Any, **_kwargs: Any) -> _PreTrainedTokenizerFast:
            return _PreTrainedTokenizerFast()

    _transformers.AutoTokenizer = _AutoTokenizer
    _transformers.PreTrainedTokenizerBase = _PreTrainedTokenizerBase
    _transformers.PreTrainedTokenizerFast = _PreTrainedTokenizerFast
    sys.modules["transformers"] = _transformers

if "streaming" not in sys.modules:
    _streaming = ModuleType("streaming")

    class _Stream:  # pragma: no cover - stub for optional dependency
        pass

    class _StreamingDataset:  # pragma: no cover - stub for optional dependency
        pass

    _streaming.Stream = _Stream
    _streaming.StreamingDataset = _StreamingDataset
    sys.modules["streaming"] = _streaming

if "streaming.base" not in sys.modules:
    _streaming_base = ModuleType("streaming.base")
    _streaming_base_util = ModuleType("streaming.base.util")

    def _clean_stale_shared_memory() -> None:  # pragma: no cover - stub
        return None

    _streaming_base_util.clean_stale_shared_memory = _clean_stale_shared_memory
    sys.modules["streaming.base"] = _streaming_base
    sys.modules["streaming.base.util"] = _streaming_base_util

if "composer" not in sys.modules:
    _composer = ModuleType("composer")
    _composer_loggers = ModuleType("composer.loggers")
    _composer_loggers.__path__ = []  # pragma: no cover - mark as package

    class _RemoteUploaderDownloader:  # pragma: no cover - stub
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    _composer_loggers.RemoteUploaderDownloader = _RemoteUploaderDownloader
    sys.modules["composer"] = _composer
    sys.modules["composer.loggers"] = _composer_loggers

    _remote_module = ModuleType("composer.loggers.remote_uploader_downloader")

    def _upload_worker(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - stub
        return None

    _remote_module._upload_worker = _upload_worker
    sys.modules["composer.loggers.remote_uploader_downloader"] = _remote_module

_schedules = import_module("torch.distributed.pipelining.schedules")
if not hasattr(_schedules, "ScheduleDualPipeV"):

    class _ScheduleDualPipeV:  # pragma: no cover - stub for missing torch API
        ...

    _schedules.ScheduleDualPipeV = _ScheduleDualPipeV

if not hasattr(_schedules, "ScheduleZBVZeroBubble"):

    class _ScheduleZBVZeroBubble:  # pragma: no cover - stub for missing torch API
        ...

    _schedules.ScheduleZBVZeroBubble = _ScheduleZBVZeroBubble

from torchtitan.experiments.fl.models.utils import (
    MosaicSpecOverrides,
    ensure_mosaic_spec,
)
from torchtitan.protocols import train_spec as train_spec_module
from torchtitan.protocols.train_spec import (
    register_train_spec,
    TrainSpec,
    unregister_train_spec,
)


@dataclass
class _DummyModel:
    name: str = "dummy"


def _dummy_parallelize(*args: Any, **kwargs: Any) -> None:  # noqa: D401
    """No-op parallelize helper."""


def _dummy_pipeline(*args: Any, **kwargs: Any) -> tuple[None, list[None], bool, bool]:
    return (None, [], False, False)


def _dummy_builder(*args: Any, **kwargs: Any) -> None:
    return None


def test_ensure_mosaic_spec_is_idempotent_for_multiple_models() -> None:
    base_spec_a = TrainSpec(
        name="test_base_a",
        model_cls=_DummyModel,
        model_args={"cfg": SimpleNamespace(vocab_size=10)},
        parallelize_fn=_dummy_parallelize,
        pipelining_fn=_dummy_pipeline,
        build_optimizers_fn=_dummy_builder,
        build_lr_schedulers_fn=lambda *args, **kwargs: None,
        build_dataloader_fn=_dummy_builder,
        build_tokenizer_fn=_dummy_builder,
        build_loss_fn=_dummy_builder,
    )
    base_spec_b = replace(
        base_spec_a,
        name="test_base_b",
        model_args={"cfg": SimpleNamespace(vocab_size=32)},
    )

    register_train_spec(base_spec_a)
    register_train_spec(base_spec_b)

    def _post_transform(base_spec: TrainSpec, mosaic_spec: TrainSpec) -> TrainSpec:
        model_args = {
            name: SimpleNamespace(
                **{**config.__dict__, "vocab_size": config.vocab_size + 5}
            )
            for name, config in base_spec.model_args.items()
        }
        return replace(mosaic_spec, model_args=model_args)

    try:
        mosaic_a = ensure_mosaic_spec(
            base_spec_a.name,
            spec_name="mosaic_test_a",
            overrides=MosaicSpecOverrides(
                dataloader=_dummy_builder,
                tokenizer=_dummy_builder,
                metrics_processor=_dummy_builder,
            ),
        )
        mosaic_b = ensure_mosaic_spec(
            base_spec_b.name,
            spec_name="mosaic_test_b",
            overrides=MosaicSpecOverrides(
                dataloader=_dummy_builder,
                tokenizer=_dummy_builder,
                metrics_processor=_dummy_builder,
                optimizers=_dummy_builder,
                validator=_dummy_builder,
                post_transform=_post_transform,
            ),
        )

        assert mosaic_a == "mosaic_test_a"
        assert mosaic_b == "mosaic_test_b"

        mosaic_spec_a = train_spec_module.get_train_spec(mosaic_a)
        mosaic_spec_b = train_spec_module.get_train_spec(mosaic_b)

        assert mosaic_spec_a.build_dataloader_fn is _dummy_builder
        assert mosaic_spec_a.build_tokenizer_fn is _dummy_builder
        assert mosaic_spec_b.build_optimizers_fn is _dummy_builder
        assert mosaic_spec_b.build_validator_fn is _dummy_builder

        base_vocab = base_spec_b.model_args["cfg"].vocab_size
        assert mosaic_spec_b.model_args["cfg"].vocab_size == base_vocab + 5

        assert (
            ensure_mosaic_spec(
                base_spec_a.name,
                spec_name="mosaic_test_a",
                overrides=MosaicSpecOverrides(
                    dataloader=_dummy_builder,
                    tokenizer=_dummy_builder,
                    metrics_processor=_dummy_builder,
                ),
            )
            == "mosaic_test_a"
        )
        assert train_spec_module.get_train_spec(mosaic_a) is mosaic_spec_a

        assert (
            ensure_mosaic_spec(
                base_spec_b.name,
                spec_name="mosaic_test_b",
                overrides=MosaicSpecOverrides(
                    dataloader=_dummy_builder,
                    tokenizer=_dummy_builder,
                    metrics_processor=_dummy_builder,
                    optimizers=_dummy_builder,
                    validator=_dummy_builder,
                    post_transform=_post_transform,
                ),
            )
            == "mosaic_test_b"
        )
        assert train_spec_module.get_train_spec(mosaic_b) is mosaic_spec_b
    finally:
        for spec_name in (
            "mosaic_test_a",
            "mosaic_test_b",
            base_spec_a.name,
            base_spec_b.name,
        ):
            unregister_train_spec(spec_name)


def test_ensure_mosaic_spec_updates_existing_spec_with_new_overrides() -> None:
    base_spec = TrainSpec(
        name="test_base_update",
        model_cls=_DummyModel,
        model_args={"cfg": SimpleNamespace(vocab_size=7)},
        parallelize_fn=_dummy_parallelize,
        pipelining_fn=_dummy_pipeline,
        build_optimizers_fn=_dummy_builder,
        build_lr_schedulers_fn=lambda *args, **kwargs: None,
        build_dataloader_fn=_dummy_builder,
        build_tokenizer_fn=_dummy_builder,
        build_loss_fn=_dummy_builder,
    )
    register_train_spec(base_spec)

    try:
        spec_name = ensure_mosaic_spec(
            base_spec.name,
            spec_name="mosaic_update",
            overrides=MosaicSpecOverrides(
                dataloader=_dummy_builder,
                tokenizer=_dummy_builder,
                metrics_processor=_dummy_builder,
            ),
        )
        initial_spec = train_spec_module.get_train_spec(spec_name)
        assert initial_spec.build_optimizers_fn is _dummy_builder

        def _post_transform(base_spec: TrainSpec, mosaic_spec: TrainSpec) -> TrainSpec:
            return replace(
                mosaic_spec,
                model_args={
                    name: SimpleNamespace(vocab_size=config.vocab_size + 1)
                    for name, config in base_spec.model_args.items()
                },
            )

        ensure_mosaic_spec(
            base_spec.name,
            spec_name=spec_name,
            overrides=MosaicSpecOverrides(
                dataloader=_dummy_builder,
                tokenizer=_dummy_builder,
                metrics_processor=_dummy_builder,
                optimizers=lambda *args, **kwargs: None,
                validator=_dummy_builder,
                post_transform=_post_transform,
            ),
        )

        updated_spec = train_spec_module.get_train_spec(spec_name)
        assert updated_spec is not initial_spec
        assert updated_spec.build_optimizers_fn is not _dummy_builder
        assert updated_spec.build_validator_fn is _dummy_builder
        assert (
            updated_spec.model_args["cfg"].vocab_size
            == base_spec.model_args["cfg"].vocab_size + 1
        )
    finally:
        unregister_train_spec("mosaic_update")
        unregister_train_spec(base_spec.name)
