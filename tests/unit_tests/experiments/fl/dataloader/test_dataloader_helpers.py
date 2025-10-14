from __future__ import annotations

import sys
import types
 

schedules_module = types.ModuleType("torch.distributed.pipelining.schedules")
schedules_module._PipelineSchedule = type("_PipelineSchedule", (), {})
schedules_module._PipelineScheduleRuntime = type("_PipelineScheduleRuntime", (), {})
schedules_module.PipelineScheduleMulti = type("PipelineScheduleMulti", (), {})
schedules_module.PipelineScheduleSingle = type("PipelineScheduleSingle", (), {})
schedules_module.ScheduleDualPipeV = type("ScheduleDualPipeV", (), {})
schedules_module.ScheduleZBVZeroBubble = type("ScheduleZBVZeroBubble", (), {})
schedules_module.get_schedule_class = lambda *_args, **_kwargs: schedules_module._PipelineSchedule
sys.modules.setdefault("torch.distributed.pipelining.schedules", schedules_module)

pipelining_module = types.ModuleType("torch.distributed.pipelining")
pipelining_module.PipelineStage = type("PipelineStage", (), {})
sys.modules.setdefault("torch.distributed.pipelining", pipelining_module)

torchmetrics_module = types.ModuleType("torchmetrics")
torchmetrics_module.Metric = type("Metric", (), {})
sys.modules.setdefault("torchmetrics", torchmetrics_module)

composer_module = types.ModuleType("composer")
composer_loggers = types.ModuleType("composer.loggers")


class _DummyRemoteUploaderDownloader:
    def __init__(self, *args, **kwargs):  # pragma: no cover - trivial stub
        pass

    def close(self) -> None:  # pragma: no cover - trivial stub
        pass


composer_loggers.RemoteUploaderDownloader = _DummyRemoteUploaderDownloader
composer_module.loggers = composer_loggers
sys.modules.setdefault("composer", composer_module)
sys.modules.setdefault("composer.loggers", composer_loggers)
remote_uploader_module = types.ModuleType("composer.loggers.remote_uploader_downloader")
remote_uploader_module._upload_worker = lambda *args, **kwargs: None  # noqa: ARG005
remote_uploader_module._download_worker = lambda *args, **kwargs: None  # noqa: ARG005
sys.modules.setdefault("composer.loggers.remote_uploader_downloader", remote_uploader_module)

llmfoundry_module = types.ModuleType("llmfoundry")
llmfoundry_module.registry = types.SimpleNamespace(tokenizers={})
sys.modules.setdefault("llmfoundry", llmfoundry_module)

registry_utils_module = types.ModuleType("llmfoundry.utils.registry_utils")
registry_utils_module.construct_from_registry = lambda *args, **kwargs: types.SimpleNamespace(eos_token="</s>")
sys.modules.setdefault("llmfoundry.utils.registry_utils", registry_utils_module)


class _DummyTokenizerBase:
    eos_token = "</s>"
    model_max_length = 0


class _DummyTokenizer(_DummyTokenizerBase):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


transformers_module = types.ModuleType("transformers")
transformers_module.AutoTokenizer = types.SimpleNamespace(
    from_pretrained=lambda *args, **kwargs: _DummyTokenizer(**kwargs)
)
transformers_module.PreTrainedTokenizerBase = _DummyTokenizerBase
transformers_module.PreTrainedTokenizerFast = _DummyTokenizerBase
sys.modules.setdefault("transformers", transformers_module)


class _DummyStream:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _DummyStreamingDataset:
    def __init__(
        self,
        *,
        streams=None,
        batch_size=None,
        split=None,
        epoch_size=None,
        shuffle=None,
        tokenizer=None,
        remote=None,
        local=None,
        **kwargs,
    ):
        self.streams = streams
        self.batch_size = batch_size
        self.split = split
        self.epoch_size = epoch_size
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.remote = remote
        self.local = local
        self.extra_kwargs = kwargs

    def __getitem__(self, idx):  # pragma: no cover - not exercised in tests
        return {"input_ids": [idx, idx + 1]}

    def state_dict(self, num_samples=None, from_beginning=True):  # pragma: no cover - simple stub
        return {
            "num_samples": num_samples,
            "from_beginning": from_beginning,
        }

    def load_state_dict(self, obj):  # pragma: no cover - simple stub
        return None


class _DummyStreamingTextDataset(_DummyStreamingDataset):
    last_init: dict[str, object] | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        result = {
            "streams": self.streams,
            "batch_size": self.batch_size,
            "split": self.split,
            "epoch_size": self.epoch_size,
            "shuffle": getattr(self, "shuffle", None),
            "kwargs": self.extra_kwargs,
        }
        type(self).last_init = result
        _DummyStreamingTextDataset.last_init = result


def _install_streaming_stubs() -> None:
    streaming_module = types.ModuleType("streaming")
    streaming_module.Stream = _DummyStream
    streaming_module.StreamingDataset = _DummyStreamingDataset
    sys.modules.setdefault("streaming", streaming_module)

    base_module = types.ModuleType("streaming.base")
    util_module = types.ModuleType("streaming.base.util")
    util_module.clean_stale_shared_memory = lambda: None  # noqa: ARG005
    base_module.util = util_module
    sys.modules.setdefault("streaming.base", base_module)
    sys.modules.setdefault("streaming.base.util", util_module)

    llmfoundry_module = types.ModuleType("llmfoundry")
    data_module = types.ModuleType("llmfoundry.data")
    text_data_module = types.ModuleType("llmfoundry.data.text_data")
    text_data_module.StreamingTextDataset = _DummyStreamingTextDataset
    llmfoundry_module.data = data_module
    data_module.text_data = text_data_module
    sys.modules.setdefault("llmfoundry", llmfoundry_module)
    sys.modules.setdefault("llmfoundry.data", data_module)
    sys.modules.setdefault("llmfoundry.data.text_data", text_data_module)


_install_streaming_stubs()

from torchtitan.experiments.fl.dataloader.dataloader import (  # noqa: E402  # isort: skip
    DatasetFactoryConfig,
    MosaicRuntimeConfig,
    StreamAssignment,
    StreamExtractionResult,
    _create_streaming_dataset,
    _normalize_mosaic_dataloader_config,
    _prepare_dataset_kwargs,
    _select_stream_subset,
    _setup_unigram_metric,
    titan_collate_fn,
)


class _DummyTokenizerForTest:
    tokenizer = object()


def _make_job_config():
    return types.SimpleNamespace(
        mosaic_dataloader={
            "num_workers": 8,
            "prefetch_factor": 2,
            "pin_memory": True,
            "persistent_workers": True,
            "drop_last": True,
            "dataset": {},
        },
        training=types.SimpleNamespace(local_batch_size=8),
        validation=types.SimpleNamespace(local_batch_size=8),
        unigram_metric=types.SimpleNamespace(
            enable=False,
            allow_failures=False,
            download_missing=False,
            num_attempts=1,
            client_config={},
            ignore_index=-100,
        ),
    )


def test_normalize_mosaic_dataloader_config_applies_split_overrides() -> None:
    job_config = _make_job_config()
    job_config.mosaic_dataloader.update(
        {
            "dataset": {
                "common": {"foo": 1},
                "train": {"bar": 2},
            },
            "train": {
                "num_workers": 2,
                "prefetch_factor": 4,
                "drop_last": False,
            },
        }
    )

    normalized = _normalize_mosaic_dataloader_config(
        job_config,
        split="train",
        default_drop_last=True,
    )

    assert normalized.dataset_config["foo"] == 1
    assert normalized.dataset_config["bar"] == 2
    assert normalized.runtime == MosaicRuntimeConfig(
        num_workers=2,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
        batch_size=job_config.training.local_batch_size,
    )


def test_select_stream_subset_falls_back_to_rank_round_robin() -> None:
    streams = [_DummyStream(name=f"stream_{idx}") for idx in range(3)]
    extraction = StreamExtractionResult(
        streams=streams,
        dataset_config={},
        sampling_group_indices=[[0, 1, 2]],
        dataset_root_remote="remote",
        dataset_split_remote="train",
    )

    assignment = _select_stream_subset(
        extraction,
        dp_rank=3,
        dp_world_size=4,
    )

    assert assignment.group_index == 3
    assert [stream.name for stream in assignment.streams] == ["stream_0"]
    assert assignment.dataset_root_remote == "remote"
    assert assignment.dataset_split_remote == "train"


def test_setup_unigram_metric_allows_failures_when_missing_counts() -> None:
    job_config = _make_job_config()
    job_config.unigram_metric.enable = True
    job_config.unigram_metric.allow_failures = True

    assignment = StreamAssignment(
        streams=[_DummyStream(name="broken")],
        group_index=0,
        dataset_root_remote=None,
        dataset_split_remote=None,
    )

    setup = _setup_unigram_metric(
        assignment,
        job_config=job_config,
        split="train",
        tokenizer=_DummyTokenizerForTest(),
    )

    assert setup.collate_fn is titan_collate_fn
    assert setup.group_key is None


def test_prepare_dataset_kwargs_sets_epoch_and_split() -> None:
    dataset_cfg = {
        "subset_num_samples": 7,
        "shuffle": True,
        "unused": "ignored",
    }

    factory_config = _prepare_dataset_kwargs(
        dataset_cfg,
        dataset_split_remote="val",
    )

    assert factory_config == DatasetFactoryConfig(
        kwargs={
            "shuffle": True,
            "epoch_size": 7,
            "split": "val",
        }
    )
    assert dataset_cfg == {
        "subset_num_samples": 7,
        "shuffle": True,
        "unused": "ignored",
    }


def test_create_streaming_dataset_uses_resolved_kwargs() -> None:
    streams = [_DummyStream(name="s0")]
    assignment = StreamAssignment(
        streams=streams,
        group_index=None,
        dataset_root_remote=None,
        dataset_split_remote=None,
    )
    dataset_config = DatasetFactoryConfig(
        kwargs={
            "shuffle": False,
        }
    )

    _DummyStreamingTextDataset.last_init = None
    dataset = _create_streaming_dataset(
        assignment=assignment,
        tokenizer=_DummyTokenizerForTest(),
        dataset_config=dataset_config,
        batch_size=4,
        split="train",
    )

    assert isinstance(dataset, _DummyStreamingTextDataset)
    assert _DummyStreamingTextDataset.last_init == {
        "streams": streams,
        "batch_size": 4,
        "split": None,
        "epoch_size": None,
        "shuffle": False,
        "kwargs": {},
    }
