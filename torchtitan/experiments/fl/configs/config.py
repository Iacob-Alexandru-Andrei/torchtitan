# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration for MosaicML training jobs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from torchtitan.components.ft.config import FaultTolerance as FTFaultTolerance

from torchtitan.config import JobConfig
from torchtitan.experiments.fl.configs.optimizers import MosaicOptimizerConfig


DEFAULT_DATASET_SPLIT_KEYS = frozenset(
    {"train", "val", "validation", "test", "eval", "train_eval"}
)


def _as_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Create a plain ``dict`` from an arbitrary mapping value."""
    if value is None:
        return {}
    return dict(value)


@dataclass
class MosaicTokenizerConfig:
    """Configuration describing how to build a Mosaic tokenizer."""

    name: str = field(
        default="",
        metadata={
            "help": "Tokenizer identifier. Either an llm-foundry registry entry or a HuggingFace model name.",
        },
    )
    # python-explicit-any
    kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Keyword arguments forwarded to the tokenizer constructor (HuggingFace or llm-foundry).",
        },
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MosaicTokenizerConfig:
        """Instantiate the configuration from a mapping payload.

        Args:
            data: Mapping describing the tokenizer ``name`` and optional
                ``kwargs`` forwarded to the underlying builder.

        Returns:
            mosaictokenizerconfig: Typed configuration wrapping the provided
            mapping.
        """
        return cls(name=str(data.get("name", "")), kwargs=_as_dict(data.get("kwargs")))

    @classmethod
    def coerce(cls, value: Any) -> MosaicTokenizerConfig:
        """Convert arbitrary inputs into a :class:`MosaicTokenizerConfig`.

        Args:
            value: Existing :class:`MosaicTokenizerConfig` instance or a mapping
                with ``name`` and ``kwargs`` keys.

        Returns:
            mosaictokenizerconfig: Valid configuration describing the tokenizer
            to instantiate.

        Raises:
            TypeError: If ``value`` cannot be interpreted as a tokenizer config.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        msg = f"Cannot convert {type(value)} to MosaicTokenizerConfig"
        raise TypeError(msg)


@dataclass
class MosaicDataLoaderConfig:
    """Configuration describing how to build the Mosaic streaming dataloader."""

    name: str = field(
        default="",
        metadata={
            "help": "Identifier of the Mosaic dataloader. Currently informational only.",
        },
    )
    # python-explicit-any
    dataset: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Nested configuration forwarded to Mosaic's StreamingDataset/StreamingTextDataset constructors.",
        },
    )
    num_workers: int = field(
        default=8,
        metadata={"help": "Number of worker processes used by the DataLoader."},
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "Prefetch factor for each DataLoader worker."},
    )
    pin_memory: bool = field(
        default=True,
        metadata={
            "help": "Pin tensors in page-locked memory for faster host→device transfers."
        },
    )
    persistent_workers: bool = field(
        default=True,
        metadata={"help": "Keep DataLoader workers alive across epochs."},
    )
    drop_last: bool | None = field(
        default=None,
        metadata={
            "help": "Override drop_last behaviour. If unset, defaults are inferred from the training/validation split.",
        },
    )
    # python-explicit-any
    split_overrides: dict[str, dict[str, Any]] = field(
        default_factory=dict,
        metadata={
            "help": "Per-split overrides (e.g. 'train', 'val') applied on top of the common DataLoader settings.",
        },
    )
    # python-explicit-any
    extras: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Additional configuration keys preserved for downstream consumers."
        },
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MosaicDataLoaderConfig:
        """Instantiate a dataloader config from an arbitrary mapping.

        Args:
            data: Mapping describing the dataset factory and runtime overrides.

        Returns:
            mosaicdataloaderconfig: Normalized dataclass mirroring the mapping
            structure.
        """
        known_keys = {
            "name",
            "dataset",
            "num_workers",
            "prefetch_factor",
            "pin_memory",
            "persistent_workers",
            "drop_last",
            "split_overrides",
        }

        inferred_split_overrides: dict[str, dict[str, Any]] = {}
        extras: dict[str, Any] = {}
        for key, value in data.items():
            if key in known_keys:
                continue
            if isinstance(value, Mapping) and key in DEFAULT_DATASET_SPLIT_KEYS:
                inferred_split_overrides[key] = _as_dict(value)
            else:
                extras[key] = _as_dict(value) if isinstance(value, Mapping) else value

        explicit_split_overrides: dict[str, dict[str, Any]] = {}
        if "split_overrides" in data and isinstance(data["split_overrides"], Mapping):
            for key, value in data["split_overrides"].items():
                if isinstance(value, Mapping):
                    explicit_split_overrides[key] = _as_dict(value)

        dataset_cfg = data.get("dataset", {})
        if dataset_cfg and not isinstance(dataset_cfg, Mapping):
            msg = "mosaic_dataloader.dataset must be a mapping"
            raise TypeError(msg)

        drop_last = data.get("drop_last")
        drop_last_bool: bool | None
        if drop_last is None:
            drop_last_bool = None
        elif isinstance(drop_last, bool):
            drop_last_bool = drop_last
        else:
            drop_last_bool = bool(drop_last)

        combined_split_overrides: dict[str, dict[str, Any]] = dict(
            inferred_split_overrides
        )
        combined_split_overrides.update(explicit_split_overrides)

        return cls(
            name=str(data.get("name", "")),
            dataset=_as_dict(dataset_cfg),
            num_workers=int(data.get("num_workers", 8)),
            prefetch_factor=int(data.get("prefetch_factor", 2)),
            pin_memory=bool(data.get("pin_memory", True)),
            persistent_workers=bool(data.get("persistent_workers", True)),
            drop_last=drop_last_bool,
            split_overrides=combined_split_overrides,
            extras=extras,
        )

    @classmethod
    def coerce(cls, value: Any) -> MosaicDataLoaderConfig:
        """Convert raw configuration payloads into a dataclass instance.

        Args:
            value: :class:`MosaicDataLoaderConfig` or a mapping read from TOML or
                CLI arguments.

        Returns:
            mosaicdataloaderconfig: Normalized dataclass ready for downstream
            consumption.

        Raises:
            TypeError: If ``value`` cannot be interpreted as a dataloader config.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        msg = f"Cannot convert {type(value)} to MosaicDataLoaderConfig"
        raise TypeError(msg)

    def get_split_overrides(self, split: str) -> dict[str, Any]:
        """Return per-split overrides merged onto the default dataset config.

        Args:
            split: Dataset split identifier, such as ``"train"`` or ``"val"``.

        Returns:
            dict[str, any]: Mapping of overrides to apply for the requested
            split.
        """
        overrides = self.split_overrides.get(split)
        if overrides is None:
            return {}
        return dict(overrides)


@dataclass
class OptimizerMonitorConfig:
    """Configuration for optimizer monitoring."""

    interval: int = field(
        default=10,
        metadata={
            "help": "Interval (in steps) for the optimizer monitor to log metrics. "
            "Set to 0 to disable optimizer monitoring."
        },
    )

    only_global: bool = field(
        default=True,
        metadata={
            "help": "If True, only log global aggregated metrics. If False, log per-parameter metrics as well."
        },
    )

    log_metrics: bool = field(
        default=True,
        metadata={
            "help": "If True, log detailed optimizer metrics (moments, updates, etc.). "
            "If False, only log gradient norms."
        },
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of gradient accumulation steps."},
    )

    enabled_metrics: set[str] | None = field(
        default=None,
        metadata={"help": "Set of enabled metrics. If None, all metrics are enabled."},
    )


@dataclass
class ActivationMonitorConfig:
    """Configuration for activation monitoring."""

    enabled: bool = field(
        default=False,
        metadata={"help": "Enable logging of full-model activation statistics."},
    )

    interval: int = field(
        default=25,
        metadata={
            "help": "Training step interval for activation monitoring. Set to 0 to disable."
        },
    )

    ignore_module_types: tuple[str, ...] = field(
        default_factory=tuple,
        metadata={
            "help": "Optional substrings of module qualified names to skip when collecting activations."
        },
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of gradient accumulation steps."},
    )

    enabled_metrics: set[str] | None = field(
        default=None,
        metadata={"help": "Set of enabled metrics. If None, all metrics are enabled."},
    )


@dataclass
class LRMonitorConfig:
    """Configuration for learning rate monitoring."""

    enabled: bool = field(
        default=True,
        metadata={"help": "Enable logging of optimizer learning rates."},
    )

    interval: int = field(
        default=1,
        metadata={"help": "Training step interval for learning rate logging."},
    )


@dataclass
class BetasMonitorConfig:
    """Configuration for optimizer beta parameter monitoring."""

    enabled: bool = field(
        default=False,
        metadata={"help": "Enable logging of optimizer beta tuples."},
    )

    interval: int = field(
        default=0,
        metadata={"help": "Training step interval for beta logging."},
    )


@dataclass
class VSMonitorConfig:
    """Configuration for quasi-hyperbolic v parameter monitoring."""

    enabled: bool = field(
        default=False,
        metadata={
            "help": "Enable logging of quasi-hyperbolic v parameters if present."
        },
    )

    interval: int = field(
        default=0,
        metadata={"help": "Training step interval for v parameter logging."},
    )


@dataclass
class FLLRSchedulerConfig:
    """Warmup-stable-decay scheduler with optional mid-training switch."""

    warmup_steps: int = field(
        default=200,
        metadata={
            "help": "Number of warmup steps before the learning rate reaches its base value."
        },
    )

    decay_ratio: float | None = field(
        default=None,
        metadata={
            "help": (
                "Portion of total steps allocated to decay (Warmup-Stable-Decay). "
                "Matches the semantics of the core WSD scheduler."
            )
        },
    )

    decay_type: Literal["linear", "sqrt", "cosine"] = field(
        default="linear",
        metadata={
            "help": "Decay shape applied after the stable region: linear, sqrt, or cosine."
        },
    )

    min_lr_factor: float = field(
        default=0.0,
        metadata={
            "help": "Minimum multiplier applied to the base learning rate during decay."
        },
    )

    switch_step: int | None = field(
        default=None,
        metadata={
            "help": (
                "1-based training step at which to apply `switch_scale`. "
                "If unset, no additional scaling is applied."
            )
        },
    )

    switch_scale: float = field(
        default=1.0,
        metadata={
            "help": (
                "Multiplicative factor applied to the learning rate once `switch_step` is reached. "
                "Use 1.0 to disable."
            )
        },
    )


@dataclass
class HyperparameterSwitchConfig:
    """Configuration for switching quasi-hyperbolic optimizer parameters mid-training."""

    enabled: bool = field(
        default=False,
        metadata={"help": "Enable switching betas/vs at specific training steps."},
    )

    steps: tuple[int, ...] = field(
        default_factory=tuple,
        metadata={
            "help": "Training steps (1-based) at which to apply the new parameters."
        },
    )

    new_vs: tuple[float, ...] | None = field(
        default=None,
        metadata={
            "help": "Replacement values for optimizer 'vs' entries. Leave empty to skip."
        },
    )

    new_betas: tuple[float, ...] | None = field(
        default=None,
        metadata={
            "help": "Replacement values for optimizer 'betas' entries. Leave empty to skip."
        },
    )

    reset_momenta: tuple[str, ...] = field(
        default_factory=tuple,
        metadata={
            "help": (
                "Optimizer state keys (e.g. 'exp_avg', 'exp_avg_sq') to zero when the switch "
                "fires. Leave empty to keep existing momenta."
            )
        },
    )

    log_metrics: bool = field(
        default=True,
        metadata={
            "help": "If True, log the updated hyperparameters as scalar metrics."
        },
    )


@dataclass
class MetricsConfig:
    """Configuration for all metrics and monitoring."""

    optimizer_monitor: OptimizerMonitorConfig = field(
        default_factory=OptimizerMonitorConfig,
        metadata={"help": "Configuration for optimizer monitoring."},
    )

    activation_monitor: ActivationMonitorConfig = field(
        default_factory=ActivationMonitorConfig,
        metadata={"help": "Configuration for activation monitoring."},
    )

    lr_monitor: LRMonitorConfig = field(
        default_factory=LRMonitorConfig,
        metadata={"help": "Configuration for learning rate monitoring."},
    )

    betas_monitor: BetasMonitorConfig = field(
        default_factory=BetasMonitorConfig,
        metadata={"help": "Configuration for optimizer beta monitoring."},
    )

    vs_monitor: VSMonitorConfig = field(
        default_factory=VSMonitorConfig,
        metadata={"help": "Configuration for quasi-hyperbolic parameter monitoring."},
    )

    hyperparameter_switch: HyperparameterSwitchConfig = field(
        default_factory=HyperparameterSwitchConfig,
        metadata={
            "help": (
                "Configuration for mid-training switches of optimizer betas/vs with optional momentum resets."
            )
        },
    )


@dataclass
class FLMetricsConfigEnvelope:
    """Wrapper that guarantees the FL metrics configuration is strongly typed."""

    metrics: MetricsConfig = field(
        default_factory=MetricsConfig,
        metadata={"help": "Typed configuration for FL-specific metrics callbacks."},
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FLMetricsConfigEnvelope:
        """Create a typed metrics configuration from a mapping payload.

        Args:
            data: Mapping containing nested dictionaries for each metrics
                section.

        Returns:
            flmetricsconfigenvelope: Envelope wrapping a fully-typed
            :class:`MetricsConfig` instance.
        """
        optimizer_monitor_dict = _as_dict(data.get("optimizer_monitor"))
        activation_monitor_dict = _as_dict(data.get("activation_monitor"))
        lr_monitor_dict = _as_dict(data.get("lr_monitor"))
        betas_monitor_dict = _as_dict(data.get("betas_monitor"))
        vs_monitor_dict = _as_dict(data.get("vs_monitor"))
        hyper_switch_dict = _as_dict(data.get("hyperparameter_switch"))

        metrics = MetricsConfig(
            optimizer_monitor=OptimizerMonitorConfig(**optimizer_monitor_dict),
            activation_monitor=ActivationMonitorConfig(**activation_monitor_dict),
            lr_monitor=LRMonitorConfig(**lr_monitor_dict),
            betas_monitor=BetasMonitorConfig(**betas_monitor_dict),
            vs_monitor=VSMonitorConfig(**vs_monitor_dict),
            hyperparameter_switch=HyperparameterSwitchConfig(**hyper_switch_dict),
        )
        return cls(metrics=metrics)

    @classmethod
    def coerce(cls, value: Any) -> FLMetricsConfigEnvelope:
        """Ensure the provided value is a typed metrics configuration.

        Args:
            value: Existing :class:`FLMetricsConfigEnvelope`, raw
                :class:`MetricsConfig`, or mapping parsed from configuration
                files.

        Returns:
            flmetricsconfigenvelope: Normalized wrapper for downstream code.

        Raises:
            TypeError: If ``value`` cannot be converted into a metrics config.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, MetricsConfig):
            return cls(metrics=value)
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        msg = f"Cannot convert {type(value)} to FLMetricsConfigEnvelope"
        raise TypeError(msg)

    def unwrap(self) -> MetricsConfig:
        """Expose the underlying :class:`MetricsConfig` instance."""
        return self.metrics

    @property
    def optimizer_monitor(self) -> OptimizerMonitorConfig:
        """Typed accessor for the optimizer monitor section."""
        return self.metrics.optimizer_monitor

    @property
    def activation_monitor(self) -> ActivationMonitorConfig:
        """Typed accessor for the activation monitor section."""
        return self.metrics.activation_monitor

    @property
    def lr_monitor(self) -> LRMonitorConfig:
        """Typed accessor for the learning-rate monitor section."""
        return self.metrics.lr_monitor

    @property
    def betas_monitor(self) -> BetasMonitorConfig:
        """Typed accessor for the optimizer betas monitor section."""
        return self.metrics.betas_monitor

    @property
    def vs_monitor(self) -> VSMonitorConfig:
        """Typed accessor for the quasi-hyperbolic ``v`` monitor section."""
        return self.metrics.vs_monitor

    @property
    def hyperparameter_switch(self) -> HyperparameterSwitchConfig:
        """Typed accessor for the hyperparameter switch configuration."""
        return self.metrics.hyperparameter_switch


@dataclass
class UnigramMetricConfig:
    """Configuration for unigram cross entropy metrics."""

    enable: bool = field(
        default=False,
        metadata={
            "help": "Enable unigram cross entropy metrics backed by 1_gram.json frequency files."
        },
    )
    download_missing: bool = field(
        default=True,
        metadata={
            "help": "Download missing 1_gram.json files from remote storage when not present locally."
        },
    )
    allow_failures: bool = field(
        default=False,
        metadata={
            "help": "Allow missing unigram files without raising if download fails."
        },
    )
    ignore_index: int = field(
        default=-100,
        metadata={
            "help": "Label value that should be ignored when accumulating unigram metrics."
        },
    )
    num_attempts: int = field(
        default=3,
        metadata={"help": "Number of attempts when downloading missing unigram files."},
    )
    # python-explicit-any
    client_config: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Optional boto client configuration used when fetching unigram files from S3."
        },
    )


@dataclass
class S3CheckpointingConfig:
    """Configuration for syncing checkpoints with S3."""

    enable: bool = field(
        default=False,
        metadata={"help": "Enable uploading and downloading checkpoints from S3."},
    )
    bucket: str = field(
        default="",
        metadata={"help": "Name of the S3 bucket used for checkpoint storage."},
    )
    prefix: str = field(
        default="",
        metadata={"help": "Prefix within the bucket for storing checkpoint artifacts."},
    )
    run_uuid: str | None = field(
        default=None,
        metadata={
            "help": "Optional unique identifier appended to remote checkpoint paths."
        },
    )
    num_attempts: int = field(
        default=3,
        metadata={
            "help": "Number of retry attempts for uploads and downloads handled by the remote uploader/downloader."
        },
    )
    # python-explicit-any
    client_config: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Optional boto client configuration forwarded to the RemoteUploaderDownloader backend."
        },
    )
    num_concurrent_uploads: int = field(
        default=1,
        metadata={"help": "Number of files to upload concurrently when syncing to S3."},
    )
    upload_staging_folder: str | None = field(
        default=None,
        metadata={
            "help": "Optional staging directory used by the remote uploader/downloader before transferring files to S3."
        },
    )
    use_procs: bool = field(
        default=True,
        metadata={
            "help": "Whether to use multiprocessing workers inside the remote uploader/downloader."
        },
    )
    remote_checkpoint_folder: str | None = field(
        default=None,
        metadata={
            "help": "Optional remote folder relative to the prefix for storing checkpoints. "
            "Defaults to the local checkpoint folder name if unset."
        },
    )
    download_on_start: bool = field(
        default=True,
        metadata={
            "help": "Download a checkpoint from S3 before training when no local checkpoints are present."
        },
    )
    resume_from_run_step: str | None = field(
        default=None,
        metadata={
            "help": (
                "Resume training from a specific run and step. "
                "Format: '{run_uuid}/step-{N}' (e.g., '16M-baseline-20251011-122516/step-10'). "
                "If not set, will look for the latest checkpoint in the current run. "
                "This completely separates the resumption path from the current run's upload path."
            )
        },
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> S3CheckpointingConfig:
        """Create an S3 checkpointing configuration from a raw mapping.

        Args:
            data: Mapping containing bucket information and runtime flags.

        Returns:
            s3checkpointingconfig: Dataclass populated with the mapping values.
        """
        run_uuid = data.get("run_uuid")
        remote_checkpoint_folder = data.get("remote_checkpoint_folder")
        upload_staging_folder = data.get("upload_staging_folder")
        resume_from_run_step = data.get("resume_from_run_step")

        return cls(
            enable=bool(data.get("enable", False)),
            bucket=str(data.get("bucket", "")),
            prefix=str(data.get("prefix", "")),
            run_uuid=None if run_uuid is None else str(run_uuid),
            num_attempts=int(data.get("num_attempts", 3)),
            client_config=_as_dict(data.get("client_config")),
            num_concurrent_uploads=int(data.get("num_concurrent_uploads", 1)),
            upload_staging_folder=(
                None if upload_staging_folder is None else str(upload_staging_folder)
            ),
            use_procs=bool(data.get("use_procs", True)),
            remote_checkpoint_folder=(
                None
                if remote_checkpoint_folder is None
                else str(remote_checkpoint_folder)
            ),
            download_on_start=bool(data.get("download_on_start", True)),
            resume_from_run_step=(
                None if resume_from_run_step is None else str(resume_from_run_step)
            ),
        )

    @classmethod
    def coerce(cls, value: Any) -> S3CheckpointingConfig:
        """Convert values into :class:`S3CheckpointingConfig` instances.

        Args:
            value: Either an existing :class:`S3CheckpointingConfig` or a mapping
                produced by configuration parsing.

        Returns:
            s3checkpointingconfig: Normalized configuration with appropriate
            defaults applied.

        Raises:
            TypeError: If ``value`` cannot be interpreted as a checkpoint config.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        msg = f"Cannot convert {type(value)} to S3CheckpointingConfig"
        raise TypeError(msg)


@dataclass
class MosaicJobConfig(JobConfig):
    """A dataclass for holding all configuration settings for a MosaicML training job.

    It inherits from the base `JobConfig` and adds Mosaic-specific sections.
    """

    # Override optimizer field to use MosaicOptimizerConfig
    optimizer: MosaicOptimizerConfig = field(  # type: ignore[assignment]
        default_factory=MosaicOptimizerConfig,
        metadata={
            "help": "Optimizer configuration with extended options for Mosaic optimizers (vs, betas)."
        },
    )

    lr_scheduler: FLLRSchedulerConfig = field(  # type: ignore[assignment]
        default_factory=FLLRSchedulerConfig,
        metadata={
            "help": (
                "Learning rate scheduler configuration with optional mid-training scaling. "
                "Extends the core WSD scheduler with switch semantics."
            )
        },
    )

    mosaic_dataloader: MosaicDataLoaderConfig = field(
        default_factory=MosaicDataLoaderConfig,
        metadata={
            "help": (
                "Configuration for the MosaicML streaming dataloader. "
                "Use 'dataset.common', 'dataset.train', and 'dataset.val' to "
                "share defaults while overriding split-specific settings. "
                "Refer to llm-foundry for documentation on available options."
            )
        },
    )

    mosaic_tokenizer: MosaicTokenizerConfig = field(
        default_factory=MosaicTokenizerConfig,
        metadata={
            "help": "Configuration for the MosaicML tokenizer. This should "
            "include the tokenizer name and any specific kwargs."
        },
    )

    fl_metrics: MetricsConfig | FLMetricsConfigEnvelope = field(
        default_factory=MetricsConfig,
        metadata={
            "help": "Configuration for FL-specific metrics and monitoring callbacks (optimizer, activation, learning rate)."
        },
    )
    unigram_metric: UnigramMetricConfig = field(
        default_factory=UnigramMetricConfig,
        metadata={
            "help": "Configuration for unigram cross entropy metrics derived from Mosaic stream frequency files."
        },
    )

    s3_checkpoint: S3CheckpointingConfig = field(
        default_factory=S3CheckpointingConfig,
        metadata={
            "help": "Configuration for synchronizing checkpoints with S3 storage."
        },
    )

    fault_tolerance: FTFaultTolerance = field(
        default_factory=FTFaultTolerance,
        metadata={
            "help": "Fault tolerance configuration with TorchFT-specific options."
        },
    )


def ensure_mosaic_job_config_types(job_config: MosaicJobConfig) -> MosaicJobConfig:
    """Convert legacy dict-based sections into their typed equivalents."""
    job_config.mosaic_dataloader = MosaicDataLoaderConfig.coerce(
        job_config.mosaic_dataloader
    )
    job_config.mosaic_tokenizer = MosaicTokenizerConfig.coerce(
        job_config.mosaic_tokenizer
    )
    job_config.fl_metrics = FLMetricsConfigEnvelope.coerce(job_config.fl_metrics)
    job_config.s3_checkpoint = S3CheckpointingConfig.coerce(job_config.s3_checkpoint)

    return job_config
