from __future__ import annotations

from pathlib import Path
import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - test environment guard
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util
import types

_S3_MODULE_PATH = PROJECT_ROOT / "torchtitan" / "experiments" / "fl" / "s3_checkpoint.py"
_COMPOSER_MODULE = types.ModuleType("composer")
_COMPOSER_LOGGERS_MODULE = types.ModuleType("composer.loggers")
_COMPOSER_RUD_MODULE = types.ModuleType("composer.loggers.remote_uploader_downloader")


class _StubRemoteUploaderDownloader:  # pragma: no cover - import-time stub
    pass


def _stub_upload_worker(*_args: object, **_kwargs: object) -> None:  # pragma: no cover
    return


_COMPOSER_LOGGERS_MODULE.RemoteUploaderDownloader = _StubRemoteUploaderDownloader
_COMPOSER_RUD_MODULE._upload_worker = _stub_upload_worker
sys.modules.setdefault("composer", _COMPOSER_MODULE)
sys.modules.setdefault("composer.loggers", _COMPOSER_LOGGERS_MODULE)
sys.modules.setdefault(
    "composer.loggers.remote_uploader_downloader", _COMPOSER_RUD_MODULE
)

_SPEC = importlib.util.spec_from_file_location("tests.s3_checkpoint", _S3_MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError(f"Unable to load s3_checkpoint module from {_S3_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
s3_module = _MODULE  # type: ModuleType

class _FakeRemoteUploaderDownloader:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, Path]] = []

    # The wrapper calls this guard before any transfers.
    def _check_workers(self) -> None:  # pragma: no cover - simple stub
        return

    def upload_file(
        self,
        *,
        state: None,
        remote_file_name: str,
        file_path: Path,
        overwrite: bool,
    ) -> None:
        del state, overwrite
        self.uploads.append((remote_file_name, Path(file_path)))

    def download_file(self, *args, **kwargs) -> None:  # pragma: no cover - unused
        raise AssertionError("download_file should not be invoked in these tests")


class _DummyCheckpointer:
    def __init__(self, base_folder: Path, *, ft_enabled: bool = False) -> None:
        self.folder = str(base_folder)
        self._base_folder = base_folder
        self._ft_enabled = ft_enabled
        self.ft_manager = SimpleNamespace(enabled=True) if ft_enabled else None
        self._ft_path = base_folder / "ft-replicat-0" if ft_enabled else None
        if self._ft_path is not None:
            self._ft_path.mkdir(parents=True, exist_ok=True)
        self.wait_calls = 0
        self.close_calls = 0

    def save(self, curr_step: int, *, last_step: bool = False) -> None:
        del last_step
        root = self._ft_path if self._ft_path is not None else self._base_folder
        step_dir = root / f"step-{curr_step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / ".metadata").write_text("meta")
        (step_dir / f"tensor-{curr_step}.bin").write_text("payload")

    def maybe_wait_for_staging(self) -> None:
        self.wait_calls += 1

    def close(self) -> None:
        self.close_calls += 1

    def _ft_folder(self) -> str:
        if self._ft_path is None:
            raise RuntimeError("TorchFT is not enabled for this dummy checkpointer")
        return str(self._ft_path)

    def _find_load_step(self, folder: str = "") -> int:  # noqa: ARG002
        return -1


@dataclass
class _S3CheckpointingConfig:
    enable: bool = True
    bucket: str = "bucket"
    prefix: str | None = "prefix"
    run_uuid: str | None = None
    num_attempts: int = 3
    client_config: dict[str, object] = field(default_factory=dict)
    num_concurrent_uploads: int = 1
    upload_staging_folder: str | None = None
    use_procs: bool = True
    remote_checkpoint_folder: str | None = None
    download_on_start: bool = True
    resume_from_run_step: str | None = None


@dataclass
class _JobSection:
    dump_folder: str = "."
    description: str | None = None


@dataclass
class _CheckpointSection:
    folder: str = "checkpoint"


@dataclass
class _MosaicJobConfig:
    job: _JobSection = field(default_factory=_JobSection)
    checkpoint: _CheckpointSection = field(default_factory=_CheckpointSection)
    s3_checkpoint: _S3CheckpointingConfig = field(
        default_factory=_S3CheckpointingConfig
    )


def _make_job_config(base_folder: Path) -> _MosaicJobConfig:
    job_config = _MosaicJobConfig()
    job_config.job.dump_folder = str(base_folder)
    job_config.s3_checkpoint = _S3CheckpointingConfig(num_concurrent_uploads=0)
    return job_config


def test_wrapper_preserves_original_methods(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "ckpt"
    base.mkdir()
    checkpointer = _DummyCheckpointer(base)
    orig_save = checkpointer.save
    orig_wait = checkpointer.maybe_wait_for_staging

    job_config = _make_job_config(tmp_path)
    config = job_config.s3_checkpoint

    remote = _FakeRemoteUploaderDownloader()
    monkeypatch.setattr(s3_module, "create_remote_up_down", lambda *args, **kwargs: remote)

    started: list[bool] = []

    def fake_start(self: s3_module.S3CheckpointWrapper) -> None:
        started.append(True)

    monkeypatch.setattr(s3_module.S3CheckpointWrapper, "_start_remote_workers", fake_start)

    wrapper = s3_module.S3CheckpointWrapper(checkpointer, config, job_config)

    assert checkpointer.save.__func__ is orig_save.__func__
    assert (
        checkpointer.maybe_wait_for_staging.__func__
        is orig_wait.__func__
    )
    assert started, "Upload workers should start when uploads are enabled"

    trainer = SimpleNamespace(checkpointer=checkpointer)
    wrapper.attach_to_trainer(trainer)
    assert trainer.checkpointer is wrapper


def test_download_only_wrapper_has_symmetric_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "ckpt"
    base.mkdir()
    checkpointer = _DummyCheckpointer(base)
    job_config = _make_job_config(tmp_path)
    config = job_config.s3_checkpoint

    remote = _FakeRemoteUploaderDownloader()
    monkeypatch.setattr(s3_module, "create_remote_up_down", lambda *args, **kwargs: remote)

    started: list[bool] = []
    monkeypatch.setattr(
        s3_module.S3CheckpointWrapper,
        "_start_remote_workers",
        lambda self: started.append(True),
    )

    wrapper = s3_module.S3CheckpointWrapper(
        checkpointer,
        config,
        job_config,
        enable_uploads=False,
    )

    assert not started, "Download-only mode must not start upload workers"

    wrapper.save(1)
    wrapper.maybe_wait_for_staging()
    wrapper.close()

    assert checkpointer.wait_calls == 2  # one from maybe_wait_for_staging, one from close()
    assert checkpointer.close_calls == 1
    assert remote.uploads == []


def test_torchft_uploads_flush_and_close_symmetrically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "ckpt"
    base.mkdir()
    checkpointer = _DummyCheckpointer(base, ft_enabled=True)
    job_config = _make_job_config(tmp_path)
    config = job_config.s3_checkpoint

    remote = _FakeRemoteUploaderDownloader()
    monkeypatch.setattr(s3_module, "create_remote_up_down", lambda *args, **kwargs: remote)

    monkeypatch.setattr(
        s3_module.S3CheckpointWrapper,
        "_start_remote_workers",
        lambda self: None,
    )

    uploads: list[tuple[str, Path]] = []

    def fake_upload(
        remote_client: _FakeRemoteUploaderDownloader,
        remote_file_name: str,
        local_file_name: Path,
    ) -> None:
        uploads.append((remote_file_name, Path(local_file_name)))
        remote_client.uploads.append((remote_file_name, Path(local_file_name)))

    monkeypatch.setattr(s3_module, "upload_file_to_s3", fake_upload)

    wrapper = s3_module.S3CheckpointWrapper(checkpointer, config, job_config)
    wrapper.save(2)
    assert wrapper._pending_steps, "Saving should queue uploads while TorchFT is enabled"  # noqa: SLF001

    wrapper.close()

    assert checkpointer.close_calls == 1
    assert checkpointer.wait_calls == 1
    assert not wrapper._pending_steps  # noqa: SLF001
    assert uploads, "Expected uploads to flush during close()"
    # Ensure the FT-specific directory is captured in remote keys.
    assert any("ft-replicat-0" in remote_key for remote_key, _ in uploads)
