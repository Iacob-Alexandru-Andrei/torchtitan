"""Helpers for downloading and converting TorchTitan ``dist.cp`` checkpoints.

This module centralises the logic for moving TorchTitan checkpoints into a
Hugging Face compatible layout so it can be reused by tooling outside of the
LightEval environment.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from composer.loggers import RemoteUploaderDownloader
from composer.utils.file_helpers import list_remote_objects
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from transformers import AutoTokenizer, LlamaConfig

import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.experiments.fl.s3_checkpoint import (
    LATEST_FILENAME,
    MANIFEST_FILENAME,
    create_remote_up_down,
    download_file_from_s3,
)

logger = logging.getLogger(__name__)

DEFAULT_S3_CLIENT_CONFIG: dict[str, Any] = {
    "connect_timeout": 3600,
    "read_timeout": 3600,
}

DEFAULT_TOKENIZER_REPO = "HuggingFaceTB/SmolLM-1.7B"


@dataclass(slots=True)
class S3CheckpointLocation:
    """Describe where a TorchTitan checkpoint lives on S3."""

    bucket: str
    remote_root: str
    prefix: str = ""
    step: int | None = None
    num_attempts: int = 3
    client_config: Mapping[str, Any] | None = None
    num_concurrent_transfers: int = 4
    use_processes: bool = False

    def normalised_remote_root(self) -> str:
        return self.remote_root.strip("/")


def _remote_key(relative_path: Path, remote_root: str | None) -> str:
    relative_key = relative_path.as_posix().lstrip("/")
    root = (remote_root or "").strip("/")
    if root and relative_key:
        return f"{root}/{relative_key}"
    if root:
        return root
    return relative_key


def _build_listing_uri(bucket: str, prefix: str, remote_key: str) -> str:
    bucket = bucket.strip("/")
    prefix = prefix.strip("/")
    remote_key = remote_key.strip("/")
    components = [part for part in (prefix, remote_key) if part]
    uri = f"s3://{bucket}"
    if components:
        uri = f"{uri}/{'/'.join(components)}"
    if not uri.endswith("/"):
        uri += "/"
    return uri


def _listing_prefix(prefix: str, remote_key: str) -> str:
    prefix = prefix.strip("/")
    remote_key = remote_key.strip("/")
    components = [part for part in (prefix, remote_key) if part]
    if not components:
        return ""
    return "/".join(components).rstrip("/") + "/"


def _enumerate_remote_step_files(
    location: S3CheckpointLocation,
    candidate_relatives: list[Path],
) -> tuple[list[str], Path] | tuple[None, None]:
    for relative_base in candidate_relatives:
        remote_key = _remote_key(relative_base, location.normalised_remote_root())
        listing_uri = _build_listing_uri(location.bucket, location.prefix or "", remote_key)
        logger.info(
            "Manifest %s not found; enumerating checkpoint files via %s",
            MANIFEST_FILENAME,
            listing_uri,
        )
        try:
            object_keys = list_remote_objects(listing_uri)
        except Exception as err:  # noqa: BLE001
            logger.warning(
                "Failed to list remote objects at %s (%s); trying alternate layout.",
                listing_uri,
                err,
            )
            continue

        listing_prefix = _listing_prefix(location.prefix or "", remote_key)
        entries: list[str] = []
        for key in object_keys:
            candidate = key
            if listing_prefix:
                if not key.startswith(listing_prefix):
                    continue
                candidate = key[len(listing_prefix) :]
            candidate = candidate.strip("/")
            if not candidate:
                continue
            entries.append(candidate)

        if entries:
            deduped_entries = sorted(set(entries))
            logger.info(
                "Identified %d file(s) for checkpoint step using manifestless discovery.",
                len(deduped_entries),
            )
            return deduped_entries, relative_base
        logger.warning("No checkpoint files found when listing %s; trying alternate layout.", listing_uri)
    return None, None


def _read_latest_step(
    remote: RemoteUploaderDownloader,
    location: S3CheckpointLocation,
    scratch_dir: Path,
) -> int | None:
    marker_path = scratch_dir / LATEST_FILENAME
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        download_file_from_s3(
            remote,
            _remote_key(Path(LATEST_FILENAME), location.normalised_remote_root()),
            marker_path,
        )
    except Exception:  # noqa: BLE001
        logger.debug("Unable to download latest marker for %s", location.remote_root)
        return None
    try:
        return int(marker_path.read_text().strip())
    except ValueError:
        logger.warning("Latest marker at %s is invalid.", marker_path)
        return None


def _ensure_remote(remote: RemoteUploaderDownloader, run_name: str) -> None:
    if getattr(remote, "_run_name", None) is None:
        remote._run_name = run_name  # type: ignore[attr-defined]
    remote._check_workers()  # type: ignore[attr-defined]


def download_dist_cp_checkpoint(
    location: S3CheckpointLocation,
    output_root: Path,
) -> tuple[Path, int]:
    """Download a TorchTitan ``dist.cp`` checkpoint from S3."""
    client_config = dict(DEFAULT_S3_CLIENT_CONFIG)
    if location.client_config:
        client_config.update(location.client_config)
    remote = create_remote_up_down(
        bucket_name=location.bucket,
        prefix=location.prefix or "",
        num_attempts=location.num_attempts,
        client_config=client_config,
        num_concurrent_uploads=max(1, location.num_concurrent_transfers),
        upload_staging_folder=None,
        use_procs=location.use_processes,
    )
    _ensure_remote(remote, f"download-{location.normalised_remote_root() or 'torchtitan'}")

    output_root.mkdir(parents=True, exist_ok=True)
    step = location.step
    if step is None:
        step = _read_latest_step(remote, location, output_root)
        if step is None:
            msg = (
                "Unable to determine latest checkpoint step. "
                "Provide --torchtitan-step to select a specific checkpoint."
            )
            raise RuntimeError(msg)
        logger.info("Resolved latest remote checkpoint step %s", step)

    step_dir = output_root / f"step-{step}"
    step_dir.mkdir(parents=True, exist_ok=True)

    relative_base = Path(f"step-{step}")
    manifest_path = step_dir / MANIFEST_FILENAME
    logger.info(
        "Downloading TorchTitan manifest: s3://%s/%s/%s",
        location.bucket,
        location.prefix,
        _remote_key(relative_base / MANIFEST_FILENAME, location.normalised_remote_root()),
    )
    manifest_entries: list[str] | None = None
    try:
        download_file_from_s3(
            remote,
            _remote_key(relative_base / MANIFEST_FILENAME, location.normalised_remote_root()),
            manifest_path,
        )
        manifest_entries = json.loads(manifest_path.read_text())
    except FileNotFoundError:
        logger.info(
            "Manifest %s not found on S3 for step %s; falling back to manifestless download.",
            MANIFEST_FILENAME,
            step,
        )
        manifest_path.unlink(missing_ok=True)
    except json.JSONDecodeError as exc:
        msg = f"Invalid manifest file downloaded for step {step}: {manifest_path}"
        raise RuntimeError(msg) from exc

    if manifest_entries is None:
        fallback_entries, fallback_relative = _enumerate_remote_step_files(
            location,
            [relative_base],
        )
        if fallback_entries is None or fallback_relative is None:
            msg = (
                f"Checkpoint manifest missing and no files could be discovered for step {step}. "
                "Ensure the remote checkpoint path is correct."
            )
            raise FileNotFoundError(msg)
        manifest_entries = fallback_entries
        if fallback_relative != relative_base:
            relative_base = fallback_relative
        manifest_path.write_text(json.dumps(manifest_entries))

    for relative in manifest_entries:
        relative_path = Path(relative)
        target_path = step_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        remote_key = _remote_key(relative_base / relative_path, location.normalised_remote_root())
        logger.debug("Downloading %s -> %s", remote_key, target_path)
        download_file_from_s3(remote, remote_key, target_path)

    metadata_file = step_dir / ".metadata"
    if not metadata_file.exists():
        logger.warning("Checkpoint metadata file missing at %s", metadata_file)

    latest_marker = output_root / LATEST_FILENAME
    latest_marker.write_text(f"{step}\n")

    logger.info(
        "Finished downloading checkpoint step %s into %s (%d files).",
        step,
        step_dir,
        len(manifest_entries),
    )
    return step_dir, step


@torch.inference_mode()
def _compute_intermediate_size(model_args: Any) -> int:
    """Match TorchTitan feed-forward sizing for config export."""
    base_hidden = 4 * model_args.dim
    hidden_dim = int(2 * base_hidden / 3)
    if getattr(model_args, "ffn_dim_multiplier", None) is not None:
        hidden_dim = int(model_args.ffn_dim_multiplier * hidden_dim)
    multiple_of = getattr(model_args, "multiple_of", None) or 1
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


def _materialize_tokenizer(
    output_dir: Path,
    tokenizer_source: str,
    *,
    revision: str | None = None,
) -> Any | None:
    """Copy or download tokenizer assets into ``output_dir``."""
    source_path = Path(tokenizer_source)
    output_dir.mkdir(parents=True, exist_ok=True)

    if source_path.exists():
        for item in source_path.iterdir():
            dest = output_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        try:
            return AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Copied tokenizer from %s but failed to re-load from %s for metadata.",
                source_path,
                output_dir,
            )
            return None

    logger.info("Downloading tokenizer assets from %s", tokenizer_source)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=revision,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(output_dir)
    return tokenizer


def _ensure_tokenizer(
    output_dir: Path,
    tokenizer: str | None,
    *,
    revision: str | None = None,
) -> tuple[Any | None, str | None]:
    """Ensure tokenizer files exist in ``output_dir`` and return metadata."""
    existing_tokenizer = None
    if (output_dir / "tokenizer.json").exists():
        try:
            existing_tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Tokenizer files already present in %s but loading failed; continuing.",
                output_dir,
            )
        else:
            if tokenizer is None:
                tokenizer = existing_tokenizer.name_or_path

    effective_tokenizer = tokenizer or DEFAULT_TOKENIZER_REPO
    if existing_tokenizer is None:
        try:
            existing_tokenizer = _materialize_tokenizer(
                output_dir,
                effective_tokenizer,
                revision=revision,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Unable to materialize tokenizer '%s': %s",
                effective_tokenizer,
                exc,
            )
            existing_tokenizer = None
        else:
            effective_tokenizer = (
                existing_tokenizer.name_or_path if existing_tokenizer else effective_tokenizer
            )
    return existing_tokenizer, effective_tokenizer


def _write_hf_config(
    output_dir: Path,
    model_args: Any,
    *,
    tokenizer_name: str | None,
    tokenizer_obj: Any | None,
    model_name: str,
) -> None:
    """Create a HuggingFace config mirroring the TorchTitan model settings."""
    intermediate_size = _compute_intermediate_size(model_args)
    num_kv_heads = (
        model_args.n_kv_heads if getattr(model_args, "n_kv_heads", None) else model_args.n_heads
    )

    if "mup" in model_name:
        # log
        logger.info("Using MuP configuration for huggingface model '%s'", model_name)
        from light_eval_photon.eval_light.patched_llama3 import Llama3MuPConfig

        config = Llama3MuPConfig(
            vocab_size=model_args.vocab_size,
            hidden_size=model_args.dim,
            intermediate_size=intermediate_size,
            num_hidden_layers=model_args.n_layers,
            num_attention_heads=model_args.n_heads,
            num_key_value_heads=num_kv_heads,
            rms_norm_eps=model_args.norm_eps,
            hidden_act="silu",
            rope_theta=model_args.rope_theta,
            max_position_embeddings=getattr(model_args, "max_seq_len", 2048),
            initializer_range=0.02,
            tie_word_embeddings=getattr(model_args, "tie_word_embeddings", True),
            torch_dtype="bfloat16",
            use_embedding_norm=getattr(model_args, "use_embedding_norm", True),
            use_peri_norm=getattr(model_args, "use_peri_norm", True),
            use_torch_layernorm=getattr(model_args, "use_torch_layernorm", True),
            use_simple_silu_ffn=getattr(model_args, "use_simple_silu_ffn", False),
            qk_norm=getattr(model_args, "qk_norm", True),
            qk_norm_bias=getattr(model_args, "qk_norm_bias", False),
            qk_norm_elementwise_affine=getattr(model_args, "qk_norm_elementwise_affine", True),
            torch_layernorm_bias=getattr(model_args, "torch_layernorm_bias", False),
            torch_layernorm_elementwise_affine=getattr(model_args, "torch_layernorm_elementwise_affine", True),
            use_flex_attn=False,
            attn_mask_type="causal",
            multiple_of=getattr(model_args, "multiple_of", 256),
            ffn_dim_multiplier=getattr(model_args, "ffn_dim_multiplier", None),
            mup_config=getattr(model_args, "mup_config", {}),
            init_config=getattr(model_args, "init_config", {}),
        )
        config.architectures = ["Llama3MuPForCausalLM"]
    else:
        config = LlamaConfig(
            vocab_size=model_args.vocab_size,
            hidden_size=model_args.dim,
            intermediate_size=intermediate_size,
            num_hidden_layers=model_args.n_layers,
            num_attention_heads=model_args.n_heads,
            num_key_value_heads=num_kv_heads,
            rms_norm_eps=model_args.norm_eps,
            hidden_act="silu",
            rope_theta=model_args.rope_theta,
            max_position_embeddings=getattr(model_args, "max_seq_len", 2048),
            initializer_range=0.02,
            tie_word_embeddings=getattr(model_args, "tie_word_embeddings", True),
            torch_dtype="bfloat16",
        )
        config.architectures = ["LlamaForCausalLM"]

    if tokenizer_name:
        config.tokenizer_name = tokenizer_name

    if tokenizer_obj is not None:
        if getattr(tokenizer_obj, "bos_token_id", None) is not None:
            config.bos_token_id = tokenizer_obj.bos_token_id
        if getattr(tokenizer_obj, "eos_token_id", None) is not None:
            config.eos_token_id = tokenizer_obj.eos_token_id
        if getattr(tokenizer_obj, "pad_token_id", None) is not None:
            config.pad_token_id = tokenizer_obj.pad_token_id
    else:
        config.eos_token_id = getattr(model_args, "eos_id", 2)

    config_path = output_dir / "config.json"
    config.save_pretrained(output_dir)
    logger.info("Wrote Hugging Face config to %s", config_path)


@torch.inference_mode()
def _format_param_count(num_params: int) -> str:
    if num_params >= 1_000_000:
        return f"{num_params:,} ({num_params / 1_000_000:.3f}M)"
    if num_params >= 1_000:
        return f"{num_params:,} ({num_params / 1_000:.3f}K)"
    return f"{num_params:,}"


@torch.inference_mode()
def _log_model_structure(model: nn.Module) -> None:
    """Emit a parameter breakdown and module tree for the constructed model."""

    def _count_params(module: nn.Module | None) -> int:
        if module is None:
            return 0
        return sum(param.numel() for param in module.parameters())

    total_params = sum(param.numel() for param in model.parameters())
    components = {
        "tok_embeddings": _count_params(getattr(model, "tok_embeddings", None)),
        "layers": _count_params(getattr(model, "layers", None)),
        "norm": _count_params(getattr(model, "norm", None)),
        "embedding_norm": _count_params(getattr(model, "embedding_norm", None)),
    }

    logger.info("Model parameter breakdown:")
    logger.info("part0: total=%s", _format_param_count(total_params))
    for name, count in components.items():
        logger.info("  - %s: %s", name, _format_param_count(count))

    for line in repr(model).splitlines():
        logger.info("%s", line)


def _verify_native_state_dict_layers(
    state_dict: Mapping[str, Any],
    expected_layers: int,
) -> None:
    """Ensure every transformer block received tensors from the checkpoint."""
    if expected_layers <= 0:
        logger.info(
            "Skipping transformer layer verification; expected_layers=%s",
            expected_layers,
        )
        return

    layer_counts = {str(idx): 0 for idx in range(expected_layers)}

    for key in state_dict.keys():
        parts = key.split(".")
        if len(parts) >= 2 and parts[0] == "layers" and parts[1].isdigit():
            layer_id = parts[1]
            if layer_id in layer_counts:
                layer_counts[layer_id] += 1

    missing = [layer_id for layer_id, count in layer_counts.items() if count == 0]
    if missing:
        msg = (
            "Checkpoint load omitted parameters for layer(s): "
            f"{', '.join(sorted(missing))}"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info("Verified state dict tensors present for all %d transformer layers.", expected_layers)


@torch.inference_mode()
def _log_hf_state_dict_summary(
    hf_state_dict: Mapping[str, torch.Tensor],
    expected_layers: int,
) -> None:
    """Log parameter counts and per-layer coverage for the HuggingFace state dict."""

    def _is_tensor(obj: Any) -> bool:
        return isinstance(obj, torch.Tensor)

    total_params = 0
    embed_params = 0
    layers_params = 0
    norm_params = 0
    embedding_norm_params = 0
    lm_head_params = 0

    layer_param_totals = {str(i): 0 for i in range(expected_layers)} if expected_layers > 0 else {}
    layer_tensor_counts = {str(i): 0 for i in range(expected_layers)} if expected_layers > 0 else {}

    for key, value in hf_state_dict.items():
        if not _is_tensor(value):
            continue
        numel = value.numel()
        total_params += numel

        if key.startswith("model.embed_tokens."):
            embed_params += numel
        elif key.startswith("model.layers."):
            layers_params += numel
            match = re.match(r"model\.layers\.(\d+)\.", key)
            if match:
                layer_id = match.group(1)
                if layer_id in layer_param_totals:
                    layer_param_totals[layer_id] += numel
                    layer_tensor_counts[layer_id] += 1
        elif key.startswith("model.norm."):
            norm_params += numel
        elif key.startswith("model.embedding_norm."):
            embedding_norm_params += numel
        elif key.startswith("lm_head."):
            lm_head_params += numel

    logger.info("HF state dict parameter breakdown:")
    logger.info("  total=%s", _format_param_count(total_params))
    logger.info("  embed_tokens=%s", _format_param_count(embed_params))
    logger.info("  layers=%s", _format_param_count(layers_params))
    logger.info("  norm=%s", _format_param_count(norm_params))
    logger.info("  embedding_norm=%s", _format_param_count(embedding_norm_params))
    logger.info("  lm_head=%s", _format_param_count(lm_head_params))

    if layer_param_totals:
        missing_layers = [layer_id for layer_id, total in layer_param_totals.items() if total == 0]
        for layer_id in sorted(layer_param_totals, key=lambda x: int(x)):
            logger.info(
                "    layer %s: params=%s tensors=%d",
                layer_id,
                _format_param_count(layer_param_totals[layer_id]),
                layer_tensor_counts[layer_id],
            )
        if missing_layers:
            msg = (
                "HF state dict is missing parameters for layer(s): "
                f"{', '.join(sorted(missing_layers))}"
            )
            logger.error(msg)
            raise RuntimeError(msg)
        logger.info("Verified HF tensors present for all %d transformer layers.", expected_layers)


@torch.inference_mode()
def convert_dist_cp_to_hf(
    checkpoint_dir: Path,
    output_dir: Path,
    model_name: str,
    model_flavor: str,
    hf_assets_path: Path | None = None,
    *,
    tokenizer: str | None = None,
    tokenizer_revision: str | None = None,
) -> Path:
    """Convert a TorchTitan ``dist.cp`` checkpoint into Hugging Face safetensors."""
    if output_dir.exists():
        msg = f"HF output directory already exists: {output_dir}"
        raise FileExistsError(msg)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    train_spec = train_spec_module.get_train_spec(model_name)
    model_args = train_spec.model_args[model_flavor]

    with torch.device("cpu"):
        model = train_spec.model_cls(model_args)
    _log_model_structure(model)
    model = ModelWrapper(model)

    sd_adapter = train_spec.state_dict_adapter(model_args, hf_assets_path)
    if sd_adapter is None:
        msg = "State dict adapter is required to convert TorchTitan checkpoint to HF."
        raise RuntimeError(msg)

    state_dict = model._get_state_dict()
    dcp.load(state_dict, checkpoint_id=str(checkpoint_dir))
    _verify_native_state_dict_layers(state_dict, getattr(model_args, "n_layers", 0))

    hf_state_dict = sd_adapter.to_hf(state_dict)
    _log_hf_state_dict_summary(hf_state_dict, getattr(model_args, "n_layers", 0))

    tied_embed = hf_state_dict.get("model.embed_tokens.weight")
    tied_head = hf_state_dict.get("lm_head.weight")
    if (
        tied_embed is not None
        and tied_head is not None
        and tied_embed.data_ptr() == tied_head.data_ptr()
    ):
        hf_state_dict["lm_head.weight"] = tied_head.clone()

    storage_writer = HuggingFaceStorageWriter(
        path=str(output_dir),
        save_distributed=True,
        fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    dcp.save(hf_state_dict, storage_writer=storage_writer)
    logger.info(
        "TorchTitan checkpoint at %s converted to HF format in %s.",
        checkpoint_dir,
        output_dir,
    )

    tokenizer_obj, effective_tokenizer = _ensure_tokenizer(
        output_dir,
        tokenizer,
        revision=tokenizer_revision,
    )
    _write_hf_config(
        output_dir,
        model_args,
        tokenizer_name=effective_tokenizer,
        tokenizer_obj=tokenizer_obj,
        model_name=model_name,
    )
    return output_dir


def prepare_torchtitan_checkpoint(
    *,
    bucket: str,
    remote_root: str,
    prefix: str = "",
    step: int | None = None,
    local_checkpoint_root: Path,
    hf_output_dir: Path,
    model_name: str,
    model_flavor: str,
    hf_assets_path: Path | None = None,
    client_config: Mapping[str, Any] | None = None,
    tokenizer: str | None = None,
    tokenizer_revision: str | None = None,
    num_attempts: int = 3,
    num_concurrent_transfers: int = 4,
    use_processes: bool = False,
) -> tuple[Path, Path, int]:
    """Download and convert a TorchTitan checkpoint, returning HF and ``dist.cp`` paths."""
    location = S3CheckpointLocation(
        bucket=bucket,
        remote_root=remote_root,
        prefix=prefix,
        step=step,
        num_attempts=num_attempts,
        client_config=client_config,
        num_concurrent_transfers=num_concurrent_transfers,
        use_processes=use_processes,
    )
    step_dir, resolved_step = download_dist_cp_checkpoint(location, local_checkpoint_root)
    hf_dir = convert_dist_cp_to_hf(
        checkpoint_dir=step_dir,
        output_dir=hf_output_dir,
        model_name=model_name,
        model_flavor=model_flavor,
        hf_assets_path=hf_assets_path,
        tokenizer=tokenizer,
        tokenizer_revision=tokenizer_revision,
    )
    return hf_dir, step_dir, resolved_step


__all__ = [
    "DEFAULT_S3_CLIENT_CONFIG",
    "DEFAULT_TOKENIZER_REPO",
    "S3CheckpointLocation",
    "convert_dist_cp_to_hf",
    "download_dist_cp_checkpoint",
    "prepare_torchtitan_checkpoint",
]
