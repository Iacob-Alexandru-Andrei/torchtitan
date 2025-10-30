#!/usr/bin/env python3

"""CLI helpers for TorchTitan checkpoint preparation and conversion."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torchtitan.tools.checkpoint_conversion import (
    DEFAULT_S3_CLIENT_CONFIG,
    convert_dist_cp_to_hf,
    prepare_torchtitan_checkpoint,
)

LOGGER = logging.getLogger(__name__)


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def _parse_client_config_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        return {}
    try:
        candidate = json.loads(cleaned)
    except json.JSONDecodeError:
        candidate_dict: dict[str, Any] | None = None
    else:
        candidate_dict = candidate if isinstance(candidate, dict) else None
    if candidate_dict is not None:
        return candidate_dict

    entries: list[str] = []
    normalised = cleaned.replace(",", "\n")
    for line in normalised.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        entries.append(stripped)

    overrides: dict[str, Any] = {}
    for entry in entries:
        if ":" in entry and "=" in entry:
            sep = ":" if entry.index(":") < entry.index("=") else "="
        elif ":" in entry:
            sep = ":"
        elif "=" in entry:
            sep = "="
        else:
            raise ValueError(f"Unable to parse client config entry '{entry}'")
        key, value = (segment.strip() for segment in entry.split(sep, 1))
        if not key or not value:
            raise ValueError(f"Invalid client config entry '{entry}'")
        overrides[key] = _coerce_scalar(value)
    return overrides


def parse_client_config(raw: str | None) -> dict[str, Any]:
    base = dict(DEFAULT_S3_CLIENT_CONFIG)
    if raw is None or not raw.strip():
        return base

    candidate_path = Path(raw)
    payloads: list[str] = []
    if candidate_path.exists():
        payloads.append(candidate_path.read_text())
    payloads.append(raw)

    for payload in payloads:
        try:
            overrides = _parse_client_config_text(payload)
        except ValueError:
            continue
        if overrides:
            base.update(overrides)
            return base
    LOGGER.warning("Unable to parse client config override '%s'; using defaults.", raw)
    return base


def _write_env(env: Mapping[str, str], env_file: Path | None) -> None:
    lines = [f"{key}={shlex.quote(value)}" for key, value in env.items()]
    output = "\n".join(lines) + ("\n" if lines else "")
    if env_file is not None:
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text(output)
    else:
        sys.stdout.write(output)


def command_prepare(args: argparse.Namespace) -> int:
    client_config = parse_client_config(args.client_config)
    hf_dir, step_dir, resolved_step = prepare_torchtitan_checkpoint(
        bucket=args.bucket,
        remote_root=args.remote_root,
        prefix=args.prefix,
        step=args.step,
        local_checkpoint_root=Path(args.local_root),
        hf_output_dir=Path(args.hf_output_dir),
        model_name=args.model_name,
        model_flavor=args.model_flavor,
        hf_assets_path=Path(args.hf_assets) if args.hf_assets else None,
        client_config=client_config,
        tokenizer=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        num_attempts=args.num_attempts,
        num_concurrent_transfers=args.num_transfers,
        use_processes=args.use_processes,
    )
    env = {
        "HF_DIR": str(hf_dir),
        "STEP_DIR": str(step_dir),
        "RESOLVED_STEP": str(resolved_step or ""),
    }
    _write_env(env, Path(args.env_file) if args.env_file else None)
    return 0


def command_convert(args: argparse.Namespace) -> int:
    hf_dir = convert_dist_cp_to_hf(
        checkpoint_dir=Path(args.checkpoint_dir),
        output_dir=Path(args.hf_output_dir),
        model_name=args.model_name,
        model_flavor=args.model_flavor,
        hf_assets_path=Path(args.hf_assets) if args.hf_assets else None,
        tokenizer=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
    )
    env = {
        "HF_DIR": str(hf_dir),
        "STEP_DIR": "",
        "RESOLVED_STEP": "",
    }
    _write_env(env, Path(args.env_file) if args.env_file else None)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TorchTitan checkpoint helpers")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Download and convert a TorchTitan checkpoint from S3.",
    )
    prepare_parser.add_argument("--bucket", required=True, help="S3 bucket name.")
    prepare_parser.add_argument(
        "--remote-root",
        default="",
        help="Remote root path within the bucket.",
    )
    prepare_parser.add_argument(
        "--prefix",
        default="",
        help="Prefix forwarded to RemoteUploaderDownloader.",
    )
    prepare_parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step to download. If omitted, resolves the latest.",
    )
    prepare_parser.add_argument(
        "--local-root",
        required=True,
        help="Local directory where dist.cp checkpoints will be stored.",
    )
    prepare_parser.add_argument(
        "--hf-output-dir",
        required=True,
        help="Destination directory for the converted Hugging Face checkpoint.",
    )
    prepare_parser.add_argument(
        "--model-name",
        required=True,
        help="TorchTitan train spec name.",
    )
    prepare_parser.add_argument(
        "--model-flavor",
        required=True,
        help="TorchTitan train spec flavour.",
    )
    prepare_parser.add_argument(
        "--hf-assets",
        default=None,
        help="Optional Hugging Face assets directory for the state dict adapter.",
    )
    prepare_parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer repo ID or local path to copy alongside the checkpoint.",
    )
    prepare_parser.add_argument(
        "--tokenizer-revision",
        default=None,
        help="Optional tokenizer revision (tag/commit) to download.",
    )
    prepare_parser.add_argument(
        "--client-config",
        default=None,
        help="boto3 client config overrides as key=value pairs, JSON, or file path.",
    )
    prepare_parser.add_argument(
        "--num-attempts",
        type=int,
        default=3,
        help="Retry attempts for S3 transfers.",
    )
    prepare_parser.add_argument(
        "--num-transfers",
        type=int,
        default=4,
        help="Concurrent S3 download workers.",
    )
    prepare_parser.add_argument(
        "--use-processes",
        action="store_true",
        help="Enable multiprocessing transfers when downloading from S3.",
    )
    prepare_parser.add_argument(
        "--env-file",
        default=None,
        help="Path to a file that receives conversion metadata as KEY=VALUE lines.",
    )

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert a local TorchTitan dist.cp checkpoint to Hugging Face format.",
    )
    convert_parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Path to the local dist.cp checkpoint directory.",
    )
    convert_parser.add_argument(
        "--hf-output-dir",
        required=True,
        help="Destination directory for the converted Hugging Face checkpoint.",
    )
    convert_parser.add_argument(
        "--model-name",
        required=True,
        help="TorchTitan train spec name.",
    )
    convert_parser.add_argument(
        "--model-flavor",
        required=True,
        help="TorchTitan train spec flavour.",
    )
    convert_parser.add_argument(
        "--hf-assets",
        default=None,
        help="Optional Hugging Face assets directory for the state dict adapter.",
    )
    convert_parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer repo ID or local path to copy alongside the checkpoint.",
    )
    convert_parser.add_argument(
        "--tokenizer-revision",
        default=None,
        help="Optional tokenizer revision (tag/commit) to download.",
    )
    convert_parser.add_argument(
        "--env-file",
        default=None,
        help="Path to a file that receives conversion metadata as KEY=VALUE lines.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.command == "prepare":
        return command_prepare(args)
    if args.command == "convert":
        return command_convert(args)
    parser.error("Command not implemented: %s" % args.command)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
