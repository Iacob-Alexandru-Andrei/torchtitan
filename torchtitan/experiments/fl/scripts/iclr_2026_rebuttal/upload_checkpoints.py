#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


STEP_PATTERN = re.compile(r"step-(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload checkpoint step folders for runs that progressed past a target step."
        )
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/home/ubuntu/projects/torchtitan/outputs"),
        help="Directory containing run_uuid subfolders (default: %(default)s).",
    )
    parser.add_argument(
        "--min-step",
        type=int,
        default=10240,
        help="Only upload runs having any step-{num} folder with num greater than this.",
    )
    parser.add_argument(
        "--endpoint-url",
        default="http://taranaki.cl.cam.ac.uk:9000",
        help="S3 endpoint to use for uploads.",
    )
    parser.add_argument(
        "--bucket",
        default="checkpoints",
        help="Destination bucket name.",
    )
    parser.add_argument(
        "--prefix",
        default="torchtitan",
        help="Prefix under the bucket to place uploads.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the commands that would run without performing uploads.",
    )
    return parser.parse_args()


def find_eligible_runs(runs_root: Path, min_step: int) -> list[str]:
    eligible = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        checkpoints_dir = run_dir / "checkpoints"
        if not checkpoints_dir.is_dir():
            continue

        if _has_step_beyond_target(checkpoints_dir.iterdir(), min_step):
            eligible.append(run_dir.name)
    return eligible


def _has_step_beyond_target(entries: Iterable[Path], min_step: int) -> bool:
    for entry in entries:
        if not entry.is_dir():
            continue
        match = STEP_PATTERN.fullmatch(entry.name)
        if match and int(match.group(1)) > min_step:
            return True
    return False


def sync_run(
    run_uuid: str,
    runs_root: Path,
    endpoint_url: str,
    bucket: str,
    prefix: str,
    dry_run: bool,
) -> int:
    checkpoints_dir = runs_root / run_uuid / "checkpoints"
    cleaned_prefix = prefix.strip("/")
    destination_prefix = f"{cleaned_prefix}/{run_uuid}" if cleaned_prefix else run_uuid
    destination = f"s3://{bucket}/{destination_prefix}/"

    cmd = [
        "aws",
        "--endpoint-url",
        endpoint_url,
        "s3",
        "sync",
        str(checkpoints_dir),
        destination,
        "--exclude",
        "*",
        "--include",
        "step-*",
    ]
    
    print(cmd)

    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return 0

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(
            f"[error] Upload failed for {run_uuid}: {result.stderr.strip()}\n"
        )
    else:
        print(f"[ok] Uploaded {run_uuid} to {destination}")
    return result.returncode


def main() -> int:
    args = parse_args()
    runs_root = args.runs_root.expanduser().resolve()
    if not runs_root.is_dir():
        sys.stderr.write(f"[error] Runs root does not exist: {runs_root}\n")
        return 2

    eligible_runs = find_eligible_runs(runs_root, args.min_step)
    if not eligible_runs:
        print("No runs found with checkpoints beyond target step.")
        return 0

    exit_code = 0
    for run_uuid in eligible_runs:
        exit_code |= sync_run(
            run_uuid=run_uuid,
            runs_root=runs_root,
            endpoint_url=args.endpoint_url,
            bucket=args.bucket,
            prefix=args.prefix,
            dry_run=args.dry_run,
        )

    print("Runs with checkpoints beyond target step:")
    for run_uuid in eligible_runs:
        print(run_uuid)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
