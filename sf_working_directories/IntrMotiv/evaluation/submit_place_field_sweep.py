#!/usr/bin/env python3
"""Plan or submit one ordinary Slurm job per place-field manifest row."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import pathlib
import re
import shlex
import subprocess
import time
from dataclasses import dataclass


WORKSPACE_ROOT = pathlib.Path("/work/classic/fr_xl1014-train")
REQUIRED_COLUMNS = (
    "condition",
    "family",
    "schedule",
    "feedback",
    "half_life",
    "seed",
    "target_frames",
    "checkpoint_frames",
    "checkpoint",
    "run_dir",
    "label_suffix",
)


@dataclass(frozen=True)
class ManifestRow:
    index: int
    values: dict[str, str]

    @property
    def label(self) -> str:
        return self.values["label_suffix"]


def workspace_path(value: str | pathlib.Path, label: str) -> pathlib.Path:
    path = pathlib.Path(value).expanduser().resolve(strict=False)
    try:
        path.relative_to(WORKSPACE_ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must be under {WORKSPACE_ROOT}, got {path}") from error
    return path


def load_manifest(path: pathlib.Path) -> list[ManifestRow]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise ValueError("Manifest columns must exactly match: " + ", ".join(REQUIRED_COLUMNS))
        rows = [ManifestRow(index, dict(values)) for index, values in enumerate(reader)]
    if not rows:
        raise ValueError("Manifest contains no data rows")
    labels = [row.label for row in rows]
    if len(labels) != len(set(labels)):
        raise ValueError("Manifest label_suffix values must be unique")
    for row in rows:
        checkpoint = workspace_path(row.values["checkpoint"], f"row {row.index} checkpoint")
        run_dir = workspace_path(row.values["run_dir"], f"row {row.index} run_dir")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
    return rows


def selected_indices(specs: list[str], row_count: int) -> list[int]:
    if not specs:
        return list(range(row_count))
    selected: set[int] = set()
    for spec in specs:
        for token in spec.split(","):
            token = token.strip()
            match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
            if not match:
                raise ValueError(f"Invalid row selector: {token!r}")
            start = int(match.group(1))
            stop = int(match.group(2) or start)
            if stop < start:
                raise ValueError(f"Descending row range is not allowed: {token}")
            selected.update(range(start, stop + 1))
    invalid = sorted(index for index in selected if index >= row_count)
    if invalid:
        raise IndexError(f"Manifest row indices out of range: {invalid}")
    return sorted(selected)


def safe_job_name(prefix: str, label: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-")
    return f"{prefix}-{clean}"[:120]


def build_sbatch_command(
    *,
    row: ManifestRow,
    manifest: pathlib.Path,
    output_dir: pathlib.Path,
    runner: pathlib.Path,
    partition: str,
    cpus: int,
    memory: str,
    time_limit: str,
    max_num_frames: int,
    job_name_prefix: str,
    record_observation_panel: pathlib.Path | None = None,
    replay_observation_panel: pathlib.Path | None = None,
) -> list[str]:
    if record_observation_panel and replay_observation_panel:
        raise ValueError("Recording and replaying a panel are mutually exclusive")
    export = ",".join(
        (
            "ALL",
            f"PLACE_FIELD_MAX_FRAMES={max_num_frames}",
            f"TMPDIR={output_dir / 'tmp'}",
            f"DMLAB_CACHE_DIR={output_dir / 'dmlab_cache'}",
            f"XDG_CACHE_HOME={output_dir / 'cache'}",
            f"WANDB_DIR={output_dir / 'wandb'}",
            "WANDB_MODE=disabled",
        )
    )
    for key, value in (("PLACE_FIELD_RECORD_PANEL", record_observation_panel),
                       ("PLACE_FIELD_REPLAY_PANEL", replay_observation_panel)):
        if value is not None:
            path = workspace_path(value, "Observation panel")
            if ',' in str(path):
                raise ValueError("Observation panel path cannot contain Slurm export delimiters")
            export += f",{key}={path}"
    return [
        "sbatch",
        f"--job-name={safe_job_name(job_name_prefix, row.label)}",
        f"--partition={partition}",
        f"--cpus-per-task={cpus}",
        f"--mem={memory}",
        f"--time={time_limit}",
        f"--output={output_dir / 'slurm' / (row.label + '-%j.out')}",
        f"--export={export}",
        str(runner),
        str(manifest),
        str(row.index),
        str(output_dir),
    ]


def parse_job_id(stdout: str) -> str:
    match = re.search(r"Submitted batch job (\d+)", stdout)
    if not match:
        raise RuntimeError(f"Could not parse sbatch job id from: {stdout!r}")
    return match.group(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print or submit one independent non-array sbatch job per manifest row."
    )
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--row",
        action="append",
        default=[],
        help="Zero-based row, comma list, or inclusive range; repeat as needed. Default: all rows.",
    )
    parser.add_argument("--max-num-frames", type=int, default=10000)
    parser.add_argument("--partition", default="cpu")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--memory", default="16G")
    parser.add_argument("--time-limit", default="01:00:00")
    parser.add_argument("--job-name-prefix", default="intrmotiv-pf")
    parser.add_argument("--pause-between", type=float, default=1.0)
    panel = parser.add_mutually_exclusive_group()
    panel.add_argument("--record-observation-panel", type=pathlib.Path)
    panel.add_argument("--replay-observation-panel", type=pathlib.Path)
    parser.add_argument(
        "--runner",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("run_place_field_sweep_single.sh"),
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually call sbatch. Without this flag, only print the validated plan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = workspace_path(args.manifest, "Manifest")
    output_dir = workspace_path(args.output_dir, "Output directory")
    runner = args.runner.expanduser().resolve(strict=True)
    rows = load_manifest(manifest)
    chosen = [rows[index] for index in selected_indices(args.row, len(rows))]
    if args.record_observation_panel and len(chosen) != 1:
        raise ValueError("Exactly one job may write a common observation panel")
    if args.record_observation_panel and args.record_observation_panel.exists():
        raise FileExistsError(args.record_observation_panel)
    if args.replay_observation_panel and not args.replay_observation_panel.is_file():
        raise FileNotFoundError(args.replay_observation_panel)
    commands = [
        build_sbatch_command(
            row=row,
            manifest=manifest,
            output_dir=output_dir,
            runner=runner,
            partition=args.partition,
            cpus=args.cpus,
            memory=args.memory,
            time_limit=args.time_limit,
            max_num_frames=args.max_num_frames,
            job_name_prefix=args.job_name_prefix,
            record_observation_panel=args.record_observation_panel,
            replay_observation_panel=args.replay_observation_panel,
        )
        for row in chosen
    ]
    print(f"Validated {len(commands)} ordinary Slurm job(s); no arrays or dependencies.")
    for command in commands:
        print(shlex.join(command))
    if not args.submit:
        print("Print-only review complete. Re-run with --submit to enqueue these jobs.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("raw", "slurm", "tmp", "dmlab_cache", "cache", "wandb"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record_path = output_dir / f"submission_manifest_{timestamp}.tsv"
    with record_path.open("w", newline="") as stream:
        fieldnames = ("row_index", "label_suffix", "job_id", "command")
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for position, (row, command) in enumerate(zip(chosen, commands)):
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            job_id = parse_job_id(result.stdout)
            writer.writerow(
                {
                    "row_index": row.index,
                    "label_suffix": row.label,
                    "job_id": job_id,
                    "command": shlex.join(command),
                }
            )
            stream.flush()
            print(f"Submitted row {row.index} as ordinary job {job_id}: {row.label}")
            if position + 1 < len(commands) and args.pause_between > 0:
                time.sleep(args.pause_between)
    print(f"Wrote submission record to {record_path}")


if __name__ == "__main__":
    main()
