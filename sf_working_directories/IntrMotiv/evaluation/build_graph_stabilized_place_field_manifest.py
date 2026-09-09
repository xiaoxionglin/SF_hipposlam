#!/usr/bin/env python3
"""Build an aligned place-field manifest for Graph-Stabilized Recruitment.

The common comparison point is the largest standard telemetry target reached
by every run.  Exact checkpoint saves are asynchronous, so ``target_frames``
records the shared target while ``checkpoint_frames`` records the nearest
retained milestone selected for each run.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from itertools import product
from pathlib import Path

from sf_working_directories.IntrMotiv.evaluation.build_place_field_sweep import (
    TARGET_FRAMES,
    checkpoint_frames,
    select_checkpoints,
)


RUN_RE = re.compile(
    r"^00_GSR_(?P<backbone>C05|C13|C15)_D(?P<distance>4|8)_"
    r"H(?P<half_life>5|10)K_S(?P<seed>8|99|123)$"
)
BACKBONES = ("C05", "C13", "C15")
DISTANCES = (4, 8)
HALF_LIVES_K = (5, 10)
SEEDS = (8, 99, 123)
WORKSPACE = Path("/work/classic/fr_xl1014-train")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--target-frames",
        type=int,
        choices=TARGET_FRAMES,
        help="standard comparison target; default is the largest reached by every run",
    )
    args = parser.parse_args()

    discovered: dict[tuple[str, int, int, int], tuple[Path, dict[int, Path]]] = {}
    for run_dir in sorted(args.batch_root.glob("GSR_*_/00_GSR_*")):
        match = RUN_RE.match(run_dir.name)
        if match is None:
            continue
        key = (
            match.group("backbone"),
            int(match.group("distance")),
            int(match.group("half_life")),
            int(match.group("seed")),
        )
        if key in discovered:
            raise RuntimeError(f"Duplicate graph-stabilized run: {key}")
        discovered[key] = (run_dir, dict(select_checkpoints(run_dir)))

    expected = set(product(BACKBONES, DISTANCES, HALF_LIVES_K, SEEDS))
    if set(discovered) != expected:
        raise RuntimeError(
            "Incomplete graph-stabilized matrix; "
            f"missing={sorted(expected - set(discovered))}, "
            f"unexpected={sorted(set(discovered) - expected)}"
        )

    latest_by_run = {
        key: checkpoint_frames(checkpoints[max(TARGET_FRAMES)])
        for key, (_, checkpoints) in discovered.items()
    }
    largest_shared_upper_bound = min(latest_by_run.values())
    reached_targets = [
        target for target in TARGET_FRAMES if all(latest >= target for latest in latest_by_run.values())
    ]
    if not reached_targets:
        raise RuntimeError("No standard telemetry target has been reached by every run")
    largest_shared_target = max(reached_targets)
    target = args.target_frames if args.target_frames is not None else largest_shared_target
    if target > largest_shared_target:
        raise ValueError(
            f"Requested target {target} exceeds the largest standard target reached "
            f"by every run ({largest_shared_target})"
        )

    rows: list[dict[str, str]] = []
    selected_frames: list[int] = []
    for (backbone, distance, half_life_k, seed), (run_dir, checkpoints) in sorted(discovered.items()):
        checkpoint = checkpoints[target]
        actual = checkpoint_frames(checkpoint)
        if checkpoint.parent.name != "milestones":
            raise RuntimeError(
                f"Shared-target checkpoint is not a stable milestone for {run_dir}: {checkpoint}"
            )
        condition = f"gsr_{backbone.lower()}_d{distance}_h{half_life_k}k"
        row = {
            "condition": condition,
            "family": "topology" if backbone == "C15" else "direct_hrl",
            "schedule": "sim",
            "feedback": "encourage",
            "half_life": str(half_life_k * 1_000),
            "seed": str(seed),
            "target_frames": str(target),
            "checkpoint_frames": str(actual),
            "checkpoint": str(checkpoint),
            "run_dir": str(run_dir),
            "label_suffix": f"{condition}__s{seed}__f{actual}",
        }
        rows.append(row)
        selected_frames.append(actual)

    if len(rows) != 36 or len({row["label_suffix"] for row in rows}) != len(rows):
        raise RuntimeError("Expected 36 rows with unique labels")
    for row in rows:
        checkpoint = Path(row["checkpoint"])
        run_dir = Path(row["run_dir"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if WORKSPACE not in checkpoint.parents or WORKSPACE not in run_dir.parents:
            raise RuntimeError(f"Non-workspace telemetry path: {row}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    manifests = {
        "analysis_manifest.tsv": rows,
        "trajectory_manifest.tsv": [row for row in rows if row["seed"] == "99"],
    }
    for name, values in manifests.items():
        with (args.output_root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(values)

    summary = {
        "runs": len(rows),
        "conditions": len({row["condition"] for row in rows}),
        "seeds": list(SEEDS),
        "largest_shared_upper_bound": largest_shared_upper_bound,
        "largest_shared_standard_target": largest_shared_target,
        "target_frames": target,
        "selected_checkpoint_min": min(selected_frames),
        "selected_checkpoint_max": max(selected_frames),
        "max_absolute_checkpoint_offset": max(abs(frame - target) for frame in selected_frames),
        "selection": "select_checkpoints(run_dir), nearest retained milestone to target_frames",
    }
    (args.output_root / "alignment_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
