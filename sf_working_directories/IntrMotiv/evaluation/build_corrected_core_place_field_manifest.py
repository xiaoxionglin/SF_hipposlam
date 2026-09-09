#!/usr/bin/env python3
"""Build the standard 112-row place-field manifest for corrected-core cells."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from sf_working_directories.IntrMotiv.evaluation.build_place_field_sweep import (
    checkpoint_frames,
    select_checkpoints,
)


RUN_RE = re.compile(r"^00_CCR_C(?P<cell>\d{2})_(?P<tag>.+)_S(?P<seed>8|99|123)$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    discovered: dict[tuple[int, int], tuple[str, Path]] = {}
    for event in sorted(args.batch_root.glob("*/*/.summary/0/events.out.tfevents.*")):
        run_dir = event.parents[2]
        run_name = event.parents[2].name
        match = RUN_RE.match(run_name)
        if match is None:
            continue
        cell, seed = int(match.group("cell")), int(match.group("seed"))
        key = (cell, seed)
        if key in discovered:
            raise RuntimeError(f"Duplicate corrected-core run for C{cell:02d}, seed {seed}")
        discovered[key] = (match.group("tag"), run_dir)

    expected = {(cell, seed) for cell in range(1, 17) for seed in (8, 99, 123)}
    if set(discovered) != expected:
        raise RuntimeError(f"Incomplete corrected-core matrix; missing={sorted(expected - set(discovered))}")

    rows: list[dict[str, str]] = []
    trajectory_rows: list[dict[str, str]] = []
    for cell, seed in sorted(discovered):
        tag, run_dir = discovered[(cell, seed)]
        selected = select_checkpoints(run_dir)
        if seed != 99:
            selected = selected[-1:]
        for target, checkpoint in selected:
            actual = checkpoint_frames(checkpoint)
            condition = f"c{cell:02d}_{tag.lower()}"
            family = "flat" if cell == 1 else "topology" if cell >= 14 else "direct_hrl"
            row = {
                "condition": condition,
                "family": family,
                "schedule": "iter" if cell == 4 else "sim",
                "feedback": "encourage",
                "half_life": "none" if cell == 1 else "10000",
                "seed": str(seed),
                "target_frames": str(target),
                "checkpoint_frames": str(actual),
                "checkpoint": str(checkpoint),
                "run_dir": str(run_dir),
                "label_suffix": f"{condition}__s{seed}__{actual // 1_000_000:03d}m",
            }
            rows.append(row)
            if seed == 99:
                trajectory_rows.append(row)

    if len(rows) != 112 or len(trajectory_rows) != 80:
        raise RuntimeError(f"Expected 112 analysis and 80 trajectory rows, got {len(rows)} and {len(trajectory_rows)}")
    if len({row["label_suffix"] for row in rows}) != len(rows):
        raise RuntimeError("Telemetry labels are not unique")
    workspace = Path("/work/classic/fr_xl1014-train")
    for row in rows:
        if not Path(row["checkpoint"]).is_file():
            raise FileNotFoundError(row["checkpoint"])
        if workspace not in Path(row["checkpoint"]).parents or workspace not in Path(row["run_dir"]).parents:
            raise RuntimeError(f"Non-workspace telemetry path: {row}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for name, values in (("analysis_manifest.tsv", rows), ("trajectory_manifest.tsv", trajectory_rows)):
        with (args.output_root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(values)
    print(f"Verified {len(rows)} tasks, {len(trajectory_rows)} seed-99 trajectory rows, and workspace-only paths")


if __name__ == "__main__":
    main()
