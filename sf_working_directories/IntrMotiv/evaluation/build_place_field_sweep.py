"""Build the representative architecture x checkpoint place-field manifest."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

TARGET_FRAMES = (5_000_000, 25_000_000, 50_000_000, 75_000_000, 100_000_000)
FRAME_RE = re.compile(r"_(\d+)\.pth$")


@dataclass(frozen=True)
class Condition:
    name: str
    batch: str
    parent: str
    run: str
    family: str
    schedule: str
    feedback: str
    half_life: str


def flat_conditions() -> list[Condition]:
    batch = "intrmotiv_flat_iterative_baseline_nopbt_20260820"
    conditions = []
    for feedback in ("encourage", "mean", "punish"):
        for schedule in ("sim", "iter"):
            stem = f"FB_F16_L64_T243_ER{feedback}_{schedule}_S99"
            conditions.append(
                Condition(
                    f"flat_{feedback}_{schedule}",
                    batch,
                    f"{stem}_",
                    f"00_{stem}",
                    "fixed_flat",
                    schedule,
                    feedback,
                    "none",
                )
            )
    return conditions


def persistence_conditions() -> list[Condition]:
    batch = "intrmotiv_hrl_persistence_comparison_20260821"
    conditions = []
    for half_life in (5000, 10000, 20000):
        for schedule in ("sim", "iter"):
            stem = f"GHRL_F16_L64_T243_HL{half_life}_{schedule}_S99"
            conditions.append(
                Condition(
                    f"global_hrl_hl{half_life}_{schedule}",
                    batch,
                    f"{stem}_",
                    f"00_{stem}",
                    "global_hrl",
                    schedule,
                    "encourage",
                    str(half_life),
                )
            )
    # The long/per-stream half-life factor was inert in this batch. Retain its
    # middle setting once per update schedule to compare memory scope, not an
    # invalid nominal half-life sweep.
    for schedule in ("sim", "iter"):
        stem = f"LHRL_F16_L64_T243_HL10000_{schedule}_S99"
        conditions.append(
            Condition(
                f"stream_long_{schedule}",
                batch,
                f"{stem}_",
                f"00_{stem}",
                "stream_long_hrl",
                schedule,
                "encourage",
                "10000_inert",
            )
        )
    for schedule in ("sim", "iter"):
        stem = f"FLATLONG_F16_L64_T243_{schedule}_S99"
        conditions.append(
            Condition(
                f"flat_long_{schedule}", batch, f"{stem}_", f"00_{stem}", "long_flat", schedule, "encourage", "none"
            )
        )
    return conditions


def checkpoint_frames(path: Path) -> int:
    match = FRAME_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse frame count from {path.name}")
    return int(match.group(1))


def select_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    files = list((run_dir / "checkpoint_p0").glob("checkpoint_*.pth"))
    files += list((run_dir / "checkpoint_p0" / "milestones").glob("checkpoint_*.pth"))
    indexed = {checkpoint_frames(path): path for path in files}
    if not indexed:
        raise FileNotFoundError(f"No checkpoints under {run_dir}")
    candidates = sorted(indexed.items())
    return [(target, min(candidates, key=lambda pair: abs(pair[0] - target))[1]) for target in TARGET_FRAMES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conditions = flat_conditions() + persistence_conditions()
    rows: list[dict[str, str]] = []
    for condition in conditions:
        run_dir = args.train_root / condition.batch / condition.parent / condition.run
        for target, checkpoint in select_checkpoints(run_dir):
            actual = checkpoint_frames(checkpoint)
            rows.append(
                {
                    "condition": condition.name,
                    "family": condition.family,
                    "schedule": condition.schedule,
                    "feedback": condition.feedback,
                    "half_life": condition.half_life,
                    "seed": "99",
                    "target_frames": str(target),
                    "checkpoint_frames": str(actual),
                    "checkpoint": str(checkpoint),
                    "run_dir": str(run_dir),
                    "label_suffix": f"{condition.name}__{actual // 1_000_000:03d}M",
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} tasks for {len(conditions)} representative conditions to {args.output}")


if __name__ == "__main__":
    main()
