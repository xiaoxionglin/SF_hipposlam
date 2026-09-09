"""Build the 10k-decision place-field manifest for the encourage regularizer batch."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sf_working_directories.IntrMotiv.dmlab.experiments.encourage_dg_regularizers import (
    BATCH_NAME,
    LOSS_ARMS,
    SEEDS,
    THRESHOLDS,
    run_name,
)
from sf_working_directories.IntrMotiv.evaluation.build_place_field_sweep import select_checkpoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    for architecture in ("flat", "global_fixed"):
        family = "flat" if architecture == "flat" else "global_fixed_hrl"
        half_life = "none" if architecture == "flat" else "10000"
        for threshold in THRESHOLDS:
            for loss_arm in LOSS_ARMS:
                condition = f"{architecture}_t{threshold:.2f}_{loss_arm.tag.lower()}".replace(".", "p")
                for seed in SEEDS:
                    name = run_name(architecture, threshold, loss_arm, seed)
                    run_dir = args.train_root / BATCH_NAME / f"{name}_" / f"00_{name}"
                    checkpoints = select_checkpoints(run_dir)
                    selected = checkpoints if seed == 99 else [checkpoints[-1]]
                    for target_frames, checkpoint in selected:
                        checkpoint_frames = int(checkpoint.stem.rsplit("_", 1)[-1])
                        rows.append(
                            {
                                "condition": condition,
                                "family": family,
                                "schedule": "simultaneous",
                                "feedback": "encourage",
                                "half_life": half_life,
                                "seed": str(seed),
                                "target_frames": str(target_frames),
                                "checkpoint_frames": str(checkpoint_frames),
                                "checkpoint": str(checkpoint),
                                "run_dir": str(run_dir),
                                "label_suffix": f"{condition}_S{seed}__{checkpoint_frames // 1_000_000:03d}M",
                            }
                        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} evaluation tasks for 20 conditions to {args.output}")


if __name__ == "__main__":
    main()
