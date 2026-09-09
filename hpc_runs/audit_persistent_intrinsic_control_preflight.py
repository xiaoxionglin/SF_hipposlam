"""Scientific runtime gates for the one-seed persistent-control preflight."""

import argparse
import json
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from hpc_runs.intrmotiv_study import load_study


TAGS = {
    "frames": "train/env_steps",
    "onset": "intrmotiv/memory/onset_fraction",
    "goal_active": "intrmotiv/memory/goal_active_fraction",
    "goal_hit": "intrmotiv/memory/goal_hit_fraction",
    "goal_ambiguous": "intrmotiv/memory/goal_ambiguous_fraction",
    "replay_mismatch": "intrmotiv/memory/replay_mismatch",
    "ppo_gradient": "intrmotiv/dg/gradient/ppo_norm",
    "encoder_gradient": "intrmotiv/dg/gradient/encoder_norm",
    "reward_nonzero": "intrmotiv/reward/intrinsic_nonzero_fraction",
}


def scalar_values(run_dir: Path):
    values = {}
    event_dirs = {path.parent for path in run_dir.rglob("events.out.tfevents.*")}
    for directory in sorted(event_dirs):
        accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0}).Reload()
        for tag in accumulator.Tags()["scalars"]:
            values.setdefault(tag, []).extend(accumulator.Scalars(tag))
    return values


def audit(study, root: Path, required_frames: int):
    runs = [run for run in study.expand_runs() if run.seed == 99]
    records = []
    for run in runs:
        candidates = list(root.glob(f"{run.name}_*/00_{run.name}")) + list(root.glob(f"{run.name}_*/*{run.name}*"))
        record = {"run": run.name, "errors": []}
        if not candidates:
            record["errors"].append("run directory missing")
            records.append(record)
            continue
        values = scalar_values(sorted(candidates)[-1])
        goal = run.metadata["control"] == "goal"
        for key, tag in TAGS.items():
            applicable = goal or key not in {"goal_active", "goal_hit", "goal_ambiguous", "replay_mismatch"}
            events = sorted(values.get(tag, []), key=lambda event: event.wall_time) if applicable else []
            record[key] = ({"last": events[-1].value, "max": max(event.value for event in events), "samples": len(events)}
                           if events else {"last": None, "max": 0.0, "samples": 0, "applicable": applicable})
            if applicable and (not events or any(not math.isfinite(event.value) for event in events)):
                record["errors"].append(f"{key}: missing or nonfinite")
        if record["frames"]["max"] < required_frames:
            record["errors"].append("incomplete training")
        if record["onset"]["max"] <= 0 or record["encoder_gradient"]["max"] <= 0:
            record["errors"].append("landmark learning signal absent")
        if goal:
            if record["goal_active"]["max"] <= 0:
                record["errors"].append("goal command never active")
            if record["replay_mismatch"]["max"] > 1e-7:
                record["errors"].append("goal replay mismatch")
        expected_joint = run.metadata["gradient"] == "joint"
        if expected_joint and record["ppo_gradient"]["max"] <= 0:
            record["errors"].append("JOINT PPO-to-DG gradient absent")
        if not expected_joint and record["ppo_gradient"]["max"] > 1e-7:
            record["errors"].append("STOP PPO-to-DG gradient nonzero")
        if not goal and record["reward_nonzero"]["max"] <= 0:
            record["errors"].append("flat intrinsic reward absent")
        records.append(record)
    return {
        **study.provenance(),
        "preflight_seed": 99,
        "expected_conditions": 11,
        "required_frames": required_frames,
        "passed": len(records) == 11 and all(not record["errors"] for record in records),
        "runs": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("--required-frames", type=int, default=2_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(load_study(args.study), args.root, args.required_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
