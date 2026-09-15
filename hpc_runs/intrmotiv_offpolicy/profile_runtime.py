"""Profile recorded DDQN stage timings using canonical StudySpec discovery."""

import argparse
import json
import statistics
from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.discovery import discover_run_directories


def profile(study, batch_root, window=250000):
    directories = discover_run_directories(study, Path(batch_root))
    rows = []
    for run in study.expand_runs():
        data = [json.loads(x) for x in (directories[run.name] / "metrics.jsonl").read_text().splitlines() if x]
        high = data[-1]
        eligible = [m for m in data if m["frames"] <= high["frames"] - window and m["updates"] > 0]
        if not eligible:
            raise ValueError(f"{run.name}: insufficient learner-active timing window")
        low = eligible[-1]
        late = [m for m in data if low["frames"] < m["frames"] <= high["frames"]]
        delta = {k: high[k] - low[k] for k in ("frames", "updates", "collection_seconds", "learning_seconds")}
        timed = delta["collection_seconds"] + delta["learning_seconds"]
        wall = high["frames"] / high["throughput_fps"] - low["frames"] / low["throughput_fps"]
        if min(timed, wall) <= 0:
            raise ValueError("nonpositive timing window")
        rows.append(
            dict(
                run=run.name,
                low_frames=low["frames"],
                high_frames=high["frames"],
                collection_fraction_of_timed=delta["collection_seconds"] / timed,
                learning_fraction_of_timed=delta["learning_seconds"] / timed,
                learner_active_fps=delta["frames"] / wall,
                sampled_prefix_batch_ms=1000 * statistics.mean(m["prefix_and_batch_seconds"] for m in late),
                sampled_learner_ms=1000 * statistics.mean(m["learner_update_seconds"] for m in late),
                optimizer_updates=delta["updates"],
                **{k: delta[k] for k in ("collection_seconds", "learning_seconds")},
            )
        )
    return dict(
        schema="intrmotiv/ddqn-stage-profile/v1",
        study_sha256=study.fingerprint,
        workflow_version=study.declared_workflow_version,
        caveat="Per-run latest windows; sampled substage means are not a full profiler trace. FPS includes log/checkpoint time after learner warmup; startup excluded.",
        runs=rows,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("study", type=Path)
    p.add_argument("batch_root", type=Path)
    p.add_argument("--window", type=int, default=250000)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = profile(load_study(args.study), args.batch_root, args.window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
