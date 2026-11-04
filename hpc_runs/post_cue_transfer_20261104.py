"""Render, check, and submit the existing evaluators after cue-transfer training.

Run this once in a compute-node job depending on all twelve production jobs.
The canonical manifest builder and ordinary-job submitter own discovery and
scheduling. This adapter supplies only this study's terminal heldout protocol.
Without --submit it records both validated print-only plans and submits no jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


def audit_manifest(study, rows: list[dict[str, str]]) -> dict:
    """Reject missing milestones and unmatched saved ages before evaluation."""
    targets = study.telemetry["target_frames"]
    trajectory_seed = study.telemetry["trajectory_seed"]
    terminal_seeds = set(study.telemetry["terminal_seeds"])
    expected = {
        (run.condition, str(run.seed), str(target))
        for run in study.expand_runs()
        for target in (targets if run.seed == trajectory_seed else [targets[-1]])
        if run.seed == trajectory_seed or run.seed in terminal_seeds
    }
    observed = [(row["condition"], row["seed"], row["target_frames"]) for row in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("Field manifest does not contain the complete declared study panel")
    saved_ages: dict[int, set[int]] = {}
    checkpoint_targets: dict[tuple[str, str, str], set[int]] = {}
    for row in rows:
        target, actual = int(row["target_frames"]), int(row["checkpoint_frames"])
        if actual < target:
            raise ValueError(f"Missing completed milestone: {row['label_suffix']} selects {actual} before {target}")
        saved_ages.setdefault(target, set()).add(actual)
        key = (row["condition"], row["seed"], row["checkpoint"])
        checkpoint_targets.setdefault(key, set()).add(target)
    if any(len(values) != 1 for values in checkpoint_targets.values()):
        raise ValueError("A checkpoint was reused for different requested milestones")
    if any(len(values) != 1 for values in saved_ages.values()):
        raise ValueError(f"Compared runs lack shared actual checkpoint ages: {saved_ages}")
    terminal = [row for row in rows if int(row["target_frames"]) == targets[-1]]
    if len(rows) != 28 or len(terminal) != 12:
        raise ValueError("This production adapter requires 28 field rows and 12 terminal rows")
    return {
        **study.provenance(),
        "field_rows": len(rows),
        "terminal_rows": len(terminal),
        "actual_frames_by_requested_target": {
            str(target): next(iter(values)) for target, values in sorted(saved_ages.items())
        },
        "heldout": {
            "trials_per_run": 100,
            "seed_base": 61000,
            "horizon_decisions": 1800,
            "policy": "stochastic with evaluator-defined per-reset policy seed",
            "cue_sampling": "sequential matched resets; report equal-cue weighted success, not balanced counts",
            "cue_shuffle": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Isolated runtime release")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--study", type=Path)
    parser.add_argument("--batch-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    python = args.python.resolve(strict=True)
    sys.path.insert(0, str(source))
    from hpc_runs.intrmotiv_study import load_study
    from hpc_runs.intrmotiv_study.telemetry import write_manifest

    study_path = (args.study or source / "hpc_runs/studies/cue_transfer_20261104.study.json").resolve(strict=True)
    study = load_study(study_path)
    if study.study_id != "cue_transfer_20261104" or study.expected_runs != 12:
        raise ValueError("This adapter requires the twelve-run cue_transfer_20261104 production study")
    workspace = Path(study.workspace_root).resolve(strict=True)
    batch = (args.batch_root or Path(study.output_root)).resolve(strict=True)
    output = (args.output_root or workspace / "IntrMotiv/SF_hipposlam/train_dir/analysis/cue_transfer_20261104/frozen").resolve()
    if not batch.is_relative_to(workspace) or not output.is_relative_to(workspace):
        raise ValueError("Batch and evaluation outputs must stay in the active study workspace")
    output.mkdir(parents=True, exist_ok=True)
    evaluation = source / "sf_working_directories/IntrMotiv/evaluation"
    submitter = evaluation / "submit_place_field_sweep.py"
    heldout_runner = Path(__file__).resolve().with_name("run_cue_transfer_heldout_single.sh")
    for path in (submitter, heldout_runner, evaluation / "fixed_reward_heldout.py"):
        if not path.is_file():
            raise FileNotFoundError(path)

    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(source) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        "INTRMOTIV_RUNTIME_SOURCE": str(source),
        "INTRMOTIV_EVAL_PYTHON": str(python),
        "INTRMOTIV_WORKSPACE_ROOT": str(workspace),
        "TMPDIR": str(workspace / "tmp" / ("ctpost_" + env.get("SLURM_JOB_ID", str(os.getpid())))),
        "XDG_CACHE_HOME": str(output / "cache"),
        "DMLAB_CACHE_DIR": str(output / "dmlab_cache"),
        "TORCH_HOME": str(workspace / "IntrMotiv/SF_hipposlam/runtime/cache/torch"),
        "WANDB_DIR": str(output / "wandb"),
        "WANDB_MODE": "disabled",
    })
    for key in ("TMPDIR", "XDG_CACHE_HOME", "DMLAB_CACHE_DIR", "TORCH_HOME", "WANDB_DIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)

    def run(command: list[str], log: str) -> None:
        with (output / log).open("w") as handle:
            subprocess.run(command, cwd=source, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)

    run([str(python), "-m", "hpc_runs.intrmotiv_study", "render-telemetry", str(study_path), str(batch), str(output)], "render_telemetry.log")
    manifest = output / "analysis_manifest.tsv"
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    audit = audit_manifest(study, rows)
    audit["execution"] = {
        "runtime_source": str(source),
        "python": str(python),
        "heldout_runner": str(heldout_runner),
        "batch_root": str(batch),
        "output_root": str(output),
    }
    terminal = [row for row in rows if int(row["target_frames"]) == study.telemetry["target_frames"][-1]]
    terminal_manifest = output / "terminal_heldout_manifest.tsv"
    write_manifest(terminal_manifest, terminal)
    (output / "evaluation_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    plans = {
        "fields": [str(python), str(submitter), "--manifest", str(manifest), "--output-dir", str(output / "fields"), "--workspace-root", str(workspace), "--max-num-frames", "10000", "--time-limit", "02:00:00", "--job-name-prefix", "ct-fields"],
        "heldout": [str(python), str(submitter), "--manifest", str(terminal_manifest), "--output-dir", str(output / "heldout"), "--workspace-root", str(workspace), "--runner", str(heldout_runner), "--time-limit", "04:00:00", "--job-name-prefix", "ct-heldout"],
    }
    # Review both existing-submitters' fully validated plans before either submit.
    for name, command in plans.items():
        run(command, name + "_print_only.log")
    if args.submit:
        for name, command in plans.items():
            run(command + ["--submit"], name + "_submission.log")
    print(json.dumps({"output_root": str(output), "field_rows": 28, "heldout_rows": 12, "submitted": args.submit}, indent=2))


if __name__ == "__main__":
    main()
