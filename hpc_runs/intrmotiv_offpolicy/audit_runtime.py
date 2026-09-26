"""Thin StudySpec-driven runtime gate; scientific qualification stays separate."""

import argparse
import json
import math
from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.discovery import discover_run_directories


def validate_native_accounting(gate, final, arguments):
    """Native SF counts consumed physical transitions, excluding transport tails."""
    if gate.get("execution") != "sample_factory_native":
        return
    accepted, decisions = gate["accepted"], gate["decisions"]
    if accepted + gate["invalid_final_exclusions"] != decisions:
        raise ValueError("SF accepted/invalid decision accounting mismatch")
    expected = max(0, (accepted - int(arguments["learning-start"])) // int(arguments["decisions-per-update"]))
    if gate["updates"] != expected or gate["update_debt"] != 0:
        raise ValueError("SF exact optimizer update budget mismatch")
    if gate["transport_emitted"] != decisions or gate["transport_received"] != decisions + gate["transport_pending"]:
        raise ValueError("SF transport accounting mismatch")
    if final["accepted"] != accepted or final["decisions"] != decisions:
        raise ValueError("SF final telemetry accounting mismatch")


def audit(study, batch_root):
    directories = discover_run_directories(study, Path(batch_root))
    rows = []
    initial_by_seed = {}
    for run in study.expand_runs():
        path = directories[run.name]
        gate = json.loads((path / "runtime_gate.json").read_text())
        conversion = json.loads((path / "conversion.json").read_text())
        metrics = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines() if line]
        if not metrics or metrics[-1]["frames"] != gate["frames"]:
            raise ValueError(f"{run.name}: gate and final metrics disagree")
        arguments = dict(arg[2:].split("=", 1) for arg in run.args if arg.startswith("--") and "=" in arg)
        validate_native_accounting(gate, metrics[-1], arguments)
        expected = int(arguments["total-frames"])
        if gate["frames"] < expected or gate["updates"] <= 0 or gate["target_copies"] <= 0:
            raise ValueError(f"{run.name}: incomplete frame/update/target-copy gate")
        if gate["invalid_final_exclusions"] <= 0 or gate["frozen_reference_unchanged"] is not True:
            raise ValueError(f"{run.name}: reset/frozen-reference gate failed")
        if float(arguments["her-fraction"]) > 0 and gate["her_samples"] <= 0:
            raise ValueError(f"{run.name}: no genuine HER samples")
        if any(not math.isfinite(v) for m in metrics for v in m.values() if isinstance(v, (int, float))):
            raise ValueError(f"{run.name}: nonfinite recorded metrics")
        if "td-positions-per-update" in arguments:
            positions = int(arguments["td-positions-per-update"])
            period = int(arguments["target-period"])
            if gate.get("valid_loss_positions") != gate["updates"] * positions:
                raise ValueError("v2 valid TD-position budget mismatch")
            if gate["target_copies"] != gate["updates"] // period:
                raise ValueError("v2 target-copy cadence mismatch")
            if metrics[-1].get("valid_loss_positions_total") != gate["valid_loss_positions"]:
                raise ValueError("v2 TD telemetry and gate disagree")
        seed = int(arguments["seed"])
        initial = conversion["worker_hash"]
        if seed in initial_by_seed and initial_by_seed[seed] != initial:
            raise ValueError("matched HER arms have different initialization")
        initial_by_seed[seed] = initial
        active = [m for m in metrics if m["updates"] > 0]
        if len(active) < 2:
            raise ValueError("insufficient learner-active throughput samples")
        first, last = active[0], active[-1]
        elapsed = last["frames"] / last["throughput_fps"] - first["frames"] / first["throughput_fps"]
        if elapsed <= 0:
            raise ValueError("nonpositive learner-active measurement duration")
        fps = (last["frames"] - first["frames"]) / elapsed
        checkpoints = list((path / "checkpoint_p0").glob("checkpoint_*.pth"))
        if not any(p.name.endswith("_0.pth") for p in checkpoints) or not any(
            p.name.endswith(f"_{gate['frames']}.pth") for p in checkpoints
        ):
            raise ValueError(f"{run.name}: missing initial or terminal checkpoint")
        rows.append(
            dict(
                run=run.name,
                **gate,
                learner_active_fps=fps,
                realized_her_fraction=last["realized_her_fraction"],
                checkpoint_files=[str(p) for p in checkpoints],
            )
        )
    return dict(
        schema="intrmotiv/ddqn-runtime-audit/v1",
        study_sha256=study.fingerprint,
        workflow_version=study.declared_workflow_version,
        runtime_passed=True,
        scientific_qualification="pending_independent_commanded_and_spatial_evaluation",
        runs=rows,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("study", type=Path)
    p.add_argument("batch_root", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = audit(load_study(args.study), args.batch_root)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
