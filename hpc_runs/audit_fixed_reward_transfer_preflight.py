"""Fail-closed runtime audit for fixed-reward transfer preflights."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.spec import SpecError


def declared_frameskip(run):
    values = [int(arg.split("=", 1)[1]) for arg in run.args if arg.startswith("--env_frameskip=")]
    if len(values) != 1 or values[0] <= 0:
        raise SpecError(f"{run.name}: expected one positive explicit env_frameskip")
    return values[0]


def source_interface_errors(source_config, destination_config):
    errors = []
    for key in (
        "env_frameskip",
        "core_name",
        "Hippo_n_feature",
        "Hippo_R",
        "Hippo_L",
        "encoder_conv_architecture",
        "depth_sensor",
        "normalize_input",
        "hrl_goal_conditioning",
    ):
        if key not in source_config or source_config[key] != destination_config.get(key):
            errors.append(
                f"source interface mismatch {key}: {source_config.get(key)!r} vs {destination_config.get(key)!r}"
            )
    for key in ("dmlab_reduced_action_set", "dmlab_extended_action_set", "dmlab_navigation_action_set"):
        if source_config.get(key, False) != destination_config.get(key, False):
            errors.append(f"source action mismatch {key}")
    return errors


def _events(run_dir: Path):
    values = {}
    for directory in sorted({p.parent for p in run_dir.rglob("events.out.tfevents.*")}):
        acc = EventAccumulator(str(directory), size_guidance={"scalars": 100_000}).Reload()
        for tag in acc.Tags().get("scalars", []):
            values.setdefault(tag, []).extend(acc.Scalars(tag))
    return values


def _checkpoint_frames(run_dir: Path):
    found = []
    for path in run_dir.glob("checkpoint_p0/checkpoint_*.pth"):
        match = re.search(r"_(\d+)\.pth$", path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def _load_model(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


def audit(study, jobs_tsv: Path, train_root: Path, required_frames: int):
    with jobs_tsv.open(newline="", encoding="utf-8") as handle:
        jobs = list(csv.DictReader(handle, delimiter="\t"))
    expected = {run.name: run for run in study.expand_runs()}
    if len(jobs) != len(expected):
        raise SpecError(f"submission has {len(jobs)} rows; expected {len(expected)}")
    source_models = {}
    records = []
    for job in jobs:
        experiment = job["experiment"]
        name = experiment[3:] if experiment.startswith("00_") else experiment
        run = expected.get(name)
        if run is None:
            raise SpecError(f"unexpected run {name}")
        run_dir = train_root / job["train_root"] / experiment
        errors = []
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        required = {
            "env": "openfield_map2_fixed_loc3",
            "env_frameskip": declared_frameskip(run),
            "dmlab_reduced_action_set": True,
            "dmlab_navigation_action_set": False,
            "num_policies": 1,
            "with_pbt": False,
            "fixed_task_conditioning": True,
            "hrl_controllable_graph": False,
            "advantage_reward_source": "external",
            "extra_encoder_losses": False,
            "reward_scale": 1.0,
            "rnn_size": 1149,
        }
        for key, value in required.items():
            if config.get(key) != value:
                errors.append(f"config {key}={config.get(key)!r}; expected {value!r}")

        events = _events(run_dir) if run_dir.is_dir() else {}
        frame_events = events.get("train/env_steps", [])
        max_frames = max((float(e.value) for e in frame_events), default=math.nan)
        if not math.isfinite(max_frames) or max_frames < required_frames:
            errors.append(f"max env frames {max_frames!r} below {required_frames}")
        for tag, tag_events in events.items():
            if "loss" in tag.lower() or "gradient" in tag.lower():
                if any(not math.isfinite(float(e.value)) for e in tag_events):
                    errors.append(f"nonfinite learning scalar {tag}")

        checkpoints = _checkpoint_frames(run_dir)
        terminal_frame, terminal_path = checkpoints[-1] if checkpoints else (0, None)
        if terminal_frame < required_frames:
            errors.append(f"terminal checkpoint {terminal_frame} below {required_frames}")
        initial = next((p for frame, p in checkpoints if frame == 0), None)

        transfer_path = config.get("transfer_model_path")
        scope = config.get("transfer_scope", "none")
        if transfer_path and Path(transfer_path).is_file():
            if study.study_metadata.get("require_source_interface_match", False):
                source_config = json.loads((Path(transfer_path).parents[1] / "config.json").read_text())
                errors.extend(source_interface_errors(source_config, config))
            if transfer_path not in source_models:
                source_models[transfer_path] = _load_model(Path(transfer_path))
            source = source_models[transfer_path]
            prefixes = ("encoder.DG_projection.linear.", "encoder.DG_projection.batchnorm1d.")
            if scope == "policy":
                prefixes += ("decoder.", "action_parameterization.")
            if initial:
                initialized = _load_model(initial)
                for key in (k for k in initialized if k.startswith(prefixes)):
                    if key not in source or not torch.equal(initialized[key], source[key]):
                        errors.append(f"initial transfer mismatch {key}")
                if scope == "policy" and torch.equal(
                    initialized["critic_linear.weight"], source["critic_linear.weight"]
                ):
                    errors.append("critic was transferred instead of freshly initialized")
            if config.get("transfer_freeze_dg") and terminal_path:
                terminal = _load_model(terminal_path)
                for key in (k for k in source if k.startswith(prefixes[:2])):
                    if key in terminal and not torch.equal(terminal[key], source[key]):
                        errors.append(f"frozen DG changed: {key}")
        elif scope != "none":
            errors.append("transfer scope has no readable source checkpoint")

        inspected = initial or terminal_path
        if inspected:
            target = _load_model(inspected).get("core.fixed_task_target")
            if target is None or not torch.isfinite(target).all() or tuple(target.shape) != (16,):
                errors.append("constant task vector is missing, nonfinite, or has the wrong shape")
            elif initial and not torch.allclose(target, torch.full((16,), 1 / 16)):
                errors.append("constant task vector is not uniformly initialized")

        stdout = Path(job["stdout"].replace("%j", job.get("job_id", "")))
        stderr = Path(job["stderr"].replace("%j", job.get("job_id", "")))
        logs = "\n".join(p.read_text(errors="replace") for p in (stdout, stderr) if p.is_file())
        if "Traceback (most recent call last)" in logs or "RuntimeError:" in logs:
            errors.append("runtime log contains an exception")
        if scope in ("dg", "policy") and f" {scope} transfer tensors from " not in logs:
            errors.append("transfer initialization was not logged")
        if config.get("transfer_freeze_dg") and "Froze DG weights and normalization statistics" not in logs:
            errors.append("DG freeze was not logged")

        records.append(
            {
                "run_name": name,
                "job_id": job.get("job_id"),
                "max_env_frames": max_frames,
                "checkpoint_frames": terminal_frame,
                "pass": not errors,
                "errors": errors,
            }
        )
    return {
        **study.provenance(),
        "protocol": "fixed-reward-transfer-preflight-v1",
        "required_frames": required_frames,
        "expected_runs": len(expected),
        "passed_runs": sum(r["pass"] for r in records),
        "passed": len(records) == len(expected) and all(r["pass"] for r in records),
        "runs": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("jobs_tsv", type=Path)
    parser.add_argument("train_root", type=Path)
    parser.add_argument("--required-frames", type=int, default=2_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(load_study(args.study), args.jobs_tsv, args.train_root, args.required_frames)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
