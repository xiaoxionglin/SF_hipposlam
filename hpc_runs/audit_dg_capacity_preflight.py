"""Runtime gates for the declared DG-capacity preflight matrix."""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

from hpc_runs.audit_navigation8_algorithm_screen_preflight import _events
from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.version import WORKFLOW_VERSION


def audit(spec, jobs, root, required_frames):
    study = load_study(spec)
    runs = {r.name: r for r in study.expand_runs()}
    rows = list(csv.DictReader(Path(jobs).open(), delimiter="\t"))
    assert len(rows) == len(runs)
    results = []
    torch.set_num_threads(1)
    for job in rows:
        run = runs[job["experiment"].removeprefix("00_")]
        directory = Path(root) / job["train_root"] / job["experiment"]
        errors = []
        cfg = json.loads((directory / "config.json").read_text())
        args = {a.split("=", 1)[0].removeprefix("--"): a.split("=", 1)[1] for a in run.args}
        for k in [
            "Hippo_n_feature",
            "env_frameskip",
            "dg_goal_input",
            "hrl_manager_mode",
            "hrl_edge_exploration",
            "save_initial_checkpoint",
            "ppo_dg_gradient",
            "advantage_reward_source",
            "dmlab_navigation_action_set",
            "dg_orthogonal_recruitment",
        ]:
            if str(cfg.get(k)) != args[k]:
                errors.append(f"config mismatch {k}")
        initial = list((directory / "checkpoint_p0").glob("initial_*.pth"))
        checkpoints = sorted((directory / "checkpoint_p0").glob("checkpoint_*.pth"))
        if not initial:
            errors.append("missing preserved initial checkpoint")
        else:
            start = torch.load(initial[0], map_location="cpu", weights_only=False)
            if start["env_steps"] != 0 or start["train_step"] != 0:
                errors.append("initial counters nonzero")
            if cfg["dg_goal_input"] == "write" and start["model"]["core.dg_goal_modulation"].count_nonzero():
                errors.append("modulator not identity initialized")
        final = torch.load(checkpoints[-1], map_location="cpu", weights_only=False) if checkpoints else {}
        frames = int(final.get("env_steps", 0))
        if frames < required_frames:
            errors.append(f"checkpoint frames {frames} below {required_frames}")
        if cfg["dg_goal_input"] == "write" and final:
            mod = final["model"]["core.dg_goal_modulation"]
            if not torch.isfinite(mod).all() or mod.norm() <= 0:
                errors.append("modulation did not learn finite nonzero weights")
        if initial and final:
            frozen = [k for k in start["model"] if k.startswith("encoder.basic_encoder.")]
            if not frozen:
                errors.append("missing frozen visual trunk tensors")
            if any(not torch.equal(start["model"][k], final["model"][k]) for k in frozen):
                errors.append("frozen trunk changed, including normalization state")
            projection = [k for k in start["model"] if k.startswith("encoder.DG_projection.") and k.endswith("weight")]
            if not projection or not any(not torch.equal(start["model"][k], final["model"][k]) for k in projection):
                errors.append("DG projection did not train")
        values = _events(directory)
        learning = [(tag, es) for tag, es in values.items() if any(x in tag.lower() for x in ["loss", "gradient"])]
        if not learning:
            errors.append("missing learning scalars")
        for tag, es in learning:
            if any(not math.isfinite(float(e.value)) for e in es):
                errors.append(f"nonfinite {tag}")

        def peak(fragment):
            return max([float(e.value) for tag, es in values.items() if fragment in tag for e in es], default=0.0)

        if cfg["dg_goal_input"] == "write" and peak("dg_goal_modulation_gradient_norm") <= 0:
            errors.append("missing live modulation gradient")
        signals = {
            k: peak(k)
            for k in ["validation_success", "waypoint_navigation_fraction", "waypoint_step_hit", "route_available"]
        }
        signals["validation_success"] = max(signals["validation_success"], peak("hrl/validation/success_rate"))
        if run.base == "WAYPOINT_DG":
            if signals["validation_success"] <= 0:
                errors.append("waypoint validation never succeeded")
            if signals["waypoint_navigation_fraction"] <= 0:
                errors.append("multi-hop routing never exercised")
        snapshots = Path(root) / "analysis" / "online_spatial" / study.batch_name / run.name / "policy_00"
        present = set()
        for snapshot in snapshots.glob("*.npz"):
            with np.load(snapshot, allow_pickle=False) as data:
                present.add(int(data["target_env_steps"]))
                if int(data["frameskip"]) != 4:
                    errors.append("snapshot frameskip mismatch")
                if not np.isfinite(data["dg_activity"]).all():
                    errors.append("nonfinite snapshot DG")
        for target in (1000000, 2000000):
            if target not in present:
                errors.append(f"missing spatial snapshot {target}")
        for key in ["stdout", "stderr"]:
            p = Path(job[key].replace("%j", job["job_id"]))
            text = p.read_text(errors="replace") if p.exists() else ""
            if "Traceback (most recent call last)" in text:
                errors.append(f"traceback in {key}")
        result = dict(
            run=run.name, job_id=job["job_id"], frames=frames, passed=not errors, errors=errors, signals=signals
        )
        results.append(result)
        print(json.dumps(result), flush=True)
    return dict(
        schema="intrmotiv/study/v1",
        workflow_version=WORKFLOW_VERSION,
        study_workflow_version=study.declared_workflow_version,
        study_id=study.study_id,
        study_sha256=study.fingerprint,
        passed=all(r["passed"] for r in results),
        runs=results,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("study")
    p.add_argument("jobs")
    p.add_argument("train_root")
    p.add_argument("--required-frames", type=int, default=2000000)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = audit(a.study, a.jobs, a.train_root, a.required_frames)
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["passed"] else 1)
