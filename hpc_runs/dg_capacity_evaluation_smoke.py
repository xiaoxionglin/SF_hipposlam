"""Slurm-only adapter exercising the canonical panel and intervention evaluators."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.discovery import discover_run_directories
from sf_working_directories.IntrMotiv.evaluation.place_fields import load_policy_env, rollout_dg
from sf_working_directories.IntrMotiv.evaluation.target_control_interventions import run_landmark_matched_interventions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("study")
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Evaluation requires a Slurm job")
    torch.set_num_threads(1)
    study = load_study(a.study)
    dirs = discover_run_directories(study, a.root)
    run = next(r for r in study.expand_runs() if r.base == "DIRECT_DG" and r.context["capacity"] == 16)
    a.output.mkdir(parents=True, exist_ok=True)
    panel = a.output / "panel.npz"
    first = rollout_dg(dirs[run.name], 128, True, 0, record_panel=panel)
    second = rollout_dg(dirs[run.name], 128, True, 0, checkpoint_path=first[1], replay_panel=panel)
    np.testing.assert_array_equal(first[3], second[3])
    np.testing.assert_array_equal(first[4], second[4])
    np.testing.assert_array_equal(first[5]["worker_dg_activity"], second[5]["worker_dg_activity"])
    weights = torch.load(first[1], map_location="cpu", weights_only=False)["model"]["core.dg_goal_modulation"]
    ids = weights.norm(dim=-1).topk(2).indices.tolist()
    forced = [
        rollout_dg(dirs[run.name], 128, True, 0, checkpoint_path=first[1], replay_panel=panel, panel_goal=g)
        for g in ids
    ]
    np.testing.assert_array_equal(forced[0][3], forced[1][3])
    np.testing.assert_array_equal(forced[0][4], forced[1][4])
    worker_difference = float(np.max(np.abs(forced[0][5]["worker_dg_activity"] - forced[1][5]["worker_dg_activity"])))
    if worker_difference <= 0:
        raise RuntimeError("Forced goals did not alter worker DG on the panel")
    cfg, env, info, actor, cp, device = load_policy_env(dirs[run.name], 100000, False, 0, first[1])
    trials, summary = run_landmark_matched_interventions(
        cfg, env, info, actor, cp, device, 20000, False, max_sources=1, targets_per_source=2, repeats=1, prefix_cap=512
    )
    if not summary["exact_start_verified"] or summary["paired_comparisons"] < 2:
        raise RuntimeError("No matched command panel completed")
    trials.to_csv(a.output / "intervention_trials.csv", index=False)
    result = dict(
        study_sha256=study.fingerprint,
        panel_exact_replay=True,
        detector_goal_invariant=True,
        worker_goal_difference=worker_difference,
        interventions=summary,
    )
    (a.output / "smoke_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
