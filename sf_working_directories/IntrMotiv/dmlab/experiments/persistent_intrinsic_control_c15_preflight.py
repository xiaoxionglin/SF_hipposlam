"""Seed-99 strict-resume smoke test for the original-C15 continuation arm."""

from pathlib import Path

from sample_factory.launcher.run_description import Experiment, RunDescription

from hpc_runs.intrmotiv_study import load_study
from sf_working_directories.IntrMotiv.dmlab.experiments.corrected_core_reevaluation import (
    CELLS,
    PROJECT as ORIGINAL_PROJECT,
    make_experiment,
)


SPEC = Path(__file__).resolve().parents[4] / "hpc_runs/studies/persistent_intrinsic_control_c15.study.json"
STUDY = load_study(SPEC)
C15 = next(cell for cell in CELLS if cell.number == 15)


def build(run):
    original = make_experiment(
        C15,
        run.seed,
        train_for_env_steps=102_000_000,
        group_suffix="pic_c15_preflight_20260908",
    )
    command = original.cmd.replace(
        f"--wandb_project={ORIGINAL_PROJECT}",
        "--wandb_project=SF_IntrMotiv_PersistentIntrinsicControl",
    ).replace(
        "--wandb_group=intrmotiv_corrected_core_reevaluation_20260901_pic_c15_preflight_20260908",
        "--wandb_group=intrmotiv_persistent_intrinsic_control_c15_20260908_preflight",
    )
    supplemental = [
        arg
        for arg in run.args
        if arg.startswith("--load_model_path=") or arg.startswith("--online_spatial_snapshot_")
    ]
    return Experiment(run.name, f"{command} {' '.join(supplemental)}", [{}])


RUN_DESCRIPTION = RunDescription(
    "intrmotiv_persistent_intrinsic_control_c15_20260908_preflight",
    experiments=[build(run) for run in STUDY.expand_runs() if run.seed == 99],
)
