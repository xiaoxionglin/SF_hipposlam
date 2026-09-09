"""One-seed-per-condition preflight derived from the production StudySpec."""

from pathlib import Path

from sample_factory.launcher.run_description import Experiment, RunDescription

from hpc_runs.intrmotiv_study import load_study


SPEC = Path(__file__).resolve().parents[4] / "hpc_runs/studies/persistent_intrinsic_control.study.json"
STUDY = load_study(SPEC)


def preflight(run):
    args = []
    for arg in run.args:
        arg = arg.replace("--train_for_env_steps=600000000", "--train_for_env_steps=2000000")
        arg = arg.replace(
            "--online_spatial_snapshot_targets=100000000,200000000,300000000,450000000,600000000",
            "--online_spatial_snapshot_targets=1000000,2000000",
        )
        arg = arg.replace("--online_spatial_snapshot_max_frames=600000000", "--online_spatial_snapshot_max_frames=2000000")
        arg = arg.replace(
            "--wandb_group=intrmotiv_persistent_intrinsic_control_20260908",
            "--wandb_group=intrmotiv_persistent_intrinsic_control_20260908_preflight_r2",
        )
        args.append(arg)
    args.append("--online_spatial_snapshot_interval_frames=1000000")
    return Experiment(run.name, " ".join(args), [{}])


RUN_DESCRIPTION = RunDescription(
    "intrmotiv_persistent_intrinsic_control_20260908_preflight_r2",
    experiments=[preflight(run) for run in STUDY.expand_runs() if run.seed == 99],
)
