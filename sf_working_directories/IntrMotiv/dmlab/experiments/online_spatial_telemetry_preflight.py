"""One-job Slurm smoke test for scalar W&B and workspace snapshot telemetry."""

from sample_factory.launcher.run_description import Experiment, RunDescription

from sf_working_directories.IntrMotiv.dmlab.experiments.flat_intrinsic_iterative_baseline import BASE_CLI


BATCH_NAME = "intrmotiv_online_spatial_graph_telemetry_preflight_20260904_r6"
PROJECT = "SF_IntrMotiv_OnlineSpatialTelemetry"


def _replace(cli: str, old: str, new: str) -> str:
    if old not in cli:
        raise ValueError(f"Expected production option is missing: {old}")
    return cli.replace(old, new)


PREFLIGHT_CLI = BASE_CLI
for old, new in (
    ("--train_for_env_steps=100000000", "--train_for_env_steps=131072"),
    ("--num_workers=32", "--num_workers=8"),
    ("--worker_num_splits=2", "--worker_num_splits=1"),
    ("--rollout=64", "--rollout=32"),
    ("--recurrence=64", "--recurrence=32"),
    ("--batch_size=2048", "--batch_size=256"),
    ("--num_batches_per_epoch=2", "--num_batches_per_epoch=1"),
    ("--decorrelate_experience_max_seconds=120", "--decorrelate_experience_max_seconds=0"),
    ("--save_every_sec=300", "--save_every_sec=120"),
    ("--save_milestones_sec=1800", "--save_milestones_sec=0"),
    ("--save_best_every_sec=300", "--save_best_every_sec=120"),
    ("--save_best_after=1200", "--save_best_after=0"),
    ("--exploration_coverage_telemetry=True", "--exploration_coverage_telemetry=False"),
    ("--hrl_controllable_graph=False", "--hrl_controllable_graph=True --hrl_graph_memory=policy_buffer"),
    ("--wandb_project=SF_IntrMotiv_FlatBaseline_Iterative", f"--wandb_project={PROJECT}"),
    ("--wandb_group=intrmotiv_flat_iterative_baseline_nopbt_20260820", f"--wandb_group={BATCH_NAME}"),
):
    PREFLIGHT_CLI = _replace(PREFLIGHT_CLI, old, new)

PREFLIGHT_CLI += (
    "--online_spatial_telemetry=True "
    "--online_spatial_window_observations=4096 "
    "--online_spatial_scalar_window_observations=1024 "
    "--online_spatial_scalar_interval_frames=32768 "
    "--online_spatial_snapshot_interval_frames=65536 "
    "--online_spatial_snapshot_max_frames=65536 "
    "--online_spatial_snapshot_targets=65536 "
)

RUN_DESCRIPTION = RunDescription(
    BATCH_NAME,
    experiments=[Experiment(
        "ONLINE_SPATIAL_GRAPH_PREFLIGHT_R6_S20260904",
        PREFLIGHT_CLI
        + "--seed=20260904 "
        + "--encoder_reward_method=punish "
        + "--iterative_update=False ",
        [{}],
    )],
)
