from sample_factory.launcher.run_description import Experiment, RunDescription
from sf_working_directories.IntrMotiv.dmlab.experiments.hrl_persistence_comparison import COMMON_CLI, _schedule_cli

BATCH_NAME = "intrmotiv_hrl_persistence_comparison_preflight_20260821"


def _preflight(cli: str) -> str:
    replacements = (
        ("--train_for_env_steps=100000000", "--train_for_env_steps=1048576"),
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
        ("--with_wandb=True", "--with_wandb=False"),
    )
    for old, new in replacements:
        if old not in cli:
            raise ValueError(f"Missing production setting: {old}")
        cli = cli.replace(old, new)
    return cli


def global_preflight() -> Experiment:
    cli = _preflight(COMMON_CLI) + (
        "--env=openfield_map2_fixed_loc3_fixedlength_noreward --save_best_metric=distance_metric "
        "--hrl_controllable_graph=True --hrl_graph_memory=policy_buffer "
        "--hrl_persistent_fast_weights=False --hrl_fast_weight_half_life_options=10000 "
        "--seed=20260821 --wandb_group=preflight_global_fixed "
        "--wandb_tags preflight hrl global_graph fixed_episode " + _schedule_cli(False)
    )
    return Experiment("GHRL_PREFLIGHT", cli, [{}])


def long_preflight() -> Experiment:
    cli = _preflight(COMMON_CLI) + (
        "--env=openfield_map2_fixed_loc3_longepisode_noreward "
        "--save_best_metric=distance_metric --exploration_window_steps=900 "
        "--hrl_controllable_graph=True --hrl_graph_memory=episode "
        "--hrl_persistent_fast_weights=False --hrl_fast_weight_half_life_options=10000 "
        "--seed=20260822 --wandb_group=preflight_stream_long "
        "--wandb_tags preflight hrl stream_graph long_episode " + _schedule_cli(False)
    )
    return Experiment("LHRL_PREFLIGHT", cli, [{}])


RUN_DESCRIPTION = RunDescription(BATCH_NAME, experiments=[global_preflight(), long_preflight()])
