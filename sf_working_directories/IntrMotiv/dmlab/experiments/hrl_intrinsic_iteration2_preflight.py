from sample_factory.launcher.run_description import Experiment, RunDescription

from sf_working_directories.IntrMotiv.dmlab.experiments.hrl_intrinsic_iteration2 import BASE_CLI


BATCH_NAME = "intrmotiv_hrl_iteration2_iterative_preflight_20260820"


def _replace(cli: str, old: str, new: str) -> str:
    if old not in cli:
        raise ValueError(f"Expected production option is missing: {old}")
    return cli.replace(old, new)


PREFLIGHT_CLI = BASE_CLI
for old, new in (
    ("--train_for_env_steps=25000000", "--train_for_env_steps=65536"),
    ("--num_workers=32", "--num_workers=8"),
    ("--num_envs_per_worker=2", "--num_envs_per_worker=2"),
    ("--worker_num_splits=2", "--worker_num_splits=1"),
    ("--rollout=64", "--rollout=32"),
    ("--recurrence=64", "--recurrence=32"),
    ("--batch_size=2048", "--batch_size=256"),
    ("--num_batches_per_epoch=2", "--num_batches_per_epoch=1"),
    ("--decorrelate_experience_max_seconds=120", "--decorrelate_experience_max_seconds=0"),
    ("--pbt_start_mutation=2500000", "--pbt_start_mutation=32768"),
    ("--pbt_period_env_steps=1250000", "--pbt_period_env_steps=8192"),
    ("--pbt_replace_reward_gap_absolute=1.0", "--pbt_replace_reward_gap_absolute=0.0"),
    ("--save_every_sec=300", "--save_every_sec=120"),
    ("--save_milestones_sec=1800", "--save_milestones_sec=0"),
    ("--save_best_every_sec=300", "--save_best_every_sec=120"),
    ("--save_best_after=1250000", "--save_best_after=0"),
    ("--with_wandb=True", "--with_wandb=False"),
):
    PREFLIGHT_CLI = _replace(PREFLIGHT_CLI, old, new)


RUN_DESCRIPTION = RunDescription(
    BATCH_NAME,
    experiments=[
        Experiment(
            "I2_PREFLIGHT_PBT4",
            PREFLIGHT_CLI
            + "--seed=20260819 "
            + "--encoder_reward_method=encourage "
            + "--hrl_worker_reward_mode=hit_distance ",
            [{}],
        )
    ],
)
