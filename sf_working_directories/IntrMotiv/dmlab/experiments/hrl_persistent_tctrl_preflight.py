from sample_factory.launcher.run_description import Experiment, RunDescription
from sf_working_directories.IntrMotiv.dmlab.experiments.hrl_persistent_tctrl_iteration2 import BASE_CLI

BATCH_NAME = "intrmotiv_hrl_persistent_tctrl_preflight_v2_20260821"


def _replace(cli: str, old: str, new: str) -> str:
    if old not in cli:
        raise ValueError(f"Expected production option is missing: {old}")
    return cli.replace(old, new)


PREFLIGHT_CLI = BASE_CLI
for old, new in (
    ("--train_for_env_steps=100000000", "--train_for_env_steps=393216"),
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
):
    PREFLIGHT_CLI = _replace(PREFLIGHT_CLI, old, new)


def preflight(mode_tag: str, iterative: bool) -> Experiment:
    return Experiment(
        f"PHRL_PREFLIGHT_{mode_tag}",
        PREFLIGHT_CLI
        + "--seed=20260821 "
        + "--hrl_fast_weight_half_life_options=10000 "
        + "--encoder_reward_method=encourage "
        + f"--iterative_update={iterative} "
        + "--iterative_initial_encoder_steps=128 "
        + "--iterative_decoder_steps=512 "
        + "--iterative_encoder_steps=128 "
        + "--iterative_start_phase=decoder ",
        [{}],
    )


RUN_DESCRIPTION = RunDescription(
    BATCH_NAME,
    experiments=[preflight("sim", False), preflight("iter", True)],
)
