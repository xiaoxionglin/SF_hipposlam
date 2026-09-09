"""Numerical-stability preflight for the empirical PPO-HER retry."""

from sample_factory.launcher.run_description import RunDescription
from sf_working_directories.IntrMotiv.dmlab.experiments.hrl_goal_condition_her import Cell, make_experiment
from sf_working_directories.IntrMotiv.dmlab.experiments.hrl_goal_condition_her_retry import BATCH_NAME

RUN_DESCRIPTION = RunDescription(
    f"{BATCH_NAME}_preflight",
    experiments=[
        make_experiment(
            Cell("delayed", 16),
            8,
            train_for_env_steps=5_000_000,
            prefix="PF",
            group_suffix="preflight",
            batch_name=BATCH_NAME,
        ),
        make_experiment(
            Cell("immediate", 64),
            123,
            train_for_env_steps=5_000_000,
            prefix="PF",
            group_suffix="preflight",
            batch_name=BATCH_NAME,
        ),
    ],
)
