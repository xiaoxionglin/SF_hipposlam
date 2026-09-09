"""Representative preflight for the direct-HRL target-timing/HER batch."""

from sample_factory.launcher.run_description import RunDescription

from sf_working_directories.IntrMotiv.dmlab.experiments.hrl_goal_condition_her import (
    BATCH_NAME,
    PROJECT,
    Cell,
    make_experiment,
)


RUN_DESCRIPTION = RunDescription(
    f"{BATCH_NAME}_preflight",
    experiments=[
        make_experiment(Cell("delayed", 64), 8, train_for_env_steps=800_000, prefix="PFA", group_suffix="preflight"),
        make_experiment(Cell("immediate", 64), 8, train_for_env_steps=800_000, prefix="PFA", group_suffix="preflight"),
    ],
)
