"""Short manager-exploration runs that exercise random and forced entry paths."""

from sample_factory.launcher.run_description import RunDescription
from sf_working_directories.IntrMotiv.dmlab.experiments.hrl_manager_exploration import (
    BATCH_NAME,
    CANDIDATES,
    ManagerSetting,
    make_experiment,
)

RUN_DESCRIPTION = RunDescription(
    f"{BATCH_NAME}_preflight",
    experiments=[
        make_experiment(
            CANDIDATES[0],
            ManagerSetting("FORCED", True, 0.0),
            8,
            train_for_env_steps=500_000,
            prefix="PF",
            group_suffix="preflight",
        ),
        make_experiment(
            CANDIDATES[1],
            ManagerSetting("P100", True, 1.0),
            99,
            train_for_env_steps=500_000,
            prefix="PF",
            group_suffix="preflight",
        ),
    ],
)
