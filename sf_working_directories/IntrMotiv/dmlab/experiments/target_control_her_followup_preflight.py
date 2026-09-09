"""Two HER-on preflights for the target-control follow-up."""

from sample_factory.launcher.run_description import RunDescription
from sf_working_directories.IntrMotiv.dmlab.experiments.target_control_her_followup import (
    BATCH_NAME,
    CELLS,
    make_experiment,
)

PREFLIGHT_CELLS = tuple(cell for cell in CELLS if cell.empirical_her)

RUN_DESCRIPTION = RunDescription(
    f"{BATCH_NAME}_preflight",
    experiments=[
        make_experiment(
            cell,
            99,
            train_for_env_steps=2_000_000,
            prefix="PF",
            group_suffix="preflight",
        )
        for cell in PREFLIGHT_CELLS
    ],
)
