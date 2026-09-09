"""Representative multi-episode preflights for the topological frontier batch."""

from sample_factory.launcher.run_description import RunDescription

from sf_working_directories.IntrMotiv.dmlab.experiments.topological_frontier_planning import (
    BATCH_NAME,
    CELLS,
    make_experiment,
)


PREFLIGHT_CELLS = (CELLS[3], CELLS[7], CELLS[15])

RUN_DESCRIPTION = RunDescription(
    f"{BATCH_NAME}_preflight",
    experiments=[
        make_experiment(
            cell,
            seed,
            train_for_env_steps=2_000_000,
            prefix="PF4",
            group_suffix="preflight",
        )
        for cell, seed in zip(PREFLIGHT_CELLS, (8, 99, 123))
    ],
)
