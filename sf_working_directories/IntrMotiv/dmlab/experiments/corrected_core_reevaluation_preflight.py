"""Four representative 2M-frame preflights for corrected-core re-evaluation."""

from sample_factory.launcher.run_description import RunDescription

from sf_working_directories.IntrMotiv.dmlab.experiments.corrected_core_reevaluation import (
    BATCH_NAME,
    CELLS,
    make_experiment,
)


PREFLIGHT_CELLS = tuple(cell for cell in CELLS if cell.number in (1, 3, 10, 16))

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
