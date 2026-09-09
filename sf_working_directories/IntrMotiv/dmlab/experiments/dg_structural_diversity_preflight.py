"""Coefficient calibration and runtime preflights for DG structural diversity."""

from sample_factory.launcher.run_description import RunDescription

from sf_working_directories.IntrMotiv.dmlab.experiments.dg_structural_diversity import (
    BACKGROUNDS,
    BATCH_NAME,
    make_experiment,
)


BACKGROUND = BACKGROUNDS[1]
PREFLIGHT_STEPS = 2_000_000

RUN_DESCRIPTION = RunDescription(
    f"{BATCH_NAME}_preflight",
    experiments=[
        make_experiment(
            "global_fixed",
            BACKGROUND,
            temporal_coeff,
            recruitment,
            seed=8,
            train_for_env_steps=PREFLIGHT_STEPS,
            prefix="PF",
            group_suffix="preflight",
        )
        for temporal_coeff, recruitment in (
            (0.03, False),
            (0.10, False),
            (0.30, False),
            (0.00, True),
            (0.10, True),
        )
    ],
)
