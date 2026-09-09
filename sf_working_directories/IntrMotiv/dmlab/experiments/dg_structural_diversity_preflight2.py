"""High-scale exclusion and corrected recruitment runtime preflights."""

from sample_factory.launcher.run_description import RunDescription

from sf_working_directories.IntrMotiv.dmlab.experiments.dg_structural_diversity import (
    BACKGROUNDS,
    BATCH_NAME,
    make_experiment,
)


BACKGROUND = BACKGROUNDS[1]
PREFLIGHT_STEPS = 2_000_000

RUN_DESCRIPTION = RunDescription(
    f"{BATCH_NAME}_preflight2",
    experiments=[
        make_experiment(
            "global_fixed",
            BACKGROUND,
            temporal_coeff,
            recruitment,
            seed=8,
            train_for_env_steps=PREFLIGHT_STEPS,
            prefix="PF2",
            group_suffix="preflight2",
        )
        for temporal_coeff, recruitment in (
            (1.0, False),
            (3.0, False),
            (0.0, True),
            (1.0, True),
        )
    ],
)
