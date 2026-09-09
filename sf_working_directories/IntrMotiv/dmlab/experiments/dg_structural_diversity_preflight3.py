"""Corrected rollout-boundary and policy-graph recruitment preflights."""

from sample_factory.launcher.run_description import RunDescription
from sf_working_directories.IntrMotiv.dmlab.experiments.dg_structural_diversity import (
    BACKGROUNDS,
    BATCH_NAME,
    make_experiment,
)

BACKGROUND = BACKGROUNDS[1]
PREFLIGHT_STEPS = 2_000_000

RUN_DESCRIPTION = RunDescription(
    f"{BATCH_NAME}_preflight3",
    experiments=[
        make_experiment(
            architecture,
            BACKGROUND,
            temporal_coeff=1.0,
            recruitment=True,
            seed=8,
            train_for_env_steps=PREFLIGHT_STEPS,
            prefix="PF3",
            group_suffix="preflight3",
        )
        for architecture in ("flat", "global_fixed")
    ],
)
