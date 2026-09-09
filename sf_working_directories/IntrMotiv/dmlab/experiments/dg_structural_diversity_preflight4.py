"""Workspace-routing smoke test for the final structural-diversity code."""

from sample_factory.launcher.run_description import RunDescription

from sf_working_directories.IntrMotiv.dmlab.experiments.dg_structural_diversity import (
    BACKGROUNDS,
    BATCH_NAME,
    make_experiment,
)


RUN_DESCRIPTION = RunDescription(
    f"{BATCH_NAME}_preflight4",
    experiments=[
        make_experiment(
            "global_fixed",
            BACKGROUNDS[1],
            temporal_coeff=1.0,
            recruitment=True,
            seed=8,
            train_for_env_steps=300_000,
            prefix="PF4",
            group_suffix="preflight4",
        )
    ],
)
