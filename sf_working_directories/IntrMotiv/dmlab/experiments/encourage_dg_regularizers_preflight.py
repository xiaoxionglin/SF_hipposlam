"""Short validation runs for the encourage DG-regularizer factorial."""

from sample_factory.launcher.run_description import RunDescription

from sf_working_directories.IntrMotiv.dmlab.experiments.encourage_dg_regularizers import (
    BATCH_NAME,
    LOSS_ARMS,
    PROJECT,
    make_experiment,
)


def arm(tag: str):
    return next(item for item in LOSS_ARMS if item.tag == tag)


PREFLIGHT_STEPS = 1_000_000
PREFLIGHTS = (
    ("flat", 2.43, arm("G003")),
    ("flat", 2.20, arm("R100")),
    ("global_fixed", 2.43, arm("G001_R100")),
    ("global_fixed", 2.20, arm("G001_R100")),
)


RUN_DESCRIPTION = RunDescription(
    f"{BATCH_NAME}_preflight",
    experiments=[
        make_experiment(
            architecture,
            threshold,
            loss_arm,
            seed=99,
            train_for_env_steps=PREFLIGHT_STEPS,
            prefix="PF",
        )
        for architecture, threshold, loss_arm in PREFLIGHTS
    ],
)
