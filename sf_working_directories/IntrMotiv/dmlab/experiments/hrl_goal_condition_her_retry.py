"""Clean retry of HER cells invalidated by the singleton-advantage NaN bug."""

from sample_factory.launcher.run_description import RunDescription

from sf_working_directories.IntrMotiv.dmlab.experiments.hrl_goal_condition_her import Cell, SEEDS, make_experiment


BATCH_NAME = "intrmotiv_hrl_goal_condition_her_20260901_retry1"
FAILED_CELLS = tuple(Cell(timing, horizon) for timing in ("delayed", "immediate") for horizon in (16, 64))


RUN_DESCRIPTION = RunDescription(
    BATCH_NAME,
    experiments=[
        make_experiment(cell, seed, group_suffix="production", batch_name=BATCH_NAME)
        for cell in FAILED_CELLS
        for seed in SEEDS
    ],
)
