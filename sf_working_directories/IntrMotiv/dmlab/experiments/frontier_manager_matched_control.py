"""Causal UCB-manager control with matched topology and local exploration."""

from sample_factory.launcher.run_description import RunDescription

from sf_working_directories.IntrMotiv.dmlab.experiments.frontier_manager_isolation import (
    FULL_SEEDS,
    make_experiment,
)
from sf_working_directories.IntrMotiv.dmlab.experiments.topological_frontier_planning import Cell


BATCH_NAME = "intrmotiv_frontier_manager_matched_control_20260831"
PROJECT = "SF_IntrMotiv_FrontierManagerCausalControl"

# Both rows keep passive collection, RETURN/VALIDATE, and the same bounded
# local dense-reward exploration. Only the landmark-selection score differs.
CONDITIONS = (
    Cell(1, "TOPOLOGY_VISIT_DIRECT", True, manager="topology_visit_direct"),
    Cell(2, "TOPOLOGY_UCB_DIRECT", True, manager="frontier_direct"),
)


RUN_DESCRIPTION = RunDescription(
    BATCH_NAME,
    experiments=[
        make_experiment(
            condition,
            seed,
            batch_name=BATCH_NAME,
            project=PROJECT,
            batch_tag="frontier_manager_causal_control",
        )
        for condition in CONDITIONS
        for seed in FULL_SEEDS
    ],
)
