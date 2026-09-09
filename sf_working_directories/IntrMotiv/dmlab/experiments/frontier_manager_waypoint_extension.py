"""Complete the paired 20-run frontier-manager and waypoint comparison."""

from sample_factory.launcher.run_description import RunDescription

from sf_working_directories.IntrMotiv.dmlab.experiments.frontier_manager_isolation import (
    BATCH_NAME,
    CONDITIONS,
    FULL_SEEDS,
    make_experiment,
)
from sf_working_directories.IntrMotiv.dmlab.experiments.topological_frontier_planning import Cell


# The initial submission already contains the first three seeds for the first
# two rows. This extension adds only the missing fourteen jobs.
EXTRA_SEEDS = (23, 57)
WAYPOINT_CONDITIONS = (
    Cell(3, "PASSIVE_UCB_FRONTIER_WAYPOINT", True, manager="frontier_waypoint"),
    Cell(4, "PASSIVE_UCB05_FRONTIER_WAYPOINT", True, manager="frontier_waypoint", uncertainty=0.5),
)


RUN_DESCRIPTION = RunDescription(
    f"{BATCH_NAME}_extension",
    experiments=[
        *(make_experiment(condition, seed) for condition in CONDITIONS for seed in EXTRA_SEEDS),
        *(make_experiment(condition, seed) for condition in WAYPOINT_CONDITIONS for seed in FULL_SEEDS),
    ],
)
