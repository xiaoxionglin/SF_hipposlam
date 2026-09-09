"""Terminal action-path integration preflight after repeated-command correction."""

from sample_factory.launcher.run_description import Experiment, RunDescription
from sf_working_directories.IntrMotiv.dmlab.experiments.topological_frontier_planning import CELLS, make_experiment


def make_terminal_motion_preflight() -> Experiment:
    base = make_experiment(
        CELLS[4],  # TOPO_ACTION_GRAPH: action graph, no motion input or scatter loss.
        99,
        train_for_env_steps=2_000_000,
        prefix="PF6",
        group_suffix="terminal_motion_preflight",
    )
    # The fixedlength Lua level intentionally suppresses physical terminals.
    # This preflight instead needs the base no-reward level's 120-second reset
    # to verify terminal action-position alignment.
    command = base.cmd.replace(
        "--env=openfield_map2_fixed_loc3_fixedlength_noreward ",
        "--env=openfield_map2_fixed_loc3_noreward ",
    )
    return Experiment(base.base_name, command, [{}])


RUN_DESCRIPTION = RunDescription(
    "intrmotiv_topological_frontier_terminal_motion_preflight_20260831",
    experiments=[make_terminal_motion_preflight()],
)
