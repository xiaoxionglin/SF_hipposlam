"""Topological frontier, waypoint planning, action integration, and SE(2) ablations."""

from __future__ import annotations

from dataclasses import dataclass

from sample_factory.launcher.run_description import Experiment, RunDescription
from sf_working_directories.IntrMotiv.dmlab.experiments.dg_structural_diversity import (
    COMMON_CLI as STRUCTURAL_COMMON_CLI,
)
from sf_working_directories.IntrMotiv.dmlab.experiments.dg_structural_diversity import PROJECT as STRUCTURAL_PROJECT

BATCH_NAME = "intrmotiv_topological_frontier_planning_20260831"
PROJECT = "SF_IntrMotiv_TopologicalFrontierPlanning"
SEEDS = (8, 99, 123)


@dataclass(frozen=True)
class Cell:
    number: int
    tag: str
    hrl: bool
    manager: str = "visit_direct"
    action_integration: bool = False
    motion_policy: bool = False
    scatter: float = 0.0
    scatter_distance: float = 8.0
    uncertainty: float = 1.0
    global_coeff: float = 0.0
    row_coeff: float = 0.0
    temporal_coeff: float = 0.0
    recruitment: bool = True
    geometry: str = "none"


CELLS = (
    Cell(1, "FLAT_G001_R100_X0_O0", False, global_coeff=0.01, row_coeff=1.0, recruitment=False),
    Cell(2, "DIRECT_CTRL_X0_O1", True),
    Cell(3, "FRONTIER_DIRECT", True, manager="frontier_direct"),
    Cell(4, "FRONTIER_WAYPOINT_TIME", True, manager="frontier_waypoint"),
    Cell(5, "TOPO_ACTION_GRAPH", True, "frontier_waypoint", True),
    Cell(6, "TOPO_ACTION_MOTION", True, "frontier_waypoint", True, True),
    Cell(7, "TOPO_GRAPH_SC001", True, "frontier_waypoint", True, False, 0.01),
    Cell(8, "TOPO_MOTION_SC001", True, "frontier_waypoint", True, True, 0.01),
    Cell(9, "TOPO_MOTION_SC0005", True, "frontier_waypoint", True, True, 0.005),
    Cell(10, "TOPO_MOTION_SC005", True, "frontier_waypoint", True, True, 0.05),
    Cell(11, "TOPO_MOTION_SC001_D4", True, "frontier_waypoint", True, True, 0.01, 4.0),
    Cell(12, "TOPO_MOTION_SC001_D12", True, "frontier_waypoint", True, True, 0.01, 12.0),
    Cell(13, "TOPO_MOTION_SC001_U05", True, "frontier_waypoint", True, True, 0.01, 8.0, 0.5),
    Cell(14, "TOPO_MOTION_SC001_U2", True, "frontier_waypoint", True, True, 0.01, 8.0, 2.0),
    Cell(
        15,
        "TOPO_G001_R100_X1_O0",
        True,
        "frontier_waypoint",
        True,
        True,
        0.01,
        global_coeff=0.01,
        row_coeff=1.0,
        temporal_coeff=1.0,
        recruitment=False,
    ),
    Cell(16, "TOPO_MOTION_SC001_SE2", True, "frontier_waypoint", True, True, 0.01, geometry="se2"),
)


COMMON_CLI = STRUCTURAL_COMMON_CLI.replace(f"--wandb_project={STRUCTURAL_PROJECT}", f"--wandb_project={PROJECT}")


def make_experiment(
    cell: Cell,
    seed: int,
    *,
    train_for_env_steps: int = 100_000_000,
    prefix: str = "",
    group_suffix: str = "",
) -> Experiment:
    name_prefix = f"{prefix}_" if prefix else ""
    name = f"{name_prefix}TFP_C{cell.number:02d}_{cell.tag}_S{seed}"
    group = BATCH_NAME + (f"_{group_suffix}" if group_suffix else "")
    cli = (
        COMMON_CLI.replace("--train_for_env_steps=100000000", f"--train_for_env_steps={train_for_env_steps}")
        + f"--seed={seed} "
        + f"--dg_global_punishment_coeff={cell.global_coeff} "
        + f"--dg_row_repulsion_coeff={cell.row_coeff} "
        + f"--dg_ca3_temporal_exclusion_coeff={cell.temporal_coeff} "
        + f"--dg_orthogonal_recruitment={cell.recruitment} "
        + "--dg_orthogonal_recruitment_margin=0.5 "
        + "--dg_orthogonal_recruitment_residual_eps=1e-6 "
        + "--dg_orthogonal_recruitment_max_per_rollout=1 "
        + f"--hrl_manager_mode={cell.manager} "
        + "--hrl_passive_edge_confidence_threshold=2 "
        + "--hrl_passive_min_displacement=2 "
        + "--hrl_passive_max_path_length=64 "
        + f"--hrl_frontier_uncertainty_weight={cell.uncertainty} "
        + f"--hrl_action_path_integration={cell.action_integration} "
        + f"--hrl_motion_policy_input={cell.motion_policy} "
        + f"--dg_path_scatter_coeff={cell.scatter} "
        + f"--dg_path_scatter_min_displacement={cell.scatter_distance} "
        + "--dg_path_scatter_min_straightness=0.8 "
        + "--dg_path_scatter_temperature=0.5 "
        + f"--hrl_landmark_geometry={cell.geometry} "
        + "--hrl_geometry_neighbors=3 --hrl_geometry_max_distance=32 "
        + "--hrl_geometry_update_steps=5 --hrl_geometry_learning_rate=0.05 "
        + f"--wandb_group={group} "
        + f"--wandb_tags topological_frontier c{cell.number:02d} {cell.tag.lower()} seed_{seed} "
        + "encourage batch_loss simultaneous nopbt "
    )
    if cell.hrl:
        cli += (
            "--hrl_controllable_graph=True --hrl_graph_memory=policy_buffer "
            "--hrl_persistent_fast_weights=False --hrl_fast_weight_half_life_options=10000 "
            "--hrl_worker_reward_mode=hit_distance --hrl_target_hit_reward=1.0 "
            "--hrl_distance_bonus_coeff=0.1 --hrl_timeout_margin_ratio=1.0 "
            "--hrl_timeout_margin_steps=8 --hrl_bootstrap_horizon=96 "
            "--hrl_min_target_visits=1 --hrl_edge_confidence_threshold=0.5 "
            "--hrl_exploration_mode=False --hrl_manager_exploration_probability=0 "
            "--hrl_exploration_horizon=64 "
        )
    else:
        cli += "--hrl_controllable_graph=False "
    return Experiment(name, cli, [{}])


RUN_DESCRIPTION = RunDescription(
    BATCH_NAME,
    experiments=[make_experiment(cell, seed) for cell in CELLS for seed in SEEDS],
)
