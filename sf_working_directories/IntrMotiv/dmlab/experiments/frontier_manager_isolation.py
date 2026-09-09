"""Isolate passive-graph UCB frontier target selection from direct-target HRL."""

from sample_factory.launcher.run_description import Experiment, RunDescription

from sf_working_directories.IntrMotiv.dmlab.experiments.topological_frontier_planning import (
    COMMON_CLI,
    Cell,
)


BATCH_NAME = "intrmotiv_frontier_manager_isolation_20260831"
PROJECT = "SF_IntrMotiv_FrontierManagerIsolation"
SEEDS = (8, 99, 123)
# The initial launch used this three-seed subset. Extensions use the complete
# matched set without resubmitting these existing runs.
FULL_SEEDS = (8, 23, 57, 99, 123)

# This historical batch compares staged managers, not UCB selection alone:
# frontier modes also include passive validation and bounded local exploration.
# The topology-matched causal UCB control is defined separately in
# ``frontier_manager_matched_control.py``.
CONDITIONS = (
    Cell(1, "DIRECT_TARGET", True, manager="visit_direct"),
    Cell(2, "PASSIVE_UCB_FRONTIER", True, manager="frontier_direct"),
)


def make_experiment(
    condition: Cell,
    seed: int,
    *,
    batch_name: str = BATCH_NAME,
    project: str = PROJECT,
    batch_tag: str = "frontier_manager_isolation",
) -> Experiment:
    name = f"FMI_{condition.tag}_S{seed}"
    cli = (
        COMMON_CLI.replace(
            "--wandb_project=SF_IntrMotiv_TopologicalFrontierPlanning",
            f"--wandb_project={project}",
        )
        + f"--seed={seed} "
        + "--dg_global_punishment_coeff=0.0 --dg_row_repulsion_coeff=0.0 "
        + "--dg_ca3_temporal_exclusion_coeff=0.0 --dg_orthogonal_recruitment=True "
        + "--dg_orthogonal_recruitment_margin=0.5 "
        + "--dg_orthogonal_recruitment_residual_eps=1e-6 "
        + "--dg_orthogonal_recruitment_max_per_rollout=1 "
        + f"--hrl_manager_mode={condition.manager} "
        + "--hrl_passive_edge_confidence_threshold=2 --hrl_passive_min_displacement=2 "
        + f"--hrl_passive_max_path_length=64 --hrl_frontier_uncertainty_weight={condition.uncertainty} "
        + "--hrl_action_path_integration=False --hrl_motion_policy_input=False "
        + "--dg_path_scatter_coeff=0.0 --hrl_landmark_geometry=none "
        + f"--wandb_group={batch_name} "
        + f"--wandb_tags {batch_tag} {condition.tag.lower()} seed_{seed} "
        + "encourage batch_loss simultaneous nopbt "
        + "--hrl_controllable_graph=True --hrl_graph_memory=policy_buffer "
        + "--hrl_persistent_fast_weights=False --hrl_fast_weight_half_life_options=10000 "
        + "--hrl_worker_reward_mode=hit_distance --hrl_target_hit_reward=1.0 "
        + "--hrl_distance_bonus_coeff=0.1 --hrl_timeout_margin_ratio=1.0 "
        + "--hrl_timeout_margin_steps=8 --hrl_bootstrap_horizon=96 "
        + "--hrl_min_target_visits=1 --hrl_edge_confidence_threshold=0.5 "
        + "--hrl_exploration_mode=False --hrl_manager_exploration_probability=0 "
        + "--hrl_exploration_horizon=64 "
    )
    return Experiment(name, cli, [{}])


RUN_DESCRIPTION = RunDescription(
    BATCH_NAME,
    experiments=[make_experiment(condition, seed) for condition in CONDITIONS for seed in SEEDS],
)
