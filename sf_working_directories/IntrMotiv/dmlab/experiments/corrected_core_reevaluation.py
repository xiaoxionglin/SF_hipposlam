"""Re-evaluate meaningful IntrMotiv designs after the 2026-09-01 core repairs."""

from __future__ import annotations

from dataclasses import dataclass

from sample_factory.launcher.run_description import Experiment, RunDescription

from sf_working_directories.IntrMotiv.dmlab.experiments.dg_structural_diversity import (
    COMMON_CLI as HISTORICAL_COMMON_CLI,
    PROJECT as HISTORICAL_PROJECT,
)


BATCH_NAME = "intrmotiv_corrected_core_reevaluation_20260901"
PROJECT = "SF_IntrMotiv_CorrectedCoreReevaluation"
SEEDS = (8, 99, 123)


@dataclass(frozen=True)
class Cell:
    number: int
    tag: str
    hrl: bool = True
    timing: str = "immediate"
    iterative: bool = False
    global_coeff: float = 0.0
    row_coeff: float = 0.0
    temporal_coeff: float = 0.0
    recruitment: bool = False
    her_horizon: int | None = None
    deadline: str = "short"
    manager: str = "visit_direct"
    exploration: bool = False
    exploration_probability: float = 0.0


CELLS = (
    Cell(1, "FLAT_CTRL", hrl=False, timing="delayed"),
    Cell(2, "DIRECT_DELAYED_CTRL", timing="delayed"),
    Cell(3, "DIRECT_IMMEDIATE_CTRL"),
    Cell(4, "DIRECT_IMMEDIATE_ITER", iterative=True),
    Cell(5, "DIRECT_IMMEDIATE_G001_R100", global_coeff=0.01, row_coeff=1.0),
    Cell(
        6,
        "DIRECT_IMMEDIATE_G001_R100_X1",
        global_coeff=0.01,
        row_coeff=1.0,
        temporal_coeff=1.0,
    ),
    Cell(7, "DIRECT_IMMEDIATE_O1", recruitment=True),
    Cell(8, "DIRECT_IMMEDIATE_X1_O1", temporal_coeff=1.0, recruitment=True),
    Cell(9, "DIRECT_IMMEDIATE_HER16", her_horizon=16),
    Cell(10, "DIRECT_IMMEDIATE_HER64", her_horizon=64),
    Cell(11, "DIRECT_DELAYED_HER64", timing="delayed", her_horizon=64),
    Cell(12, "DIRECT_DELAYED_X1_O1_LONG", timing="delayed", temporal_coeff=1.0, recruitment=True, deadline="long"),
    Cell(
        13,
        "DIRECT_DELAYED_X1_O1_P010",
        timing="delayed",
        temporal_coeff=1.0,
        recruitment=True,
        deadline="long",
        exploration=True,
        exploration_probability=0.10,
    ),
    Cell(14, "TOPOLOGY_VISIT_O1", timing="delayed", recruitment=True, deadline="long", manager="topology_visit_direct"),
    Cell(15, "TOPOLOGY_UCB_DIRECT_O1", timing="delayed", recruitment=True, deadline="long", manager="frontier_direct"),
    Cell(16, "TOPOLOGY_UCB_WAYPOINT_O1", timing="delayed", recruitment=True, deadline="long", manager="frontier_waypoint"),
)


COMMON_CLI = (
    HISTORICAL_COMMON_CLI.replace(
        f"--wandb_project={HISTORICAL_PROJECT}", f"--wandb_project={PROJECT}"
    )
    .replace(
        "--save_best_metric=distance_metric",
        "--save_best_metric=z_00_openfield_map2_fixed_loc3_fixedlength_noreward_coverage_auc",
    )
    .replace("--encoder_multi_activation_loss=False", "--encoder_multi_activation_loss=True")
    + "--encoder_batch_loss_temperature=0.5 "
)


def _deadline_flags(profile: str) -> str:
    if profile == "short":
        return "--hrl_timeout_margin_ratio=0.20 --hrl_timeout_margin_steps=2 --hrl_bootstrap_horizon=64 "
    if profile == "long":
        return "--hrl_timeout_margin_ratio=1.0 --hrl_timeout_margin_steps=8 --hrl_bootstrap_horizon=96 "
    raise ValueError(f"Unknown deadline profile: {profile}")


def make_experiment(
    cell: Cell,
    seed: int,
    *,
    train_for_env_steps: int = 100_000_000,
    prefix: str = "",
    group_suffix: str = "production",
) -> Experiment:
    prefix = f"{prefix}_" if prefix else ""
    name = f"{prefix}CCR_C{cell.number:02d}_{cell.tag}_S{seed}"
    group = f"{BATCH_NAME}_{group_suffix}" if group_suffix else BATCH_NAME
    cli = (
        COMMON_CLI.replace("--train_for_env_steps=100000000", f"--train_for_env_steps={train_for_env_steps}")
        .replace("--iterative_update=False", f"--iterative_update={cell.iterative}")
        + f"--seed={seed} "
        + f"--dg_global_punishment_coeff={cell.global_coeff} "
        + f"--dg_row_repulsion_coeff={cell.row_coeff} "
        + f"--dg_ca3_temporal_exclusion_coeff={cell.temporal_coeff} "
        + f"--dg_orthogonal_recruitment={cell.recruitment} "
        + "--dg_orthogonal_recruitment_margin=0.5 "
        + "--dg_orthogonal_recruitment_residual_eps=1e-6 "
        + "--dg_orthogonal_recruitment_max_per_rollout=1 "
        + "--iterative_initial_encoder_steps=128 "
        + "--iterative_decoder_steps=512 --iterative_encoder_steps=128 "
        + "--iterative_start_phase=decoder "
        + f"--wandb_group={group} "
        + f"--wandb_tags corrected_core c{cell.number:02d} {cell.tag.lower()} seed_{seed} "
        + "pretrained_layer2 encourage batch_loss multi_activation_penalty one_policy nopbt "
    )

    if not cell.hrl:
        return Experiment(name, cli + "--hrl_controllable_graph=False ", [{}])

    cli += (
        "--hrl_controllable_graph=True --hrl_graph_memory=policy_buffer "
        "--hrl_persistent_fast_weights=False --hrl_fast_weight_half_life_options=10000 "
        f"--hrl_target_timing={cell.timing} --hrl_manager_mode={cell.manager} "
        "--hrl_worker_reward_mode=hit_distance --hrl_target_hit_reward=1.0 "
        "--hrl_distance_bonus_coeff=0.1 --hrl_min_target_visits=1 "
        "--hrl_edge_confidence_threshold=0.5 "
        f"--hrl_exploration_mode={cell.exploration} "
        f"--hrl_manager_exploration_probability={cell.exploration_probability} "
        "--hrl_exploration_horizon=64 "
        f"--hrl_empirical_her={cell.her_horizon is not None} "
        f"--hrl_empirical_her_horizon={cell.her_horizon or 64} --hrl_empirical_her_coeff=0.5 "
        + _deadline_flags(cell.deadline)
    )

    if cell.manager != "visit_direct":
        cli += (
            "--hrl_passive_edge_confidence_threshold=2 "
            "--hrl_passive_min_displacement=2 --hrl_passive_max_path_length=64 "
            "--hrl_frontier_uncertainty_weight=1.0 "
            "--hrl_action_path_integration=False --hrl_motion_policy_input=False "
            "--dg_path_scatter_coeff=0.0 --hrl_landmark_geometry=none "
        )

    return Experiment(name, cli, [{}])


RUN_DESCRIPTION = RunDescription(
    BATCH_NAME,
    experiments=[make_experiment(cell, seed) for cell in CELLS for seed in SEEDS],
)
