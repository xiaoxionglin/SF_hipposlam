"""Replicated HRL manager-exploration study on the strongest current backgrounds."""

from __future__ import annotations

from dataclasses import dataclass

from sample_factory.launcher.run_description import Experiment, RunDescription
from sf_working_directories.IntrMotiv.dmlab.experiments.dg_structural_diversity import (
    COMMON_CLI as STRUCTURAL_COMMON_CLI,
)
from sf_working_directories.IntrMotiv.dmlab.experiments.dg_structural_diversity import PROJECT as STRUCTURAL_PROJECT

BATCH_NAME = "intrmotiv_hrl_manager_exploration_20260826"
PROJECT = "SF_IntrMotiv_HRLManagerExploration"
SEEDS = (8, 99, 123)


@dataclass(frozen=True)
class Candidate:
    tag: str
    global_coeff: float
    row_coeff: float
    temporal_coeff: float
    recruitment: bool


@dataclass(frozen=True)
class ManagerSetting:
    tag: str
    enabled: bool
    probability: float


CANDIDATES = (
    Candidate("CTRL_X0_O1", 0.0, 0.0, 0.0, True),
    Candidate("CTRL_X1_O1", 0.0, 0.0, 1.0, True),
)

MANAGER_SETTINGS = (
    ManagerSetting("DCTRL", False, 0.0),
    ManagerSetting("FORCED", True, 0.0),
    ManagerSetting("P010", True, 0.10),
    ManagerSetting("P025", True, 0.25),
)

COMMON_CLI = STRUCTURAL_COMMON_CLI.replace(f"--wandb_project={STRUCTURAL_PROJECT}", f"--wandb_project={PROJECT}")


def make_experiment(
    candidate: Candidate,
    manager: ManagerSetting,
    seed: int,
    *,
    train_for_env_steps: int = 100_000_000,
    prefix: str = "",
    group_suffix: str = "production",
) -> Experiment:
    name_prefix = f"{prefix}_" if prefix else ""
    name = f"{name_prefix}ME_GSD_{candidate.tag}_{manager.tag}_S{seed}"
    cli = (
        COMMON_CLI.replace("--train_for_env_steps=100000000", f"--train_for_env_steps={train_for_env_steps}")
        + f"--seed={seed} "
        + f"--dg_global_punishment_coeff={candidate.global_coeff} "
        + f"--dg_row_repulsion_coeff={candidate.row_coeff} "
        + f"--dg_ca3_temporal_exclusion_coeff={candidate.temporal_coeff} "
        + f"--dg_orthogonal_recruitment={candidate.recruitment} "
        + "--dg_orthogonal_recruitment_margin=0.5 "
        + "--dg_orthogonal_recruitment_residual_eps=1e-6 "
        + "--dg_orthogonal_recruitment_max_per_rollout=1 "
        + "--hrl_controllable_graph=True --hrl_graph_memory=policy_buffer "
        + "--hrl_persistent_fast_weights=False --hrl_fast_weight_half_life_options=10000 "
        + "--hrl_worker_reward_mode=hit_distance --hrl_target_hit_reward=1.0 "
        + "--hrl_distance_bonus_coeff=0.1 --hrl_timeout_margin_ratio=1.0 "
        + "--hrl_timeout_margin_steps=8 --hrl_bootstrap_horizon=96 --hrl_min_target_visits=1 "
        + "--hrl_edge_confidence_threshold=0.5 "
        + f"--hrl_exploration_mode={manager.enabled} "
        + f"--hrl_manager_exploration_probability={manager.probability} "
        + "--hrl_exploration_horizon=64 "
        + f"--wandb_group={BATCH_NAME}_{group_suffix}_{candidate.tag.lower()} "
        + f"--wandb_tags manager_exploration {candidate.tag.lower()} {manager.tag.lower()} "
        + "encourage batch_loss global_fixed simultaneous nopbt "
    )
    return Experiment(name, cli, [{}])


RUN_DESCRIPTION = RunDescription(
    BATCH_NAME,
    experiments=[
        make_experiment(candidate, manager, seed)
        for candidate in CANDIDATES
        for manager in MANAGER_SETTINGS
        for seed in SEEDS
    ],
)
