"""Direct-HRL target-timing and empirical PPO-HER comparison."""

from __future__ import annotations

from dataclasses import dataclass

from sample_factory.launcher.run_description import Experiment, RunDescription
from sf_working_directories.IntrMotiv.dmlab.experiments.dg_structural_diversity import COMMON_CLI

BATCH_NAME = "intrmotiv_hrl_goal_condition_her_20260901"
PROJECT = "SF_IntrMotiv_HRLGoalConditionHER"
SEEDS = (8, 99, 123)


@dataclass(frozen=True)
class Cell:
    timing: str
    her_horizon: int | None

    @property
    def tag(self) -> str:
        return f"{self.timing.upper()}_HER{self.her_horizon}" if self.her_horizon else f"{self.timing.upper()}_HER_OFF"


CELLS = tuple(Cell(timing, horizon) for timing in ("delayed", "immediate") for horizon in (None, 16, 64))


def make_experiment(
    cell: Cell,
    seed: int,
    *,
    train_for_env_steps: int = 100_000_000,
    prefix: str = "",
    group_suffix: str = "production",
    batch_name: str = BATCH_NAME,
) -> Experiment:
    name_prefix = f"{prefix}_" if prefix else ""
    name = f"{name_prefix}GCH_{cell.tag}_S{seed}"
    cli = (
        COMMON_CLI.replace("--train_for_env_steps=100000000", f"--train_for_env_steps={train_for_env_steps}").replace(
            "--wandb_project=SF_IntrMotiv_DGStructuralDiversity", f"--wandb_project={PROJECT}"
        )
        + f"--seed={seed} "
        + "--dg_global_punishment_coeff=0 --dg_row_repulsion_coeff=0 "
        + "--dg_ca3_temporal_exclusion_coeff=0 --dg_orthogonal_recruitment=False "
        + "--hrl_controllable_graph=True --hrl_graph_memory=policy_buffer "
        + f"--hrl_target_timing={cell.timing} "
        + "--hrl_manager_mode=visit_direct --hrl_persistent_fast_weights=False "
        + "--hrl_fast_weight_half_life_options=10000 --hrl_worker_reward_mode=hit_distance "
        + "--hrl_target_hit_reward=1 --hrl_distance_bonus_coeff=0.1 "
        + "--hrl_timeout_margin_ratio=1 --hrl_timeout_margin_steps=8 --hrl_bootstrap_horizon=96 "
        + "--hrl_min_target_visits=1 --hrl_edge_confidence_threshold=0.5 "
        + "--hrl_exploration_mode=False --hrl_manager_exploration_probability=0 "
        + "--hrl_exploration_horizon=64 "
        + f"--hrl_empirical_her={cell.her_horizon is not None} "
        + f"--hrl_empirical_her_horizon={cell.her_horizon or 64} --hrl_empirical_her_coeff=0.5 "
        + f"--wandb_group={batch_name}_{group_suffix} "
        + f"--wandb_tags hrl goal_timing {cell.timing} empirical_ppo_her her_{cell.her_horizon or 'off'} "
        + f"seed_{seed} encourage batch_loss simultaneous nopbt "
    )
    return Experiment(name, cli, [{}])


RUN_DESCRIPTION = RunDescription(
    BATCH_NAME,
    experiments=[make_experiment(cell, seed) for cell in CELLS for seed in SEEDS],
)
