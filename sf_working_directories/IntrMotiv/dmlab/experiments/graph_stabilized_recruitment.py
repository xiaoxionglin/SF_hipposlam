"""Matched 36-run sweep for graph-stabilized DG recruitment."""

from __future__ import annotations

from sample_factory.launcher.run_description import Experiment, RunDescription

from sf_working_directories.IntrMotiv.dmlab.experiments.corrected_core_reevaluation import (
    BATCH_NAME as CORRECTED_BATCH,
    CELLS,
    PROJECT as CORRECTED_PROJECT,
    make_experiment as make_corrected_experiment,
)


BATCH_NAME = "intrmotiv_graph_stabilized_recruitment_20260903"
PROJECT = "SF_IntrMotiv_GraphStabilizedRecruitment"
BACKBONE_NUMBERS = (5, 13, 15)
REDUNDANCY_THRESHOLDS = (4, 8)
HALF_LIVES = (5000, 10000)
SEEDS = (8, 99, 123)
BACKBONES = {cell.number: cell for cell in CELLS if cell.number in BACKBONE_NUMBERS}


def make_experiment(
    backbone_number: int,
    redundancy_max_steps: int,
    half_life: int,
    seed: int,
    *,
    train_for_env_steps: int = 100_000_000,
    prefix: str = "",
    group_suffix: str = "production",
) -> Experiment:
    cell = BACKBONES[backbone_number]
    base = make_corrected_experiment(
        cell,
        seed,
        train_for_env_steps=train_for_env_steps,
        group_suffix=group_suffix,
    )
    group = f"{BATCH_NAME}_{group_suffix}" if group_suffix else BATCH_NAME
    cli = (
        base.cmd.replace(f"--wandb_project={CORRECTED_PROJECT}", f"--wandb_project={PROJECT}")
        .replace(f"--wandb_group={CORRECTED_BATCH}_{group_suffix}", f"--wandb_group={group}")
        .replace("--dg_orthogonal_recruitment=False", "--dg_orthogonal_recruitment=True")
        .replace(
            "--hrl_fast_weight_half_life_options=10000",
            f"--hrl_fast_weight_half_life_options={half_life}",
        )
        + "--dg_orthogonal_recruitment_mode=graph "
        + "--dg_recruitment_connectivity_threshold=0.25 "
        + f"--dg_recruitment_redundancy_max_steps={redundancy_max_steps} "
        + f"--dg_recruitment_passive_half_life_events={half_life} "
    )
    prefix = f"{prefix}_" if prefix else ""
    name = f"{prefix}GSR_C{backbone_number:02d}_D{redundancy_max_steps}_H{half_life // 1000}K_S{seed}"
    return Experiment(name, cli, [{}])


RUN_DESCRIPTION = RunDescription(
    BATCH_NAME,
    experiments=[
        make_experiment(backbone, redundancy, half_life, seed)
        for backbone in BACKBONE_NUMBERS
        for redundancy in REDUNDANCY_THRESHOLDS
        for half_life in HALF_LIVES
        for seed in SEEDS
    ],
)
