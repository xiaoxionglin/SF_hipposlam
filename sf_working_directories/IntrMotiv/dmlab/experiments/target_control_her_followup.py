"""Matched C12/C13 empirical-HER follow-up for target-conditioned control."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sample_factory.launcher.run_description import Experiment, RunDescription
from sf_working_directories.IntrMotiv.dmlab.experiments.corrected_core_reevaluation import (
    BATCH_NAME as CORRECTED_CORE_BATCH_NAME,
)
from sf_working_directories.IntrMotiv.dmlab.experiments.corrected_core_reevaluation import CELLS as CORRECTED_CORE_CELLS
from sf_working_directories.IntrMotiv.dmlab.experiments.corrected_core_reevaluation import PROJECT
from sf_working_directories.IntrMotiv.dmlab.experiments.corrected_core_reevaluation import Cell as CorrectedCoreCell
from sf_working_directories.IntrMotiv.dmlab.experiments.corrected_core_reevaluation import (
    make_experiment as make_corrected_core_experiment,
)

BATCH_NAME = "intrmotiv_target_control_her_20260902"
SEEDS = (8, 99, 123)
HER_HORIZON = 64
HER_COEFFICIENT = 0.5


def _corrected_core_cell(number: int) -> CorrectedCoreCell:
    return next(cell for cell in CORRECTED_CORE_CELLS if cell.number == number)


@dataclass(frozen=True)
class Cell:
    backbone_number: int
    empirical_her: bool

    @property
    def backbone(self) -> CorrectedCoreCell:
        return _corrected_core_cell(self.backbone_number)

    @property
    def tag(self) -> str:
        return f"C{self.backbone_number:02d}_HER{'64' if self.empirical_her else 'OFF'}"


CELLS = (
    Cell(12, False),
    Cell(12, True),
    Cell(13, False),
    Cell(13, True),
)


def _replace_single_flag(command: str, flag: str, value: str) -> str:
    updated, count = re.subn(rf"{re.escape(flag)}=[^ ]+", f"{flag}={value}", command)
    if count != 1:
        raise ValueError(f"Expected exactly one {flag} setting, found {count}")
    return updated


def make_experiment(
    cell: Cell,
    seed: int,
    *,
    train_for_env_steps: int = 100_000_000,
    prefix: str = "",
    group_suffix: str = "production",
) -> Experiment:
    """Reuse C12/C13 exactly, changing only W&B grouping and HER enablement."""
    source = make_corrected_core_experiment(
        cell.backbone,
        seed,
        train_for_env_steps=train_for_env_steps,
        group_suffix=group_suffix,
    )
    group = f"{BATCH_NAME}_{group_suffix}" if group_suffix else BATCH_NAME
    source_group = f"{CORRECTED_CORE_BATCH_NAME}_{group_suffix}" if group_suffix else CORRECTED_CORE_BATCH_NAME
    command = source.cmd.replace(f"--wandb_group={source_group}", f"--wandb_group={group}")
    if command == source.cmd:
        raise ValueError("Could not replace corrected-core W&B group")
    command = _replace_single_flag(command, "--hrl_empirical_her", str(cell.empirical_her))
    command = _replace_single_flag(command, "--hrl_empirical_her_horizon", str(HER_HORIZON))
    command = _replace_single_flag(command, "--hrl_empirical_her_coeff", str(HER_COEFFICIENT))

    name_prefix = f"{prefix}_" if prefix else ""
    return Experiment(f"{name_prefix}TCH_{cell.tag}_S{seed}", command, [{}])


RUN_DESCRIPTION = RunDescription(
    BATCH_NAME,
    experiments=[make_experiment(cell, seed) for cell in CELLS for seed in SEEDS],
)
