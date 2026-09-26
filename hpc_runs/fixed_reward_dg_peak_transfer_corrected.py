"""Corrected fixed-reward transfer StudySpec launcher adapter."""

from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description

SPEC = Path(__file__).with_name("studies") / "fixed_reward_dg_peak_transfer_corrected_20260925.study.json"
STUDY = load_study(SPEC)
RUN_DESCRIPTION = build_run_description(STUDY)
