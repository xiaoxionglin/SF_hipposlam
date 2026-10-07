"""Corrected short legacy gain-9 qualification for the episode-long study."""

from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description


SPEC = Path(__file__).resolve().parents[4] / "hpc_runs/studies/episode_long_dg_legacy9_qualification_20261008.study.json"
STUDY = load_study(SPEC)
RUN_DESCRIPTION = build_run_description(STUDY)
