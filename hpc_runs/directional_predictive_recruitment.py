"""Canonical Sample Factory adapter for the 54-run recruitment study."""

from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description


SPEC = Path(__file__).with_name("studies") / "directional_predictive_recruitment.study.json"
STUDY = load_study(SPEC)
RUN_DESCRIPTION = build_run_description(STUDY)
