"""Validated six-run four-field DG controller study for NEMO2."""

from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description


SPEC = Path(__file__).resolve().parents[4] / "hpc_runs/studies/four_prescribed_dg_controller_20261007.study.json"
STUDY = load_study(SPEC)
RUN_DESCRIPTION = build_run_description(STUDY)
