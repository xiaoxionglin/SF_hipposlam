"""CPU StudySpec adapter for the direct-z/relation follow-up."""

from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description

ROOT = Path(__file__).resolve().parents[4]
STUDY = load_study(ROOT / "hpc_runs/studies/ca3_zrelation_followup_20260923_cpu.study.json")
RUN_DESCRIPTION = build_run_description(STUDY)
