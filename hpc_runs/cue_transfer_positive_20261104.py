"""Resume the single declared positive-reward qualification cell in place."""

from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description

SPEC = Path(__file__).with_name("studies") / "cue_transfer_positive_20261104.study.json"
STUDY = load_study(SPEC)
RUN_DESCRIPTION = build_run_description(STUDY)
