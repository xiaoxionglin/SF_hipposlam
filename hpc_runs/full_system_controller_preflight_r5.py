from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description

SPEC = Path(__file__).with_name("studies") / "full_system_controller_preflight_r5.study.json"
STUDY = load_study(SPEC)
RUN_DESCRIPTION = build_run_description(STUDY)
