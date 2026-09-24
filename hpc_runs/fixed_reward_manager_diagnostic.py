"""Sample Factory adapter for reward-manager state attribution diagnosis."""
from pathlib import Path
from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description
SPEC = Path(__file__).with_name("studies") / "fixed_reward_manager_diagnostic_20260924.study.json"
STUDY = load_study(SPEC)
RUN_DESCRIPTION = build_run_description(STUDY)
