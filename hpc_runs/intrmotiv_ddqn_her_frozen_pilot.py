from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description

STUDY = load_study(Path(__file__).with_name("studies") / "intrmotiv_ddqn_her_frozen_pilot.study.json")
RUN_DESCRIPTION = build_run_description(STUDY)
