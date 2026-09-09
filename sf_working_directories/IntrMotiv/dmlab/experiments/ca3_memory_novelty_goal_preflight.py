from pathlib import Path
from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description

SPEC = Path(__file__).resolve().parents[4] / "hpc_runs/studies/ca3_memory_novelty_goal_preflight.study.json"
RUN_DESCRIPTION = build_run_description(load_study(SPEC))
