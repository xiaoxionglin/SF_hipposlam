"""Resume only PPO cells after the terminal-telemetry compatibility fix.

The canonical nine-cell StudySpec still owns every argument and run identity.
Audit this submission together with the three continuing DDQN job records.
"""

from hpc_runs.corridor_geometry_preflight import STUDY
from hpc_runs.intrmotiv_study.sample_factory import build_run_description

RUN_DESCRIPTION = build_run_description(STUDY)
selected = {run.name for run in STUDY.expand_runs() if run.base in {"SAT", "DGP"}}
RUN_DESCRIPTION.experiments = [item for item in RUN_DESCRIPTION.experiments if item.base_name in selected]
