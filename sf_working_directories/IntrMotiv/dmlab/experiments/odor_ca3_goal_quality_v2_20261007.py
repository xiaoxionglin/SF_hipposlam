"""StudySpec-driven NEMO2 launch adapter for the odor/CA3 40-cell study."""

from __future__ import annotations

import os
from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description


SPEC = Path(__file__).resolve().parents[4] / "hpc_runs/studies/odor_ca3_goal_quality_v2_20261007.study.json"
STUDY = load_study(SPEC)

# The four declared first-production cells cover both odor levels and the
# ALL/RANDOM/HEBB paths. Filtering consumes StudySpec rows; it adds no second
# run list or factor metadata source.
FIRST_NAMES = {
    "ODOR_OFF_ALL16_S8_V2",
    "ODOR_OFF_HEBB4_S8_V2",
    "ODOR_ON_RANDOM4_S8_V2",
    "ODOR_ON_HEBB8_S8_V2",
}
TRANCHE = os.environ.get("INTRMOTIV_ODOR_TRANCHE", "all")
if TRANCHE not in ("all", "first", "remaining"):
    raise ValueError("INTRMOTIV_ODOR_TRANCHE must be all, first, or remaining")


def selected(run) -> bool:
    if TRANCHE == "all":
        return True
    return (run.name in FIRST_NAMES) == (TRANCHE == "first")


RUN_DESCRIPTION = build_run_description(STUDY, run_filter=selected)
