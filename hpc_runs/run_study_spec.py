"""Sample Factory launcher adapter for a validated declarative StudySpec.

Set ``INTRMOTIV_STUDY_SPEC`` to the repository-relative StudySpec path before
invoking the launcher. The runtime jobs receive the expanded arguments from
the canonical study package and do not need this environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description


ROOT = Path(__file__).resolve().parent.parent
STUDY_PATH = os.environ.get("INTRMOTIV_STUDY_SPEC")
if not STUDY_PATH:
    raise RuntimeError("INTRMOTIV_STUDY_SPEC must name a validated StudySpec")
SPEC = ROOT / STUDY_PATH
if not SPEC.is_file() or not SPEC.resolve().is_relative_to(ROOT):
    raise RuntimeError(f"StudySpec must be a file inside the source repository: {STUDY_PATH}")

RUN_DESCRIPTION = build_run_description(load_study(SPEC))
