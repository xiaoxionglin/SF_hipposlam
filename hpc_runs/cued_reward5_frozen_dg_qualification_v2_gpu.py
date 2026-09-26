"""GPU shard for the calibrated frozen-DG qualification."""

from hpc_runs.cued_reward5_frozen_dg_qualification_v2 import STUDY
from hpc_runs.intrmotiv_study.sample_factory import build_run_description

RUN_DESCRIPTION = build_run_description(
    STUDY,
    run_filter=lambda run: run.metadata["site"] == "dg51",
    batch_name=STUDY.batch_name + "_gpu",
)
