"""CPU shard for the DG-capacity frozen-DG controls."""

from hpc_runs.cued_reward5_frozen_dg_controls import STUDY
from hpc_runs.intrmotiv_study.sample_factory import build_run_description

RUN_DESCRIPTION = build_run_description(
    STUDY,
    run_filter=lambda run: run.metadata["site"] == "dg50",
    batch_name=STUDY.batch_name + "_cpu",
)
