"""CPU submission shard of the one fixed-reward transfer StudySpec."""

from hpc_runs.fixed_reward_dg_peak_transfer import STUDY
from hpc_runs.intrmotiv_study.sample_factory import build_run_description

RUN_DESCRIPTION = build_run_description(
    STUDY,
    run_filter=lambda run: run.metadata["site"] == "dg50",
    batch_name=STUDY.batch_name + "_cpu",
)
