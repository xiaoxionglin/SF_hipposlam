"""GPU resource shard of the corrected fixed-reward transfer study."""

from hpc_runs.fixed_reward_dg_peak_transfer_corrected import STUDY
from hpc_runs.intrmotiv_study.sample_factory import build_run_description

RUN_DESCRIPTION = build_run_description(
    STUDY,
    run_filter=lambda run: run.metadata["site"] == "dg51",
    batch_name=STUDY.batch_name + "_gpu",
)
