"""GPU shard of flat-mixture jitter qualification."""

from hpc_runs.fixed_reward_flat_mixture_jitter_qualification import STUDY
from hpc_runs.intrmotiv_study.sample_factory import build_run_description

RUN_DESCRIPTION = build_run_description(
    STUDY,
    run_filter=lambda run: run.metadata["site"] == "dg51",
    batch_name=STUDY.batch_name + "_gpu",
)
