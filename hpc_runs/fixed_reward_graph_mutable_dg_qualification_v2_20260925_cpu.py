"""dg50 source-graph transfer with trainable DG qualification."""

from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description

STUDY = load_study(Path(__file__).with_name("studies") / "fixed_reward_graph_mutable_dg_qualification_v2_20260925.study.json")
RUN_DESCRIPTION = build_run_description(
    STUDY, run_filter=lambda run: run.metadata["site"] == "dg50",
    batch_name=STUDY.batch_name + "_cpu",
)
