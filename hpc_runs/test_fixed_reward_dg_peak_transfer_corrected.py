"""Checks the corrected reward and transfer factors before submission."""

from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description

SPEC = Path(__file__).with_name("studies") / "fixed_reward_dg_peak_transfer_corrected_20260925.study.json"


def test_corrected_matrix_and_uninformed_flat_start():
    study = load_study(SPEC)
    runs = study.expand_runs()
    assert len(runs) == 48
    assert len({run.name for run in runs}) == 48
    assert {run.seed for run in runs} == {42, 1234, 9999}
    graph_arms = {"W_GRAPH", "W_WORKER", "W_FULL"}
    for run in runs:
        arm = run.factors["arm"]
        args = set(run.args)
        assert "--advantage_reward_source=external" in args
        assert "--fixed_task_goal_init=uniform_jitter" in args
        assert "--train_for_env_steps=100000000" in args
        assert "--with_pos_obs=False" in args
        assert run.metadata["arm_graph_init"] == ("source" if arm in graph_arms else "empty")
        if arm.startswith("F_"):
            assert "--fixed_task_goal_mixture=true" in args
            assert run.metadata["arm_goal_mixture_init"] == "uniform_jitter"
            assert not any(arg.startswith("--fixed_task_target_id=") for arg in args)
        assert ("--transfer_graph=true" in args) == (arm in graph_arms)


def test_incremental_transfer_contrasts_are_preregistered():
    study = load_study(SPEC)
    names = {contrast["name"] for contrast in study.analysis["contrasts"]}
    assert {
        "W_GRAPH_minus_W_FIXED",
        "W_WORKER_minus_W_GRAPH",
        "W_FULL_minus_W_WORKER",
        "F_DG_minus_F_SCRATCH",
        "F_FULL_minus_F_DG",
    } <= names


def test_corrected_resource_shards_partition_study():
    study = load_study(SPEC)
    cpu = build_run_description(study, run_filter=lambda run: run.metadata["site"] == "dg50", batch_name="cpu")
    gpu = build_run_description(study, run_filter=lambda run: run.metadata["site"] == "dg51", batch_name="gpu")
    assert len(cpu.experiments) == len(gpu.experiments) == 24
