import json
from pathlib import Path

from hpc_runs.intrmotiv_study import load_study

STUDIES = Path(__file__).with_name("studies")


def _raw(name):
    return json.loads((STUDIES / name).read_text())


def test_followup_preflight_is_four_fresh_seed99_two_million_frame_runs():
    name = "ca3_state_goal_followup_20260922_preflight.study.json"
    study = load_study(STUDIES / name)
    raw = _raw(name)
    assert len(study.expand_runs()) == 4
    assert raw["seeds"] == [99]
    assert raw["workflow_version"] == "1.11.0"
    assert raw["metadata"]["project"] == "SF_IntrMotiv_CA3StateGoals"
    assert "--train_for_env_steps=2000000" in raw["training"]["common_args"]
    assert raw["telemetry"]["target_frames"] == [2_000_000]
    assert "intervention" not in raw["telemetry"]


def test_followup_production_is_complete_factorial_with_flat_tracking():
    name = "ca3_state_goal_followup_20260922_production.study.json"
    study = load_study(STUDIES / name)
    raw = _raw(name)
    runs = study.expand_runs()
    assert len(runs) == 12
    assert raw["seeds"] == [8, 99, 123]
    assert raw["analysis"]["group_by"] == ["base", "anchor_update", "candidate_rule", "controller"]
    assert len(raw["analysis"]["contrasts"]) == 5
    assert raw["telemetry"]["target_frames"] == [5_000_000, 25_000_000, 75_000_000, 150_000_000, 300_000_000]
    assert raw["telemetry"]["intervention"]["target_frames"] == [75_000_000, 300_000_000]
    for run in runs:
        assert f"--study_condition={run.condition}" in run.args
        assert f"--wandb_tags={run.condition}" in run.args
        assert "--wandb_project=SF_IntrMotiv_CA3StateGoals" in run.args
        assert "--ca3_state_readout_horizon=32" in run.args
        assert "--ca3_state_readout_var_coeff=0.1" in run.args
        assert "--ca3_state_readout_cov_coeff=0.01" in run.args
