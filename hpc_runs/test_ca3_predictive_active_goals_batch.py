import json
from pathlib import Path

import torch

from hpc_runs.audit_ca3_predictive_active_goals_preflight import _model_buffer
from hpc_runs.intrmotiv_study import load_study


STUDIES = Path(__file__).with_name("studies")


def _raw(name):
    return json.loads((STUDIES / name).read_text())


def test_release_qualification_is_seven_fresh_two_million_frame_runs():
    path = STUDIES / "ca3_predictive_active_goals_preflight.study.json"
    study = load_study(path)
    raw = _raw(path.name)

    assert len(study.expand_runs()) == 7
    assert raw["seeds"] == [99]
    assert raw["training"]["batch_name"].endswith("_preflight")
    assert "--train_for_env_steps=2000000" in raw["training"]["common_args"]
    assert "--checkpoint_frame_targets=2000000" in raw["training"]["common_args"]
    assert raw["telemetry"]["target_frames"] == [2_000_000]
    assert "intervention" not in raw["telemetry"]


def test_production_is_full_paired_matrix_with_declared_metrics_and_contrasts():
    path = STUDIES / "ca3_predictive_active_goals_production.study.json"
    study = load_study(path)
    raw = _raw(path.name)
    architectures = raw["factors"][0]["levels"]

    assert len(architectures) == 7
    assert raw["seeds"] == [8, 99, 123]
    assert len(study.expand_runs()) == 21
    assert raw["analysis"]["group_by"] == ["base", "architecture", "controller"]
    assert len(raw["analysis"]["contrasts"]) == 7
    assert raw["telemetry"]["target_frames"] == [5_000_000, 25_000_000, 75_000_000, 150_000_000, 300_000_000]
    assert raw["telemetry"]["terminal_seeds"] == [8, 99, 123]
    assert "readout_action_shuffle_delta" in raw["analysis"]["window_metrics"]
    assert "active_goal_count" in raw["analysis"]["cumulative_metrics"]
    assert all(f"command_slot_{slot:02d}" in raw["analysis"]["cumulative_metrics"] for slot in range(64))


def test_checkpoint_buffer_lookup_requires_one_exact_suffix():
    model = {"core.policy_graph.active_goal_mask": torch.tensor([False, True])}
    assert _model_buffer(model, "policy_graph.active_goal_mask").tolist() == [False, True]
