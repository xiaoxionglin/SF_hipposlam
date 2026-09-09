from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from hpc_runs.intrmotiv_study.spatial_contract import SpatialContractError, load_spatial_snapshot
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.shared_buffers import policy_output_shapes
from sample_factory.algo.utils.tensor_dict import TensorDict
from sf_working_directories.IntrMotiv.dmlab.custom_actor_critic import IntrMotivActorCriticSharedWeights
from sf_working_directories.IntrMotiv.dmlab.custom_params import add_hipposlam_env_args
from sf_working_directories.IntrMotiv.dmlab.custom_learner import DistanceLearnerReward
from sf_working_directories.IntrMotiv.dmlab.online_spatial_telemetry import TrainingSpatialTelemetry
from sf_working_directories.IntrMotiv.dmlab.hrl_controllable_graph import PolicyControllableGraph
from sf_working_directories.IntrMotiv.dmlab.dg_recruitment_graph import PassiveRecruitmentGraph
from sf_working_directories.IntrMotiv.dmlab.reward_summaries import INTRMOTIV_SUMMARY_TAGS


def test_online_spatial_default_windows_are_100k_snapshot_and_10k_scalar():
    import argparse

    parser = argparse.ArgumentParser()
    add_hipposlam_env_args(parser)
    cfg = parser.parse_args([])
    assert cfg.online_spatial_window_observations == 100_000
    assert cfg.online_spatial_scalar_window_observations == 10_000


class _PrivacyModel(nn.Module):
    privileged_obs_keys = ("telemetry_pose",)

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(3, 2) / 10)
        self.seen_keys = None

    def device_for_input_tensor(self, _key):
        return torch.device("cpu")

    def type_for_input_tensor(self, _key):
        return torch.float32

    def normalize_obs(self, obs):
        self.seen_keys = tuple(sorted(obs))
        assert "telemetry_pose" not in obs
        return obs

    def outputs(self, obs):
        encoded = obs["obs"] @ self.weight
        dg = torch.relu(encoded)
        logits = dg @ torch.tensor(((1.0, -1.0), (0.5, 0.25)))
        reward = dg.sum(dim=-1)
        return encoded, dg, logits, reward


def _privacy_pass(pose):
    model = _PrivacyModel()
    normalized = prepare_and_normalize_obs(model, {
        "obs": torch.tensor(((1.0, 2.0, 3.0),)),
        "telemetry_pose": torch.tensor((pose,), dtype=torch.float32),
    })
    outputs = model.outputs(normalized)
    sum(value.sum() for value in outputs).backward()
    return model.seen_keys, tuple(value.detach().clone() for value in outputs), model.weight.grad.detach().clone()


def test_pose_changes_cannot_change_encoder_dg_logits_reward_or_gradients():
    first = _privacy_pass((100.0, 200.0, 10.0))
    second = _privacy_pass((1900.0, 1800.0, 350.0))
    assert first[0] == second[0] == ("obs",)
    for left, right in zip(first[1], second[1]):
        torch.testing.assert_close(left, right)
    torch.testing.assert_close(first[2], second[2])


def test_privileged_filter_preserves_tensordict_bootstrap_slicing():
    model = _PrivacyModel()
    normalized = prepare_and_normalize_obs(model, TensorDict({
        "obs": torch.ones((2, 3, 3), dtype=torch.float32),
        "telemetry_pose": torch.zeros((2, 3, 3), dtype=torch.float32),
    }))

    assert isinstance(normalized, TensorDict)
    assert tuple(normalized[:, -1]["obs"].shape) == (2, 3)
    assert "telemetry_pose" not in normalized


class _TinyEncoder(nn.Module):
    def forward(self, obs):
        return obs["obs"]


class _TinyOnlineActor(IntrMotivActorCriticSharedWeights):
    def __init__(self):
        nn.Module.__init__(self)
        self.cfg = SimpleNamespace(online_spatial_telemetry=True, Hippo_n_feature=2)
        self.encoder = _TinyEncoder()

    def forward_core(self, head_output, rnn_states):
        return head_output, rnn_states

    def forward_tail(self, core_output, values_only, sample_actions, action_mask=None):
        result = TensorDict(values=core_output.sum(dim=-1))
        if not values_only:
            result["action_logits"] = core_output[:, :2]
            result["actions"] = torch.zeros(core_output.shape[0])
            result["log_prob_actions"] = torch.zeros(core_output.shape[0])
        return result


def test_actor_exports_exact_behavior_time_thresholded_dg_activity():
    actor = _TinyOnlineActor()
    head = torch.tensor(((0.0, 1.0, 7.0), (1.0, 0.0, 8.0)), requires_grad=True)
    result = actor({"obs": head}, torch.zeros(2, 1))
    torch.testing.assert_close(result["dg_activity"], head[:, :2])
    assert not result["dg_activity"].requires_grad
    assert "dg_activity" not in actor({"obs": head}, torch.zeros(2, 1), values_only=True)


def test_policy_output_contract_adds_dg_only_when_enabled():
    enabled = SimpleNamespace(double_value=False, online_spatial_telemetry=True, Hippo_n_feature=16)
    disabled = SimpleNamespace(double_value=False, online_spatial_telemetry=False, Hippo_n_feature=16)
    assert ("dg_activity", [16]) in policy_output_shapes(enabled, 1, 3)
    assert all(name != "dg_activity" for name, _ in policy_output_shapes(disabled, 1, 3))


def _cfg(root: Path, experiment: str = "batch/run"):
    return SimpleNamespace(
        online_spatial_window_observations=4,
        online_spatial_scalar_interval_frames=1_000_000,
        online_spatial_snapshot_interval_frames=25_000_000,
        online_spatial_snapshot_max_frames=100_000_000,
        online_spatial_grid_grain=19,
        online_spatial_stationary_distance=1.0,
        online_spatial_x_min=100.0,
        online_spatial_x_max=2000.0,
        online_spatial_y_min=100.0,
        online_spatial_y_max=2000.0,
        online_spatial_workspace_root=str(root),
        online_spatial_output_root=str(root / "analysis" / "online_spatial"),
        train_dir=str(root / "train_dir"),
        experiment=experiment,
        env="dmlab_test",
        online_spatial_telemetry=True,
        max_policy_lag=100,
    )


def _fill_window(telemetry):
    telemetry.window.append_rollouts(
        np.asarray((((100, 100, 359), (200, 100, 1), (300, 100, 2), (400, 100, 3)),), dtype=np.float32),
        np.asarray((((1, 0), (1, 0), (0, 1), (0, 1)),), dtype=np.float32),
        np.asarray(((0, 1, 2, 3),)),
        np.asarray(((False, False, False, True),)),
        np.asarray(((1, 1, 2, 2),)),
        np.ones((1, 4), dtype=bool),
    )


def test_batch_capture_aligns_pose_dg_actions_and_policy_lag_exactly(tmp_path):
    telemetry = TrainingSpatialTelemetry(_cfg(tmp_path), SimpleNamespace(frameskip=4), 0, 0)
    buff = {
        "obs": {"telemetry_pose": torch.tensor((
            ((100.0, 101.0, 1.0), (200.0, 201.0, 2.0), (300.0, 301.0, 3.0), (999.0, 999.0, 9.0)),
        ))},
        "dg_activity": torch.tensor((((1.0, 0.0), (0.0, 2.0), (3.0, 0.0)),)),
        "actions": torch.tensor(((4, 5, 6),)),
        "dones": torch.tensor(((False, False, True),)),
        "policy_version": torch.tensor(((10.0, 4.0, 10.0),)),
    }
    valid = np.asarray(((True, False, True),))
    assert telemetry.append_batch(buff, valid) == 2
    arrays = telemetry.window.arrays()
    assert arrays["actions"].tolist() == [4, 6]
    assert arrays["pose"].tolist() == [[100.0, 101.0, 1.0], [300.0, 301.0, 3.0]]
    assert arrays["dg_activity"].tolist() == [[1.0, 0.0], [3.0, 0.0]]
    assert arrays["policy_version"].tolist() == [10, 10]
    assert arrays["segment_id"][0] != arrays["segment_id"][1]


def test_training_window_splits_unmarked_large_pose_relocation(tmp_path):
    telemetry = TrainingSpatialTelemetry(_cfg(tmp_path), SimpleNamespace(frameskip=4), 0, 0)
    telemetry.window.append_rollouts(
        np.asarray((((100, 100, 0), (140, 100, 0), (1500, 1600, 0), (1540, 1600, 0)),), dtype=np.float32),
        np.ones((1, 4, 2), dtype=np.float32),
        np.zeros((1, 4), dtype=np.int16),
        np.zeros((1, 4), dtype=bool),
        np.zeros((1, 4), dtype=np.int64),
        np.ones((1, 4), dtype=bool),
        telemetry.max_segment_jump_distance,
    )
    segments = telemetry.window.arrays()["segment_id"]
    assert segments[0] == segments[1]
    assert segments[1] != segments[2]
    assert segments[2] == segments[3]


def test_lazy_capture_survives_distance_learner_constructor_bypass(tmp_path):
    """Regression test for specialized learners that call BaseLearner directly."""
    learner = DistanceLearnerReward.__new__(DistanceLearnerReward)
    learner.cfg = _cfg(tmp_path, "00_EXACT_STUDY_RUN")
    learner.env_info = SimpleNamespace(frameskip=4)
    learner.policy_id = 0
    learner.env_steps = 0
    learner.train_step = 10
    buff = TensorDict({
        "obs": TensorDict({"telemetry_pose": torch.tensor((
            ((100.0, 101.0, 1.0), (200.0, 201.0, 2.0), (300.0, 301.0, 3.0)),
        ))}),
        "dg_activity": torch.tensor((((1.0, 0.0), (0.0, 2.0)),)),
        "actions": torch.tensor(((4, 5),)),
        "dones": torch.tensor(((False, True),)),
        "policy_version": torch.tensor(((10.0, 10.0),)),
        "policy_id": torch.tensor(((0, 0),)),
    })

    learner._capture_online_spatial(buff)

    assert learner._online_spatial.run_name == "EXACT_STUDY_RUN"
    assert len(learner._online_spatial.window) == 2


def test_one_and_twenty_five_million_cadence_atomic_resume_and_multi_policy(tmp_path):
    env_info = SimpleNamespace(frameskip=4)
    first = TrainingSpatialTelemetry(_cfg(tmp_path), env_info, 0, restored_env_steps=0)
    _fill_window(first)
    stats = first.on_env_steps(1_000_000)
    assert stats["online_spatial_scalar_target_env_steps"] == 1_000_000
    assert not list(tmp_path.rglob("*.npz"))
    first.on_env_steps(25_000_128)
    snapshots = list(tmp_path.rglob("*.npz"))
    assert len(snapshots) == 2
    payloads = [load_spatial_snapshot(path) for path in snapshots]
    assert sorted(payload["target_env_steps"].item() for payload in payloads) == [5_000_000, 25_000_000]
    payload = next(payload for payload in payloads if payload["target_env_steps"].item() == 25_000_000)
    assert payload["actual_env_steps"].item() == 25_000_128
    assert payload["pose"].dtype == np.float32
    assert payload["dg_activity"].dtype == np.float32

    resumed = TrainingSpatialTelemetry(_cfg(tmp_path), env_info, 0, restored_env_steps=25_000_000)
    assert resumed.next_snapshot_target == 50_000_000
    _fill_window(resumed)
    resumed.on_env_steps(49_999_999)
    assert len(list(tmp_path.rglob("*.npz"))) == 2
    resumed.on_env_steps(50_000_000)
    assert len(list(tmp_path.rglob("*.npz"))) == 3

    policy_one = TrainingSpatialTelemetry(_cfg(tmp_path), env_info, 1, restored_env_steps=0)
    _fill_window(policy_one)
    policy_one.on_env_steps(25_000_000)
    assert len(list(tmp_path.rglob("policy_01/*.npz"))) == 2


def test_snapshot_uses_full_window_while_scalars_use_latest_tail(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.online_spatial_scalar_window_observations = 2
    telemetry = TrainingSpatialTelemetry(cfg, SimpleNamespace(frameskip=4), 0, restored_env_steps=0)
    _fill_window(telemetry)

    stats = telemetry.on_env_steps(5_000_000)
    snapshot = load_spatial_snapshot(next(tmp_path.rglob("*.npz")))

    assert stats["online_spatial_place_valid_sample_count"] == 2
    assert snapshot["pose"].shape[0] == 4
    assert snapshot["window_limit"].item() == 4
    assert snapshot["scalar_window_limit"].item() == 2
    assert snapshot["pose"][:, 0].tolist() == [100.0, 200.0, 300.0, 400.0]


def test_graph_buffers_diagnostics_and_compact_wandb_tags(tmp_path):
    graph = PolicyControllableGraph(2)
    graph.tctrl.copy_(torch.tensor(((0.0, 4.0), (5.0, 0.0))))
    graph.edge_confidence.copy_(torch.tensor(((0.0, 3.0), (3.0, 0.0))))
    graph.control_attempts.copy_(torch.tensor(((0.0, 4.0), (4.0, 0.0))))
    graph.prospective_attempts.copy_(torch.tensor(((0.0, 2.0), (2.0, 0.0))))
    graph.prospective_successes.copy_(torch.tensor(((0.0, 2.0), (1.0, 0.0))))
    actor = SimpleNamespace(
        core=SimpleNamespace(policy_graph=graph, passive_recruitment_graph=None),
        encoder=SimpleNamespace(DG_projection=None),
    )
    telemetry = TrainingSpatialTelemetry(
        _cfg(tmp_path), SimpleNamespace(frameskip=4), 0, restored_env_steps=0, actor_critic=actor
    )
    _fill_window(telemetry)
    stats = telemetry.on_env_steps(5_000_000)
    snapshot = load_spatial_snapshot(next(tmp_path.rglob("*.npz")))

    assert snapshot["control_attempts"].shape == (2, 2)
    assert snapshot["graph_reliable_global_efficiency"].item() == 1.0
    assert stats["online_spatial_graph_reliable_global_efficiency"] == 1.0
    assert stats["online_spatial_graph_grounded_controllability"] == 0.0
    assert INTRMOTIV_SUMMARY_TAGS["online_spatial_graph_reliable_global_efficiency"] == (
        "intrmotiv/hrl/summary/reliable_global_efficiency"
    )
    assert INTRMOTIV_SUMMARY_TAGS["online_spatial_graph_grounded_controllability"] == (
        "intrmotiv/hrl/summary/grounded_controllability"
    )


def test_passive_recruitment_buffers_are_saved_without_control_graph(tmp_path):
    passive = PassiveRecruitmentGraph(2)
    passive.confidence[0, 1] = 2.0
    actor = SimpleNamespace(
        core=SimpleNamespace(policy_graph=None, passive_recruitment_graph=passive),
        encoder=SimpleNamespace(DG_projection=None),
    )
    telemetry = TrainingSpatialTelemetry(
        _cfg(tmp_path), SimpleNamespace(frameskip=4), 0, restored_env_steps=0, actor_critic=actor
    )
    _fill_window(telemetry)
    stats = telemetry.on_env_steps(5_000_000)
    snapshot = load_spatial_snapshot(next(tmp_path.rglob("*.npz")))

    assert snapshot["passive_recruitment_confidence"][0, 1] == 2.0
    assert "control_tctrl" not in snapshot
    assert stats["online_spatial_graph_reliable_global_efficiency"] == 0.0
    assert stats["online_spatial_graph_grounded_controllability"] == 0.0


def test_missing_telemetry_and_workspace_escape_fail_fast(tmp_path):
    env_info = SimpleNamespace(frameskip=4)
    telemetry = TrainingSpatialTelemetry(_cfg(tmp_path), env_info, 0, restored_env_steps=0)
    with pytest.raises(SpatialContractError, match="telemetry_pose"):
        telemetry.append_batch({}, np.ones((1, 2), dtype=bool))
    cfg = _cfg(tmp_path)
    cfg.online_spatial_output_root = str(tmp_path.parent / "escape")
    with pytest.raises(SpatialContractError, match="inside workspace"):
        TrainingSpatialTelemetry(cfg, env_info, 0, restored_env_steps=0)
