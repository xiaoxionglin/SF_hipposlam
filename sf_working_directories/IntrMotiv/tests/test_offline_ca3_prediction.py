"""Temporal and held-out contracts for the offline prediction diagnostic."""

import json
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from sf_working_directories.IntrMotiv.evaluation.offline_ca3_prediction import (
    episode_splits,
    evaluate,
    file_sha256,
    fit_pair,
    prediction_metrics,
    prediction_objective,
    run_fit,
    support,
    support_gate,
    transition_indices,
    validate_dataset,
    validation_selection_metric,
)


def fixture_data(episodes=12, length=20):
    rng = np.random.default_rng(12)
    actions = rng.integers(0, 2, episodes * length)
    states = np.zeros((episodes * length, 4), np.float32)
    for episode in range(episodes):
        for step in range(1, length):
            index = episode * length + step
            states[index, int(actions[index - 1]) * 2] = 3
    data = dict(
        ca3=states,
        actions=actions,
        dones_after=np.tile(np.arange(length) == length - 1, episodes),
        episode_id=np.repeat(np.arange(episodes), length),
        step_in_episode=np.tile(np.arange(length), episodes),
    )
    return data, dict(n_dg=2, expanded=2, action_count=2)


def test_windows_use_executed_action_and_future_target_without_reset_leakage():
    data, metadata = fixture_data(6, 5)
    validate_dataset(data, metadata)
    h1, h2 = transition_indices(data, 1), transition_indices(data, 2)
    assert len(h1) == 6 * 4 and len(h2) == 6 * 3
    assert not np.any(h1 % 5 == 4)
    assert np.all(data["ca3"][h1 + 1, ::2].argmax(1) == data["actions"][h1])
    assert np.all(data["episode_id"][h2] == data["episode_id"][h2 + 2])


@pytest.mark.parametrize("defect", ["action", "nonfinite", "noncontiguous", "missing_reset", "internal_reset", "steps"])
def test_malformed_trajectory_fails_closed(defect):
    data, metadata = fixture_data(6, 5)
    if defect == "action":
        data["actions"][0] = 2
    if defect == "nonfinite":
        data["ca3"][0, 0] = np.nan
    if defect == "noncontiguous":
        data["episode_id"][-5:] = 0
    if defect == "missing_reset":
        data["dones_after"][4] = False
    if defect == "internal_reset":
        data["dones_after"][2] = True
    if defect == "steps":
        data["step_in_episode"][2] = 7
    with pytest.raises(ValueError):
        validate_dataset(data, metadata)


def test_split_is_episode_disjoint_and_independent_of_fit_seed():
    data, _ = fixture_data()
    first, again = episode_splits(data["episode_id"]), episode_splits(data["episode_id"])
    for name in first:
        np.testing.assert_array_equal(first[name], again[name])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["validation"]).isdisjoint(first["test"])
    assert sum(map(len, first.values())) == 12
    with pytest.raises(ValueError):
        episode_splits(np.arange(5))


def test_sparse_support_gate_and_active_zero_metrics():
    data, _ = fixture_data()
    starts = transition_indices(data, 1)
    coverage = {"test": support(data, starts, 2, 2, 1)}
    assert not support_gate(coverage)
    data["ca3"][:] = 0
    assert "active transitions" in support_gate({"test": support(data, starts, 2, 2, 1)})[0]
    result = prediction_metrics(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 0.0]]))
    assert result["active_mse"] == 4 and result["zero_mse"] == 4


def test_loss_mode_controls_loss_and_checkpoint_selection_without_changing_default():
    target = torch.tensor([[3.0, 0.0, 0.0, 0.0]])
    predicts_activity_everywhere = torch.full_like(target, 3.0)
    predicts_zero = torch.zeros_like(target)
    assert prediction_objective(predicts_activity_everywhere, target) == pytest.approx(0.25)
    assert prediction_objective(predicts_zero, target) == pytest.approx(2.5)
    assert prediction_objective(predicts_activity_everywhere, target, "elementwise_mse") == pytest.approx(6.75)
    assert prediction_objective(predicts_zero, target, "elementwise_mse") == pytest.approx(2.25)
    for mode, expected in (("stratified", "selection_loss"), ("elementwise_mse", "mse")):
        assert validation_selection_metric(mode) == expected
    with pytest.raises(ValueError):
        prediction_objective(predicts_zero, target, "unknown")


def test_paired_fit_learns_action_signal_and_keeps_dataset_immutable():
    torch.set_num_threads(1)
    data, metadata = fixture_data(20, 20)
    before = data["ca3"].copy()
    states, actions = torch.from_numpy(data["ca3"]), torch.from_numpy(data["actions"])
    valid = transition_indices(data, 1)
    splits = episode_splits(data["episode_id"])
    indices = {name: valid[np.isin(data["episode_id"][valid], ids)] for name, ids in splits.items()}
    rows, models = fit_pair(states, actions, indices, metadata, 1, 13, epochs=25, batch_size=64, lr=0.01)
    test = {row["model"]: row for row in rows if row["split"] == "test"}
    assert test["action"]["active_mse"] < test["state_only"]["active_mse"]
    assert test["action"]["action_shuffle_active_mse_delta"] > 0
    assert test["state_only"]["action_shuffle_mse_delta"] == pytest.approx(0, abs=1e-9)
    assert models.keys() == {"action", "state_only"}
    np.testing.assert_array_equal(data["ca3"], before)
    baseline = evaluate(None, None, states, actions, indices["test"], 1, 2)
    assert test["action"]["active_mse"] < baseline["active_mse"]


def test_cli_fit_reports_low_support_without_fitting(tmp_path):
    data, metadata = fixture_data()
    data["ca3"][:] = 0
    dataset = tmp_path / "trajectory.npz"
    np.savez(dataset, **data)
    metadata["dataset_sha256"] = file_sha256(dataset)
    dataset.with_suffix(".json").write_text(json.dumps(metadata))
    args = SimpleNamespace(
        dataset=dataset,
        output=tmp_path / "fit",
        split_seed=901,
        epochs=1,
        batch_size=32,
        lr=0.001,
        seeds=[13],
        horizons=[1],
        minimum_active=20,
        device="cpu",
    )
    assert run_fit(args) == 2
    result = json.loads((args.output / "results.json").read_text())
    assert not result["passed"] and result["unsupported"]["1"] and not result["records"]


@pytest.mark.parametrize("loss_mode", ["stratified", "elementwise_mse"])
def test_fit_artifacts_preserve_provenance_and_random_action_subset(tmp_path, loss_mode):
    data, metadata = fixture_data()
    data["random_action"] = np.ones(len(data["actions"]), dtype=bool)
    dataset = tmp_path / "trajectory.npz"
    np.savez(dataset, **data)
    metadata["dataset_sha256"] = file_sha256(dataset)
    dataset.with_suffix(".json").write_text(json.dumps(metadata))
    args = SimpleNamespace(
        dataset=dataset,
        output=tmp_path / "fit",
        split_seed=901,
        epochs=1,
        batch_size=64,
        lr=0.001,
        seeds=[13],
        horizons=[1],
        minimum_active=20,
        device="cpu",
        loss_mode=loss_mode,
    )
    assert run_fit(args) == 0
    result = json.loads((args.output / "results.json").read_text())
    assert result["passed"] and result["test_comparisons"]
    assert len([row for row in result["records"] if row["split"] == "test_random_action"]) == 4
    assert result["fit_settings"]["loss_mode"] == loss_mode
    expected = validation_selection_metric(loss_mode)
    assert result["fit_settings"]["validation_selection_metric"] == expected
    for row in result["records"]:
        if row["model"] in ("action", "state_only"):
            assert row["validation_selection_metric"] == expected
            if row["split"] == "validation":
                assert row["best_validation_score"] == pytest.approx(row[expected])
        if row["model"] == "zero":
            assert row["zero_mse"] == 0
    saved = torch.load(args.output / "predictors_h1_s13.pth", weights_only=True)
    assert saved["dataset_sha256"] == metadata["dataset_sha256"] == file_sha256(dataset)
    assert set(saved["models"]) == {"action", "state_only"}
    assert saved["fit_settings"]["loss_mode"] == loss_mode


def test_collector_hook_tracks_actual_random_actions_and_resets(tmp_path, monkeypatch):
    from sf_working_directories.IntrMotiv.evaluation import collect_ca3_prediction as collector
    from sf_working_directories.IntrMotiv.evaluation import place_fields

    class Core(torch.nn.Module):
        def forward(self, obs, state):
            return obs["activity"], obs["activity"].clone()

    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = Core()
            self.action_space = gym.spaces.Discrete(2)

        def model_to_device(self, device):
            self.to(device)

        def device_for_input_tensor(self, _key):
            return torch.device("cpu")

        def type_for_input_tensor(self, _key):
            return torch.float32

        def normalize_obs(self, obs):
            return obs

        def forward(self, obs, state):
            core, new_state = self.core(obs, state)
            return dict(actions=torch.zeros(1, 1, dtype=torch.long), new_rnn_states=new_state)

    class Env:
        num_agents = 1

        def __init__(self):
            self.unwrapped = self
            self.step_id = 0
            self.last_reset_seed = 1
            self.executed = []

        def seed(self, seed):
            self.last_reset_seed = seed

        def obs(self, value=0):
            return dict(
                activity=torch.tensor([[float(value), 0.0, 0.0, 0.0]]), pos=torch.zeros(1, 3), rot=torch.zeros(1, 3)
            )

        def reset(self):
            self.step_id = 0
            return self.obs(), {}

        def step(self, actions):
            action = int(actions[0])
            self.executed.append(action)
            self.step_id += 1
            done = self.step_id % 4 == 0
            return self.obs(0 if done else action + 1), [0], [done], [False], {}

        def close(self):
            pass

    run = tmp_path / "run"
    run.mkdir()
    cfg = dict(core_name="BypassSS", transfer_freeze_dg=True, dmlab_use_level_cache=True)
    (run / "config.json").write_text(json.dumps(cfg))
    checkpoint = run / "checkpoint.pth"
    checkpoint.write_bytes(b"fixture")
    env, actor = Env(), Actor()
    loaded = SimpleNamespace(
        Hippo_n_feature=2,
        Hippo_R=1,
        Hippo_L=2,
        env="fixture",
        env_frameskip=4,
        use_rnn=True,
        rnn_type="gru",
        rnn_size=4,
        rnn_num_layers=1,
        actor_critic_share_weights=True,
    )

    def load(loader, *_args):
        private = json.loads((loader / "config.json").read_text())
        assert private["dmlab_use_level_cache"] is False
        return (
            loaded,
            env,
            SimpleNamespace(action_space=actor.action_space, gpu_actions=False, all_discrete=True),
            actor,
            checkpoint,
            torch.device("cpu"),
        )

    monkeypatch.setattr(place_fields, "load_policy_env", load)
    monkeypatch.setitem(sys.modules, "deepmind_lab", SimpleNamespace(set_runfiles_path=lambda _path: None))
    monkeypatch.setenv("INTRMOTIV_WORKSPACE_ROOT", str(tmp_path))
    args = SimpleNamespace(
        output=tmp_path / "data.npz",
        run_dir=run,
        checkpoint=checkpoint,
        runfiles=tmp_path,
        decisions=24,
        random_action_probability=1.0,
        seed=41,
        device="cpu",
    )
    collector.collect(args)
    with np.load(args.output) as archive:
        data = dict(archive)
    np.testing.assert_array_equal(data["actions"], env.executed)
    valid = transition_indices(data, 1)
    np.testing.assert_array_equal(data["ca3"][valid + 1, 0], data["actions"][valid] + 1)
    assert data["random_action"].all() and data["dones_after"].sum() == 6
    assert json.loads((run / "config.json").read_text()) == cfg
