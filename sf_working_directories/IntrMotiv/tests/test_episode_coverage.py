from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sf_working_directories.IntrMotiv.evaluation import episode_coverage as module


class Actor(torch.nn.Module):
    def __init__(self, mutate=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.action_space = SimpleNamespace(n=8)
        self.mutate = mutate
        self.calls = 0

    def forward(self, obs, state):
        assert set(obs) == {"obs"}
        self.calls += 1
        if self.mutate:
            self.weight.add_(1)
        return {"actions": torch.zeros((1, 1), dtype=torch.long), "new_rnn_states": state}


class Env:
    def __init__(self, env_name):
        self.unwrapped = self
        self.env_name = env_name
        self.last_debug_position = np.array([150, 150, 25])
        self.seeds = []

    def seed(self, value):
        self.seeds.append(value)

    def reset(self):
        self.step_count = 0
        return {"obs": torch.zeros(1, 3)}, {}

    def step(self, action):
        self.step_count += 1
        done = self.step_count == 3
        stats = {
            f"z_00_{self.env_name}_" + k: v
            for k, v in dict(
                accessible_coverage_auc=0.25,
                accessible_coverage_fraction=0.5,
                coverage_unique_cells=2,
                geometry_invalid_pose_steps=0,
            ).items()
        }
        return (
            {"obs": torch.zeros(1, 3)},
            torch.zeros(1),
            torch.tensor([done]),
            torch.tensor([False]),
            [{"episode_extra_stats": stats}],
        )

    def close(self):
        pass


def setup_mock_environment(monkeypatch, env_name="corridor_geometry_noreward"):
    envs = []

    def make(cfg, *args, **kwargs):
        assert cfg.dmlab_use_level_cache is False
        env = Env(cfg.env)
        envs.append(env)
        return env

    monkeypatch.setattr(module, "make_env_func_batched", make)
    monkeypatch.setattr(module, "get_rnn_size", lambda cfg: 2)
    monkeypatch.setattr(module, "preprocess_actions", lambda info, action: action)
    monkeypatch.setattr(module, "prepare_and_normalize_obs", lambda actor, obs: obs)
    return (
        SimpleNamespace(
            env=env_name,
            dmlab_map_seed=1001,
            dmlab_wall_removal_probability=0.85 if env_name == "easy_landmark_maze_noreward" else 0,
            dmlab_map_rows=11,
            dmlab_map_cols=11,
            dmlab_cue_layout_seed=20260923,
            dmlab_landmark_cues="rich",
        ),
        envs,
    )


@pytest.mark.parametrize("random_actions", [False, True])
def test_matched_seeds_episode_counts_and_frozen_state(monkeypatch, tmp_path, random_actions):
    cfg, envs = setup_mock_environment(monkeypatch)
    actor = Actor()
    result = module.evaluate_episodes(cfg, actor, None, tmp_path, episodes=2, random_actions=random_actions)
    assert len(result["episodes"]) == 2 and len(result["coverage_curves"]) == 2
    assert [e.seeds for e in envs] == [[51000], [51001]]
    assert result["mean_accessible_coverage_auc"] == 0.25
    assert actor.calls == (0 if random_actions else 6)
    assert result["frozen_state_verified"]


def test_coverage_metric_namespace_follows_environment(monkeypatch, tmp_path):
    cfg, _ = setup_mock_environment(monkeypatch, "easy_landmark_maze_noreward")
    result = module.evaluate_episodes(cfg, Actor(), None, tmp_path, episodes=1)
    assert result["episodes"][0]["accessible_coverage_fraction"] == 0.5


def test_frozen_probe_rejects_mutation(monkeypatch, tmp_path):
    cfg, _ = setup_mock_environment(monkeypatch)
    with pytest.raises(RuntimeError, match="changed model"):
        module.evaluate_episodes(cfg, Actor(mutate=True), None, tmp_path, episodes=1)
    assert not list(tmp_path.glob("*.json"))
