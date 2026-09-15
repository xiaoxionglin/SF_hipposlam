from dataclasses import replace

import gymnasium as gym
import numpy as np

from sample_factory.algo.utils.make_env import NonBatchedMultiAgentWrapper
from sf_working_directories.IntrMotiv.dmlab.controller_replay import PhysicalDecision, PhysicalReplay
from sf_working_directories.IntrMotiv.dmlab.controller_transport import ControllerIdentity, record_terminal


class TerminalEnv(gym.Env):
    observation_space = gym.spaces.Dict({"obs": gym.spaces.Box(0, 255, (1,), np.uint8)})
    action_space = gym.spaces.Discrete(2)

    def __init__(self, valid=True):
        self.valid = valid
        self.seeds = []

    def reset(self, **kwargs):
        self.seeds.append(kwargs)
        return {"obs": np.array([1], np.uint8)}, {}

    def step(self, action):
        return {"obs": np.array([99], np.uint8)}, 3.0, True, False, {"intrmotiv_final_observation_valid": self.valid}


def test_terminal_image_survives_native_autoreset_and_seeding_is_unchanged():
    base = TerminalEnv()
    env = NonBatchedMultiAgentWrapper(ControllerIdentity(base, 7))
    env.reset(seed=123)
    obs, _, _, _, infos = env.step([0])
    info = infos[0]
    assert obs[0]["obs"].item() == 1
    assert info["controller_final_observation"]["obs"].item() == 99
    assert base.seeds == [{"seed": 123}, {}]
    buffer = {
        "controller_terminated": np.zeros(1, bool),
        "controller_final_valid": np.ones(1, bool),
        "controller_final_obs": {"obs": np.zeros((1, 1), np.uint8), "controller_identity": np.zeros((1, 4), np.int64)},
    }
    record_terminal(buffer, 0, True, info)
    assert buffer["controller_final_obs"]["obs"][0, 0] == 99
    record_terminal(buffer, 0, False, {})
    assert not buffer["controller_final_valid"][0]


def test_stale_terminal_image_is_never_certified():
    env = NonBatchedMultiAgentWrapper(ControllerIdentity(TerminalEnv(False), 0))
    env.reset()
    *_, infos = env.step([0])
    assert not infos[0]["controller_final_observation_valid"]
    assert "controller_final_observation" not in infos[0]


def row(i, **kw):
    return PhysicalDecision(
        (0, 4),
        0,
        i,
        i,
        {"obs": np.array([i], np.uint8)},
        0,
        np.array([1.0, 0.0]),
        np.zeros(3),
        2,
        0,
        False,
        False,
        4,
        **kw,
    )


def test_out_of_order_packets_join_across_rollouts_and_keep_certified_terminal():
    replay = PhysicalReplay(100, 1)
    last = replace(row(2), terminated=True, successor={"obs": np.array([17], np.uint8)}, successor_valid=True)
    replay.receive(row(1))
    assert replay.accepted == 0
    replay.receive(row(0))
    assert replay.accepted == 1
    replay.receive(last)
    assert replay.accepted == 3
    assert replay.rows[row(0).key].successor["obs"][0] == 1
    assert replay.rows[last.key].successor["obs"][0] == 17
    assert replay.physical_frames == 12


def test_restart_retains_replay_but_starts_new_physical_stream_session():
    replay = PhysicalReplay(100, 1)
    replay.receive(row(0))
    replay.receive(row(1))
    saved = replay.state_dict()
    restored = PhysicalReplay(100, 2)
    restored.load_state_dict(saved)
    assert restored.session == 1 and restored.accepted == 1 and restored.physical_frames == 8
    assert restored.rejected["restart_pending_tail"] == 1
    restored.receive(replace(row(0), stream=(1, 4)))
    assert restored.accepted == 1
    restored.receive(replace(row(1), stream=(1, 4)))
    assert restored.accepted == 2 and len(restored.rows) == 2


def test_engine_terminal_read_is_certified_only_when_available():
    from types import SimpleNamespace

    from sf_working_directories.IntrMotiv.dmlab.dmlab_gym import DmlabGymEnv_custom

    env = object.__new__(DmlabGymEnv_custom)
    env.benchmark_mode = False
    env.action_list = [0]
    env.action_repeat = 4
    env.with_pos_telemetry = False
    env.main_observation = "RGBD_INTERLEAVED"
    env.capture_terminal_observation = True
    env.format_obs_dict = lambda obs: {"obs": obs["RGBD_INTERLEAVED"]}
    env.dmlab = SimpleNamespace(
        step=lambda *a, **k: 0.0, is_running=lambda: False, observations=lambda: {"RGBD_INTERLEAVED": np.array([99])}
    )
    env.last_observation = {"obs": np.array([1])}
    obs, _, terminated, _, info = env.step(0)
    assert terminated and info["intrmotiv_final_observation_valid"] and obs["obs"][0] == 99

    def unavailable():
        raise RuntimeError("engine has stopped")

    env.dmlab.observations = unavailable
    env.last_observation = {"obs": np.array([1])}
    obs, _, _, _, info = env.step(0)
    assert not info["intrmotiv_final_observation_valid"] and obs["obs"][0] == 1
    env.dmlab.terminal_observations = lambda: {"RGBD_INTERLEAVED": np.array([101])}
    obs, _, _, _, info = env.step(0)
    assert info["intrmotiv_final_observation_valid"] and obs["obs"][0] == 101


def test_replay_checkpoint_uses_safe_tensor_serialization(tmp_path):
    import torch

    replay = PhysicalReplay(100, 1)
    replay.receive(row(0))
    replay.receive(row(1))
    path = tmp_path / "replay.pth"
    torch.save(replay.state_dict(), path)
    state = torch.load(path, weights_only=True)
    restored = PhysicalReplay(100, 2)
    restored.load_state_dict(state)
    np.testing.assert_array_equal(restored.rows[row(0).key].observation["obs"], row(0).observation["obs"])
    assert restored.session == 1


def test_unrecorded_startup_walk_is_rejected():
    from types import SimpleNamespace

    import pytest

    from sf_working_directories.IntrMotiv.dmlab.controller_transport import validate_controller_config

    with pytest.raises(ValueError, match="decorrelate_envs_on_one_worker"):
        validate_controller_config(SimpleNamespace(decorrelate_envs_on_one_worker=True))
