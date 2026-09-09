import gymnasium as gym
import numpy as np

from sf_working_directories.IntrMotiv.dmlab.dmlab_gym import DmlabGymEnv_custom
from sf_working_directories.IntrMotiv.dmlab.wrappers.reward_shaping import DmlabRewardShapingWrapper


class PositionEnv(gym.Env):
    def __init__(self):
        self.level_name = "fixedlength"
        self.task_id = 0
        self.index = 0
        self.poses = ([0.0, 0.0, 0.0], [150.0, 0.0, 30.0], [150.0, 0.0, 60.0])
        self.observation_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(1)

    def reset(self, **kwargs):
        self.index = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        pose = np.asarray(self.poses[self.index], dtype=np.float64)
        self.index += 1
        terminated = self.index == len(self.poses)
        return (
            np.zeros(1, dtype=np.float32),
            0.0,
            terminated,
            False,
            {"num_frames": 8, "intrmotiv_pose": pose},
        )


def test_training_telemetry_is_not_returned_as_an_observation():
    env = DmlabGymEnv_custom.__new__(DmlabGymEnv_custom)
    env.main_observation = "RGB"
    env.instructions_observation = "INSTR"
    env.instructions = np.zeros(1, dtype=np.int32)
    env.with_number_instruction = True
    env.with_pos_obs = False
    env.action_path_integration = False
    env.last_debug_position = None
    env.last_debug_rotation = None

    obs = env.format_obs_dict(
        {
            "RGB": np.zeros((2, 2, 3), dtype=np.uint8),
            "DEBUG.POS.TRANS": np.asarray((7.0, 11.0, 13.0)),
            "DEBUG.POS.ROT": np.asarray((1.0, 45.0, 2.0)),
        }
    )

    assert set(obs) == {"obs"}
    assert np.array_equal(env.last_debug_position, np.asarray((7.0, 11.0, 13.0)))
    assert np.array_equal(env.last_debug_rotation, np.asarray((1.0, 45.0, 2.0)))


def test_online_spatial_pose_is_a_float32_privileged_observation():
    env = DmlabGymEnv_custom.__new__(DmlabGymEnv_custom)
    env.main_observation = "RGB"
    env.instructions_observation = "INSTR"
    env.instructions = np.zeros(1, dtype=np.int32)
    env.with_number_instruction = True
    env.with_pos_obs = False
    env.with_online_spatial_telemetry = True
    env.action_path_integration = False
    env.last_debug_position = None
    env.last_debug_rotation = None

    obs = env.format_obs_dict({
        "RGB": np.zeros((2, 2, 3), dtype=np.uint8),
        "DEBUG.POS.TRANS": np.asarray((7.0, 11.0, 13.0)),
        "DEBUG.POS.ROT": np.asarray((1.0, 45.0, 2.0)),
    })

    assert set(obs) == {"obs", "telemetry_pose"}
    assert obs["telemetry_pose"].dtype == np.float32
    assert np.array_equal(obs["telemetry_pose"], np.asarray((7.0, 11.0, 45.0), dtype=np.float32))


def test_coverage_metrics_are_emitted_without_forwarding_position():
    env = DmlabRewardShapingWrapper(PositionEnv(), coverage_telemetry=True, coverage_grid_size=100.0)
    env.reset()

    final_info = None
    for _ in range(3):
        obs, _, _, _, final_info = env.step(0)
        assert "intrmotiv_pose" not in final_info
        assert "intrmotiv_position" not in final_info
        assert np.array_equal(obs, np.zeros(1, dtype=np.float32))

    stats = final_info["episode_extra_stats"]
    assert stats["z_00_fixedlength_coverage_unique_cells"] == 2.0
    assert np.isclose(stats["z_00_fixedlength_coverage_auc"], 5.0 / 3.0)
    assert stats["z_00_fixedlength_coverage_entropy"] > 0.0
    assert stats["z_00_fixedlength_pose_unique_bins"] == 3.0
    assert np.isclose(stats["z_00_fixedlength_pose_auc"], 2.0)
    assert np.isclose(stats["z_00_fixedlength_pose_entropy"], np.log(3.0))


def test_nonterminal_exploration_window_does_not_reset_environment_state():
    env = DmlabRewardShapingWrapper(
        PositionEnv(), coverage_telemetry=True, coverage_grid_size=100.0, exploration_window_steps=2
    )
    env.reset()
    _, _, terminated, _, first = env.step(0)
    assert not terminated
    assert "intrmotiv_periodic_stats" not in first
    _, _, terminated, _, second = env.step(0)
    assert not terminated
    stats = second["intrmotiv_periodic_stats"]
    assert stats["intrmotiv/exploration/window/length_policy_steps"] == 2.0
    assert stats["intrmotiv/exploration/window/length_frames"] == 16.0
    assert stats["intrmotiv/exploration/window/coverage_unique_cells"] == 2.0
    assert stats["intrmotiv/exploration/window/pose_unique_bins"] == 2.0
    assert stats["intrmotiv/exploration/window/pose_entropy"] > 0.0


def test_invalid_heading_bin_width_is_rejected():
    with np.testing.assert_raises(ValueError):
        DmlabRewardShapingWrapper(PositionEnv(), coverage_heading_bin_degrees=0.0)
