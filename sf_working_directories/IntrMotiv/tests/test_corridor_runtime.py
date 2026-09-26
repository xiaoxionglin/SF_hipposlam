"""Corridor integration contracts, plus opt-in real-engine qualification."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from hpc_runs.intrmotiv_study.geometry import accessible_cell, geometry_from_config, verify_entity
from sf_working_directories.IntrMotiv.dmlab.dmlab_gym import DmlabGymEnv_custom
from sf_working_directories.IntrMotiv.dmlab.wrappers.reward_shaping import DmlabRewardShapingWrapper
from sf_working_directories.IntrMotiv.tests.test_exploration_coverage import PositionEnv


def config(seed=1001, q=0):
    return SimpleNamespace(env="corridor_geometry_noreward", dmlab_map_seed=seed, dmlab_wall_removal_probability=q)


def test_geometry_verification_is_stripped_before_model_input():
    record = geometry_from_config(config())
    env = DmlabGymEnv_custom.__new__(DmlabGymEnv_custom)
    env.main_observation = "RGB"
    env.instructions_observation = "INSTR"
    env.instructions = np.zeros(1, dtype=np.int32)
    env.with_number_instruction = True
    env.with_pos_obs = False
    env.with_online_spatial_telemetry = False
    env.action_path_integration = False
    env.geometry_expected_hash = record["sha256"]
    env.geometry_verified = False
    env.last_debug_position = None
    env.last_debug_rotation = None
    result = env.format_obs_dict(
        {
            "RGB": np.zeros((2, 2, 3), np.uint8),
            "GEOMETRY.ENTITY_LAYER": record["entity_layer"],
            "DEBUG.POS.TRANS": np.array([150, 150, 25]),
            "DEBUG.POS.ROT": np.zeros(3),
        }
    )
    assert set(result) == {"obs"}
    assert env.geometry_verified
    env.geometry_verified = False
    with pytest.raises(ValueError, match="SHA-256"):
        env.format_obs_dict({"RGB": np.zeros((2, 2, 3), np.uint8), "GEOMETRY.ENTITY_LAYER": "bad"})


def test_wrapper_emits_accessible_episode_metrics():
    base = PositionEnv()
    base.geometry_record = {"accessible_mask": [[1, 1], [0, 1]], "accessible_cells": 3, "cell_size": 100}
    base.poses = ([150, 150, 0], [250, 150, 0], [250, 150, 0])
    env = DmlabRewardShapingWrapper(base, coverage_telemetry=True)
    env.reset()
    for _ in range(3):
        obs, rew, done, trunc, info = env.step(0)
    assert info["episode_extra_stats"]["z_00_fixedlength_accessible_coverage_fraction"] == 2 / 3
    assert info["episode_extra_stats"]["z_00_fixedlength_accessible_coverage_auc"] == 5 / 9
    assert rew == 0 and done
    env.reset()
    assert not env.accessible_coverage.visited


def test_wrapper_does_not_label_missing_terminal_pose_out_of_bounds():
    class MissingTerminalPoseEnv(PositionEnv):
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            if terminated:
                info["intrmotiv_terminal_pose_fresh"] = False
            return obs, reward, terminated, truncated, info

    base = MissingTerminalPoseEnv()
    base.geometry_record = {
        "accessible_mask": [[1, 1], [0, 1]],
        "accessible_cells": 3,
        "cell_size": 100,
    }
    base.poses = ([150, 150, 0], [250, 150, 0], [250, 150, 0])
    env = DmlabRewardShapingWrapper(base, coverage_telemetry=True)
    env.reset()
    for _ in range(3):
        _, _, _, _, info = env.step(0)
    stats = info["episode_extra_stats"]
    assert stats["z_00_fixedlength_geometry_invalid_pose_steps"] == 0
    assert stats["z_00_fixedlength_accessible_coverage_auc"] == 5 / 9


@pytest.mark.skipif(not os.environ.get("CORRIDOR_DMLAB_RUNFILES"), reason="Opt-in real DMLab map compilation")
def test_native_maps_prefix_replay_spawns_and_timeout(tmp_path):
    import deepmind_lab

    deepmind_lab.set_runfiles_path(os.environ["CORRIDOR_DMLAB_RUNFILES"])
    starts = {}
    import shutil

    class Cache:
        def fetch(self, key, path):
            cached = tmp_path / key
            if not cached.is_file():
                return False
            shutil.copyfile(cached, path)
            return True

        def write(self, key, path):
            shutil.copyfile(path, tmp_path / key)

    cache = Cache()
    for seed in (1001, 1002, 1003):
        for q in (0, 0.35, 0.75):
            record = geometry_from_config(config(seed, q))
            lab = deepmind_lab.Lab(
                "corridor_geometry_noreward",
                ["RGB_INTERLEAVED", "GEOMETRY.ENTITY_LAYER", "DEBUG.POS.TRANS", "DEBUG.POS.ROT"],
                config={
                    "geometrySeed": str(seed),
                    "wallRemovalProbability": str(q),
                    "geometryHash": record["sha256"],
                    "width": "96",
                    "height": "72",
                },
                renderer="software",
                level_cache=cache,
            )
            try:
                for reset_seed in (51000, 51001):
                    signatures = []
                    for repeat in range(2):
                        # Match the causal evaluator: reconstruct the engine, not
                        # only its episode. Renderer animation can persist in a Lab.
                        lab.close()
                        lab = deepmind_lab.Lab(
                            "corridor_geometry_noreward",
                            ["RGB_INTERLEAVED", "GEOMETRY.ENTITY_LAYER", "DEBUG.POS.TRANS", "DEBUG.POS.ROT"],
                            config={
                                "geometrySeed": str(seed),
                                "wallRemovalProbability": str(q),
                                "geometryHash": record["sha256"],
                                "width": "96",
                                "height": "72",
                            },
                            renderer="software",
                            level_cache=cache,
                        )
                        lab.reset(seed=reset_seed)
                        obs = lab.observations()
                        verify_entity(obs["GEOMETRY.ENTITY_LAYER"], record)
                        pose = np.r_[obs["DEBUG.POS.TRANS"], obs["DEBUG.POS.ROT"]]
                        if reset_seed not in starts:
                            starts[reset_seed] = pose
                        np.testing.assert_array_equal(starts[reset_seed], pose)
                        assert accessible_cell(obs["DEBUG.POS.TRANS"], record) is not None
                        for _ in range(12):
                            assert lab.step(np.array([20, 0, 0, 1, 0, 0, 0], np.intc), num_steps=4) == 0
                        obs = lab.observations()
                        signatures.append((obs["RGB_INTERLEAVED"], obs["DEBUG.POS.TRANS"], obs["DEBUG.POS.ROT"]))
                    for a, b in zip(*signatures):
                        np.testing.assert_array_equal(a, b)
                if seed == 1001 and q == 0:
                    lab.reset(seed=51000)
                    steps = 0
                    while lab.is_running() and steps <= 1801:
                        assert lab.step(np.zeros(7, np.intc), num_steps=4) == 0
                        steps += 1
                    assert 1799 <= steps <= 1801 and not lab.is_running()
            finally:
                lab.close()


def test_geometry_snapshot_preserves_contract_and_orientation(tmp_path):
    from hpc_runs.intrmotiv_study.spatial_contract import validate_snapshot_payload
    from sf_working_directories.IntrMotiv.dmlab.online_spatial_telemetry import TrainingSpatialTelemetry
    from sf_working_directories.IntrMotiv.tests.test_online_spatial_telemetry import _cfg, _fill_window

    cfg = _cfg(tmp_path)
    cfg.env = "corridor_geometry_noreward"
    cfg.dmlab_map_seed = 1001
    cfg.dmlab_wall_removal_probability = 0
    telem = TrainingSpatialTelemetry(cfg, SimpleNamespace(frameskip=4), 0, 0)
    _fill_window(telem)
    telem.on_env_steps(5_000_000)
    path = next(tmp_path.rglob("snapshot*.npz"))
    with np.load(path, allow_pickle=False) as archive:
        payload = dict(archive)
    validate_snapshot_payload(payload)
    assert payload["geometry_accessible_mask"].shape == (19, 19)
    assert payload["geometry_field_components_half_peak"].shape == (2,)
    payload["geometry_accessible_mask"] = ~payload["geometry_accessible_mask"]
    with pytest.raises(ValueError, match="disagrees"):
        validate_snapshot_payload(payload)


def test_terminal_pose_uses_certified_binding_without_policy_capture():
    class TerminalLab:
        def terminal_observations(self):
            return {"DEBUG.POS.TRANS": np.array([250, 350, 25]), "DEBUG.POS.ROT": np.array([0, 90, 0])}

        def observations(self):
            raise RuntimeError("engine ended")

    env = DmlabGymEnv_custom.__new__(DmlabGymEnv_custom)
    env.dmlab = TerminalLab()
    env.capture_terminal_observation = False
    assert env._refresh_terminal_debug_pose()
    np.testing.assert_array_equal(env.last_debug_position, [250, 350, 25])
    np.testing.assert_array_equal(env.last_debug_rotation, [0, 90, 0])
