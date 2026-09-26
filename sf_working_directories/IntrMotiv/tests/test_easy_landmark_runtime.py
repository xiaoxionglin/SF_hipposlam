"""Easy-landmark runtime contracts, including opt-in native DMLab checks."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hpc_runs.intrmotiv_study.geometry import geometry_from_config, geometry_payload, verify_cue_manifest, verify_entity
from sf_working_directories.IntrMotiv.dmlab.dmlab_gym import DmlabGymEnv_custom


def _config(mode: str = "rich") -> SimpleNamespace:
    return SimpleNamespace(
        env="easy_landmark_maze_noreward",
        dmlab_map_seed=1001,
        dmlab_wall_removal_probability=0.85,
        dmlab_map_rows=11,
        dmlab_map_cols=11,
        dmlab_cue_layout_seed=20260923,
        dmlab_landmark_cues=mode,
    )


def _native_manifest(record: dict) -> str:
    lines = [
        "schema\teasy-landmark-maze/v2",
        "shape\t11\t11",
        f"mode\t{record['cue_mode']}",
        f"seed\t{record['cue_layout_seed']}",
        "cue_id\ttype\tasset\twall_row\twall_col\tfloor_row\tfloor_col\torientation",
    ]
    for cue in record["cue_sites"]:
        lines.append(
            "\t".join(
                map(
                    str,
                    (
                        cue["cue_id"],
                        cue["cue_type"],
                        cue["asset"],
                        *cue["wall_rc"],
                        *cue["floor_rc"],
                        cue["orientation"],
                    ),
                )
            )
        )
    return "\n".join(lines)


def test_rich_and_control_share_exact_reserved_sites():
    rich = geometry_from_config(_config("rich"))
    control = geometry_from_config(_config("none"))
    assert rich["schema"] == control["schema"] == "intrmotiv/map-geometry/v2"
    assert rich["sha256"] == control["sha256"]
    assert rich["entity_shape"] == control["entity_shape"] == [11, 11]
    assert np.asarray(rich["accessible_mask"]).shape == (9, 9)
    assert rich["cue_sites"] == control["cue_sites"]
    assert len({tuple(cue["wall_rc"]) for cue in rich["cue_sites"]}) == 20
    assert len({tuple(cue["floor_rc"]) for cue in rich["cue_sites"]}) == 20
    assert [cue["cue_type"] for cue in rich["cue_sites"]].count("decal") == 10
    assert [cue["cue_type"] for cue in rich["cue_sites"]].count("color") == 10
    assert geometry_payload(rich)["geometry_cue_rendered"].all()
    assert not geometry_payload(control)["geometry_cue_rendered"].any()


def test_all_six_preflight_commands_parse_through_training_entrypoint(tmp_path):
    from hpc_runs.intrmotiv_study import load_study
    from sf_working_directories.IntrMotiv.dmlab.train_hipposlam import parse_dmlab_args

    runtime_root = Path(__file__).resolve().parents[3]
    study = load_study(runtime_root / "hpc_runs/studies/easy_landmark_maze_preflight.study.json")
    runs = study.expand_runs()
    assert len(runs) == 6
    for run in runs:
        cfg = parse_dmlab_args(
            argv=[
                *run.args,
                f"--experiment={run.name}",
                f"--train_dir={tmp_path / run.name}",
            ]
        )
        assert cfg.study_condition == run.condition
        assert cfg.env_frameskip == 4 and cfg.dmlab_navigation_action_set
        assert cfg.dmlab_map_rows == cfg.dmlab_map_cols == 11
        assert cfg.online_spatial_grid_grain == 9
        assert cfg.online_spatial_x_max == cfg.online_spatial_y_max == 1000
        assert cfg.dmlab_landmark_cues == run.factors["cue_mode"]
        if run.base == "WAYPOINT_F64_DDQN_HER":
            assert cfg.controller_learning == "ddqn"
            assert cfg.controller_replay_state == "stored"
            assert cfg.controller_decisions_per_update == 2048
            assert cfg.controller_td_positions == cfg.controller_her_positions == 1024


@pytest.mark.parametrize("mode", ["rich", "none"])
def test_privileged_geometry_and_cues_are_verified_then_stripped(mode):
    record = geometry_from_config(_config(mode))
    env = DmlabGymEnv_custom.__new__(DmlabGymEnv_custom)
    env.main_observation = "RGB_INTERLEAVED"
    env.instructions_observation = "INSTR"
    env.instructions = np.zeros(1, dtype=np.int32)
    env.with_number_instruction = True
    env.with_pos_obs = False
    env.with_online_spatial_telemetry = False
    env.action_path_integration = False
    env.geometry_expected_hash = record["sha256"]
    env.cue_layout_expected_hash = record["cue_layout_sha256"]
    env.geometry_record = record
    env.geometry_verified = False
    env.last_debug_position = None
    env.last_debug_rotation = None
    result = env.format_obs_dict(
        {
            "RGB_INTERLEAVED": np.zeros((2, 2, 3), np.uint8),
            "GEOMETRY.ENTITY_LAYER": record["entity_layer"],
            "GEOMETRY.CUE_MANIFEST": _native_manifest(record),
            "DEBUG.POS.TRANS": np.array([150, 150, 25]),
            "DEBUG.POS.ROT": np.zeros(3),
        }
    )
    assert set(result) == {"obs"}
    assert env.geometry_verified


def test_online_snapshot_uses_native_nine_by_nine_geometry(tmp_path):
    from hpc_runs.intrmotiv_study.spatial_contract import validate_snapshot_payload
    from sf_working_directories.IntrMotiv.dmlab.online_spatial_telemetry import TrainingSpatialTelemetry
    from sf_working_directories.IntrMotiv.tests.test_online_spatial_telemetry import _cfg, _fill_window

    cfg = _cfg(tmp_path)
    cfg.env = "easy_landmark_maze_noreward"
    cfg.dmlab_map_seed = 1001
    cfg.dmlab_wall_removal_probability = 0.85
    cfg.dmlab_map_rows = 11
    cfg.dmlab_map_cols = 11
    cfg.dmlab_cue_layout_seed = 20260923
    cfg.dmlab_landmark_cues = "rich"
    cfg.dmlab_geometry_manifest = ""
    cfg.online_spatial_grid_grain = 9
    cfg.online_spatial_x_max = cfg.online_spatial_y_max = 1000.0
    telemetry = TrainingSpatialTelemetry(cfg, SimpleNamespace(frameskip=4), 0, 0)
    _fill_window(telemetry)
    telemetry.on_env_steps(5_000_000)
    path = next(tmp_path.rglob("snapshot*.npz"))
    with np.load(path, allow_pickle=False) as archive:
        payload = dict(archive)
    validate_snapshot_payload(payload)
    assert payload["geometry_accessible_mask"].shape == (9, 9)
    assert payload["occupancy"].shape == (9, 9)
    assert payload["geometry_cue_ids"].shape == (20,)
    assert payload["geometry_entity_shape"].tolist() == [11, 11]


@pytest.mark.skipif(not os.environ.get("EASY_LANDMARK_DMLAB_RUNFILES"), reason="Opt-in real DMLab map compilation")
def test_native_modes_geometry_reset_reward_and_timeout():
    import deepmind_lab

    deepmind_lab.set_runfiles_path(os.environ["EASY_LANDMARK_DMLAB_RUNFILES"])
    pose_signatures = {}
    image_signatures = {}
    for mode in ("rich", "none"):
        record = geometry_from_config(_config(mode))
        lab = deepmind_lab.Lab(
            "easy_landmark_maze_noreward",
            ["RGB_INTERLEAVED", "GEOMETRY.ENTITY_LAYER", "GEOMETRY.CUE_MANIFEST", "DEBUG.POS.TRANS", "DEBUG.POS.ROT"],
            config={
                "geometrySeed": "1001",
                "wallRemovalProbability": "0.85",
                "mapRows": "11",
                "mapCols": "11",
                "cueLayoutSeed": "20260923",
                "landmarkCues": mode,
                "geometryHash": record["sha256"],
                "cueLayoutHash": record["cue_layout_sha256"],
                "width": "96",
                "height": "72",
            },
            renderer="software",
        )
        try:
            # The first software-renderer reset warms DMLab's texture cache.
            # Compare the next two resets so the assertion measures level
            # determinism rather than one-time renderer initialization.
            for repeat in range(3):
                lab.reset(seed=51000)
                lab.step(np.zeros(7, np.intc), num_steps=1)
                obs = lab.observations()
                verify_entity(obs["GEOMETRY.ENTITY_LAYER"], record)
                verify_cue_manifest(obs["GEOMETRY.CUE_MANIFEST"], record)
                pose = np.r_[obs["DEBUG.POS.TRANS"], obs["DEBUG.POS.ROT"]]
                image = np.asarray(obs["RGB_INTERLEAVED"]).copy()
                if repeat == 1:
                    pose_signatures[mode] = pose
                    image_signatures[mode] = image
                elif repeat == 2:
                    np.testing.assert_array_equal(pose_signatures[mode], pose)
                    np.testing.assert_array_equal(image_signatures[mode], image)
                for _ in range(12):
                    assert lab.step(np.array([20, 0, 0, 1, 0, 0, 0], np.intc), num_steps=4) == 0
            lab.reset(seed=51000)
            decisions = 0
            while lab.is_running() and decisions <= 1801:
                assert lab.step(np.zeros(7, np.intc), num_steps=4) == 0
                decisions += 1
            assert 1799 <= decisions <= 1801
            assert not lab.is_running()
        finally:
            lab.close()
    np.testing.assert_array_equal(pose_signatures["rich"], pose_signatures["none"])
    assert not np.array_equal(image_signatures["rich"], image_signatures["none"])
