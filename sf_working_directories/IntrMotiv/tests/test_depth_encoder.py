"""Depth bypass response and sampling contract, without loading visual weights."""

import argparse
import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.utils.normalize import ObservationNormalizer
from sample_factory.utils.utils import cfg_file
from sf_working_directories.IntrMotiv.dmlab.custom_encoder import DepthEncoder
from sf_working_directories.IntrMotiv.dmlab.custom_params import add_hipposlam_env_args
from sf_working_directories.IntrMotiv.dmlab.train_hipposlam import parse_dmlab_args


def test_capped_inverse_depth_response():
    encoder = DepthEncoder(SimpleNamespace(encoder_conv_architecture="layer2_resnet18", depth_sensor_inverse=True))
    depth = torch.tensor([0.0, 0.5, 1.0, 2.0, 10.0, 50.0, 100.0, 150.0, 200.0, 255.0])
    obs = depth.reshape(1, 1, 1, 10)
    original = obs.clone()
    actual = encoder(obs).flatten()
    torch.testing.assert_close(actual[:5], torch.tensor([10.0, 10.0, 10.0, 5.0, 1.0]))
    assert torch.isfinite(actual).all()
    assert (actual[1:] <= actual[:-1]).all()
    assert actual[7:].max() <= 10 / 150 + 1e-8
    assert abs(actual[7] - actual[9]) < 0.03
    torch.testing.assert_close(obs, original)


def test_depth_sampling_preserves_batch_shape_and_dtype():
    encoder = DepthEncoder(SimpleNamespace(encoder_conv_architecture="layer2_resnet18", depth_sensor_inverse=True))
    obs = torch.full((3, 1, 72, 96), 2.0, dtype=torch.float64)
    actual = encoder(obs)
    assert actual.shape == (3, 1, 1, encoder.get_out_size())
    assert actual.dtype == obs.dtype
    torch.testing.assert_close(actual, torch.full_like(actual, 5.0))


def test_fixed_gain_gives_order_one_norm_without_erasing_distance():
    encoder = DepthEncoder(SimpleNamespace(encoder_conv_architecture="layer2_resnet18", depth_sensor_inverse=True))
    depths = torch.tensor([30.0, 100.0, 150.0, 255.0])
    obs = depths[:, None, None, None].expand(-1, 1, 1, 10)
    norms = encoder(obs).flatten(1).norm(dim=1)
    torch.testing.assert_close(norms, torch.tensor([1.0540926, 0.3162278, 0.2108185, 0.1240109]))
    assert (norms[1:] < norms[:-1]).all()


@pytest.mark.parametrize("explicit", [False, True])
def test_legacy_config_preserves_original_sampling_and_checkpoint(explicit):
    cfg = SimpleNamespace(encoder_conv_architecture="layer2_resnet18", normalize_input=True)
    if explicit:
        cfg.depth_sensor_inverse = False
    encoder = DepthEncoder(cfg)
    obs = torch.randn(2, 1, 72, 96)
    torch.testing.assert_close(encoder(obs), torch.nn.functional.interpolate(obs, size=(1, 10)), rtol=0, atol=0)
    encoder.load_state_dict({}, strict=True)
    assert not encoder.state_dict()


@pytest.mark.parametrize("scale,mean", [(1.0, 0.0), (255.0, 0.0), (255.0, 127.5)])
def test_inverse_mode_restores_fixed_preprocessing(scale, mean):
    cfg = SimpleNamespace(
        encoder_conv_architecture="layer2_resnet18",
        depth_sensor_mode="capped_inverse",
        depth_sensor_gain=3.0,
        obs_scale=scale,
        obs_subtract_mean=mean,
        normalize_input=False,
    )
    space = gym.spaces.Dict({"obs": gym.spaces.Box(0, 255, (1, 1, 10), dtype=np.uint8)})
    raw = torch.tensor([0.0, 1.0, 2.0, 10.0, 30.0, 50.0, 100.0, 150.0, 200.0, 255.0]).reshape(1, 1, 1, 10)
    processed = ObservationNormalizer(space, cfg)({"obs": raw})["obs"]
    torch.testing.assert_close(DepthEncoder(cfg)(processed), 3.0 / raw.clamp_min(1), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("gain", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_inverse_gain_is_rejected(gain):
    with pytest.raises(ValueError, match="depth_sensor_gain"):
        DepthEncoder(
            SimpleNamespace(
                encoder_conv_architecture="layer2_resnet18", depth_sensor_mode="capped_inverse", depth_sensor_gain=gain
            )
        )


def test_running_depth_normalization_is_rejected():
    with pytest.raises(ValueError, match="normalize_input=False"):
        DepthEncoder(
            SimpleNamespace(
                encoder_conv_architecture="layer2_resnet18",
                depth_sensor_mode="capped_inverse",
                normalize_input=True,
                normalize_input_keys=["obs"],
            )
        )


def test_cli_defaults_and_inverse_toggle():
    parser = argparse.ArgumentParser()
    add_hipposlam_env_args(parser)
    defaults = parser.parse_args([])
    assert defaults.depth_sensor_inverse is True
    assert defaults.online_spatial_workspace_root == "/work/classic/fr_xl1014-corridor-geometry"
    assert defaults.online_spatial_output_root == ""
    assert not hasattr(defaults, "depth_sensor_mode")
    assert not hasattr(defaults, "depth_sensor_gain")
    assert parser.parse_args(["--depth_sensor_inverse=True"]).depth_sensor_inverse is True
    assert parser.parse_args(["--depth_sensor_inverse=False"]).depth_sensor_inverse is False


def test_fresh_inverse_default_preserves_legacy_saved_configs(tmp_path):
    argv = [
        "--env=openfield_map2_fixed_loc3_fixedlength_noreward",
        "--experiment=depth_default_migration",
        f"--train_dir={tmp_path}",
        "--depth_sensor=True",
        "--normalize_input=False",
        "--encoder_conv_architecture=layer2_resnet18",
    ]
    fresh = parse_dmlab_args(argv)
    assert fresh.depth_sensor_inverse is True
    assert fresh.checkpoint_frame_targets == "auto"
    assert fresh.dmlab_level_cache_path == str(tmp_path / "runtime" / "dmlab_cache")
    assert DepthEncoder(fresh).depth_mode == "capped_inverse"

    # A pre-switch run has no depth_sensor_inverse field in config.json.
    with open(cfg_file(fresh), "w") as stream:
        json.dump({"depth_sensor": True}, stream)
    resumed = load_from_checkpoint(fresh)
    assert resumed.depth_sensor_inverse is None
    assert resumed.checkpoint_frame_targets == ""
    assert DepthEncoder(resumed).depth_mode == "legacy"

    timed = parse_dmlab_args([*argv, "--save_milestones_sec=1800"])
    assert timed.checkpoint_frame_targets == ""

    explicit = parse_dmlab_args([*argv, "--depth_sensor_inverse=True"])
    assert load_from_checkpoint(explicit).depth_sensor_inverse is True


@pytest.mark.parametrize("inverse,expected", [(None, 1.5), (True, 5.0), (False, 2.0)])
def test_saved_mode_gain_compatibility_and_explicit_toggle_precedence(inverse, expected):
    cfg = SimpleNamespace(
        encoder_conv_architecture="layer2_resnet18",
        depth_sensor_mode="capped_inverse",
        depth_sensor_gain=3.0,
        depth_sensor_inverse=inverse,
    )
    actual = DepthEncoder(cfg)(torch.full((1, 1, 1, 10), 2.0))
    torch.testing.assert_close(actual, torch.full_like(actual, expected))
