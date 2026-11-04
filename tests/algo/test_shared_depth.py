"""Compatibility of shared depth sampling and reversible family CLI switches."""

import argparse
import importlib
import json
from types import SimpleNamespace

import pytest
import torch

from sample_factory.cfg.arguments import checkpoint_override_defaults, load_from_checkpoint
from sample_factory.utils.utils import cfg_file
from sf_shared.depth import SampledDepthEncoder

FAMILIES = (
    "sf_xxl.dmlab",
    "sf_working_directories.default.dmlab",
    "sf_working_directories.jannek.dmlab",
    "sf_working_directories.IntrMotiv.dmlab",
)


@pytest.fixture(params=FAMILIES)
def family(request):
    return request.param


@pytest.fixture(params=("shared", *FAMILIES))
def encoder_family(request):
    return request.param


def family_parser(family):
    parser = argparse.ArgumentParser()
    importlib.import_module(f"{family}.custom_params").add_hipposlam_env_args(parser)
    return parser


def depth_encoder_class(family):
    if family == "shared":
        return SampledDepthEncoder
    # Adapter modules also contain visual encoders. Keep exercising their real
    # classes locally, without making TorchVision a dependency of core CI tests.
    pytest.importorskip("torchvision", reason="Full visual encoder adapters require TorchVision")
    return importlib.import_module(f"{family}.custom_encoder").DepthEncoder


def depth_encoder(family, **kwargs):
    cfg = SimpleNamespace(encoder_conv_architecture="pretrained_resnet", **kwargs)
    return depth_encoder_class(family)(cfg)


@pytest.mark.parametrize("inverse", [None, False])
def test_legacy_depth_is_exact_and_state_dict_compatible(encoder_family, inverse):
    config = dict(normalize_input=True, obs_scale=255.0, obs_subtract_mean=127.5)
    if inverse is not None:
        config["depth_sensor_inverse"] = inverse
    encoder = depth_encoder(encoder_family, **config)
    obs = torch.linspace(-1, 1, 2 * 72 * 96, dtype=torch.float64).reshape(2, 1, 72, 96)
    original = obs.clone()
    torch.testing.assert_close(encoder(obs), torch.nn.functional.interpolate(obs, size=(1, 10)), rtol=0, atol=0)
    torch.testing.assert_close(obs, original, rtol=0, atol=0)
    assert encoder.get_out_size() == 10
    assert list(encoder.named_children())[0][0] == "downsample"
    assert not encoder.state_dict()
    encoder.load_state_dict({}, strict=True)
    expected_module = "sf_shared.depth" if encoder_family == "shared" else f"{encoder_family}.custom_encoder"
    assert type(encoder).__module__ == expected_module


@pytest.mark.parametrize("scale,mean", [(1.0, 0.0), (255.0, 0.0), (255.0, 127.5)])
def test_inverse_depth_matches_raw_codes_after_fixed_preprocessing(encoder_family, scale, mean):
    encoder = depth_encoder(
        encoder_family, depth_sensor_inverse=True, normalize_input=False, obs_scale=scale, obs_subtract_mean=mean
    )
    raw = torch.tensor([0.0, 0.5, 1.0, 2.0, 10.0, 30.0, 50.0, 100.0, 150.0, 255.0], dtype=torch.float64)
    raw = raw.reshape(1, 1, 1, 10).expand(3, -1, -1, -1)
    actual = encoder((raw - mean) / scale)
    torch.testing.assert_close(actual, 10.0 / raw.clamp_min(1.0))
    assert actual.shape == (3, 1, 1, 10)
    assert actual.dtype == raw.dtype
    assert torch.isfinite(actual).all()


def test_inverse_depth_rejects_running_observation_normalization(encoder_family):
    with pytest.raises(ValueError, match="normalize_input=False"):
        depth_encoder(encoder_family, depth_sensor_inverse=True, normalize_input=True, normalize_input_keys=["obs"])


def test_depth_architecture_support_stays_family_specific(family):
    encoder_type = depth_encoder_class(family)
    cfg = SimpleNamespace(encoder_conv_architecture="layer2_resnet18")
    if "IntrMotiv" in family:
        assert encoder_type(cfg).get_out_size() == 10
    else:
        with pytest.raises(NotImplementedError, match="layer2_resnet18"):
            encoder_type(cfg)


def test_family_boolean_switches_are_reversible_without_changing_defaults(family):
    parser = family_parser(family)
    expected = dict(
        simple_sequence=False,
        DG_batch_q=None,
        DG_softmax=None,
        fix_encoder_when_load=True,
        depth_sensor=False,
        dmlab_reduced_action_set=False,
        depth_sensor_inverse="IntrMotiv" in family,
    )
    if family != "sf_xxl.dmlab":
        expected["rec_distances"] = None
    if "IntrMotiv" in family:
        expected.update(shared_dg_projection=False, recover_empty_ppo_minibatches=False)
    defaults = parser.parse_args([])
    for name, default in expected.items():
        assert getattr(defaults, name) is default, name
        assert getattr(parser.parse_args([f"--{name}=False"]), name) is False, name
        assert getattr(parser.parse_args([f"--{name}=True"]), name) is True, name
    # Prevent a later added boolean argument from silently restoring bool("False").
    assert not any(action.type is bool for action in parser._actions)


@pytest.mark.parametrize("explicit", [None, False, True])
def test_old_saved_config_preserves_legacy_depth_and_explicit_override(family, explicit, tmp_path):
    parser = family_parser(family)
    argv = [] if explicit is None else [f"--depth_sensor_inverse={explicit}"]
    fresh = parser.parse_args(argv)
    fresh.train_dir = str(tmp_path)
    fresh.experiment = "legacy_depth"
    fresh.cli_args = {} if explicit is None else {"depth_sensor_inverse": explicit}
    saved = {
        "depth_sensor": True,
        "encoder_conv_architecture": "pretrained_resnet",
        "normalize_input": False,
        # The saved effective value remains authoritative for old bool parsing.
        "command_line": "--depth_sensor=False",
    }
    with open(cfg_file(fresh), "w") as stream:
        json.dump(saved, stream)
    resumed = load_from_checkpoint(fresh)
    assert resumed.depth_sensor_inverse is explicit
    assert resumed.depth_sensor is True
    assert SampledDepthEncoder(resumed).depth_mode == ("capped_inverse" if explicit else "legacy")
    checkpoint_override_defaults(fresh, parser)
    restored = parser.parse_args(argv)
    assert restored.depth_sensor_inverse is explicit
    assert restored.depth_sensor is True


@pytest.mark.parametrize("saved_inverse", [None, False, True])
def test_saved_depth_choice_is_not_replaced_by_fresh_default(family, saved_inverse, tmp_path):
    parser = family_parser(family)
    fresh = parser.parse_args([])
    fresh.train_dir = str(tmp_path)
    fresh.experiment = "saved_depth"
    fresh.cli_args = {}
    with open(cfg_file(fresh), "w") as stream:
        json.dump({"depth_sensor_inverse": saved_inverse}, stream)
    assert load_from_checkpoint(fresh).depth_sensor_inverse is saved_inverse
    checkpoint_override_defaults(fresh, parser)
    assert parser.parse_args([]).depth_sensor_inverse is saved_inverse
