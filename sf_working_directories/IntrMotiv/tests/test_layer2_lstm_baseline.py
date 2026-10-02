"""Matched sparse/dense layer2 ResNet + LSTM baseline contracts."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from sample_factory.algo.utils.model_context import global_model_factory, reset_global_model_context
from sample_factory.model.core import ModelCoreRNN
from sf_working_directories.IntrMotiv.dmlab import custom_encoder
from sf_working_directories.IntrMotiv.dmlab.custom_actor_critic import make_hipposlam_actor_critic
from sf_working_directories.IntrMotiv.dmlab.custom_core import LstmDGWithBypassCore, make_hipposlam_core
from sf_working_directories.IntrMotiv.dmlab.custom_decoder import make_hipposlam_decoder
from sf_working_directories.IntrMotiv.dmlab.train_hipposlam import parse_dmlab_args


def _baseline_cfg(tmp_path, mode, *extra):
    return parse_dmlab_args(
        [
            "--env=openfield_map2_cued_reward5",
            "--experiment=layer2_lstm_test",
            f"--train_dir={tmp_path}",
            "--device=cpu",
            f"--layer2_lstm_baseline={mode}",
            "--depth_sensor=True",
            "--normalize_input=False",
            "--reward_instruction_count=5",
            "--rnn_size=0",
            *extra,
        ]
    )


@pytest.mark.parametrize("mode", ["sparse", "dense"])
def test_layer2_lstm_baseline_keeps_inputs_and_trains_projection_and_lstm(tmp_path, monkeypatch, mode):
    # Check the pretrained request without a network/cache dependency.
    original_resnet18 = custom_encoder.models.resnet18
    requested_weights = []

    def local_resnet18(*, weights=None, pretrained=None):
        requested_weights.append(weights)
        return original_resnet18(weights=None)

    monkeypatch.setattr(custom_encoder.models, "resnet18", local_resnet18)
    cfg = _baseline_cfg(tmp_path, mode)
    assert cfg.rnn_size == cfg.cli_args["rnn_size"] == 256
    assert cfg.rnn_persistent_state_size == 0
    assert (cfg.encoder_conv_architecture, cfg.DG_name, cfg.core_name, cfg.rnn_type) == (
        "layer2_resnet18",
        "batchnorm_relu",
        "LstmDGBypass",
        "lstm",
    )
    assert cfg.ppo_dg_gradient == "joint"
    assert not cfg.distance_learning and not cfg.online_spatial_telemetry
    assert cfg.advantage_reward_source == "external"

    reset_global_model_context()
    factory = global_model_factory()
    factory.register_encoder_factory(custom_encoder.make_hipposlam_encoder)
    factory.register_model_core_factory(make_hipposlam_core)
    factory.register_decoder_factory(make_hipposlam_decoder)
    obs_space = gym.spaces.Dict(
        {
            "obs": gym.spaces.Box(0, 255, (4, cfg.res_h, cfg.res_w), dtype=np.uint8),
            "INSTR": gym.spaces.Box(0, 255, (1,), dtype=np.int32),
        }
    )
    torch.manual_seed(7)
    actor = make_hipposlam_actor_critic(cfg, obs_space, gym.spaces.Discrete(9))
    assert requested_weights == [custom_encoder.models.ResNet18_Weights.IMAGENET1K_V1]
    assert isinstance(actor.encoder.basic_encoder, custom_encoder.ResNet18Layer2)
    assert isinstance(actor.core, LstmDGWithBypassCore)
    assert isinstance(actor.core.dg_core, ModelCoreRNN)
    assert isinstance(actor.core.dg_core.core, torch.nn.LSTM)
    assert actor.encoder.get_out_size() == 64 + 10 + 3 + 5
    assert not any(p.requires_grad for p in actor.encoder.basic_encoder.parameters())
    actor.train()
    assert not actor.encoder.basic_encoder.features.training

    batch = 32  # Enough examples for a threshold-2 sparse DG to activate.
    torch.manual_seed(11)
    obs = {
        "obs": torch.rand(batch, 4, cfg.res_h, cfg.res_w),
        "INSTR": torch.randint(1, 6, (batch, 1)),
    }
    head = actor.forward_head(obs)
    assert head.shape == (batch, 82)
    expected_depth = actor.encoder.depth_encoder(obs["obs"][:, -1:]).flatten(1)
    torch.testing.assert_close(head[:, 64:74], expected_depth)
    torch.testing.assert_close(head[:, 74:77], torch.tensor([0.0, 0.0, 1.0]).expand(batch, -1))
    torch.testing.assert_close(
        head[:, 77:], torch.nn.functional.one_hot(obs["INSTR"].flatten() - 1, 5).float()
    )
    if mode == "sparse":
        assert (head[:, :64] == 0).float().mean() > 0.8
        assert (head[:, :64] > 0).any()
    else:
        assert (head[:, :64] < 0).any() and (head[:, :64] > 0).any()
        assert (head[:, :64] == 0).sum() == 0

    core_out, state = actor.forward_core(head, torch.zeros(batch, 2 * cfg.rnn_size))
    assert core_out.shape == (batch, 256 + 18) and state.shape == (batch, 512)
    torch.testing.assert_close(core_out[:, 256:], head[:, 64:])
    actor.forward_tail(core_out, values_only=False, sample_actions=True)["values"].sum().backward()
    assert actor.core.dg_core.core.weight_ih_l0.grad.norm() > 0
    assert actor.encoder.DG_projection.linear.weight.grad.norm() > 0


def test_baseline_rejects_intrinsic_and_ca3_specific_training(tmp_path):
    with pytest.raises(ValueError, match="distance_learning=False"):
        _baseline_cfg(tmp_path, "dense", "--distance_learning=True")
    with pytest.raises(ValueError, match="hrl_controllable_graph=False"):
        _baseline_cfg(tmp_path, "sparse", "--hrl_controllable_graph=True")
    with pytest.raises(ValueError, match="online_spatial_telemetry=False"):
        _baseline_cfg(tmp_path, "dense", "--online_spatial_telemetry=True")


def test_lstm_baseline_packed_replay_keeps_bypass_at_each_step(tmp_path):
    cfg = _baseline_cfg(tmp_path, "dense")
    core = LstmDGWithBypassCore(cfg, input_size=82)
    values = torch.randn(3, 2, 82)
    packed = torch.nn.utils.rnn.pack_padded_sequence(values, [3, 2], enforce_sorted=False)
    output, state = core(packed, torch.zeros(2, 2 * cfg.rnn_size))
    assert output.data.shape == (5, 256 + 18)
    assert state.shape == (2, 512)
    torch.testing.assert_close(output.data[:, 256:], packed.data[:, 64:])


def test_dense_projection_removes_only_the_sparse_activation():
    torch.manual_seed(17)
    sparse = custom_encoder.DGProjection_batchnorm_relu(4, 8, intercept=2)
    dense = custom_encoder.DGProjection_batchnorm_relu(4, 8, intercept=2, output_activation="identity")
    dense.load_state_dict(sparse.state_dict(), strict=True)
    sparse.eval()
    dense.eval()
    features = torch.randn(6, 4)
    signed = dense(features)
    torch.testing.assert_close(sparse(features), torch.relu(signed - 2))
    assert (signed < 0).any()


def test_legacy_architecture_defaults_are_unchanged(tmp_path):
    cfg = parse_dmlab_args(
        ["--env=openfield_map2_fixed_loc3", "--experiment=legacy", f"--train_dir={tmp_path}"]
    )
    assert cfg.layer2_lstm_baseline == "off"
    assert cfg.encoder_conv_architecture == "convnet_impala"
    assert cfg.distance_learning and cfg.online_spatial_telemetry


def test_false_distance_recording_flag_parses_as_false(tmp_path):
    cfg = _baseline_cfg(tmp_path, "sparse", "--rec_distances=False")
    assert cfg.rec_distances is False


def test_baseline_switch_updates_inherited_architecture_cli_metadata(tmp_path):
    cfg = _baseline_cfg(
        tmp_path,
        "dense",
        "--core_name=BypassSS",
        "--encoder_conv_architecture=convnet_impala",
        "--rnn_type=gru",
        "--ppo_dg_gradient=stop",
    )
    for name, expected in {
        "core_name": "LstmDGBypass",
        "encoder_conv_architecture": "layer2_resnet18",
        "rnn_type": "lstm",
        "ppo_dg_gradient": "joint",
    }.items():
        assert getattr(cfg, name) == cfg.cli_args[name] == expected
