"""Real encoder-factory selection and audit capture with a tiny local visual trunk."""

import argparse
import json
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from sf_shared.dg import DGProjection_batchnorm_relu as SharedDG
from sf_working_directories.IntrMotiv.dmlab import custom_encoder
from sf_working_directories.IntrMotiv.dmlab.custom_params import add_hipposlam_env_args
from sf_working_directories.IntrMotiv.evaluation import normalization_gradient_audit as normalization_audit


class TinyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(-0.5, 0.5, 60).reshape(5, 12))

    def forward(self, images):
        return images.flatten(1) @ self.weight.T

    def get_out_size(self):
        return 5


@pytest.fixture(autouse=True)
def tiny_visual(monkeypatch):
    monkeypatch.setattr(custom_encoder, "make_img_encoder", lambda cfg, space: TinyVisual())


def config(flag="default", **overrides):
    parser = argparse.ArgumentParser()
    add_hipposlam_env_args(parser)
    args = [] if flag in ("default", "missing") else [f"--shared_dg_projection={flag}"]
    cfg = parser.parse_args(args)
    if flag == "missing":
        del cfg.shared_dg_projection
    vars(cfg).update(
        encoder_name="Hipposlam",
        Hippo_n_feature=4,
        DG_name="batchnorm_relu",
        DG_BN_intercept=0.2,
        core_name="BypassSS",
        res_h=2,
        res_w=2,
        with_number_instruction=True,
        number_instruction_coef=1.0,
        depth_sensor=False,
        dg_batchnorm_semantics="running_poststep_atomic",
    )
    vars(cfg).update(overrides)
    return cfg


def make_encoder(flag="default", **overrides):
    cfg = config(flag, **overrides)
    space = gym.spaces.Dict(
        {
            "obs": gym.spaces.Box(0, 255, (3, 2, 2), dtype=np.uint8),
            "INSTR": gym.spaces.Box(1, 3, (1,), dtype=np.int32),
        }
    )
    return custom_encoder.make_hipposlam_encoder(cfg, space)


def observations(batch=6):
    return {
        "obs": torch.linspace(-1, 1, batch * 12).reshape(batch, 3, 2, 2),
        "INSTR": (torch.arange(batch) % 3 + 1)[:, None],
    }


def assert_state_equal(left, right):
    assert left.state_dict().keys() == right.state_dict().keys()
    for key, value in left.state_dict().items():
        torch.testing.assert_close(value, right.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("flag", ("default", "missing", "False", "True"))
def test_real_factory_selects_requested_class_and_legacy_default(flag):
    encoder = make_encoder(flag)
    assert isinstance(encoder, custom_encoder.HipposlamEncoder)
    expected = SharedDG if flag == "True" else custom_encoder.DGProjection_batchnorm_relu
    assert type(encoder.DG_projection) is expected
    assert encoder.get_out_size() == 12
    if flag == "default":
        assert encoder.cfg.shared_dg_projection is False


@pytest.mark.parametrize(
    "mode", ("legacy_batch", "running_consistent", "running_poststep_atomic", "input_centered_atomic")
)
@pytest.mark.parametrize("context", ("none", "gate", "additive"))
def test_real_encoder_rng_output_state_gradients_and_cross_load_match(mode, context):
    torch.manual_seed(71)
    legacy = make_encoder("False", dg_batchnorm_semantics=mode, dg_context_feedback=context)
    legacy_rng = torch.get_rng_state().clone()
    torch.manual_seed(71)
    shared = make_encoder("True", dg_batchnorm_semantics=mode, dg_context_feedback=context)
    assert torch.equal(torch.get_rng_state(), legacy_rng)
    assert_state_equal(legacy, shared)
    outputs, input_gradients = [], []
    for encoder in (legacy, shared):
        obs = observations()
        obs["obs"].requires_grad_()
        with encoder.DG_projection.running_stats_update(True):
            output = encoder(obs)
        output.square().mean().backward()
        outputs.append(output.detach())
        input_gradients.append(obs["obs"].grad)
        if mode.endswith("atomic"):
            encoder.DG_projection.post_step_update_running_stats()
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(input_gradients[0], input_gradients[1], rtol=0, atol=0)
    for (name, old), (_, new) in zip(legacy.named_parameters(), shared.named_parameters()):
        assert old.grad is not None, name
        torch.testing.assert_close(old.grad, new.grad, rtol=0, atol=0)
    assert_state_equal(legacy, shared)
    shared.load_state_dict(legacy.state_dict(), strict=True)
    legacy.load_state_dict(shared.state_dict(), strict=True)
    legacy.eval()
    shared.eval()
    torch.testing.assert_close(legacy(observations(1)), shared(observations(1)), rtol=0, atol=0)


@pytest.mark.parametrize("context", ("none", "gate"))
@pytest.mark.parametrize("initialize_live", (False, True))
def test_frozen_reference_uses_selected_class_and_stays_frozen(tmp_path, context, initialize_live):
    torch.manual_seed(82)
    source = make_encoder("False")
    path = tmp_path / "reference.pth"
    torch.save({"model": {"encoder." + key: value for key, value in source.state_dict().items()}}, path)
    outputs = []
    for flag in ("False", "True"):
        torch.manual_seed(93)
        encoder = make_encoder(
            flag,
            intrinsic_goal_reference_checkpoint=str(path),
            intrinsic_goal_initialize_live_from_reference=initialize_live,
            dg_context_feedback=context,
        )
        reference = encoder.goal_reference_projection
        assert type(reference) is type(encoder.DG_projection)
        assert not any(p.requires_grad for p in reference.parameters())
        assert_state_equal(reference, source.DG_projection)
        if initialize_live:
            assert_state_equal(encoder.DG_projection, source.DG_projection)
        before = {key: value.clone() for key, value in reference.state_dict().items()}
        encoder.train()
        with encoder.DG_projection.running_stats_update(True):
            output = encoder(observations())
        output.square().mean().backward()
        encoder.DG_projection.post_step_update_running_stats()
        assert output.shape == (6, encoder.get_out_size()) == (6, 16)
        assert not reference.training
        assert all(p.grad is None for p in reference.parameters())
        for key, value in reference.state_dict().items():
            assert torch.equal(value, before[key])
        expected = reference(encoder.projection_input(observations()).detach())
        torch.testing.assert_close(output[:, 4:8], expected, rtol=0, atol=0)
        outputs.append(output)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


@pytest.mark.parametrize("flag", ("False", "True"))
@pytest.mark.parametrize("context", ("none", "gate"))
def test_normalization_audit_captures_live_backend_once_and_preserves_results(tmp_path, monkeypatch, flag, context):
    torch.manual_seed(13)
    encoder = make_encoder(flag, dg_context_feedback=context, DG_BN_intercept=-0.2).eval()
    # A second same-type detector must not double the captured sample count.
    import copy

    encoder.goal_reference_projection = copy.deepcopy(encoder.DG_projection).requires_grad_(False).eval()
    projection = encoder.DG_projection
    before = {key: value.clone() for key, value in encoder.state_dict().items()}
    obs = observations(6)
    expected_features = encoder.projection_input(obs).detach()[:5]
    expected = normalization_audit.audit(expected_features, projection, 2e-4)
    checkpoint = tmp_path / "checkpoint.pth"

    def fake_loader(*args, **kwargs):
        return encoder.cfg, None, None, SimpleNamespace(encoder=encoder), checkpoint, torch.device("cpu")

    def fake_rollout(*args, **kwargs):
        assert kwargs == {"max_num_frames": 9, "deterministic": False, "checkpoint_rank": 2}
        loaded = normalization_audit.place_fields.load_policy_env(*args, **kwargs)
        with torch.no_grad():
            for start in (0, 3):
                encoder({key: value[start : start + 3] for key, value in obs.items()})
        return loaded[0], loaded[4]

    monkeypatch.setattr(normalization_audit.place_fields, "load_policy_env", fake_loader)
    monkeypatch.setattr(normalization_audit.place_fields, "rollout_dg", fake_rollout)
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--run-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--max-num-frames",
            "9",
            "--max-features",
            "5",
            "--checkpoint-rank",
            "2",
        ],
    )
    normalization_audit.main()
    result = json.loads(output.read_text())
    assert result == {**expected, "run_dir": str(tmp_path), "checkpoint": str(checkpoint)}
    assert result["feature_count"] == 5
    assert "preactivation" not in projection.__dict__
    assert normalization_audit.place_fields.load_policy_env is fake_loader
    for key, value in encoder.state_dict().items():
        assert torch.equal(value, before[key])


@pytest.mark.parametrize("flag", ("False", "True"))
def test_normalization_audit_restores_instance_and_loader_on_rollout_failure(tmp_path, monkeypatch, flag):
    encoder = make_encoder(flag).eval()
    original_forward = type(encoder.DG_projection).forward

    def fake_loader(*args, **kwargs):
        return encoder.cfg, None, None, SimpleNamespace(encoder=encoder), tmp_path / "checkpoint.pth", "cpu"

    def failed_rollout(*args, **kwargs):
        normalization_audit.place_fields.load_policy_env(*args, **kwargs)
        encoder(observations())
        raise RuntimeError("rollout failed")

    monkeypatch.setattr(normalization_audit.place_fields, "load_policy_env", fake_loader)
    monkeypatch.setattr(normalization_audit.place_fields, "rollout_dg", failed_rollout)
    with pytest.raises(RuntimeError, match="rollout failed"):
        normalization_audit._capture_rollout_features(tmp_path, 9, 5, 0)
    assert "preactivation" not in encoder.DG_projection.__dict__
    assert type(encoder.DG_projection).forward is original_forward
    assert normalization_audit.place_fields.load_policy_env is fake_loader
