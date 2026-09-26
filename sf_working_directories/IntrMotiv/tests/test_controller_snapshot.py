import pytest
import torch
from torch import nn

from sf_working_directories.IntrMotiv.dmlab.controller_snapshot import ControllerSnapshot, differentiable_replay
from sf_working_directories.IntrMotiv.dmlab.custom_encoder import DGProjection_batchnorm_relu


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dg = DGProjection_batchnorm_relu(4, 3, intercept=-2.0, batchnorm_semantics="legacy_batch")
        self.head = nn.Linear(3, 2)
        self.register_buffer("graph_evidence", torch.tensor([4.0]))

    def forward(self, x):
        return self.head(self.dg(x))


def test_replay_preserves_live_bn_graph_cache_and_routes_gradients():
    model = SmallModel().train()
    model(torch.randn(8, 4))
    old_cache = model.dg.last_pre_threshold_logits
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    snap = ControllerSnapshot(SmallModel, model, 0)
    assert torch.equal(rng, torch.get_rng_state())
    x = torch.randn(5, 4)
    result = differentiable_replay(snap, model, lambda m, x: m(x), x, source_version=0)
    result.square().mean().backward()
    assert model.dg.linear.weight.grad.abs().sum() > 0
    assert model.dg.last_pre_threshold_logits is old_cache
    assert model.training and model.dg.training
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    assert all(p.grad is None for p in snap.model.parameters())


def test_replay_cannot_accumulate_graph_evidence_even_if_operation_mutates_buffer():
    model = SmallModel()
    snap = ControllerSnapshot(SmallModel, model, 1)

    def mistaken_operation(m, x):
        m.graph_evidence.add_(1.0)
        return m(x)

    differentiable_replay(snap, model, mistaken_operation, torch.ones(4, 4), source_version=1)
    assert model.graph_evidence.item() == 4.0
    assert snap.model.graph_evidence.item() == 4.0


def test_target_owns_representation_and_normalization_until_explicit_refresh():
    model = SmallModel()
    snap = ControllerSnapshot(SmallModel, model, 1)
    x = torch.ones(4, 4)
    before = snap.model(x).clone()
    with torch.no_grad():
        model.dg.linear.weight.add_(2.0)
        model.dg.batchnorm1d.running_mean.add_(3.0)
    torch.testing.assert_close(snap.model(x), before, rtol=0, atol=0)
    snap.refresh(model, 2)
    assert not torch.equal(snap.model(x), before)
    with pytest.raises(ValueError, match="advance"):
        snap.refresh(model, 2)


def test_online_snapshot_version_mismatch_is_rejected():
    model = SmallModel()
    snap = ControllerSnapshot(SmallModel, model, 1)
    with pytest.raises(ValueError, match="same version"):
        differentiable_replay(snap, model, lambda m, x: m(x), torch.ones(4, 4), source_version=2)


def test_replay_does_not_advance_parent_random_stream():
    model = SmallModel()
    snap = ControllerSnapshot(SmallModel, model, 0)
    before = torch.get_rng_state().clone()
    differentiable_replay(snap, model, lambda m, x: m(x) + torch.rand(1), torch.ones(4, 4), source_version=0)
    assert torch.equal(before, torch.get_rng_state())
