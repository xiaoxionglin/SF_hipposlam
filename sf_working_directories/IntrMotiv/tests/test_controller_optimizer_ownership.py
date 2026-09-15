import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sf_working_directories.IntrMotiv.dmlab.controller_snapshot import ControllerSnapshot, differentiable_replay


class Model(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.cfg = SimpleNamespace(ppo_dg_gradient=mode)
        self.encoder = nn.Module()
        self.encoder.DG_projection = nn.Linear(2, 2, bias=False)
        self.worker = nn.Linear(2, 1, bias=False)

    def forward(self, x):
        # Like a stopped DG slice plus a live bypass in a concatenated head.
        head = torch.cat((self.encoder.DG_projection(x), x), -1)
        dg = head[:, :2]
        if self.cfg.ppo_dg_gradient == "stop":
            dg = dg.detach()
        return self.worker(dg + head[:, 2:])


def equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    else:
        assert a == b


@pytest.mark.parametrize("mode", ["stop", "joint"])
def test_replay_optimizer_preserves_stop_dg_momentum_owner(mode):
    torch.manual_seed(99)
    source = Model(mode).eval()
    optimizer = torch.optim.Adam(source.parameters(), lr=0.01)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    # Seed real Adam momentum using a fresh DG training step.
    source.encoder.DG_projection(x).square().mean().backward()
    optimizer.step()
    p = source.encoder.DG_projection.weight
    original = p.detach().clone()
    state = copy.deepcopy(optimizer.state[p])
    worker = source.worker.weight.detach().clone()
    for version in range(3):
        snapshot = ControllerSnapshot(lambda: Model(mode), source, version)
        optimizer.zero_grad(set_to_none=True)
        loss = (
            differentiable_replay(snapshot, source, lambda model, x: model(x), x, source_version=version)
            .square()
            .mean()
        )
        loss.backward()
        if mode == "stop":
            assert p.grad is None
        else:
            assert p.grad is not None and p.grad.count_nonzero() > 0
        optimizer.step()
    assert not torch.equal(worker, source.worker.weight)
    if mode == "stop":
        assert torch.equal(original, p)
        equal(state, optimizer.state[p])
    else:
        assert not torch.equal(original, p)


def test_fresh_dg_step_preserves_worker_optimizer_state():
    from sf_working_directories.IntrMotiv.dmlab.controller_snapshot import fresh_dg_parameter_owner

    source = Model("stop").eval()
    optimizer = torch.optim.Adam(source.parameters(), lr=0.01)
    sum(p.sum() for p in source.parameters()).backward()
    optimizer.step()
    original = source.worker.weight.detach().clone()
    state = copy.deepcopy(optimizer.state[source.worker.weight])
    dg = source.encoder.DG_projection.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    with fresh_dg_parameter_owner(source):
        x = torch.ones(2, 2)
        mixed = torch.cat((source.encoder.DG_projection(x), source.worker(x)), -1)
        mixed[:, :2].square().mean().backward()
        assert source.worker.weight.grad is None
        optimizer.step()
    assert source.worker.weight.requires_grad
    assert torch.equal(original, source.worker.weight)
    equal(state, optimizer.state[source.worker.weight])
    assert not torch.equal(dg, source.encoder.DG_projection.weight)


def test_fresh_owner_restores_flags_after_exception():
    from sf_working_directories.IntrMotiv.dmlab.controller_snapshot import fresh_dg_parameter_owner

    model = Model("stop")
    model.worker.weight.requires_grad_(False)
    before = [p.requires_grad for p in model.parameters()]
    with pytest.raises(RuntimeError):
        with fresh_dg_parameter_owner(model):
            raise RuntimeError("test")
    assert before == [p.requires_grad for p in model.parameters()]
