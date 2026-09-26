import torch
from torch import nn

import sf_working_directories.IntrMotiv.dmlab.controller_history as module


class BatchSensitiveHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.sizes = []

    def forward_head(self, obs):
        self.sizes.append(len(obs["x"]))
        assert torch.all(obs["instruction"] > 0)
        return self.scale * obs["x"] + (len(obs["x"]) - 256) * 1e-7


def test_reconstruction_head_is_invariant_to_regrouping(monkeypatch):
    monkeypatch.setattr(module, "prepare_and_normalize_obs", lambda model, obs: obs)
    model = BatchSensitiveHead().eval()
    x = torch.linspace(-1, 1, 513).reshape(-1, 1)
    obs = {"x": x, "instruction": torch.ones(513, 1, dtype=torch.long)}
    whole = module.reconstruction_head(model, obs)
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(99))
    grouped = []
    for ids in order.split(71):
        grouped.append(module.reconstruction_head(model, {k: v[ids] for k, v in obs.items()}))
    restored = torch.empty_like(whole)
    restored[order] = torch.cat(grouped)
    assert torch.equal(whole, restored)
    assert set(model.sizes) == {256}


def test_padding_does_not_add_gradient_positions(monkeypatch):
    monkeypatch.setattr(module, "prepare_and_normalize_obs", lambda model, obs: obs)
    model = BatchSensitiveHead().eval()
    x = torch.arange(1.0, 301.0).reshape(-1, 1)
    result = module.reconstruction_head(model, {"x": x, "instruction": torch.ones(300, 1, dtype=torch.long)})
    result.sum().backward()
    assert result.shape == x.shape
    assert model.scale.grad == x.sum()
