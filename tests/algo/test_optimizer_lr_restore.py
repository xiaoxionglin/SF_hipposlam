"""Exercise real learner initialization and restart with multiple LR groups."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from sample_factory.algo.learning.learner import BaseLearner
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.utils.attr_dict import AttrDict


class TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Parameter(torch.tensor([1.0]))
        self.second = nn.Parameter(torch.tensor([2.0]))
        self.first._intrmotiv_lr_scale = 0.25

    def model_to_device(self, device):
        self.to(device)


def config(path, enabled=None):
    argv = [
        "--env=lr_restore_test",
        "--experiment=restart",
        f"--train_dir={path}",
        "--device=cpu",
        "--serial_mode=true",
        "--learning_rate=0.001",
        "--lr_schedule=constant",
        "--seed=7",
    ]
    if enabled is not None:
        argv.append(f"--preserve_loaded_optimizer_lr={enabled}")
    parser, _ = parse_sf_args(argv)
    return AttrDict(vars(parse_full_cfg(parser, argv)))


def initialize(path, enabled, monkeypatch):
    monkeypatch.setattr("sample_factory.algo.learning.learner.create_actor_critic", lambda *args: TinyActor())
    cfg = config(path, enabled)
    info = EnvInfo(
        gym.spaces.Dict(obs=gym.spaces.Box(-1, 1, (1,), dtype=np.float32)),
        gym.spaces.Discrete(2),
        1,
        False,
        False,
        None,
        True,
        1,
    )
    versions = torch.zeros(1, dtype=torch.int32)
    learner = BaseLearner(cfg, info, versions, 0, ParameterServer(0, versions, serial_mode=True))
    learner.init()
    return learner


@pytest.mark.parametrize("enabled", [None, False, True])
def test_fresh_initialization_keeps_nominal_group_rates(tmp_path, monkeypatch, enabled):
    learner = initialize(tmp_path, enabled, monkeypatch)
    assert [g["lr"] for g in learner.optimizer.param_groups] == [0.00025, 0.001]
    assert not learner._optimizer_state_loaded


@pytest.mark.parametrize("enabled", [None, False, True])
def test_actual_restart_retains_effective_rates_only_when_enabled(tmp_path, monkeypatch, enabled):
    original = initialize(tmp_path, False, monkeypatch)
    sum(p.square().sum() for p in original.actor_critic.parameters()).backward()
    original.optimizer.step()
    saved_rates = [0.000111, 0.000777]
    for group, rate in zip(original.optimizer.param_groups, saved_rates):
        group["lr"] = rate
    original.train_step = 17
    original.env_steps = 80
    checkpoint = original._get_checkpoint_dict()
    torch.save(checkpoint, Path(original.checkpoint_dir(original.cfg, 0)) / "checkpoint_000000017.pth")

    resumed = initialize(tmp_path, enabled, monkeypatch)
    assert resumed._optimizer_state_loaded
    assert resumed.train_step == 17 and resumed.env_steps == 80
    assert resumed.curr_lr == original.curr_lr == 0.001
    assert [g["lr"] for g in resumed.optimizer.param_groups] == (saved_rates if enabled else [0.00025, 0.001])
    for key, value in original.actor_critic.state_dict().items():
        assert torch.equal(resumed.actor_critic.state_dict()[key], value)
    for left, right in zip(original.optimizer.state.values(), resumed.optimizer.state.values()):
        for key in left:
            assert torch.equal(left[key], right[key])

    # The option affects initialization only. An explicit later configuration
    # update still applies a new nominal LR with the existing group scales.
    resumed.set_new_cfg({"learning_rate": 0.002})
    resumed._maybe_update_cfg()
    assert resumed.curr_lr == 0.002
    assert [g["lr"] for g in resumed.optimizer.param_groups] == [0.0005, 0.002]


def test_cli_false_really_disables_restore(tmp_path):
    assert config(tmp_path).preserve_loaded_optimizer_lr is False
    assert config(tmp_path, True).preserve_loaded_optimizer_lr is True
    assert config(tmp_path, False).preserve_loaded_optimizer_lr is False
