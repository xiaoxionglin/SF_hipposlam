"""Run the actual IntrMotiv update loop around a small differentiable model."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.timing import Timing
from sf_working_directories.IntrMotiv.dmlab.custom_learner import DistanceLearnerReward, make_hipposlam_learner
from sf_working_directories.IntrMotiv.dmlab.iterative_update import DECODER


class RecordingScheduler:
    def __init__(self, per_minibatch=False):
        self.per_minibatch = per_minibatch
        self.calls = []

    def invoke_after_each_minibatch(self):
        return self.per_minibatch

    def invoke_after_each_epoch(self):
        return not self.per_minibatch

    def update(self, lr, kls):
        self.calls.append(list(kls))
        return lr


class UpdateHarness:
    """Replace environment/model loss construction, retaining real SGD control flow."""

    def __init__(self, counts, recover=True, *, fixed_losses=None, per_minibatch=False):
        self.counts = counts
        self.fixed_losses = fixed_losses
        self.cfg = AttrDict(
            num_batches_per_epoch=len(counts[0]),
            num_epochs=len(counts),
            with_vtrace=False,
            recurrence=1,
            rollout=1,
            max_grad_norm=0.0,
            serial_mode=True,
            recover_empty_ppo_minibatches=recover,
        )
        self.timing = Timing(name="recovery-test")
        self.actor_critic = nn.Linear(1, 1, bias=False)
        self.actor_critic.encoder = SimpleNamespace(DG_projection=self.actor_critic)
        with torch.no_grad():
            self.actor_critic.weight.fill_(1.0)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=0.01)
        self.curr_lr = 0.01
        self.lr_scheduler = RecordingScheduler(per_minibatch)
        self.param_server = SimpleNamespace(policy_lock=nullcontext())
        self.train_step = 0
        self.policy_id = 0
        self.policy_versions_tensor = torch.zeros(1, dtype=torch.int64)
        self.device = torch.device("cpu")
        self.epoch = -1
        self.forward_calls = 0
        self.recruitments = 0

    def _get_minibatches(self, *_):
        self.epoch += 1
        return list(range(len(self.counts[self.epoch])))

    def _get_minibatch(self, _, index):
        self.index = index
        return {"valids": torch.arange(4) < self.counts[self.epoch][index]}

    def _should_save_summaries(self):
        return False

    def _iterative_phase(self):
        return DECODER

    def _calculate_losses(self, mb, *_, **__):
        if int(mb.valids.sum()) < 2:
            raise RuntimeError("fewer than two valid decisions reached PPO")
        self.forward_calls += 1
        weight = self.actor_critic.weight
        policy = weight.square().sum()
        if self.fixed_losses is not None:
            policy = weight.sum() * 0 + self.fixed_losses[self.epoch][self.index]
        zero = weight.sum() * 0
        return None, policy, zero, torch.zeros(2), zero, zero, zero, zero, {"additional_stats": {}}

    def _apply_lr(self, lr):
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _after_optimizer_step(self):
        self.train_step += 1

    def _apply_dg_recruitment(self, _):
        self.recruitments += 1

    def run(self):
        np.random.seed(23)
        return DistanceLearnerReward._train(self, {}, 4, 4 * len(self.counts[0]), 0)


@pytest.mark.parametrize("valid_count", [0, 1])
def test_empty_update_has_no_forward_optimizer_scheduler_or_recruitment(valid_count):
    learner = UpdateHarness([[valid_count, valid_count], [valid_count, valid_count]])
    before = learner.actor_critic.weight.detach().clone()
    assert learner.run() is None
    assert torch.equal(before, learner.actor_critic.weight)
    assert not learner.optimizer.state
    assert learner.forward_calls == learner.train_step == learner.recruitments == 0
    assert not learner.lr_scheduler.calls
    assert learner.policy_versions_tensor.item() == 0


@pytest.mark.parametrize("valid_count", [0, 1])
def test_disabled_recovery_retains_strict_failure(valid_count):
    learner = UpdateHarness([[valid_count, 4]], recover=False)
    with pytest.raises(RuntimeError, match="fewer than two"):
        learner.run()
    assert learner.train_step == learner.recruitments == 0


@pytest.mark.parametrize("per_minibatch", [False, True])
def test_partial_then_all_empty_epoch_does_not_advance_scheduler(per_minibatch):
    learner = UpdateHarness([[2, 0], [0, 1]], per_minibatch=per_minibatch)
    learner.run()
    assert learner.forward_calls == learner.train_step == 1
    assert len(learner.lr_scheduler.calls) == 1
    assert learner.recruitments == 1
    assert learner.policy_versions_tensor.item() == 1


def test_empty_first_epoch_can_be_followed_by_supported_updates():
    learner = UpdateHarness([[0, 1], [2, 4]])
    learner.run()
    assert learner.train_step == 2
    assert len(learner.lr_scheduler.calls) == learner.recruitments == 1


def test_early_stopping_uses_only_executed_losses():
    learner = UpdateHarness([[2, 0], [0, 2], [2, 2]], fixed_losses=[[2, 0], [0, 2], [3, 3]])
    learner.run()
    assert learner.train_step == 2
    assert len(learner.lr_scheduler.calls) == 2
    assert learner.recruitments == 1


def test_supported_batches_match_disabled_path_exactly():
    legacy, enabled = UpdateHarness([[2, 4], [4, 2]], False), UpdateHarness([[2, 4], [4, 2]], True)
    legacy.run()
    enabled.run()
    assert torch.equal(legacy.actor_critic.weight, enabled.actor_critic.weight)
    assert legacy.train_step == enabled.train_step == 4
    assert legacy.lr_scheduler.calls == enabled.lr_scheduler.calls
    assert legacy.recruitments == enabled.recruitments == 1
    for left, right in zip(legacy.optimizer.state.values(), enabled.optimizer.state.values()):
        for key in left:
            assert torch.equal(left[key], right[key])


@pytest.mark.parametrize(
    "changes", [dict(distance_learning=False), dict(double_value=True), dict(controller_learning="sac")]
)
def test_unsupported_recovery_modes_fail_at_factory(changes):
    cfg = AttrDict(
        recover_empty_ppo_minibatches=True, distance_learning=True, double_value=False, controller_learning="ppo"
    )
    cfg.update(changes)
    with pytest.raises(ValueError, match="recover_empty_ppo_minibatches requires"):
        make_hipposlam_learner(cfg, None, None, 0, None)


def test_summary_selection_on_skipped_minibatch_is_safe(monkeypatch):
    learner = UpdateHarness([[4, 0]])
    learner._should_save_summaries = lambda: True
    monkeypatch.setattr(np.random, "rand", lambda: 0.75)  # Select the final, skipped minibatch.
    assert learner.run() is None
    assert learner.train_step == 1 and learner.recruitments == 1
