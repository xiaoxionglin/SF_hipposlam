from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sf_working_directories.IntrMotiv.dmlab.controller_learner import ControllerLearner
from sf_working_directories.IntrMotiv.dmlab.controller_schedule import UpdateClock


class Replay:
    def __init__(self):
        self.accepted = 2
        self.rng = np.random.default_rng(19)
        self.rejected = Counter()

    def candidate_order(self, excluded):
        return [i for i in range(20) if i not in excluded]

    def reject(self, reason):
        self.rejected[reason] += 1


@pytest.mark.parametrize(
    "routing,target_interval,expected_checks", [("stop", 100, 23), ("joint", 100, 40), ("stop", 1, 40)]
)
def test_complete_search_fills_main_budget_and_invalidates_cached_rejections(routing, target_interval, expected_checks):
    learner = object.__new__(ControllerLearner)
    learner.cfg = SimpleNamespace(
        controller_td_positions=4,
        controller_her=False,
        controller_her_loss_coeff=1.0,
        max_grad_norm=0.0,
        ppo_dg_gradient=routing,
    )
    learner.device = torch.device("cpu")
    learner.replay = Replay()
    learner.clock = UpdateClock(0, 1, target_interval)
    learner.controller_version = 0
    learner.controller_stats = {}
    learner.actor_critic = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        learner.actor_critic.weight.fill_(1.0)
    learner.optimizer = torch.optim.SGD(learner.actor_critic.parameters(), lr=0.1)
    learner.curr_lr = 0.1
    learner._apply_lr = lambda lr: None
    learner.online_snapshot = learner.target_snapshot = SimpleNamespace(refresh=lambda *args: None)
    learner._example = lambda key: key
    evaluated = []
    gradient_batches = []

    def evaluate(keys):
        if torch.is_grad_enabled():
            gradient_batches.append(len(keys))
        else:
            evaluated.extend(keys)
        return [
            (
                "history_recognition_changed"
                if key < 17
                else (learner.actor_critic.weight.square().sum(), {"q": learner.actor_critic.weight.detach()})
            )
            for key in keys
        ]

    learner._evaluate_pairs = evaluate
    learner._controller_updates()
    assert learner.clock.completed == 2 and learner.clock.main_positions == 8
    assert learner.clock.due(learner.replay.accepted) == 0
    assert len(evaluated) == expected_checks
    assert gradient_batches == [4, 4]
    torch.testing.assert_close(learner.actor_critic.weight, torch.tensor([[0.64]]))


def test_uniform_search_covers_remaining_replay_once():
    from sf_working_directories.IntrMotiv.dmlab.controller_replay import PhysicalReplay

    replay = PhysicalReplay(100, 99)
    replay.rows = {i: None for i in range(100)}
    result = replay.candidate_order({1, 2, 3})
    assert set(result) == set(range(100)) - {1, 2, 3}
    assert len(result) == len(set(result))
    assert result != sorted(result)
