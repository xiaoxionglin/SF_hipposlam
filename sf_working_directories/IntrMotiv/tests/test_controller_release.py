from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sf_working_directories.IntrMotiv.dmlab import custom_learner
from sf_working_directories.IntrMotiv.dmlab.controller_learner import controller_policy_lag


@pytest.mark.parametrize("state,preflight", [("stored", False), ("stored", True), ("reconstruct", True)])
def test_release_routing(state, preflight):
    cfg = SimpleNamespace(controller_learning="ddqn", controller_preflight=preflight, controller_replay_state=state)
    with patch(
        "sf_working_directories.IntrMotiv.dmlab.controller_transport.validate_controller_config"
    ) as validate, patch("sf_working_directories.IntrMotiv.dmlab.controller_learner.ControllerLearner") as factory:
        assert custom_learner.make_hipposlam_learner(cfg, None, None, 0, None) is factory.return_value
        validate.assert_called_once_with(cfg)
        factory.assert_called_once_with(cfg, None, None, 0, None)


def test_reconstruction_still_guarded():
    cfg = SimpleNamespace(controller_learning="ddqn", controller_preflight=False, controller_replay_state="reconstruct")
    with pytest.raises(RuntimeError, match="production remains guarded"):
        custom_learner.make_hipposlam_learner(cfg, None, None, 0, None)


def test_ppo_factory_unchanged():
    cfg = SimpleNamespace(
        controller_learning="ppo",
        controller_preflight=False,
        controller_her=False,
        distance_learning=True,
        double_value=False,
    )
    with patch.object(custom_learner, "DistanceLearnerReward") as factory:
        assert custom_learner.make_hipposlam_learner(cfg, None, None, 0, None) is factory.return_value


def test_controller_policy_lag_uses_fresh_optimizer_version():
    minibatch = {
        "policy_id": torch.tensor([0, 0, 1]),
        "policy_version": torch.tensor([4, 4, 4]),
        "controller_fresh_version": torch.tensor([[118], [115], [117]]),
    }

    lag = controller_policy_lag(120, minibatch, policy_id=0)

    assert lag.tolist() == [2, 5]


def test_controller_policy_lag_keeps_legacy_fallback():
    minibatch = {
        "policy_id": torch.tensor([0, 1]),
        "policy_version": torch.tensor([17, 18]),
    }

    lag = controller_policy_lag(20, minibatch, policy_id=0)

    assert lag.tolist() == [3]
