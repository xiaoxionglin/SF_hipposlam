from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sf_working_directories.IntrMotiv.dmlab import custom_learner


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
