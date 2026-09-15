from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sample_factory.algo.learning.learner_worker import LearnerWorker
from sf_working_directories.IntrMotiv.dmlab.controller_learner import ControllerLearner


def test_controller_progress_uses_existing_heartbeat_cadence():
    learner = object.__new__(ControllerLearner)
    learner.cfg = SimpleNamespace(heartbeat_interval=40)
    callback = Mock()
    with patch(
        "sf_working_directories.IntrMotiv.dmlab.controller_learner.time.monotonic", side_effect=[0, 39, 40, 50, 80]
    ):
        learner.set_progress_callback(callback)
        for _ in range(4):
            learner._report_controller_progress()
    assert callback.call_count == 2


@pytest.mark.parametrize("controller", [False, True])
def test_worker_connects_existing_watchdog_without_changing_default_ppo(controller):
    learner = SimpleNamespace(init=Mock(return_value="model"), env_steps=0, policy_id=0)
    if controller:
        learner.set_progress_callback = Mock()
    worker = SimpleNamespace(
        cfg=SimpleNamespace(serial_mode=True),
        learner=learner,
        _report_heartbeat=Mock(),
        model_initialized=Mock(),
        report_msg=Mock(),
        initialized=Mock(),
        object_id="test",
    )
    LearnerWorker.init(worker)
    learner.init.assert_called_once()
    if controller:
        learner.set_progress_callback.assert_called_once_with(worker._report_heartbeat)
    worker.model_initialized.emit.assert_called_once_with("model")
