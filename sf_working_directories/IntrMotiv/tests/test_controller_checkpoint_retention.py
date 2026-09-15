from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sf_working_directories.IntrMotiv.dmlab.controller_learner import ControllerLearner
from sf_working_directories.IntrMotiv.dmlab.custom_learner import DistanceLearnerReward


def test_periodic_retention_pins_canonical_checkpoints_and_keeps_latest(tmp_path):
    learner = object.__new__(ControllerLearner)
    learner.cfg = SimpleNamespace(keep_checkpoints=2)
    learner.policy_id = 0
    learner.checkpoint_dir = lambda *args: str(tmp_path)
    directory = tmp_path / "milestones"
    directory.mkdir()
    names = [f"checkpoint_{i:09d}_{i*100}.pth" for i in range(1, 7)]
    for name in names:
        (directory / name).write_text("full replay checkpoint fixture")
    learner.frame_milestones = {names[0], names[2]}
    with patch.object(DistanceLearnerReward, "save_milestone") as save:
        learner.save_milestone()
        save.assert_called_once()
    assert sorted(p.name for p in directory.iterdir()) == [names[0], names[2], names[4], names[5]]
    assert all(p.read_text() == "full replay checkpoint fixture" for p in directory.iterdir())


def test_canonical_frame_is_pinned_before_parent_save_and_retained_in_state():
    learner = object.__new__(ControllerLearner)
    learner.cfg = SimpleNamespace(checkpoint_frame_targets="100,200")
    learner.env_steps = 112
    learner.train_step = 7
    learner.frame_milestones = set()
    seen = []
    learner.save_milestone = lambda: seen.append(set(learner.frame_milestones))
    learner._save_completed_frame_targets(96)
    assert seen == [{"checkpoint_000000007_112.pth"}]
    learner.env_steps = 128
    learner._save_completed_frame_targets(112)
    assert len(seen) == 1
