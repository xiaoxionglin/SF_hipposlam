from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sf_working_directories.IntrMotiv.dmlab.checkpoint_schedule import checkpoint_targets
from sf_working_directories.IntrMotiv.dmlab.controller_learner import ControllerLearner
from sf_working_directories.IntrMotiv.dmlab.custom_learner import DistanceLearnerReward


@pytest.mark.parametrize("frames", [8, 80, 100_000_000])
def test_auto_milestones_cover_planned_run_in_eight_steps(frames):
    targets = checkpoint_targets(SimpleNamespace(checkpoint_frame_targets="auto", train_for_env_steps=frames))
    assert len(targets) == 8
    assert targets[-1] == frames
    assert (
        max(b - a for a, b in zip((0, *targets[:-1]), targets))
        - min(b - a for a, b in zip((0, *targets[:-1]), targets))
        <= 1
    )


def test_auto_milestones_are_pinned_without_periodic_extras(tmp_path):
    learner = object.__new__(ControllerLearner)
    learner.cfg = SimpleNamespace(keep_checkpoints=2, checkpoint_frame_targets="auto")
    learner.policy_id = 0
    learner.checkpoint_dir = lambda *args: str(tmp_path)
    directory = tmp_path / "milestones"
    directory.mkdir()
    names = [f"checkpoint_{i:09d}_{i*100}.pth" for i in range(1, 13)]
    for name in names:
        (directory / name).write_text("evaluation artifact")
    learner.frame_milestones = set(names[::2][:8])
    with patch.object(DistanceLearnerReward, "save_milestone"):
        learner.save_milestone()
    assert {path.name for path in directory.iterdir()} == learner.frame_milestones


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
    assert sorted(p.name for p in directory.iterdir()) == [names[0], names[2], names[5]]
    assert all(p.read_text() == "full replay checkpoint fixture" for p in directory.iterdir())


def test_five_frame_targets_fit_inside_eight_milestones(tmp_path):
    learner = object.__new__(ControllerLearner)
    learner.cfg = SimpleNamespace(keep_checkpoints=8)
    learner.policy_id = 0
    learner.checkpoint_dir = lambda *args: str(tmp_path)
    directory = tmp_path / "milestones"
    directory.mkdir()
    names = [f"checkpoint_{i:09d}_{i*100}.pth" for i in range(20)]
    for name in names:
        (directory / name).write_text("evaluation artifact")
    learner.frame_milestones = set(names[::4]) | {"checkpoint_missing.pth"}
    with patch.object(DistanceLearnerReward, "save_milestone"):
        learner.save_milestone()
    assert {path.name for path in directory.iterdir()} == set(names[::4]) | set(names[-3:])


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


def _checkpoint_learner():
    learner = object.__new__(ControllerLearner)
    learner.target_snapshot = SimpleNamespace(model=SimpleNamespace(state_dict=lambda: {}), version=3)
    learner.cfg = SimpleNamespace(controller_replay_state="stored")
    learner.replay = SimpleNamespace(state_dict=lambda: {"rows": ["large replay"]})
    learner.clock = SimpleNamespace(state_dict=lambda: {"completed": 7})
    learner.frame_milestones = set()
    learner.her_rng = SimpleNamespace(bit_generator=SimpleNamespace(state={"state": 1}))
    learner.controller_version = 2
    learner.publication = 3
    learner.fresh_dg_steps = 4
    learner.fresh_graph_batches = 5
    learner.optimizer = SimpleNamespace(state={})
    learner._optimizer_step_counts = lambda: {}
    return learner


def test_milestone_omits_replay_but_rolling_checkpoint_keeps_it():
    learner = _checkpoint_learner()
    with patch.object(DistanceLearnerReward, "_get_checkpoint_dict", side_effect=lambda: {"model": {}}):
        rolling = learner._get_checkpoint_dict()
        learner._evaluation_checkpoint = True
        milestone = learner._get_checkpoint_dict()
    assert rolling["controller"]["checkpoint_role"] == "restart"
    assert rolling["controller"]["replay_included"] is True
    assert "replay" in rolling["controller"]
    assert milestone["controller"]["checkpoint_role"] == "evaluation"
    assert milestone["controller"]["replay_included"] is False
    assert "replay" not in milestone["controller"]


def test_evaluation_checkpoint_cannot_resume_training():
    learner = object.__new__(ControllerLearner)
    learner.cfg = SimpleNamespace(controller_replay_state="stored")
    checkpoint = {"controller": {"checkpoint_role": "evaluation", "replay_included": False}}
    with patch.object(DistanceLearnerReward, "_load_state"):
        with pytest.raises(RuntimeError, match="omits controller replay"):
            learner._load_state(checkpoint, load_progress=True)


def test_ddqn_keeps_one_full_restart_checkpoint():
    learner = object.__new__(ControllerLearner)
    learner.cfg = SimpleNamespace(controller_learning="ddqn")
    with patch.object(learner, "_save_impl", return_value=True) as save:
        assert learner.save() is True
    save.assert_called_once_with("checkpoint", "", 1)
