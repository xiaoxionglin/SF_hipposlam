from types import SimpleNamespace

import pytest
import torch

from sf_working_directories.IntrMotiv.dmlab.custom_learner import DistanceLearnerReward


class RecordingBuffer:
    def __init__(self, invalidated_mass=0.0):
        self.rows = []
        self.invalidated_mass = invalidated_mass

    def invalidate_node(self, row):
        self.rows.append(row)
        return torch.tensor(self.invalidated_mass)


def learner_for(mode, *, with_evidence=True):
    learner = object.__new__(DistanceLearnerReward)
    learner.cfg = SimpleNamespace(
        dg_orthogonal_recruitment=True,
        dg_orthogonal_recruitment_mode=mode,
        hrl_controllable_graph=True,
        hrl_graph_memory="policy_buffer",
    )
    core = SimpleNamespace(
        passive_recruitment_graph=RecordingBuffer(),
        policy_graph=RecordingBuffer(),
        predictive_recruitment_evidence=(RecordingBuffer(7.0) if with_evidence else None),
    )
    learner.actor_critic = SimpleNamespace(core=core)
    learner._last_recruitment_stats = {"predictive_invalidation_mass": 0.0}
    return learner, core


def test_legacy_recruitment_invalidates_only_policy_graph():
    learner, core = learner_for("legacy", with_evidence=False)

    learner._invalidate_recruited_node(3)

    assert core.policy_graph.rows == [3]
    assert core.passive_recruitment_graph.rows == []
    assert learner._last_recruitment_stats["predictive_invalidation_mass"] == 0.0


def test_graph_recruitment_invalidates_all_dependent_state():
    learner, core = learner_for("graph")

    learner._invalidate_recruited_node(2)

    assert core.passive_recruitment_graph.rows == [2]
    assert core.policy_graph.rows == [2]
    assert core.predictive_recruitment_evidence.rows == [2]
    assert learner._last_recruitment_stats["predictive_invalidation_mass"] == 7.0


def test_graph_recruitment_still_requires_predictive_evidence():
    learner, _ = learner_for("graph", with_evidence=False)

    with pytest.raises(RuntimeError, match="Persistent PRED requires evidence buffers"):
        learner._invalidate_recruited_node(1)
