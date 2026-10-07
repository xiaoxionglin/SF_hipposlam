"""Contracts for the fixed four-field DG intervention."""

import numpy as np
import pytest
import torch

from sf_working_directories.IntrMotiv.dmlab.custom_learner import dg_unused_batch_recruitment_loss
from sf_working_directories.IntrMotiv.dmlab.hrl_controllable_graph import (
    current_dg_from_activity,
    restrict_goal_identities,
)
from sf_working_directories.IntrMotiv.dmlab.prescribed_dg import (
    FIELD_CENTERS,
    FIELD_RADIUS,
    prescribed_activity,
    replace_prescribed_channels,
)


def test_four_fields_are_narrow_disjoint_and_exactly_zero_outside_support():
    for i, center in enumerate(FIELD_CENTERS):
        at_center = prescribed_activity(np.array([*center, 90.0]), "gaussian4")
        assert at_center[i] == pytest.approx(1.0)
        assert np.count_nonzero(at_center) == 1
        at_edge = prescribed_activity(center + np.array([FIELD_RADIUS, 0.0]), "gaussian4")
        assert at_edge[i] == 0.0
        assert np.count_nonzero(at_edge) == 0
    np.testing.assert_array_equal(prescribed_activity(None, "zero"), np.zeros(4))
    with pytest.raises(ValueError):
        prescribed_activity(None, "gaussian4")


def test_fixed_rows_have_no_gradient_and_context_rows_remain_trainable():
    learned = torch.ones((2, 16), requires_grad=True)
    fields = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    activity = replace_prescribed_channels(learned, fields)
    activity.sum().backward()
    assert torch.count_nonzero(learned.grad[:, :4]) == 0
    torch.testing.assert_close(learned.grad[:, 4:], torch.ones((2, 12)))
    torch.testing.assert_close(activity[:, :4], fields)


def test_only_four_goal_units_can_define_manager_events():
    activity = torch.tensor([[0.7, 0, 0, 0, 3.0, 0], [0, 0, 0, 0, 2.0, 0]])
    goal_activity = restrict_goal_identities(activity, 4)
    ids, active, count = current_dg_from_activity(goal_activity)
    assert ids.tolist() == [0, -1]
    assert active.tolist() == [True, False]
    assert count.tolist() == [1, 0]
    assert activity[0, 4] == 3.0  # Context remains available to CA3.


def test_batch_recruitment_ignores_fixed_rows():
    logits = torch.zeros((2, 6), requires_grad=True)
    activity = torch.zeros_like(logits)
    prior = torch.zeros_like(logits, dtype=torch.bool)
    eligible = torch.tensor([False, False, False, False, True, True])
    loss, unused = dg_unused_batch_recruitment_loss(
        logits, activity, prior, torch.ones(2, dtype=torch.bool), 0.0, 0.5,
        eligible_features=eligible,
    )
    assert unused.tolist() == eligible.tolist()
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, :4]) == 0
    assert torch.count_nonzero(logits.grad[:, 4:]) == 4
