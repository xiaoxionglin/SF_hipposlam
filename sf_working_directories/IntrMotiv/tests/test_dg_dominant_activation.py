from __future__ import annotations

import torch

from sf_working_directories.IntrMotiv.dmlab.custom_learner import (
    dominant_new_activation_masks,
    predecessor_distance_for_dominant_events,
)


def test_simultaneous_event_selects_strongest_behavior_activation():
    r = 2
    progression = torch.tensor(
        [
            [
                [5, 5, 5],
                [0, 0, 3],
                [0, 0, 0],
            ]
        ]
    )
    sequence_core = torch.zeros(1, 3, 3, 5)
    sequence_core[0, 1, 0, 0] = 0.4
    sequence_core[0, 1, 1, 0] = 0.9
    sequence_core[0, 1, 2, 3] = 1.0
    sequence_core[0, 2, :, 0] = torch.tensor([0.5, 0.6, 0.7])

    candidates, dominant, non_dominant = dominant_new_activation_masks(sequence_core, progression, r)

    assert torch.equal(candidates[0, 1], torch.tensor([True, True, False]))
    assert torch.equal(dominant[0, 1], torch.tensor([False, True, False]))
    assert torch.equal(non_dominant[0, 1], torch.tensor([True, False, False]))
    # Units 0 and 1 were still inside their refractory region at t=1. Only
    # unit 2 is a new event at t=2, despite all three having progression zero.
    assert torch.equal(candidates[0, 2], torch.tensor([False, False, True]))
    assert torch.equal(dominant[0, 2], torch.tensor([False, False, True]))


def test_first_sequence_state_does_not_wrap_to_future_state():
    progression = torch.zeros(1, 2, 2, dtype=torch.long)
    sequence_core = torch.ones(1, 2, 2, 3)

    candidates, dominant, non_dominant = dominant_new_activation_masks(sequence_core, progression, 2)

    assert not candidates[:, 0].any()
    assert not dominant[:, 0].any()
    assert not non_dominant[:, 0].any()


def test_dominant_tie_break_is_deterministic():
    progression = torch.tensor([[[5, 5], [0, 0]]])
    sequence_core = torch.zeros(1, 2, 2, 3)
    sequence_core[0, 1, :, 0] = 0.5

    _, dominant, non_dominant = dominant_new_activation_masks(sequence_core, progression, 2)

    assert torch.equal(dominant[0, 1], torch.tensor([True, False]))
    assert torch.equal(non_dominant[0, 1], torch.tensor([False, True]))


def test_simultaneous_candidates_are_not_each_others_predecessors():
    baseline = 7
    progression = torch.tensor([[[7, 7, 7], [0, 0, 4], [0, 0, 7]]])
    candidates = torch.tensor(
        [[[False, False, False], [True, True, False], [True, True, False]]]
    )
    dominant = torch.tensor(
        [[[False, False, False], [False, True, False], [True, False, False]]]
    )

    distance = predecessor_distance_for_dominant_events(progression, candidates, dominant, baseline)

    # The prior third-unit trace determines the first event's distance.
    assert distance[0, 1].item() == 4
    # With no other prior trace, simultaneous activation receives the capped
    # no-predecessor value rather than zero or baseline + 1.
    assert distance[0, 2].item() == baseline
