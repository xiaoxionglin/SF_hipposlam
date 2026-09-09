import math

import numpy as np
import torch

from sf_working_directories.IntrMotiv.dmlab.hrl_controllable_graph import PolicyControllableGraph
from sf_working_directories.IntrMotiv.evaluation.target_control_interventions import (
    Trial,
    add_matched_shuffled_targets,
    balanced_target,
    classify_trial_completion,
    pair_deadline,
    summarize_trials,
)


def test_balanced_target_covers_all_alternatives_before_repeating():
    counts = np.zeros((4, 4), dtype=np.int64)
    np.fill_diagonal(counts, 5)
    selected = []
    for _ in range(3):
        target = balanced_target(counts, 0, 5)
        selected.append(target)
        counts[0, target] += 1
    assert selected == [1, 2, 3]


def test_balanced_target_can_restrict_commands_to_observed_landmarks():
    counts = np.zeros((4, 4), dtype=np.int64)
    observed = np.array([True, False, True, True])
    assert balanced_target(counts, 0, 5, observed) == 2
    counts[0, 2] = 1
    assert balanced_target(counts, 0, 5, observed) == 3


def test_intervention_deadline_uses_reliable_then_passive_then_bootstrap():
    graph = PolicyControllableGraph(3)
    graph.tctrl[0, 1] = 5.1
    graph.edge_confidence[0, 1] = 1
    graph.control_attempts[0, 1] = 1
    assert pair_deadline(graph, 0, 1) == 9
    graph.control_attempts[0, 1] = 4
    graph.passive_time[0, 1] = 4.1
    assert pair_deadline(graph, 0, 1) == 64
    assert pair_deadline(graph, 1, 2) == 64


def test_first_distinct_completion_stops_on_wrong_outcome_but_ignores_source_and_multi_active():
    trial = Trial(0, 2, np.zeros(3), np.zeros(3), deadline=8)
    assert classify_trial_completion(trial, 0, True) is None
    assert classify_trial_completion(trial, -1, True) is None
    assert classify_trial_completion(trial, 1, True) == (1, "first_distinct")
    assert classify_trial_completion(trial, 2, True) == (2, "first_distinct")
    assert classify_trial_completion(trial, 1, False) is None
    assert classify_trial_completion(trial, 2, False) == (2, "target_hit")
    trial.elapsed = 8
    assert classify_trial_completion(trial, -1, True) == (-1, "timeout")


def test_context_matched_shuffle_uses_hit_mask_and_summary_rates():
    rows = [
        {
            "source": 0,
            "target": 1,
            "source_x_bin": 1,
            "source_y_bin": 2,
            "source_orientation_bin": 3,
            "hit_mask": 1 << 2,
            "success": 1,
            "counterfactual_action_sensitivity": 0.2,
        },
        {
            "source": 0,
            "target": 2,
            "source_x_bin": 1,
            "source_y_bin": 2,
            "source_orientation_bin": 4,
            "hit_mask": 0,
            "success": 0,
            "counterfactual_action_sensitivity": 0.4,
        },
    ]
    add_matched_shuffled_targets(rows, 3)
    assert rows[0]["shuffled_target"] == 2
    assert rows[0]["shuffled_success"] == 1
    assert rows[1]["shuffled_target"] == 1
    assert rows[1]["shuffled_success"] == 0
    counts = np.zeros((3, 3), dtype=np.int64)
    eligible = ~np.eye(3, dtype=bool)
    summary = summarize_trials(rows, counts, eligible, 100000)
    assert summary["executed_target_success_rate"] == 0.5
    assert summary["matched_shuffled_target_success_rate"] == 0.5
    assert math.isclose(summary["mean_counterfactual_action_sensitivity"], 0.3)
