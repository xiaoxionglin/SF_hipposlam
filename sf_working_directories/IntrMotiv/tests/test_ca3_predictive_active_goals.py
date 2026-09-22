from types import SimpleNamespace
from unittest.mock import patch

import torch

from sf_working_directories.IntrMotiv.dmlab.ca3_state_readout import (
    CA3StateReadout,
    CausalDGInnovationPredictor,
    predictive_readout_loss,
)
from sf_working_directories.IntrMotiv.dmlab.hrl_controllable_graph import PolicyControllableGraph
from sf_working_directories.IntrMotiv.dmlab.topological_frontier import (
    select_least_tested_successor,
    select_least_tested_target,
    validated_paths,
)


def test_anchor_requires_calibrated_independent_confirmation_and_replacement_deactivates():
    graph = PolicyControllableGraph(4, ca3_size=8, contextual=True, calibration_capacity=4, prediction_horizon=2)
    readout = CA3StateReadout(8, 3)
    predictor = CausalDGInnovationPredictor(3, 2, 2, 4, hidden_size=8)
    with torch.no_grad():
        readout.linear.weight.zero_()
        readout.linear.weight[:, :3] = torch.eye(3)
    anchor = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    graph.register_anchor(2, anchor, 0)
    assert not graph.selectable_mask().any()
    actions = torch.tensor([0, 1])

    def predicted_targets(state):
        latent = readout(state.unsqueeze(0)).expand(2, -1)
        prefixes = torch.tensor([[0, 0], [0, 1]])
        return predictor(latent, prefixes, torch.tensor([1, 2])).detach()

    anchor_targets = predicted_targets(anchor)
    assert not graph.confirm_anchor(2, anchor, readout, predictor, actions, anchor_targets, 1.0, 0.1)

    for _ in range(4):
        graph.add_positive_pair(anchor, anchor, actions, anchor_targets)
    assert graph.recalibrate(readout, predictor, 10, 1, 4, 0.1, 1.0, 0.1)
    wrong = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert not graph.confirm_anchor(2, wrong, readout, predictor, actions, predicted_targets(wrong), 1.0, 0.1)
    assert not graph.selectable_mask()[2]
    assert graph.confirm_anchor(2, anchor, readout, predictor, actions, anchor_targets, 1.0, 0.1)
    assert graph.selectable_mask()[2]
    activity = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    recognized = graph.contextual_activity(activity, torch.stack((anchor, wrong)), readout, predictor)
    assert recognized[0, 2] == 1
    assert recognized[1].sum() == 0
    assert not hasattr(graph, "anchor_eval_position")

    structural_generation = int(graph.representation_generation)
    anchor_generation = int(graph.anchor_generation[2])
    graph.node_visits[2] = 5
    graph.edge_confidence[2, 1] = 3
    graph.replace_anchor(2, wrong, 20, 0.25)
    assert int(graph.representation_generation) == structural_generation
    assert int(graph.anchor_generation[2]) == anchor_generation + 1
    assert not graph.selectable_mask()[2]
    assert graph.node_visits[2] == 0 and graph.edge_confidence[2, 1] == 0


def test_all_topological_candidates_and_paths_obey_active_endpoints():
    graph = PolicyControllableGraph(4, ca3_size=8, contextual=True)
    graph.node_visits[:] = 10
    graph.passive_confidence[:] = 2
    graph.tctrl[0, 1] = 2
    graph.tctrl[1, 2] = 2
    graph.edge_confidence[0, 1] = 4
    graph.edge_confidence[1, 2] = 4
    graph.active_goal_mask[[0, 2]] = True
    graph.anchor_valid[[0, 2]] = True
    graph.active_generation[[0, 2]] = graph.anchor_generation[[0, 2]]

    # Slot 1 has abundant stale evidence but is not selectable, so it cannot
    # be a direct target, successor, endpoint, or intermediate waypoint.
    assert select_least_tested_target(graph, 0) == 2
    successor, count = select_least_tested_successor(graph, 0)
    assert successor == 2 and count == 1
    distance, next_hop, _ = validated_paths(graph, 0.5)
    assert torch.isinf(distance[0, 2])
    assert next_hop[0, 2] == -1


def test_prediction_windows_are_causal_and_gradients_stop_at_canonical_ca3():
    n, expanded, recurrence, horizon = 3, 2, 5, 2
    ca3 = torch.zeros(recurrence, n * expanded, requires_grad=True)
    with torch.no_grad():
        ca3[:, 0] = torch.arange(recurrence)
        ca3[:, 2] = 1
    actions = torch.tensor([[0], [1], [2], [1], [0]])
    valid = torch.ones(recurrence, dtype=torch.bool)
    done = torch.zeros(recurrence, dtype=torch.bool)
    readout = CA3StateReadout(n * expanded, 4)
    predictor = CausalDGInnovationPredictor(4, 3, horizon, n)
    result = predictive_readout_loss(readout, predictor, ca3, actions, valid, done, recurrence, n, expanded, horizon)
    result.loss.backward()
    assert ca3.grad is None
    assert readout.linear.weight.grad is not None
    assert any(parameter.grad is not None for parameter in predictor.parameters())
    assert int(result.valid_targets) == (recurrence - 1) + (recurrence - 2)


def test_readout_regularizers_are_finite_and_only_update_the_readout():
    n, expanded, recurrence, horizon = 3, 2, 4, 2
    ca3 = torch.ones(recurrence, n * expanded, requires_grad=True)
    readout = CA3StateReadout(n * expanded, 4)
    predictor = CausalDGInnovationPredictor(4, 2, horizon, n)
    result = predictive_readout_loss(
        readout,
        predictor,
        ca3,
        torch.zeros(recurrence, 1, dtype=torch.long),
        torch.ones(recurrence, dtype=torch.bool),
        torch.zeros(recurrence, dtype=torch.bool),
        recurrence,
        n,
        expanded,
        horizon,
    )
    assert torch.isfinite(torch.stack((result.loss, result.var_loss, result.cov_loss))).all()
    assert result.var_loss > 0
    result.var_loss.backward()
    assert ca3.grad is None
    assert readout.linear.weight.grad is not None
    assert all(parameter.grad is None for parameter in predictor.parameters())


def test_ema_refinement_preserves_generation_and_graph_evidence():
    graph = PolicyControllableGraph(2, ca3_size=4, contextual=True, signature_dim=8, candidate_mode="dominant")
    readout = CA3StateReadout(4, 2)
    predictor = CausalDGInnovationPredictor(2, 2, 2, 2, hidden_size=4)
    anchor = torch.tensor([1.0, 0.0, 0.0, 0.0])
    candidate = torch.tensor([0.0, 1.0, 0.0, 0.0])
    graph.register_anchor(0, anchor, 1)
    graph.active_goal_mask[0] = True
    graph.active_generation[0] = graph.anchor_generation[0]
    graph.confirmation_count[0] = 8
    graph.node_visits[0] = 7
    graph.edge_confidence[0, 1] = 3
    generation = int(graph.anchor_generation[0])
    graph.refine_anchor_ema(0, candidate, readout, predictor, 10, 0.05, 8, -1.0)
    assert int(graph.anchor_generation[0]) == generation
    assert graph.selectable_mask()[0]
    assert graph.node_visits[0] == 7 and graph.edge_confidence[0, 1] == 3
    assert int(graph.anchor_refinements) == 1


def test_unique_contextual_recognition_rescues_one_multi_active_match_and_abstains_on_ambiguity():
    graph = PolicyControllableGraph(3, ca3_size=3, contextual=True, candidate_mode="unique_contextual")
    graph.active_goal_mask[:2] = True
    graph.anchor_valid[:2] = True
    graph.active_generation[:2] = graph.anchor_generation[:2]
    graph.recognition_threshold.fill_(0.5)
    activity = torch.tensor([[1.0, 2.0, 0.0]])
    with torch.no_grad(), patch.object(
        graph, "recognition_similarity", side_effect=[torch.tensor(0.9), torch.tensor(0.1)]
    ):
        result = graph.contextual_activity(activity, torch.ones(1, 3), None, None)
    assert result.tolist() == [[2.0, 0.0, 0.0]]
    assert int(graph.context_unique_rescues) == 1
    with torch.no_grad(), patch.object(
        graph, "recognition_similarity", side_effect=[torch.tensor(0.9), torch.tensor(0.8)]
    ):
        result = graph.contextual_activity(activity, torch.ones(1, 3), None, None)
    assert not result.any()
    assert int(graph.context_multi_match) == 1


def test_contextual_checkpoint_round_trip_preserves_calibration_and_generations():
    graph = PolicyControllableGraph(2, ca3_size=4, contextual=True, calibration_capacity=3, prediction_horizon=2)
    graph.register_anchor(1, torch.ones(4), 7)
    graph.add_positive_pair(torch.ones(4), torch.ones(4), torch.zeros(2, dtype=torch.long), torch.zeros(2, 2))
    graph.anchor_generation[1] = 3
    graph.active_goal_mask[1] = True
    graph.active_generation[1] = 3
    clone = PolicyControllableGraph(2, ca3_size=4, contextual=True, calibration_capacity=3, prediction_horizon=2)
    clone.load_state_dict(graph.state_dict())
    assert clone.selectable_mask().tolist() == [False, True]
    assert int(clone.calibration_count) == 1
    assert int(clone.anchor_last_update[1]) == 7
