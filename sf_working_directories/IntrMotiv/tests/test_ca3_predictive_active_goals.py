from types import SimpleNamespace

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
    result = predictive_readout_loss(
        readout, predictor, ca3, actions, valid, done, recurrence, n, expanded, horizon
    )
    result.loss.backward()
    assert ca3.grad is None
    assert readout.linear.weight.grad is not None
    assert any(parameter.grad is not None for parameter in predictor.parameters())
    assert int(result.valid_targets) == (recurrence - 1) + (recurrence - 2)


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
