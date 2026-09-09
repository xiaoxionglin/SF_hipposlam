from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import gymnasium as gym

from sf_working_directories.IntrMotiv.dmlab.custom_core import SimpleSequenceWithBypassCore
from sf_working_directories.IntrMotiv.dmlab.dmlab_gym import DmlabGymEnv_custom
from sf_working_directories.IntrMotiv.dmlab.hrl_controllable_graph import (
    HRLStateLayout,
    PolicyControllableGraph,
    hrl_option_state_size,
)
from sf_working_directories.IntrMotiv.dmlab.topological_frontier import (
    ACTION_FEATURE_SIZE,
    MODE_EXPLORE,
    MODE_RETURN,
    MODE_VALIDATE,
    N_MANAGER_MODES,
    TopologicalStateLayout,
    _fit_se2_pose_graph,
    advance_topological_manager,
    dg_path_scatter_loss,
    frontier_scores,
    integrate_motion,
    reduced_action_features,
    reliable_edges,
    select_connectivity_probe,
    select_least_tested_target,
    select_least_tested_successor,
    topological_state_size,
    update_topological_graph_from_rollout,
    validated_paths,
    visit_scores,
)
from sf_working_directories.IntrMotiv.dmlab.wrappers.reward_shaping import (
    DmlabRewardShapingWrapper,
    repeated_command_transform,
    similarity_trajectory_error,
)


def _manager_step(
    option,
    topo,
    dg,
    graph,
    action=5,
    *,
    waypoint=True,
    geometry="none",
    target_timing="delayed",
    **manager_kwargs,
):
    return advance_topological_manager(
        option,
        topo,
        dg,
        reduced_action_features(torch.tensor([action])),
        graph,
        fallback_horizon=64,
        margin_ratio=0.2,
        margin_steps=2,
        confidence_threshold=0.5,
        passive_threshold=2.0,
        passive_min_displacement=2.0,
        passive_max_length=64.0,
        use_motion_filter=True,
        frontier_uncertainty_weight=1.0,
        waypoint_planning=waypoint,
        exploration_horizon=64,
        geometry=geometry,
        target_timing=target_timing,
        **manager_kwargs,
    )


def test_least_tested_target_requires_observation_excludes_source_and_breaks_ties_by_index():
    graph = PolicyControllableGraph(4)
    graph.node_visits[:] = torch.tensor([2.0, 1.0, 0.0, 1.0])
    graph.control_attempts[0, 1] = 3.0
    graph.control_attempts[0, 3] = 1.0
    assert select_least_tested_target(graph, 0, 1.0) == 3
    graph.control_attempts[0, 1] = 1.0
    assert select_least_tested_target(graph, 0, 1.0) == 1
    assert select_least_tested_target(graph, 2, 1.0) == 0


def test_least_tested_successor_is_directed_passive_local_and_reports_behavior_set_size():
    graph = PolicyControllableGraph(4)
    graph.node_visits[:] = 1.0
    graph.passive_confidence[0, 1] = 0.01
    graph.passive_confidence[0, 3] = 2.0
    graph.passive_confidence[2, 0] = 4.0  # Incoming evidence is irrelevant.
    graph.control_attempts[0, 1] = 2.0
    graph.control_attempts[0, 3] = 1.0
    assert select_least_tested_successor(graph, 0) == (3, 2)
    graph.control_attempts[0, 1] = 1.0
    assert select_least_tested_successor(graph, 0) == (1, 2)
    assert select_least_tested_successor(graph, 1) == (None, 0)


def test_local_successor_manager_uses_passive_candidates_and_behavior_time_count():
    n_nodes = 4
    graph = PolicyControllableGraph(n_nodes)
    graph.passive_confidence[0, 1] = 1.0
    graph.passive_confidence[0, 2] = 1.0
    graph.control_attempts[0, 1] = 2.0
    graph.control_attempts[0, 2] = 0.0
    graph.tctrl[0, 2] = 4.0
    graph.edge_confidence[0, 2] = 2.0
    option = torch.zeros(1, hrl_option_state_size(n_nodes))
    topo = torch.zeros(1, topological_state_size(n_nodes))
    option, topo, _ = _manager_step(
        option,
        topo,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        graph,
        waypoint=False,
        target_timing="immediate",
        direct_target_selection="local_successor",
        edge_exploration=False,
    )
    option_layout = HRLStateLayout(n_nodes)
    topo_layout = TopologicalStateLayout(n_nodes)
    assert int(option[0, option_layout.target].item()) - 1 == 2
    assert option[0, option_layout.selected_deadline].item() == 7.0
    assert topo[0, topo_layout.local_candidate_count].item() == 2.0

    # Later graph mutation cannot retrospectively alter the stored behavior count.
    graph.passive_confidence[0, 3] = 1.0
    assert topo[0, topo_layout.local_candidate_count].item() == 2.0


def test_local_successor_manager_explores_when_source_has_no_candidate():
    n_nodes = 3
    graph = PolicyControllableGraph(n_nodes)
    option = torch.zeros(1, hrl_option_state_size(n_nodes))
    topo = torch.zeros(1, topological_state_size(n_nodes))
    option, topo, _ = _manager_step(
        option,
        topo,
        torch.tensor([[1.0, 0.0, 0.0]]),
        graph,
        waypoint=False,
        target_timing="immediate",
        direct_target_selection="local_successor",
        edge_exploration=False,
    )
    option_layout = HRLStateLayout(n_nodes)
    topo_layout = TopologicalStateLayout(n_nodes)
    assert int(option[0, option_layout.target].item()) - 1 == n_nodes
    assert topo[0, topo_layout.local_candidate_count].item() == 0.0


def test_first_distinct_wrong_outcome_terminates_without_changing_state_shape():
    n_nodes = 3
    graph = PolicyControllableGraph(n_nodes)
    graph.node_visits[:] = 1.0
    graph.control_attempts[0, 1] = 0.0
    graph.control_attempts[0, 2] = 1.0
    option = torch.zeros(1, hrl_option_state_size(n_nodes))
    topo = torch.zeros(1, topological_state_size(n_nodes))

    option, topo, _ = _manager_step(
        option,
        topo,
        torch.tensor([[1.0, 0.0, 0.0]]),
        graph,
        waypoint=False,
        target_timing="immediate",
        direct_target_selection="least_tested",
        control_outcome="first_distinct",
    )
    layout = HRLStateLayout(n_nodes)
    assert int(option[0, layout.source].item()) - 1 == 0
    assert int(option[0, layout.target].item()) - 1 == 1
    original_shape = option.shape

    option, _, _ = _manager_step(
        option,
        topo,
        torch.tensor([[0.0, 0.0, 1.0]]),
        graph,
        waypoint=False,
        target_timing="immediate",
        direct_target_selection="least_tested",
        control_outcome="first_distinct",
    )
    assert option.shape == original_shape
    assert option[0, layout.option_expired].item() == 1.0
    assert option[0, layout.target_hit].item() == 0.0
    assert option[0, layout.completion_elapsed].item() < 0.0


def test_first_distinct_ignores_same_source_and_multiactive_observations():
    n_nodes = 3
    graph = PolicyControllableGraph(n_nodes)
    graph.node_visits[:] = 1.0
    option = torch.zeros(1, hrl_option_state_size(n_nodes))
    topo = torch.zeros(1, topological_state_size(n_nodes))
    option, topo, _ = _manager_step(
        option,
        topo,
        torch.tensor([[1.0, 0.0, 0.0]]),
        graph,
        waypoint=False,
        target_timing="immediate",
        direct_target_selection="least_tested",
        control_outcome="first_distinct",
    )
    layout = HRLStateLayout(n_nodes)
    for activity in (torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[1.0, 1.0, 0.0]])):
        option, topo, _ = _manager_step(
            option,
            topo,
            activity,
            graph,
            waypoint=False,
            target_timing="immediate",
            direct_target_selection="least_tested",
            control_outcome="first_distinct",
        )
        assert option[0, layout.option_expired].item() == 0.0
        assert option[0, layout.target_hit].item() == 0.0


def test_reliable_edge_uses_success_attempt_ratio_and_can_drop_out():
    graph = PolicyControllableGraph(3)
    graph.tctrl[0, 1] = 4.0
    graph.edge_confidence[0, 1] = 1.0
    graph.control_attempts[0, 1] = 1.0
    assert reliable_edges(graph, 0.5, 0.5)[0, 1]

    graph.control_attempts[0, 1] = 3.0
    assert not reliable_edges(graph, 0.5, 0.5)[0, 1]


def test_connectivity_probe_prefers_edge_that_expands_reachability():
    graph = PolicyControllableGraph(3)
    graph.tctrl[0, 1] = 3.0
    graph.edge_confidence[0, 1] = 2.0
    graph.control_attempts[0, 1] = 2.0
    candidates = torch.zeros(3, 3, dtype=torch.bool)
    candidates[0, 1] = True  # already reliable; supplied only as a low-gain control
    candidates[1, 2] = True
    selected, score = select_connectivity_probe(
        graph,
        candidates,
        torch.tensor([True, True, False]),
        confidence_threshold=0.5,
        reliability_threshold=0.5,
        connectivity_weight=0.25,
    )
    assert selected == (1, 2)
    assert score > 0


def test_immediate_manager_condition_contains_current_mode():
    n_nodes = 3
    graph = PolicyControllableGraph(n_nodes)
    graph.node_visits[:] = 1.0
    option = torch.zeros(1, hrl_option_state_size(n_nodes))
    topo = torch.zeros(1, topological_state_size(n_nodes))
    _, next_topo, condition = advance_topological_manager(
        option,
        topo,
        torch.tensor([[1.0, 0.0, 0.0]]),
        reduced_action_features(torch.tensor([5])),
        graph,
        fallback_horizon=64,
        margin_ratio=0.2,
        margin_steps=0,
        confidence_threshold=0.5,
        passive_threshold=2.0,
        passive_min_displacement=2.0,
        passive_max_length=64.0,
        use_motion_filter=False,
        frontier_uncertainty_weight=1.0,
        waypoint_planning=True,
        exploration_horizon=64,
        edge_exploration=False,
        reliability_threshold=0.5,
        target_timing="immediate",
        include_behavior_mode=True,
    )
    assert condition.shape[-1] == n_nodes + 4 + N_MANAGER_MODES
    mode = int(next_topo[0, TopologicalStateLayout(n_nodes).mode].item())
    assert condition[0, n_nodes + 4 + mode].item() == 1.0


def test_fixed_action_integration_uses_midpoint_heading_and_wraps_angle():
    actions = reduced_action_features(torch.tensor([0, 1, 2, 3, 4, 5]))
    assert torch.equal(actions[0], torch.tensor([1.0, 0.0, 0.0, 1.0]))
    assert torch.equal(actions[1], torch.tensor([0.0, -1.0, 0.0, 1.0]))
    assert torch.equal(actions[2], torch.tensor([0.0, 1.0, 0.0, 1.0]))
    assert torch.equal(actions[5], torch.zeros(4))

    x, y, heading, path, turn = integrate_motion(
        torch.zeros(1), torch.zeros(1), torch.tensor([3.1]), torch.zeros(1), torch.zeros(1), actions[4:5]
    )
    assert -torch.pi <= heading.item() <= torch.pi
    assert torch.allclose(path, torch.ones(1))
    assert torch.allclose(turn, torch.tensor([torch.deg2rad(torch.tensor(20.0))]))
    expected_midpoint = 3.1 + 0.5 * torch.deg2rad(torch.tensor(20.0))
    assert torch.allclose(x, torch.cos(expected_midpoint).view(1), atol=1e-6)
    assert torch.allclose(y, torch.sin(expected_midpoint).view(1), atol=1e-6)


def test_fixed_action_integration_applies_yaw_for_every_repeated_engine_step():
    actions = reduced_action_features(torch.tensor([3, 4]), action_repeat=4)
    expected = torch.deg2rad(torch.tensor([-80.0, 80.0]))
    assert torch.allclose(actions[:, 2], expected)
    scale = torch.sin(torch.deg2rad(torch.tensor(40.0))) / torch.sin(torch.deg2rad(torch.tensor(10.0)))
    assert torch.allclose(actions[:, 0], torch.full((2,), scale.item()))
    assert torch.equal(actions[:, 3], torch.full((2,), 4.0))

    forward, strafe, yaw, path_length = repeated_command_transform(4, 4)
    assert np.isclose(forward, scale.item())
    assert strafe == 0.0
    assert np.isclose(yaw, expected[1].item())
    assert path_length == 4.0


def test_terminal_debug_position_replaces_the_last_live_observation():
    class FakeDmlab:
        def observations(self):
            return {"DEBUG.POS.TRANS": np.asarray((7.0, 11.0, 13.0))}

    env = DmlabGymEnv_custom.__new__(DmlabGymEnv_custom)
    env.dmlab = FakeDmlab()
    env.last_debug_position = np.asarray((1.0, 2.0, 3.0))

    assert env._refresh_terminal_debug_position()
    assert np.array_equal(env.last_debug_position, np.asarray((7.0, 11.0, 13.0)))


def test_terminal_path_telemetry_skips_a_stale_position():
    class FakeEnv(gym.Env):
        action_space = gym.spaces.Discrete(5)
        observation_space = gym.spaces.Dict({})
        action_repeat = 1
        last_debug_position = np.asarray((0.0, 0.0, 0.0))
        level_name = "openfield"
        task_id = 0

        def reset(self, **kwargs):
            return {}, {}

        def step(self, action):
            return {}, 0.0, True, False, {
                "num_frames": 1,
                "intrmotiv_position": np.asarray((0.0, 0.0, 0.0)),
                "intrmotiv_terminal_position_fresh": False,
            }

    wrapped = DmlabRewardShapingWrapper(FakeEnv(), action_path_integration=True)
    wrapped.reset()
    _, _, _, _, info = wrapped.step(0)
    assert len(wrapped.command_positions) == 2
    assert len(wrapped.actual_positions) == 1
    assert "intrmotiv_periodic_stats" not in info


def test_previous_action_observation_uses_reset_sentinel_then_executed_action():
    env = DmlabGymEnv_custom.__new__(DmlabGymEnv_custom)
    env.main_observation = "RGBD_INTERLEAVED"
    env.with_pos_obs = False
    env.action_path_integration = True
    env.instructions_observation = "INSTR"
    env.with_number_instruction = True
    env.instructions = np.zeros(1, dtype=np.int32)
    env.previous_action = 5
    obs = env.format_obs_dict({"RGBD_INTERLEAVED": np.zeros((2, 2, 4), dtype=np.uint8)})
    assert obs["prev_action"].tolist() == [5]

    env.previous_action = 3
    obs = env.format_obs_dict({"RGBD_INTERLEAVED": np.zeros((2, 2, 4), dtype=np.uint8)})
    assert obs["prev_action"].tolist() == [3]


def test_path_telemetry_error_ignores_global_scale_and_rotation():
    predicted = np.asarray(((0, 0), (1, 0), (2, 1), (3, 1)), dtype=np.float64)
    angle = np.deg2rad(37.0)
    rotation = np.asarray(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    actual = 23.0 * predicted @ rotation + np.asarray((400.0, -170.0))
    assert similarity_trajectory_error(predicted, actual) < 1e-12

    perturbed = actual.copy()
    perturbed[2] += np.asarray((0.0, 10.0))
    assert similarity_trajectory_error(predicted, perturbed) > 0


def test_validated_shortest_path_returns_explicit_first_hop():
    graph = PolicyControllableGraph(4)
    graph.tctrl[0, 1] = 3
    graph.tctrl[1, 2] = 4
    graph.tctrl[0, 2] = 20
    graph.edge_confidence[0, 1] = graph.edge_confidence[1, 2] = graph.edge_confidence[0, 2] = 1
    graph.control_attempts[0, 1] = graph.control_attempts[1, 2] = graph.control_attempts[0, 2] = 1

    distance, next_hop, hops = validated_paths(graph, 0.5)

    assert distance[0, 2].item() == 7
    assert next_hop[0, 2].item() == 1
    assert hops[0, 2].item() == 2
    assert not torch.isfinite(distance[0, 3])


def test_frontier_score_is_deterministic_and_uses_attempt_uncertainty():
    graph = PolicyControllableGraph(3)
    graph.node_visits.copy_(torch.tensor([1.0, 2.0, 3.0]))
    graph.frontier_attempts.copy_(torch.tensor([20.0, 0.0, 4.0]))
    graph.frontier_discoveries.copy_(torch.tensor([1.0, 0.0, 4.0]))

    first = frontier_scores(graph, 1.0)
    second = frontier_scores(graph, 1.0)

    assert torch.equal(first, second)
    assert first.argmax().item() == 1


def test_direct_frontier_can_command_observed_node_without_validated_route():
    graph = PolicyControllableGraph(3)
    graph.node_visits.copy_(torch.tensor([20.0, 1.0, 0.0]))
    option = torch.zeros(1, hrl_option_state_size(3))
    topo = torch.zeros(1, topological_state_size(3))

    option, _, condition = _manager_step(
        option,
        topo,
        torch.tensor([[1.0, 0.0, 0.0]]),
        graph,
        waypoint=False,
        target_timing="immediate",
    )

    layout = HRLStateLayout(3)
    assert option[0, layout.target].item() == 2
    assert condition[0, 1].item() == 1


def test_topology_matched_visit_score_ignores_ucb_and_discovery_history():
    graph = PolicyControllableGraph(3)
    graph.node_visits.copy_(torch.tensor([3.0, 2.0, 1.0]))
    graph.frontier_attempts.copy_(torch.tensor([0.0, 100.0, 0.0]))
    graph.frontier_discoveries.copy_(torch.tensor([0.0, 100.0, 0.0]))

    scores = visit_scores(graph)

    assert scores.argmax().item() == 2


def test_passive_edge_is_not_routable_until_deliberately_validated():
    graph = PolicyControllableGraph(3)
    graph.node_visits[:2] = 1
    graph.passive_confidence[0, 1] = 2
    graph.passive_path_length[0, 1] = 6
    graph.passive_time[0, 1] = 6

    distance, _, _ = validated_paths(graph, 0.5)
    assert not torch.isfinite(distance[0, 1])

    option = torch.zeros(1, hrl_option_state_size(3))
    topo = torch.zeros(1, topological_state_size(3))
    option, topo, _ = _manager_step(option, topo, torch.tensor([[1.0, 0.0, 0.0]]), graph)
    option_layout = HRLStateLayout(3)
    topo_layout = TopologicalStateLayout(3)
    assert topo[0, topo_layout.mode].item() == MODE_VALIDATE
    assert option[0, option_layout.target].item() == 2

    option, topo, behavior = _manager_step(option, topo, torch.tensor([[0.0, 1.0, 0.0]]), graph)
    assert behavior[0, 1].item() == 1
    assert topo[0, topo_layout.validation_success].item() == 1


def test_first_passive_observation_discovers_but_second_queues_validation():
    n_nodes = 3
    graph = PolicyControllableGraph(n_nodes)
    graph.node_visits[:2] = 1
    option_layout = HRLStateLayout(n_nodes)
    topo_layout = TopologicalStateLayout(n_nodes)

    def exploring_transition():
        option = torch.zeros(1, hrl_option_state_size(n_nodes))
        option[0, option_layout.target] = n_nodes + 1
        option[0, option_layout.source] = 1
        option[0, option_layout.countdown] = 20
        topo = torch.zeros(1, topological_state_size(n_nodes))
        topo[0, topo_layout.mode] = MODE_EXPLORE
        topo[0, topo_layout.last_landmark] = 1
        topo[0, topo_layout.last_landmark_age] = 3
        topo[0, topo_layout.segment_x] = 3
        topo[0, topo_layout.segment_path] = 3
        return _manager_step(option, topo, torch.tensor([[0.0, 1.0, 0.0]]), graph)

    _, first_topo, _ = exploring_transition()
    assert first_topo[0, topo_layout.discovery].item() == 1
    assert first_topo[0, topo_layout.mode].item() != MODE_RETURN

    graph.passive_confidence[0, 1] = 1
    _, second_topo, _ = exploring_transition()
    assert second_topo[0, topo_layout.discovery].item() == 0
    assert second_topo[0, topo_layout.mode].item() == MODE_RETURN


def test_common_manager_does_not_infer_reverse_travel_from_passive_edge():
    n_nodes = 3
    graph = PolicyControllableGraph(n_nodes)
    graph.node_visits[:2] = 1
    graph.passive_confidence[0, 1] = 1
    option_layout = HRLStateLayout(n_nodes)
    topo_layout = TopologicalStateLayout(n_nodes)
    option = torch.zeros(1, hrl_option_state_size(n_nodes))
    option[0, option_layout.target] = n_nodes + 1
    option[0, option_layout.source] = 1
    option[0, option_layout.countdown] = 20
    topo = torch.zeros(1, topological_state_size(n_nodes))
    topo[0, topo_layout.mode] = MODE_EXPLORE
    topo[0, topo_layout.last_landmark] = 1
    topo[0, topo_layout.last_landmark_age] = 3

    _, next_topo, _ = advance_topological_manager(
        option,
        topo,
        torch.tensor([[0.0, 1.0, 0.0]]),
        reduced_action_features(torch.tensor([5])),
        graph,
        fallback_horizon=64,
        margin_ratio=0.2,
        margin_steps=0,
        confidence_threshold=0.5,
        passive_threshold=2,
        passive_min_displacement=2,
        passive_max_length=64,
        use_motion_filter=False,
        frontier_uncertainty_weight=1,
        waypoint_planning=True,
        exploration_horizon=64,
        edge_exploration=True,
        target_timing="immediate",
        common_manager=True,
    )
    assert next_topo[0, topo_layout.mode].item() != MODE_RETURN


def test_common_manager_node_competition_uses_only_visit_rank_novelty():
    n_nodes = 3
    graph = PolicyControllableGraph(n_nodes)
    graph.node_visits.copy_(torch.tensor([3.0, 2.0, 1.0]))
    graph.frontier_attempts.copy_(torch.tensor([0.0, 0.0, 100.0]))
    graph.frontier_discoveries.copy_(torch.tensor([0.0, 0.0, 100.0]))
    for destination in (1, 2):
        graph.tctrl[0, destination] = 4
        graph.edge_confidence[0, destination] = 1
        graph.control_attempts[0, destination] = 1
    option = torch.zeros(1, hrl_option_state_size(n_nodes))
    topo = torch.zeros(1, topological_state_size(n_nodes))

    option, _, _ = advance_topological_manager(
        option,
        topo,
        torch.tensor([[1.0, 0.0, 0.0]]),
        reduced_action_features(torch.tensor([5])),
        graph,
        fallback_horizon=64,
        margin_ratio=0.2,
        margin_steps=0,
        confidence_threshold=0.5,
        passive_threshold=2,
        passive_min_displacement=2,
        passive_max_length=64,
        use_motion_filter=False,
        frontier_uncertainty_weight=1,
        waypoint_planning=True,
        exploration_horizon=64,
        edge_exploration=False,
        target_timing="immediate",
        common_manager=True,
    )
    # Node 2 has the lowest visit rank. The legacy frontier-UCB score prefers
    # node 1 under these histories, so selecting node 2 isolates pure novelty.
    assert option[0, HRLStateLayout(n_nodes).target].item() == 3


def test_equal_edge_ucb_uses_deterministic_row_major_tie_break():
    graph = PolicyControllableGraph(3)
    graph.node_visits.copy_(torch.tensor([3.0, 2.0, 1.0]))
    graph.passive_confidence[0, 1] = graph.passive_confidence[0, 2] = 2.0
    option = torch.zeros(1, hrl_option_state_size(3))
    topo = torch.zeros(1, topological_state_size(3))

    option, _, _ = _manager_step(option, topo, torch.tensor([[1.0, 0.0, 0.0]]), graph)

    # Both probes have identical posterior, uncertainty, and connectivity gain.
    # The documented deterministic tie-break therefore chooses edge 0 -> 1.
    assert option[0, HRLStateLayout(3).target].item() == 2


def test_node_only_objective_does_not_turn_passive_events_into_edge_probes():
    graph = PolicyControllableGraph(3)
    graph.node_visits[:2] = 1
    graph.passive_confidence[0, 1] = 2
    graph.passive_time[0, 1] = 5
    option = torch.zeros(1, hrl_option_state_size(3))
    topo = torch.zeros(1, topological_state_size(3))
    option, topo, _ = advance_topological_manager(
        option,
        topo,
        torch.tensor([[1.0, 0.0, 0.0]]),
        reduced_action_features(torch.tensor([5])),
        graph,
        fallback_horizon=64,
        margin_ratio=0.2,
        margin_steps=0,
        confidence_threshold=0.5,
        passive_threshold=2,
        passive_min_displacement=2,
        passive_max_length=64,
        use_motion_filter=False,
        frontier_uncertainty_weight=1,
        waypoint_planning=True,
        exploration_horizon=64,
        edge_exploration=False,
        target_timing="immediate",
    )
    assert topo[0, TopologicalStateLayout(3).mode].item() == MODE_EXPLORE
    assert option[0, HRLStateLayout(3).target].item() == 4


def test_reliable_and_candidate_deadlines_have_only_the_1p2_multiplier():
    option_layout = HRLStateLayout(3)
    topo_layout = TopologicalStateLayout(3)

    reliable_graph = PolicyControllableGraph(3)
    reliable_graph.node_visits.copy_(torch.tensor([10.0, 1.0, 0.0]))
    reliable_graph.tctrl[0, 1] = 5.0
    reliable_graph.edge_confidence[0, 1] = 1.0
    reliable_graph.control_attempts[0, 1] = 1.0
    option = torch.zeros(1, hrl_option_state_size(3))
    topo = torch.zeros(1, topological_state_size(3))
    option, _, _ = advance_topological_manager(
        option,
        topo,
        torch.tensor([[1.0, 0.0, 0.0]]),
        reduced_action_features(torch.tensor([5])),
        reliable_graph,
        fallback_horizon=64,
        margin_ratio=0.2,
        margin_steps=0,
        confidence_threshold=0.5,
        passive_threshold=2,
        passive_min_displacement=2,
        passive_max_length=64,
        use_motion_filter=False,
        frontier_uncertainty_weight=1,
        waypoint_planning=True,
        exploration_horizon=64,
        edge_exploration=False,
        reliability_threshold=0.5,
        target_timing="immediate",
    )
    assert option[0, option_layout.countdown].item() == 6

    candidate_graph = PolicyControllableGraph(3)
    candidate_graph.node_visits[:2] = 1
    candidate_graph.passive_confidence[0, 1] = 2
    candidate_graph.passive_time[0, 1] = 5.1
    option = torch.zeros(1, hrl_option_state_size(3))
    topo = torch.zeros(1, topological_state_size(3))
    option, topo, _ = advance_topological_manager(
        option,
        topo,
        torch.tensor([[1.0, 0.0, 0.0]]),
        reduced_action_features(torch.tensor([5])),
        candidate_graph,
        fallback_horizon=64,
        margin_ratio=0.2,
        margin_steps=0,
        confidence_threshold=0.5,
        passive_threshold=2,
        passive_min_displacement=2,
        passive_max_length=64,
        use_motion_filter=False,
        frontier_uncertainty_weight=1,
        waypoint_planning=True,
        exploration_horizon=64,
        edge_exploration=True,
        reliability_threshold=0.5,
        target_timing="immediate",
    )
    assert topo[0, topo_layout.mode].item() == MODE_VALIDATE
    assert option[0, option_layout.countdown].item() == 7

    candidate_graph.passive_time[0, 1] = 0
    option.zero_()
    topo.zero_()
    option, _, _ = advance_topological_manager(
        option,
        topo,
        torch.tensor([[1.0, 0.0, 0.0]]),
        reduced_action_features(torch.tensor([5])),
        candidate_graph,
        fallback_horizon=64,
        margin_ratio=0.2,
        margin_steps=0,
        confidence_threshold=0.5,
        passive_threshold=2,
        passive_min_displacement=2,
        passive_max_length=64,
        use_motion_filter=False,
        frontier_uncertainty_weight=1,
        waypoint_planning=True,
        exploration_horizon=64,
        edge_exploration=True,
        reliability_threshold=0.5,
        target_timing="immediate",
    )
    assert option[0, option_layout.countdown].item() == 64


def test_failed_probe_is_not_reselected_during_stream_cooldown():
    n_nodes = 3
    graph = PolicyControllableGraph(n_nodes)
    graph.node_visits[:2] = 1
    graph.passive_confidence[0, 1] = 2
    graph.passive_time[0, 1] = 5
    option_layout = HRLStateLayout(n_nodes)
    topo_layout = TopologicalStateLayout(n_nodes)
    option = torch.zeros(1, hrl_option_state_size(n_nodes))
    topo = torch.zeros(1, topological_state_size(n_nodes))
    option[0, option_layout.target] = 2
    option[0, option_layout.source] = 1
    option[0, option_layout.countdown] = 1
    topo[0, topo_layout.mode] = MODE_VALIDATE
    topo[0, topo_layout.pending_source] = 1
    topo[0, topo_layout.pending_destination] = 2

    option, topo, _ = advance_topological_manager(
        option,
        topo,
        torch.tensor([[1.0, 0.0, 0.0]]),
        reduced_action_features(torch.tensor([5])),
        graph,
        fallback_horizon=64,
        margin_ratio=0.2,
        margin_steps=0,
        confidence_threshold=0.5,
        passive_threshold=2,
        passive_min_displacement=2,
        passive_max_length=64,
        use_motion_filter=False,
        frontier_uncertainty_weight=1,
        waypoint_planning=True,
        exploration_horizon=64,
        edge_exploration=True,
        reliability_threshold=0.5,
        target_timing="immediate",
    )
    assert topo[0, topo_layout.validation_timeout].item() == 1
    assert topo[0, topo_layout.validation_defer_countdown].item() == 64
    assert topo[0, topo_layout.mode].item() != MODE_VALIDATE


def test_immediate_target_geometry_and_mode_match_packed_replay():
    n_nodes = 3
    cfg = SimpleNamespace(
        Hippo_R=2,
        Hippo_L=3,
        Hippo_n_feature=n_nodes,
        hrl_controllable_graph=True,
        hrl_graph_memory="policy_buffer",
        hrl_manager_mode="control_graph",
        hrl_target_timing="immediate",
        hrl_action_path_integration=True,
        hrl_motion_policy_input=True,
        hrl_landmark_geometry="se2",
        hrl_behavior_mode_condition=True,
        hrl_edge_exploration=True,
        hrl_exploration_mode=False,
        hrl_bootstrap_horizon=64,
        hrl_timeout_margin_ratio=0.2,
        hrl_timeout_margin_steps=0,
        hrl_edge_confidence_threshold=0.5,
        hrl_edge_reliability_threshold=0.5,
        hrl_passive_edge_confidence_threshold=2,
        hrl_passive_min_displacement=2,
        hrl_passive_max_path_length=64,
        hrl_frontier_uncertainty_weight=1,
        hrl_exploration_horizon=64,
        hrl_geometry_neighbors=3,
        hrl_geometry_max_distance=32,
        ca3_predictor_shadow=False,
    )
    core = SimpleSequenceWithBypassCore(cfg, n_nodes + 16)
    core.policy_graph.node_visits[:2] = 1
    core.policy_graph.passive_confidence[0, 1] = 2
    core.policy_graph.passive_time[0, 1] = 5
    core.policy_graph.pose_valid[:2] = True
    core.policy_graph.landmark_pose[1, 0] = 8
    state = torch.zeros(1, core.total_state_size)
    sequence = torch.zeros(3, 1, n_nodes + 16)
    sequence[0, 0, 0] = 1
    sequence[1, 0, 1] = 1
    sequence[2, 0, 0] = 1
    sequence[..., -ACTION_FEATURE_SIZE:] = reduced_action_features(torch.tensor([5])).view(1, 1, -1)

    sampled_outputs = []
    sampled_states = []
    sampled_state = state.clone()
    for step in sequence:
        output, sampled_state = core(step, sampled_state)
        sampled_outputs.append(output)
        sampled_states.append(sampled_state.clone())
    sampled_outputs = torch.stack(sampled_outputs)

    packed = torch.nn.utils.rnn.pack_padded_sequence(
        sequence, torch.tensor([sequence.size(0)]), enforce_sorted=False
    )
    packed_output, replay_state = core(packed, state.clone())
    replay_output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output)
    assert torch.equal(replay_output, sampled_outputs)
    assert torch.equal(replay_state, sampled_state)

    descriptor = sampled_states[0][0, -core.behavior_goal_state_size :]
    final_condition = sampled_outputs[0, 0, core.policy_base_output_size :]
    assert descriptor[0].item() == final_condition[:n_nodes].argmax().item() + 1
    assert torch.equal(
        descriptor[1 : 1 + 4],
        final_condition[core.geometry_condition_start - core.policy_base_output_size :][:4],
    )
    assert descriptor[-1].item() == final_condition[-N_MANAGER_MODES:].argmax().item()


def test_rollout_passive_update_is_applied_once_and_checkpointed():
    graph = PolicyControllableGraph(3)
    option = torch.zeros(1, 2, hrl_option_state_size(3))
    topo = torch.zeros(1, 2, topological_state_size(3))
    layout = TopologicalStateLayout(3)
    topo[0, 1, layout.passive_event] = 1
    topo[0, 1, layout.passive_source] = 1
    topo[0, 1, layout.passive_destination] = 2
    topo[0, 1, layout.passive_elapsed] = 5
    topo[0, 1, layout.passive_path_length] = 4
    topo[0, 1, layout.passive_dx] = 4

    stats = update_topological_graph_from_rollout(graph, option, topo, torch.ones(1, 1, dtype=torch.bool))

    assert stats["passive_update_count"].item() == 1
    assert graph.passive_confidence[0, 1].item() == 1
    assert graph.passive_time[0, 1].item() == 5
    restored = PolicyControllableGraph(3)
    restored.load_state_dict(graph.state_dict())
    assert torch.equal(restored.passive_confidence, graph.passive_confidence)
    assert list(restored.parameters()) == []


def test_se2_pose_graph_gauge_fixes_each_disconnected_component():
    graph = PolicyControllableGraph(4)
    graph.passive_confidence[0, 1] = 2
    graph.passive_confidence[2, 3] = 2
    graph.passive_dx[0, 1] = 4
    graph.passive_dy[2, 3] = 5
    graph.passive_dtheta_cos[0, 1] = 1
    graph.passive_dtheta_cos[2, 3] = 1

    _fit_se2_pose_graph(graph, 5, 0.05)

    assert torch.allclose(graph.landmark_pose[0], torch.zeros(3), atol=1e-6)
    assert torch.allclose(graph.landmark_pose[2], torch.zeros(3), atol=1e-6)
    assert graph.pose_valid.all()
    assert torch.isfinite(graph.pose_stress)


def test_recruitment_invalidation_clears_all_graph_scopes_and_active_generation():
    graph = PolicyControllableGraph(3)
    for buffer in (
        graph.tctrl,
        graph.edge_confidence,
        graph.control_attempts,
        graph.passive_confidence,
        graph.passive_time,
        graph.passive_path_length,
        graph.passive_dx,
        graph.passive_dy,
    ):
        buffer.fill_(1)
    graph.pose_valid.fill_(True)
    before = graph.representation_generation.item()

    graph.invalidate_node(1)

    for buffer in (graph.tctrl, graph.edge_confidence, graph.control_attempts, graph.passive_confidence):
        assert buffer[1].eq(0).all()
        assert buffer[:, 1].eq(0).all()
    assert not graph.pose_valid[1]
    assert graph.representation_generation.item() == before + 1


def test_scatter_loss_penalizes_far_straight_reactivation_not_nearby_or_loop_return():
    n_nodes = 2
    layout = TopologicalStateLayout(n_nodes)
    states = torch.zeros(1, layout.size)
    states[0, layout.anchors_start + 2] = 1
    logits = torch.tensor([[3.0, 0.0]], requires_grad=True)
    action = reduced_action_features(torch.tensor([5]))

    states[0, layout.episode_x] = 9
    states[0, layout.segment_x] = 9
    states[0, layout.segment_path] = 9
    far_loss, far_conflict = dg_path_scatter_loss(
        logits, states, action, 0, n_nodes, 2.43, 0.01, 8, 0.8, 0.5, torch.ones(1, dtype=torch.bool)
    )
    assert far_loss.item() > 0
    assert far_conflict.item() == 1

    nearby = states.clone()
    nearby[0, layout.episode_x] = 2
    nearby_loss, _ = dg_path_scatter_loss(
        logits, nearby, action, 0, n_nodes, 2.43, 0.01, 8, 0.8, 0.5, torch.ones(1, dtype=torch.bool)
    )
    assert nearby_loss.item() == 0

    loop = states.clone()
    loop[0, layout.segment_x] = 0
    loop[0, layout.segment_path] = 20
    loop_loss, _ = dg_path_scatter_loss(
        logits, loop, action, 0, n_nodes, 2.43, 0.01, 8, 0.8, 0.5, torch.ones(1, dtype=torch.bool)
    )
    assert loop_loss.item() == 0


def test_stored_worker_condition_is_invariant_to_newer_policy_graph():
    cfg = SimpleNamespace(
        Hippo_R=2,
        Hippo_L=3,
        Hippo_n_feature=3,
        hrl_controllable_graph=True,
        hrl_graph_memory="policy_buffer",
        hrl_manager_mode="frontier_waypoint",
        hrl_action_path_integration=True,
        hrl_motion_policy_input=True,
        hrl_landmark_geometry="se2",
        hrl_exploration_mode=False,
        hrl_bootstrap_horizon=64,
        hrl_timeout_margin_ratio=0.2,
        hrl_timeout_margin_steps=2,
        hrl_edge_confidence_threshold=0.5,
        hrl_passive_edge_confidence_threshold=2,
        hrl_passive_min_displacement=2,
        hrl_passive_max_path_length=64,
        hrl_frontier_uncertainty_weight=1,
        hrl_exploration_horizon=64,
        hrl_geometry_neighbors=3,
        hrl_geometry_max_distance=32,
        ca3_predictor_shadow=False,
    )
    core = SimpleSequenceWithBypassCore(cfg, 19)
    option_layout = HRLStateLayout(3)
    topo_layout = TopologicalStateLayout(3)
    states = torch.zeros(1, core.total_state_size)
    option_offset = core.base_state_size
    topo_offset = option_offset + hrl_option_state_size(3)
    states[0, option_offset + option_layout.target] = 2
    states[0, option_offset + option_layout.source] = 1
    states[0, topo_offset + topo_layout.geometry_start : topo_offset + topo_layout.geometry_end] = torch.tensor(
        [0.25, -0.5, 0.0, 1.0]
    )
    head = torch.zeros(1, 19)
    head[0, -ACTION_FEATURE_SIZE:] = reduced_action_features(torch.tensor([5]))

    first, _ = core(head, states.clone())
    core.policy_graph.node_visits.fill_(10)
    core.policy_graph.tctrl.fill_(7)
    core.policy_graph.edge_confidence.fill_(5)
    core.policy_graph.landmark_pose.normal_()
    core.policy_graph.pose_valid.fill_(True)
    second, _ = core(head, states.clone())

    assert torch.equal(first, second)


def test_default_core_state_size_is_unchanged_when_new_flags_are_disabled():
    cfg = SimpleNamespace(
        Hippo_R=2,
        Hippo_L=3,
        Hippo_n_feature=3,
        hrl_controllable_graph=True,
        hrl_graph_memory="policy_buffer",
        hrl_manager_mode="visit_direct",
        hrl_action_path_integration=False,
        hrl_exploration_mode=False,
        ca3_predictor_shadow=False,
    )
    core = SimpleSequenceWithBypassCore(cfg, 16)
    assert core.topological_state_size == 0
    assert core.action_feature_size == 0
    assert core.total_state_size == 3 * (2 + 3 - 1) + 13 + hrl_option_state_size(3)
