from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sample_factory.algo.learning.rnn_utils import reset_rnn_states_on_episode_boundary

from sf_working_directories.IntrMotiv.dmlab.custom_core import SimpleSequenceWithBypassCore
from sf_working_directories.IntrMotiv.dmlab.custom_learner import (
    control_outcome_labels,
    future_target_labels,
    legacy_reward_streams,
    selected_deadline_stats,
    target_gate_decoder_reward,
    target_reward_magnitude,
    target_success_worker_reward,
)
from sf_working_directories.IntrMotiv.dmlab.hrl_controllable_graph import (
    HRLStateLayout,
    PolicyControllableGraph,
    controllability_distances,
    exploration_mode_mask,
    hrl_option_state_size,
    hrl_persistent_state_size,
    hrl_state_size,
    initial_hrl_state,
    option_target_one_hot,
    option_deadline,
    select_target_for_layout,
    split_hrl_option_state,
    split_hrl_state,
    source_from_trace,
    update_option_state_from_policy_graph,
    update_hrl_state,
)


class ControllableGraphHRLTest(unittest.TestCase):
    def test_exploration_target_uses_zero_worker_conditioning(self):
        n_nodes = 3
        layout = HRLStateLayout(n_nodes)
        option = torch.zeros(1, hrl_option_state_size(n_nodes))
        option[:, layout.target] = n_nodes + 1

        self.assertTrue(exploration_mode_mask(option, n_nodes).item())
        self.assertTrue(option_target_one_hot(option, n_nodes).eq(0).all())

    def test_target_timeout_forces_exploration(self):
        n_nodes = 3
        layout = HRLStateLayout(n_nodes)
        state = initial_hrl_state(1, n_nodes)
        state[:, layout.target] = 2.0
        state[:, layout.source] = 1.0
        state[:, layout.age] = 4.0
        state[:, layout.countdown] = 1.0
        state[:, layout.visits_start : layout.visits_end] = 1.0

        next_state, worker_target = update_hrl_state(
            state,
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.zeros(1, n_nodes, 4),
            fallback_horizon=8,
            exploration_mode=True,
            manager_exploration_probability=0.0,
            exploration_horizon=7,
        )

        self.assertEqual(next_state[0, layout.target].item(), n_nodes + 1)
        self.assertEqual(next_state[0, layout.option_expired].item(), 1.0)
        self.assertEqual(next_state[0, layout.completion_elapsed].item(), 5.0)
        self.assertEqual(next_state[0, layout.selected_deadline].item(), 7.0)
        self.assertEqual(next_state[0, layout.countdown].item(), 7.0)
        self.assertTrue(worker_target.eq(0).all())

    def test_exploration_timeout_returns_to_target_selection(self):
        n_nodes = 3
        layout = HRLStateLayout(n_nodes)
        state = initial_hrl_state(1, n_nodes)
        state[:, layout.target] = n_nodes + 1
        state[:, layout.source] = 1.0
        state[:, layout.age] = 6.0
        state[:, layout.countdown] = 1.0
        state[:, layout.visits_start : layout.visits_end] = 1.0

        trace = torch.zeros(1, n_nodes, 4)
        trace[0, 0, 0] = 1.0
        next_state, _ = update_hrl_state(
            state,
            torch.zeros(1, n_nodes),
            trace,
            fallback_horizon=8,
            exploration_mode=True,
            manager_exploration_probability=0.0,
            exploration_horizon=7,
        )

        self.assertEqual(next_state[0, layout.target].item(), 2.0)
        self.assertEqual(next_state[0, layout.option_expired].item(), 1.0)
        self.assertEqual(next_state[0, layout.completion_elapsed].item(), -7.0)

    def test_probability_one_selects_exploration(self):
        n_nodes = 3
        layout = HRLStateLayout(n_nodes)
        state = initial_hrl_state(1, n_nodes)
        state[:, layout.visits_start : layout.visits_end] = 1.0

        next_state, _ = update_hrl_state(
            state,
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.zeros(1, n_nodes, 4),
            fallback_horizon=8,
            exploration_mode=True,
            manager_exploration_probability=1.0,
            exploration_horizon=7,
        )

        self.assertEqual(next_state[0, layout.target].item(), n_nodes + 1)

    def test_exploration_requires_policy_graph_replay(self):
        cfg = SimpleNamespace(
            Hippo_R=2,
            Hippo_L=3,
            Hippo_n_feature=3,
            hrl_controllable_graph=True,
            hrl_graph_memory="episode",
            hrl_exploration_mode=True,
        )
        with self.assertRaisesRegex(ValueError, "policy_buffer"):
            SimpleSequenceWithBypassCore(cfg, 16)

    def test_positive_selected_deadline_mean_excludes_zero_deadlines(self):
        deadlines = torch.tensor([0.0, 0.0, 8.0, 12.0])
        resets = torch.tensor([1.0, 0.0, 1.0, 1.0])

        mean, selection_fraction = selected_deadline_stats(deadlines, resets)

        self.assertEqual(mean.item(), 10.0)
        self.assertAlmostEqual(selection_fraction.item(), 2.0 / 3.0, places=6)

    def test_state_packing_round_trip(self):
        base = torch.arange(10, dtype=torch.float32).view(2, 5)
        hrl = initial_hrl_state(2, 3)
        packed = torch.cat([base, hrl], dim=1)
        base_out, hrl_out = split_hrl_state(packed, 5, 3)
        self.assertTrue(torch.equal(base_out, base))
        self.assertEqual(hrl_out.shape, (2, hrl_state_size(3)))

    def test_compact_option_state_packing_round_trip(self):
        base = torch.arange(10, dtype=torch.float32).view(2, 5)
        option = torch.zeros(2, hrl_option_state_size(3))
        packed = torch.cat([base, option], dim=1)
        base_out, option_out = split_hrl_option_state(packed, 5, 3)
        self.assertTrue(torch.equal(base_out, base))
        self.assertEqual(option_out.shape, (2, hrl_option_state_size(3)))

    def test_policy_graph_is_a_buffer_and_serializes(self):
        graph = PolicyControllableGraph(3)
        self.assertEqual(list(graph.parameters()), [])
        graph.node_visits[1] = 3.0
        restored = PolicyControllableGraph(3)
        restored.load_state_dict(graph.state_dict())
        self.assertTrue(torch.equal(restored.node_visits, graph.node_visits))

    def test_old_policy_graph_checkpoint_loads_without_generation(self):
        graph = PolicyControllableGraph(3)
        old_state = {
            key: value for key, value in graph.state_dict().items()
            if key != "representation_generation" and not key.startswith("prospective_")
        }
        restored = PolicyControllableGraph(3)
        restored.load_state_dict(old_state, strict=True)
        self.assertEqual(restored.representation_generation.item(), 0)
        self.assertTrue(restored.prospective_attempts.eq(0).all())

    def test_policy_graph_node_invalidation_clears_incident_memory(self):
        graph = PolicyControllableGraph(3)
        graph.node_visits.copy_(torch.tensor([1.0, 2.0, 3.0]))
        graph.tctrl.copy_(torch.arange(9, dtype=torch.float32).view(3, 3) + 1)
        graph.edge_confidence.fill_(2.0)

        graph.invalidate_node(1)

        self.assertEqual(graph.node_visits.tolist(), [1.0, 0.0, 3.0])
        self.assertTrue(graph.tctrl[1].eq(0).all())
        self.assertTrue(graph.tctrl[:, 1].eq(0).all())
        self.assertTrue(graph.edge_confidence[1].eq(0).all())
        self.assertTrue(graph.edge_confidence[:, 1].eq(0).all())
        self.assertEqual(graph.representation_generation.item(), 1)

    def test_policy_graph_target_is_stored_before_graph_can_change(self):
        layout = HRLStateLayout(3)
        graph = PolicyControllableGraph(3)
        graph.node_visits.copy_(torch.tensor([2.0, 1.0, 1.0]))
        option = torch.zeros(1, hrl_option_state_size(3))
        option[:, layout.target] = 2.0
        option[:, layout.source] = 1.0
        dg = torch.tensor([[1.0, 0.0, 0.0]])
        trace = torch.zeros(1, 3, 4)
        _, behavior_target = update_option_state_from_policy_graph(option, dg, trace, graph, fallback_horizon=8)
        graph.node_visits.copy_(torch.tensor([100.0, 0.0, 0.0]))
        self.assertTrue(torch.equal(behavior_target, option_target_one_hot(option, 3)))

    def test_policy_graph_immediate_timing_conditions_current_selection(self):
        layout = HRLStateLayout(3)
        graph = PolicyControllableGraph(3)
        graph.node_visits.copy_(torch.tensor([2.0, 1.0, 1.0]))
        option = torch.zeros(1, hrl_option_state_size(3))
        option[:, layout.target] = 2.0
        option[:, layout.source] = 1.0
        option[:, layout.age] = 1.0
        option[:, layout.countdown] = 8.0
        dg = torch.tensor([[0.0, 1.0, 0.0]])
        trace = torch.zeros(1, 3, 4)

        next_option, worker_target = update_option_state_from_policy_graph(
            option, dg, trace, graph, fallback_horizon=8, target_timing="immediate"
        )

        self.assertTrue(torch.equal(worker_target, option_target_one_hot(next_option, 3)))
        self.assertGreater(next_option[0, layout.target].item(), 0.0)

    def test_policy_graph_local_selection_uses_configured_half_life(self):
        layout = HRLStateLayout(3)
        graph = PolicyControllableGraph(3)
        graph.node_visits.copy_(torch.tensor([10.0, 2.0, 100.0]))
        graph.edge_confidence[0, 1] = 0.75
        graph.tctrl[0, 1] = 4.0
        option = torch.zeros(1, hrl_option_state_size(3))
        trace = torch.zeros(1, 3, 5)
        trace[0, 0, 0] = 1.0

        next_option, _ = update_option_state_from_policy_graph(
            option,
            torch.zeros(1, 3),
            trace,
            graph,
            fallback_horizon=8,
            fast_weight_half_life_options=10000.0,
        )

        self.assertEqual(next_option[0, layout.target].item(), 2.0)
        self.assertEqual(next_option[0, layout.deadline_learned].item(), 1.0)
        self.assertEqual(next_option[0, layout.selected_deadline].item(), 7.0)

    def test_policy_graph_hebbian_update_and_decay(self):
        layout = HRLStateLayout(3)
        graph = PolicyControllableGraph(3)
        graph.edge_confidence[0, 1] = 3.0
        graph.tctrl[0, 1] = 7.0
        states = torch.zeros(1, 3, hrl_option_state_size(3))
        states[0, 0, layout.source] = 1.0
        states[0, 0, layout.target] = 2.0
        states[0, 1, layout.active_dg] = 2.0
        states[0, 1, layout.target_hit] = 1.0
        states[0, 1, layout.completion_elapsed] = 5.0
        result = graph.update_from_option_rollout(states, torch.tensor([[True, False]]), half_life_options=1)
        self.assertEqual(result["completion_count"].item(), 1.0)
        self.assertAlmostEqual(graph.edge_confidence[0, 1].item(), 2.5)
        self.assertAlmostEqual(graph.control_attempts[0, 1].item(), 1.0)
        self.assertAlmostEqual(graph.tctrl[0, 1].item(), (1.5 * 7.0 + 5.0) / 2.5, places=6)

    def test_policy_graph_records_pre_update_prospective_outcome_and_timing(self):
        layout = HRLStateLayout(3)
        graph = PolicyControllableGraph(3)
        graph.edge_confidence[0, 1] = 3.0
        graph.control_attempts[0, 1] = 4.0
        graph.tctrl[0, 1] = 7.0
        states = torch.zeros(1, 2, hrl_option_state_size(3))
        states[0, 0, layout.source] = 1.0
        states[0, 0, layout.target] = 2.0
        states[0, 1, layout.active_dg] = 2.0
        states[0, 1, layout.target_hit] = 1.0
        states[0, 1, layout.completion_elapsed] = 5.0

        graph.update_from_option_rollout(
            states, torch.tensor([[True]]), half_life_options=10000,
            confidence_threshold=0.5, reliability_threshold=0.5,
        )

        self.assertEqual(graph.prospective_attempts[0, 1].item(), 1.0)
        self.assertEqual(graph.prospective_successes[0, 1].item(), 1.0)
        self.assertAlmostEqual(graph.prospective_probability_sum[0, 1].item(), 4.0 / 6.0, places=6)
        self.assertAlmostEqual(graph.prospective_brier_sum[0, 1].item(), (1.0 / 3.0) ** 2, places=6)
        self.assertEqual(graph.prospective_timing_count[0, 1].item(), 1.0)
        self.assertEqual(graph.prospective_timing_sum[0, 1].item(), 5.0)
        self.assertEqual(graph.prospective_predicted_timing_sum[0, 1].item(), 7.0)
        self.assertEqual(graph.prospective_timing_absolute_error_sum[0, 1].item(), 2.0)

    def test_policy_graph_updates_directed_success_and_timeout_independently(self):
        layout = HRLStateLayout(3)
        graph = PolicyControllableGraph(3)
        states = torch.zeros(1, 3, hrl_option_state_size(3))
        states[0, 0, layout.source] = 1
        states[0, 0, layout.target] = 2
        states[0, 1, layout.target_hit] = 1
        states[0, 1, layout.completion_elapsed] = 4
        states[0, 1, layout.source] = 2
        states[0, 1, layout.target] = 1
        states[0, 2, layout.option_expired] = 1
        states[0, 2, layout.completion_elapsed] = 6

        graph.update_from_option_rollout(
            states,
            torch.tensor([[True, True]]),
            half_life_options=5000,
            confidence_threshold=0.5,
            reliability_threshold=0.5,
        )

        self.assertGreater(graph.edge_confidence[0, 1].item(), 0)
        self.assertEqual(graph.edge_confidence[1, 0].item(), 0)
        self.assertGreater(graph.control_attempts[0, 1].item(), 0)
        self.assertGreater(graph.control_attempts[1, 0].item(), 0)
        self.assertEqual(graph.tctrl[1, 0].item(), 0)

    def test_policy_graph_classifies_exploration_timeout_without_success(self):
        n_nodes = 3
        layout = HRLStateLayout(n_nodes)
        graph = PolicyControllableGraph(n_nodes)
        states = torch.zeros(1, 3, hrl_option_state_size(n_nodes))
        states[0, 0, layout.source] = 1.0
        states[0, 0, layout.target] = n_nodes + 1
        states[0, 1, layout.option_expired] = 1.0
        states[0, 1, layout.completion_elapsed] = -64.0

        result = graph.update_from_option_rollout(
            states, torch.tensor([[True, False]]), half_life_options=10000
        )

        self.assertEqual(result["completion_count"].item(), 1.0)
        self.assertEqual(result["success_count"].item(), 0.0)
        self.assertEqual(result["target_timeout_count"].item(), 0.0)
        self.assertEqual(result["exploration_timeout_count"].item(), 1.0)
        self.assertTrue(graph.control_attempts.eq(0).all())
        self.assertTrue(graph.edge_confidence.eq(0).all())

    def test_policy_graph_ignores_rollout_from_stale_representation(self):
        layout = HRLStateLayout(3)
        graph = PolicyControllableGraph(3)
        graph.invalidate_node(2)
        states = torch.zeros(1, 3, hrl_option_state_size(3))
        states[0, 0, layout.source] = 1.0
        states[0, 0, layout.target] = 2.0
        states[0, 1, layout.active_dg] = 2.0
        states[0, 1, layout.target_hit] = 1.0
        states[0, 1, layout.completion_elapsed] = 5.0

        result = graph.update_from_option_rollout(states, torch.tensor([[True, False]]), half_life_options=1)

        self.assertEqual(result["completion_count"].item(), 0.0)
        self.assertTrue(graph.edge_confidence.eq(0).all())

    def test_policy_graph_generation_change_resets_option_without_false_hit(self):
        layout = HRLStateLayout(3)
        graph = PolicyControllableGraph(3)
        option = torch.zeros(1, hrl_option_state_size(3))
        option[:, layout.target] = 2.0
        option[:, layout.source] = 1.0
        option[:, layout.age] = 4.0
        option[:, layout.countdown] = 3.0
        graph.node_visits.copy_(torch.tensor([2.0, 2.0, 2.0]))
        graph.invalidate_node(2)
        dg = torch.tensor([[0.0, 1.0, 0.0]])
        trace = torch.zeros(1, 3, 4)

        next_option, behavior_target = update_option_state_from_policy_graph(
            option, dg, trace, graph, fallback_horizon=8
        )

        self.assertEqual(next_option[0, layout.target_hit].item(), 0.0)
        self.assertEqual(next_option[0, layout.persistent_start].item(), 1.0)
        self.assertTrue(torch.equal(behavior_target, option_target_one_hot(option, 3)))

    def test_persistent_graph_is_a_contiguous_hrl_suffix(self):
        layout = HRLStateLayout(4)
        self.assertEqual(layout.persistent_start + layout.persistent_size, layout.size)
        self.assertEqual(hrl_persistent_state_size(4), 4 + 2 * 4 * 4)

    def test_terminal_reset_keeps_only_configured_rnn_suffix(self):
        state = torch.arange(16, dtype=torch.float32).view(2, 8)
        terminal = torch.zeros(2, 1)
        preserved = reset_rnn_states_on_episode_boundary(state, terminal, persistent_state_size=3)
        self.assertTrue(torch.equal(preserved[:, :-3], torch.zeros_like(preserved[:, :-3])))
        self.assertTrue(torch.equal(preserved[:, -3:], state[:, -3:]))
        default_reset = reset_rnn_states_on_episode_boundary(state, terminal)
        self.assertTrue(torch.equal(default_reset, torch.zeros_like(state)))

    def test_target_selection_is_least_visited_and_deterministic(self):
        layout = HRLStateLayout(4)
        state = initial_hrl_state(1, 4)
        state[:, layout.visits_start : layout.visits_end] = torch.tensor([[3.0, 2.0, 2.0, 1.0]])
        source = torch.tensor([0])
        selected = select_target_for_layout(state, source, layout)
        self.assertEqual(selected.item(), 3)

        tied = initial_hrl_state(1, 4)
        self.assertEqual(select_target_for_layout(tied, source, layout).item(), -1)

    def test_target_selection_requires_observation_evidence(self):
        layout = HRLStateLayout(4)
        state = initial_hrl_state(1, 4)
        state[:, layout.visits_start : layout.visits_end] = torch.tensor([[2.0, 0.0, 1.0, 0.0]])
        selected = select_target_for_layout(state, torch.tensor([0]), layout)
        self.assertEqual(selected.item(), 2)

    def test_reachable_targets_are_prioritized_without_cost_ranking(self):
        layout = HRLStateLayout(4)
        state = initial_hrl_state(1, 4)
        state[:, layout.visits_start : layout.visits_end] = torch.tensor([[2.0, 2.0, 2.0, 1.0]])
        source = torch.tensor([0])
        tctrl = state[:, layout.tctrl_start : layout.tctrl_end].view(1, 4, 4)
        tctrl[0, 0] = torch.tensor([0.0, 0.0, 100.0, 0.0])
        selected = select_target_for_layout(state, source, layout)
        self.assertEqual(selected.item(), 2)

    def test_reachable_cost_does_not_change_exploration_ranking(self):
        layout = HRLStateLayout(4)
        state = initial_hrl_state(1, 4)
        state[:, layout.visits_start : layout.visits_end] = torch.tensor([[2.0, 3.0, 1.0, 2.0]])
        source = torch.tensor([0])
        tctrl = state[:, layout.tctrl_start : layout.tctrl_end].view(1, 4, 4)
        tctrl[0, 0] = torch.tensor([0.0, 100.0, 1.0, 50.0])
        selected = select_target_for_layout(state, source, layout)
        self.assertEqual(selected.item(), 2)

    def test_multi_hop_controllability_prioritizes_reachable_frontier(self):
        layout = HRLStateLayout(4)
        state = initial_hrl_state(1, 4)
        state[:, layout.visits_start : layout.visits_end] = torch.tensor([[2.0, 2.0, 1.0, 2.0]])
        tctrl = state[:, layout.tctrl_start : layout.tctrl_end].view(1, 4, 4)
        tctrl[0, 0, 1] = 5.0
        tctrl[0, 1, 2] = 5.0
        selected = select_target_for_layout(state, torch.tensor([0]), layout)
        self.assertEqual(selected.item(), 2)

    def test_multi_hop_deadline_uses_shortest_known_path(self):
        layout = HRLStateLayout(3)
        state = initial_hrl_state(1, 3)
        tctrl = state[:, layout.tctrl_start : layout.tctrl_end].view(1, 3, 3)
        tctrl[0, 0, 1] = 5.0
        tctrl[0, 1, 2] = 5.0
        deadline, learned = option_deadline(
            state, torch.tensor([0]), torch.tensor([2]), layout,
            fallback_horizon=64, margin_ratio=0.20, margin_steps=2,
        )
        self.assertEqual(deadline.item(), 14.0)
        self.assertTrue(learned.item())

    def test_known_and_unknown_deadlines(self):
        layout = HRLStateLayout(3)
        state = initial_hrl_state(2, 3)
        tctrl = state[:, layout.tctrl_start : layout.tctrl_end].view(2, 3, 3)
        tctrl[0, 0, 2] = 10.0
        deadline, learned = option_deadline(
            state,
            torch.tensor([0, 0]),
            torch.tensor([2, 2]),
            layout,
            fallback_horizon=64,
            margin_ratio=0.20,
            margin_steps=2,
        )
        self.assertTrue(torch.equal(deadline, torch.tensor([14.0, 64.0])))
        self.assertTrue(torch.equal(learned, torch.tensor([True, False])))

    def test_intended_target_hits_keep_best_successful_time(self):
        layout = HRLStateLayout(3)
        state = initial_hrl_state(1, 3)
        state[:, layout.target] = 2.0
        state[:, layout.source] = 1.0
        state[:, layout.age] = 4.0
        state[:, layout.countdown] = 4.0
        dg = torch.tensor([[0.0, 1.0, 0.0]])
        trace = torch.zeros(1, 3, 6)
        updated, _ = update_hrl_state(state, dg, trace, fallback_horizon=8)
        tctrl = updated[:, layout.tctrl_start : layout.tctrl_end].view(1, 3, 3)
        counts = updated[:, layout.tctrl_count_start : layout.tctrl_count_end].view(1, 3, 3)
        self.assertEqual(tctrl[0, 0, 1].item(), 5.0)
        self.assertEqual(counts[0, 0, 1].item(), 1.0)
        self.assertEqual(updated[0, layout.target_hit].item(), 1.0)

        updated[:, layout.target] = 2.0
        updated[:, layout.source] = 1.0
        updated[:, layout.age] = 7.0
        updated[:, layout.countdown] = 4.0
        slower, _ = update_hrl_state(updated, dg, trace, fallback_horizon=8)
        tctrl_slower = slower[:, layout.tctrl_start : layout.tctrl_end].view(1, 3, 3)
        counts_slower = slower[:, layout.tctrl_count_start : layout.tctrl_count_end].view(1, 3, 3)
        self.assertEqual(tctrl_slower[0, 0, 1].item(), 5.0)
        self.assertEqual(counts_slower[0, 0, 1].item(), 2.0)
        self.assertEqual(slower[0, layout.tctrl_updated].item(), 0.0)

    def test_persistent_fast_weight_updates_decayed_time_mean(self):
        layout = HRLStateLayout(3)
        state = initial_hrl_state(1, 3)
        state[:, layout.target] = 2.0
        state[:, layout.source] = 1.0
        state[:, layout.age] = 4.0
        state[:, layout.countdown] = 4.0
        tctrl = state[:, layout.tctrl_start : layout.tctrl_end].view(1, 3, 3)
        strength = state[:, layout.edge_strength_start : layout.edge_strength_end].view(1, 3, 3)
        tctrl[0, 0, 1] = 7.0
        strength[0, 0, 1] = 3.0

        updated, _ = update_hrl_state(
            state,
            torch.tensor([[0.0, 1.0, 0.0]]),
            torch.zeros(1, 3, 6),
            fallback_horizon=8,
            persistent_fast_weights=True,
            fast_weight_half_life_options=1,
            edge_confidence_threshold=0.5,
        )
        tctrl = updated[:, layout.tctrl_start : layout.tctrl_end].view(1, 3, 3)
        strength = updated[:, layout.edge_strength_start : layout.edge_strength_end].view(1, 3, 3)
        self.assertAlmostEqual(strength[0, 0, 1].item(), 2.5)
        self.assertAlmostEqual(tctrl[0, 0, 1].item(), (1.5 * 7.0 + 5.0) / 2.5, places=6)
        self.assertEqual(updated[0, layout.tctrl_updated].item(), 1.0)

    def test_decayed_edge_becomes_infeasible(self):
        layout = HRLStateLayout(3)
        state = initial_hrl_state(1, 3)
        tctrl = state[:, layout.tctrl_start : layout.tctrl_end].view(1, 3, 3)
        strength = state[:, layout.edge_strength_start : layout.edge_strength_end].view(1, 3, 3)
        tctrl[0, 0, 1] = 5.0
        strength[0, 0, 1] = 0.6
        self.assertTrue(torch.isfinite(controllability_distances(state, layout, 0.5)[0, 0, 1]))

        updated, _ = update_hrl_state(
            state,
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.zeros(1, 3, 6),
            fallback_horizon=8,
            persistent_fast_weights=True,
            fast_weight_half_life_options=1,
            edge_confidence_threshold=0.5,
        )
        self.assertFalse(torch.isfinite(controllability_distances(updated, layout, 0.5)[0, 0, 1]))

    def test_accidental_arrival_does_not_claim_controllability(self):
        layout = HRLStateLayout(4)
        state = initial_hrl_state(1, 4)
        state[:, layout.target] = 4.0
        state[:, layout.source] = 1.0
        state[:, layout.age] = 2.0
        state[:, layout.countdown] = 5.0
        dg = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        updated, _ = update_hrl_state(state, dg, torch.zeros(1, 4, 6), fallback_horizon=8)
        tctrl = updated[:, layout.tctrl_start : layout.tctrl_end].view(1, 4, 4)
        self.assertEqual(updated[0, layout.target_hit].item(), 0.0)
        self.assertEqual(updated[0, layout.tctrl_updated].item(), 0.0)
        self.assertEqual(tctrl[0, 0, 1].item(), 0.0)

    def test_expired_option_excludes_failed_target_for_replan(self):
        layout = HRLStateLayout(4)
        state = initial_hrl_state(1, 4)
        state[:, layout.target] = 2.0
        state[:, layout.source] = 1.0
        state[:, layout.countdown] = 1.0
        state[:, layout.visits_start : layout.visits_end] = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
        dg = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        updated, target = update_hrl_state(state, dg, torch.zeros(1, 4, 6), fallback_horizon=6)
        self.assertEqual(updated[0, layout.option_expired].item(), 1.0)
        self.assertEqual(updated[0, layout.option_reset].item(), 1.0)
        self.assertEqual(updated[0, layout.countdown].item(), 6.0)
        self.assertEqual(target.argmax(dim=-1).item(), 2)

    def test_replay_reconstructs_same_target_sequence(self):
        trace = torch.zeros(1, 4, 6)
        sequence = [
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
            torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        ]

        def run_once():
            state = initial_hrl_state(1, 4)
            targets = []
            for dg in sequence:
                state, target = update_hrl_state(state, dg, trace, fallback_horizon=3)
                targets.append(target.clone())
            return state, torch.cat(targets, dim=0)

        state_a, targets_a = run_once()
        state_b, targets_b = run_once()
        self.assertTrue(torch.equal(state_a, state_b))
        self.assertTrue(torch.equal(targets_a, targets_b))

    def test_sequence_core_packed_replay_matches_sampling_targets(self):
        n_nodes = 4
        cfg = SimpleNamespace(
            Hippo_R=2,
            Hippo_L=3,
            Hippo_n_feature=n_nodes,
            hrl_controllable_graph=True,
            hrl_timeout_margin_ratio=0.20,
            hrl_timeout_margin_steps=2,
        )
        core = SimpleSequenceWithBypassCore(cfg, n_nodes + 13)
        state = torch.zeros(1, core.total_state_size)
        sequence = torch.zeros(3, 1, n_nodes + 13)
        sequence[0, 0, 0] = 1.0
        sequence[1, 0, 2] = 1.0
        sequence[2, 0, 1] = 1.0

        sampled_targets = []
        sampled_state = state.clone()
        for t in range(sequence.size(0)):
            out, sampled_state = core(sequence[t], sampled_state)
            sampled_targets.append(out[:, -n_nodes:].clone())
        sampled_targets = torch.cat(sampled_targets, dim=0)

        packed = pack_padded_sequence(sequence, torch.tensor([sequence.size(0)]), enforce_sorted=False)
        packed_out, replay_state = core(packed, state.clone())
        unpacked, lengths = pad_packed_sequence(packed_out)

        self.assertEqual(lengths.item(), sequence.size(0))
        self.assertTrue(torch.equal(unpacked[:, 0, -n_nodes:], sampled_targets))
        self.assertTrue(torch.equal(replay_state, sampled_state))

    def test_persistent_core_replay_matches_after_terminal_reset(self):
        n_nodes = 4
        cfg = SimpleNamespace(
            Hippo_R=2,
            Hippo_L=3,
            Hippo_n_feature=n_nodes,
            hrl_controllable_graph=True,
            hrl_persistent_fast_weights=True,
            hrl_fast_weight_half_life_options=10000,
            hrl_edge_confidence_threshold=0.5,
            hrl_timeout_margin_ratio=0.20,
            hrl_timeout_margin_steps=2,
        )
        core = SimpleSequenceWithBypassCore(cfg, n_nodes + 13)
        state = torch.zeros(1, core.total_state_size)
        for dg_id in (0, 2, 1):
            step = torch.zeros(1, n_nodes + 13)
            step[0, dg_id] = 1.0
            _, state = core(step, state)

        persistent_size = hrl_persistent_state_size(n_nodes)
        terminal_state = state.clone()
        terminal_state[:, :-persistent_size] = 0.0
        next_step = torch.zeros(1, n_nodes + 13)
        next_step[0, 3] = 1.0
        sampled_out, sampled_state = core(next_step, terminal_state.clone())
        packed = pack_padded_sequence(next_step.unsqueeze(0), torch.tensor([1]), enforce_sorted=False)
        replay_out, replay_state = core(packed, terminal_state.clone())
        unpacked, _ = pad_packed_sequence(replay_out)

        self.assertTrue(torch.equal(unpacked[0], sampled_out))
        self.assertTrue(torch.equal(replay_state, sampled_state))

    def test_legacy_encoder_rewards_and_target_gated_worker_reward(self):
        internal = torch.tensor([[7.0, 3.0, 5.0, 4.0]])
        decoder, encourage = legacy_reward_streams(internal, 7.0, 0.1, "encourage")
        _, punish = legacy_reward_streams(internal, 7.0, 0.1, "punish")
        _, mean = legacy_reward_streams(internal, 7.0, 0.1, "mean")
        self.assertTrue(torch.allclose(decoder, torch.tensor([[0.2, 0.3]])))
        self.assertTrue(torch.allclose(encourage, torch.tensor([[0.3, 0.5]])))
        self.assertTrue(torch.allclose(punish, torch.tensor([[-0.4, -0.2]])))
        self.assertTrue(torch.allclose(mean, torch.tensor([[-0.175, 0.025]])))

        hit = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        gated = target_gate_decoder_reward(decoder, hit)
        self.assertTrue(torch.allclose(gated, torch.tensor([[0.2, 0.0]])))

        success = target_success_worker_reward(internal, hit, 7.0, 0.1, "hit", 1.0, 0.1)
        success_distance = target_success_worker_reward(
            internal, hit, 7.0, 0.1, "hit_distance", 1.0, 0.5
        )
        self.assertTrue(torch.allclose(success, torch.tensor([[1.0, 0.0]])))
        self.assertTrue(torch.all(success_distance >= success))
        self.assertTrue(torch.all(success_distance >= 0))

        exploration = torch.tensor([[False, True, False, False]])
        exploration_reward = target_success_worker_reward(
            internal,
            torch.zeros_like(hit),
            7.0,
            0.1,
            "hit_distance",
            1.0,
            0.5,
            exploration,
            decoder,
        )
        self.assertTrue(torch.allclose(exploration_reward, torch.tensor([[0.2, 0.0]])))

    def test_first_distinct_reward_is_signed_chance_centered_and_distance_monotonic(self):
        internal = torch.tensor([[7.0, 7.0, 2.0, 6.0]])
        hit = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        wrong = torch.tensor([[False, True]])
        magnitude = target_reward_magnitude(internal, 7.0, 0.1, "hit_distance", 1.0, 0.1)
        reward = target_success_worker_reward(
            internal,
            hit,
            7.0,
            0.1,
            "hit_distance",
            1.0,
            0.1,
            control_outcome="first_distinct",
            wrong_outcome=wrong,
            n_targets=16,
        )
        self.assertGreater(magnitude[0, 0].item(), magnitude[0, 1].item())
        self.assertTrue(torch.allclose(reward[0, 0], magnitude[0, 0]))
        self.assertTrue(torch.allclose(reward[0, 1], -magnitude[0, 1] / 14.0))
        for value in magnitude.flatten():
            balanced = value + 14.0 * (-value / 14.0)
            self.assertAlmostEqual(balanced.item(), 0.0, places=6)

    def test_first_distinct_reward_uses_behavior_time_local_command_count(self):
        internal = torch.tensor([[7.0, 7.0, 4.0, 4.0]])
        hit = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        wrong = torch.tensor([[False, True]])
        sizes = torch.tensor([[3.0, 3.0]])
        reward = target_success_worker_reward(
            internal,
            hit,
            7.0,
            0.1,
            "hit",
            1.0,
            0.1,
            control_outcome="first_distinct",
            wrong_outcome=wrong,
            command_set_size=sizes,
        )
        self.assertEqual(reward[0, 0].item(), 1.0)
        self.assertEqual(reward[0, 1].item(), -0.5)
        self.assertEqual((reward[0, 0] + 2.0 * reward[0, 1]).item(), 0.0)

        singleton = target_success_worker_reward(
            internal,
            hit,
            7.0,
            0.1,
            "hit",
            1.0,
            0.1,
            control_outcome="first_distinct",
            wrong_outcome=wrong,
            command_set_size=torch.ones_like(sizes),
        )
        self.assertEqual(singleton[0, 1].item(), -1.0)

    def test_control_outcome_labels_separate_correct_wrong_timeout_exploration_and_censoring(self):
        n_nodes = 3
        layout = HRLStateLayout(n_nodes)
        states = torch.zeros(5, 3, hrl_option_state_size(n_nodes))
        states[:, 1, layout.source] = 1.0
        states[:4, 1, layout.target] = 2.0
        states[4, 1, layout.target] = n_nodes + 1.0
        states[0, 2, layout.target_hit] = 1.0
        states[0, 2, layout.active_dg] = 2.0
        states[1, 2, layout.option_expired] = 1.0
        states[1, 2, layout.completion_elapsed] = -4.0
        states[1, 2, layout.active_dg] = 3.0
        states[2, 2, layout.option_expired] = 1.0
        states[2, 2, layout.completion_elapsed] = 64.0
        states[4, 2, layout.option_expired] = 1.0
        states[4, 2, layout.completion_elapsed] = -64.0
        labels = control_outcome_labels(
            states, torch.tensor([[False], [False], [False], [True], [False]]), layout
        )

        self.assertEqual(labels["correct"].flatten().tolist(), [True, False, False, False, False])
        self.assertEqual(labels["wrong"].flatten().tolist(), [False, True, False, False, False])
        self.assertEqual(labels["timeout"].flatten().tolist(), [False, False, True, False, False])
        self.assertEqual(labels["censored"].flatten().tolist(), [False, False, False, True, False])
        self.assertEqual(labels["exploration_timeout"].flatten().tolist(), [False, False, False, False, True])

    def test_wrong_first_outcome_updates_attempt_but_not_confidence(self):
        n_nodes = 3
        layout = HRLStateLayout(n_nodes)
        graph = PolicyControllableGraph(n_nodes)
        states = torch.zeros(1, 2, hrl_option_state_size(n_nodes))
        states[0, 0, layout.source] = 1.0
        states[0, 0, layout.target] = 2.0
        states[0, 1, layout.active_dg] = 3.0
        states[0, 1, layout.option_expired] = 1.0
        states[0, 1, layout.completion_elapsed] = -4.0

        result = graph.update_from_option_rollout(
            states, torch.tensor([[True]]), half_life_options=5000
        )

        self.assertEqual(result["wrong_outcome_count"].item(), 1.0)
        self.assertEqual(result["timeout_count"].item(), 0.0)
        self.assertGreater(graph.control_attempts[0, 1].item(), 0.0)
        self.assertEqual(graph.edge_confidence[0, 1].item(), 0.0)

    def test_future_target_labels_respect_episode_boundaries(self):
        target = torch.tensor(
            [[[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]]
        )
        dg = torch.tensor(
            [[[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]
        )
        dones = torch.tensor([[False, True, False, False]])
        hit, hit_time, valid = future_target_labels(target, dg, dones, horizon=3)
        self.assertTrue(valid.all())
        self.assertEqual(hit[0, 0].item(), 0.0)
        self.assertEqual(hit[0, 1].item(), 0.0)
        self.assertEqual(hit[0, 2].item(), 1.0)
        self.assertEqual(hit_time[0, 2].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
