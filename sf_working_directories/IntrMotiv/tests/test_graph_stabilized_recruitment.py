from types import SimpleNamespace

import pytest
import torch

from sf_working_directories.IntrMotiv.dmlab.custom_core import SimpleSequenceWithBypassCore
from sf_working_directories.IntrMotiv.dmlab.custom_learner import DistanceLearnerReward
from sf_working_directories.IntrMotiv.dmlab.dg_recruitment_graph import (
    PassiveRecruitmentGraph,
    batch_predictive_events,
    directional_recruitment_eligibility,
    graph_recruitment_eligibility,
    predictive_recruitment_eligibility,
    update_recruitment_history,
)
from sf_working_directories.IntrMotiv.dmlab.hrl_controllable_graph import (
    HRLStateLayout,
    PolicyControllableGraph,
    hrl_option_state_size,
)


def _history(last_id, age, generation=0):
    return torch.tensor([last_id, age, generation], dtype=torch.float32)


def test_behavior_history_tracks_exclusive_events_without_ca3():
    state = torch.zeros(1, 3)
    generation = torch.tensor(0)
    state = update_recruitment_history(state, torch.tensor([[1.0, 0.0, 0.0]]), generation, 4)
    assert state.tolist() == [[1.0, 0.0, 0.0]]
    state = update_recruitment_history(state, torch.zeros(1, 3), generation, 4)
    assert state.tolist() == [[1.0, 1.0, 0.0]]
    state = update_recruitment_history(state, torch.tensor([[0.0, 1.0, 0.0]]), generation, 4)
    assert state.tolist() == [[2.0, 0.0, 0.0]]


def test_passive_history_crosses_rollout_boundary_and_rejects_episode_reset_and_long_gap():
    graph = PassiveRecruitmentGraph(3)
    # The first state is actor history carried from the preceding rollout.
    states = torch.stack((_history(1, 2), _history(2, 0))).view(1, 2, 3)
    result = graph.update_from_rollout(states, torch.ones(1, 1, dtype=torch.bool), 4, 5000)
    assert result["accepted_count"].item() == 1
    assert graph.confidence[0, 1].item() == 1
    reset_states = torch.stack((_history(0, 0), _history(2, 0))).view(1, 2, 3)
    assert graph.update_from_rollout(reset_states, torch.ones(1, 1), 4, 5000)["accepted_count"].item() == 0
    long_states = torch.stack((_history(1, 4), _history(2, 0))).view(1, 2, 3)
    assert graph.update_from_rollout(long_states, torch.ones(1, 1), 4, 5000)["over_gap_count"].item() == 1


def test_stale_generation_is_rejected_after_reassignment():
    graph = PassiveRecruitmentGraph(3)
    graph.invalidate_node(1)
    states = torch.stack((_history(1, 0, 0), _history(2, 0, 0))).view(1, 2, 3)
    result = graph.update_from_rollout(states, torch.ones(1, 1), 4, 5000)
    assert result["accepted_count"].item() == 0
    assert result["stale_count"].item() == 1


@pytest.mark.parametrize("half_life", (5000, 10000))
def test_birth_support_uses_exact_configured_half_life(half_life):
    graph = PassiveRecruitmentGraph(2)
    graph.decay_birth_support(half_life, half_life)
    assert torch.allclose(graph.birth_support, torch.full((2,), 0.5))
    graph.decay_birth_support(half_life, half_life)
    assert torch.allclose(graph.birth_support, torch.full((2,), 0.25))


def test_connected_nonredundant_vertex_is_protected_and_isolated_vertex_waits_for_birth_expiry():
    confidence = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    elapsed = torch.tensor([[0.0, 5.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    protected = graph_recruitment_eligibility(confidence, elapsed, torch.ones(3), 0.25, 4)
    assert protected.victim is None
    mature = graph_recruitment_eligibility(confidence, elapsed, torch.zeros(3), 0.25, 4)
    assert mature.victim == 2
    assert mature.reason == "isolated"
    assert not mature.eligible[0]
    assert not mature.eligible[1]


@pytest.mark.parametrize("threshold,inside,outside", ((4, 4.0, 5.0), (8, 8.0, 9.0)))
def test_redundancy_requires_mutual_supported_edges_at_inclusive_boundary(threshold, inside, outside):
    confidence = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    elapsed = torch.tensor([[0.0, inside], [inside, 0.0]])
    result = graph_recruitment_eligibility(confidence, elapsed, torch.zeros(2), 0.25, threshold)
    assert result.redundant_pair_count == 1
    assert result.redundant_loser.tolist() == [False, True]
    elapsed[1, 0] = outside
    result = graph_recruitment_eligibility(confidence, elapsed, torch.zeros(2), 0.25, threshold)
    assert result.redundant_pair_count == 0
    confidence[1, 0] = 0.25
    elapsed[1, 0] = inside
    result = graph_recruitment_eligibility(confidence, elapsed, torch.zeros(2), 0.25, threshold)
    assert result.redundant_pair_count == 0


def test_redundancy_loser_uses_supported_incident_confidence_and_tie_breaks_high_index():
    confidence = torch.tensor([[0.0, 1.0, 0.8], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    elapsed = torch.tensor([[0.0, 4.0, 9.0], [4.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    result = graph_recruitment_eligibility(confidence, elapsed, torch.zeros(3), 0.25, 4)
    assert result.redundant_loser[1]
    confidence[0, 2] = 0.25  # exactly at threshold is not incident support
    tie = graph_recruitment_eligibility(confidence, elapsed, torch.zeros(3), 0.25, 4)
    assert tie.redundant_loser.tolist() == [False, True, False]


def test_passive_graph_old_checkpoint_defaults_to_protected_empty_state():
    restored = PassiveRecruitmentGraph(3)
    restored.load_state_dict({}, strict=True)
    assert restored.confidence.eq(0).all()
    assert restored.elapsed.eq(0).all()
    assert restored.birth_support.eq(1).all()


def test_policy_buffer_hrl_prefers_tctrl_and_flat_mode_uses_passive_evidence():
    learner = DistanceLearnerReward.__new__(DistanceLearnerReward)
    passive = PassiveRecruitmentGraph(2)
    passive.birth_support.zero_()
    policy = PolicyControllableGraph(2)
    policy.edge_confidence[0, 1] = 1.0
    policy.tctrl[0, 1] = 5.0
    learner.actor_critic = SimpleNamespace(core=SimpleNamespace(policy_graph=policy, passive_recruitment_graph=passive))
    learner.cfg = SimpleNamespace(
        hrl_controllable_graph=True,
        hrl_graph_memory="policy_buffer",
        dg_recruitment_connectivity_threshold=0.25,
        dg_recruitment_redundancy_max_steps=4,
    )
    assert learner._graph_recruitment_eligibility().victim is None
    learner.cfg.hrl_controllable_graph = False
    assert learner._graph_recruitment_eligibility().victim == 0


def _directional_inputs(n_nodes=16):
    confidence = torch.zeros(n_nodes, n_nodes)
    attempts = torch.full((n_nodes, n_nodes), 0.5)
    attempts.fill_diagonal_(0)
    tctrl = torch.zeros(n_nodes, n_nodes)
    birth = torch.ones(n_nodes)
    birth[0] = 0.25
    return confidence, attempts, tctrl, birth


def test_directional_bad_source_requires_all_fifteen_outgoing_attempts():
    confidence, attempts, tctrl, birth = _directional_inputs()
    result = directional_recruitment_eligibility(confidence, attempts, tctrl, birth)
    assert result.fully_tested[0]
    assert result.bad_source[0]
    assert result.victim == 0
    attempts[0, 15] = 0.499
    protected = directional_recruitment_eligibility(confidence, attempts, tctrl, birth)
    assert not protected.fully_tested[0]
    assert not protected.bad_source[0]
    assert protected.victim is None


def test_directional_ignores_incoming_edges_and_requires_reliable_outgoing_edge_for_protection():
    confidence, attempts, tctrl, birth = _directional_inputs()
    confidence[1:, 0] = 4.0
    tctrl[1:, 0] = 2.0
    result = directional_recruitment_eligibility(confidence, attempts, tctrl, birth)
    assert result.bad_source[0]
    confidence[0, 1] = 0.49
    tctrl[0, 1] = 2.0
    assert directional_recruitment_eligibility(confidence, attempts, tctrl, birth).bad_source[0]
    confidence[0, 1] = 0.5
    protected = directional_recruitment_eligibility(confidence, attempts, tctrl, birth)
    assert protected.adjacency[0, 1]
    assert not protected.bad_source[0]


def test_directional_duplicate_uses_mutual_reliable_d4_and_outgoing_tie_breaks():
    confidence = torch.zeros(3, 3)
    attempts = torch.ones(3, 3)
    tctrl = torch.zeros(3, 3)
    birth = torch.tensor([0.0, 0.0, 1.0])
    confidence[0, 1] = confidence[1, 0] = 1.0
    tctrl[0, 1] = tctrl[1, 0] = 4.0
    result = directional_recruitment_eligibility(confidence, attempts, tctrl, birth)
    assert result.redundant_pair_count == 1
    assert result.redundant_loser.tolist() == [False, True, False]
    assert result.reason == "redundant"
    assert result.victim == 1
    tctrl[1, 0] = 4.01
    assert directional_recruitment_eligibility(confidence, attempts, tctrl, birth).redundant_pair_count == 0


def test_predictive_rule_is_batch_local_and_context_conditional():
    source = torch.tensor([0, 0, 0, 0])
    target = torch.tensor([3, 3, 3, 3])
    context = torch.tensor([1, 1, 2, 2])
    success = torch.tensor([True, True, False, False])
    result = predictive_recruitment_eligibility(source, target, context, success, torch.zeros(4), n_nodes=4)
    assert result.event_count == 4
    assert result.context_group_count == 2
    assert result.eligible.tolist() == [True, False, False, False]
    assert result.victim == 0
    assert result.reason == "predictive"
    empty = predictive_recruitment_eligibility(
        source[:0], target[:0], context[:0], success[:0], torch.zeros(4), n_nodes=4
    )
    assert empty.victim is None
    assert empty.event_count == 0


def test_batch_predictive_events_excludes_exploration_and_uses_distinct_ca3_context():
    n_nodes, R, expanded = 4, 2, 5
    layout = HRLStateLayout(n_nodes)
    option = torch.zeros(2, 2, hrl_option_state_size(n_nodes))
    ca3 = torch.zeros(2, 2, n_nodes, expanded)
    valid = torch.ones(2, 1, dtype=torch.bool)

    option[0, 0, layout.option_reset] = 1
    option[0, 0, layout.source] = 1
    option[0, 0, layout.target] = 4
    ca3[0, 0, 0, 0] = 1
    ca3[0, 0, 0, 1] = 1  # persistent source trace is not a predecessor
    ca3[0, 0, 2, 1] = 1
    option[0, 1, layout.target_hit] = 1

    option[1, 0, layout.option_reset] = 1
    option[1, 0, layout.source] = 1
    option[1, 0, layout.target] = n_nodes + 1
    ca3[1, 0, 0, 0] = 1
    ca3[1, 0, 1, 1] = 1
    option[1, 1, layout.option_expired] = 1
    option[1, 1, layout.completion_elapsed] = -64

    source, target, context, success = batch_predictive_events(option, ca3, valid, n_nodes, R)
    assert source.tolist() == [0]
    assert target.tolist() == [3]
    assert context.tolist() == [2]
    assert success.tolist() == [True]


def test_core_adds_history_only_in_graph_mode_and_refreshes_same_landmark_age():
    common = dict(Hippo_R=2, Hippo_L=4, Hippo_n_feature=3, hrl_controllable_graph=False)
    legacy = SimpleSequenceWithBypassCore(SimpleNamespace(**common), 16)
    graph = SimpleSequenceWithBypassCore(
        SimpleNamespace(
            **common,
            dg_orthogonal_recruitment=True,
            dg_orthogonal_recruitment_mode="graph",
        ),
        16,
    )
    assert graph.total_state_size == legacy.total_state_size + 3
    state = torch.zeros(1, graph.total_state_size)
    step = torch.zeros(1, 16)
    step[0, 1] = 1
    _, state = graph(step, state)
    assert graph.recruitment_history_from_rnn(state).tolist() == [[2.0, 0.0, 0.0]]
    _, state = graph(torch.zeros_like(step), state)
    assert graph.recruitment_history_from_rnn(state).tolist() == [[2.0, 1.0, 0.0]]
    _, state = graph(step, state)
    assert graph.recruitment_history_from_rnn(state).tolist() == [[2.0, 0.0, 0.0]]
