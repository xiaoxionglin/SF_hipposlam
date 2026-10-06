"""Focused contracts for the odor input and fixed CA3 goal-quality rule."""

import numpy as np
import torch
from types import SimpleNamespace

from sample_factory.utils.attr_dict import AttrDict

from sf_working_directories.IntrMotiv.dmlab.ca3_goal_quality import CA3GoalQuality, candidate_mask
from sf_working_directories.IntrMotiv.dmlab.hrl_controllable_graph import HRLStateLayout
from sf_working_directories.IntrMotiv.dmlab.hrl_controllable_graph import PolicyControllableGraph, hrl_option_state_size
from sf_working_directories.IntrMotiv.dmlab.odor import ODOR_CENTERS, clean_odor, odor_observation
from sf_working_directories.IntrMotiv.dmlab.custom_learner import DistanceLearnerReward
from sf_working_directories.IntrMotiv.dmlab.custom_encoder import HipposlamEncoder
from sf_working_directories.IntrMotiv.dmlab.topological_frontier import (
    TopologicalStateLayout,
    advance_topological_manager,
    reduced_action_features,
    topological_state_size,
)


def _rollout(contexts, events, valid):
    layout = HRLStateLayout(3)
    ca3 = torch.tensor(contexts, dtype=torch.float32)[None]
    option = torch.zeros((1, len(contexts), layout.persistent_start), dtype=torch.float32)
    option[0, :, layout.active_dg] = torch.tensor(events, dtype=torch.float32)
    return ca3, option, torch.tensor(valid, dtype=torch.bool)[None]


def test_new_exclusive_event_uses_pre_action_ca3_and_only_accepted_rows():
    memory = CA3GoalQuality(3, 2)
    ca3, option, valid = _rollout(
        [[3, 0], [0, 4], [0, 5], [9, 0], [1, 1]],
        [0, 1, 1, 2, 3],
        [True, True, True, False],
    )
    assert memory.update_from_rollout(ca3, option, valid) == 2
    assert memory.event_counts.tolist() == [1, 1, 0]
    torch.testing.assert_close(memory.prototypes[0], torch.tensor([0.01, 0.0]))
    torch.testing.assert_close(memory.prototypes[1], torch.tensor([0.0, 0.01]))
    assert memory.quality_counts.sum() == 0  # no own prototype at first event
    assert not any(parameter.requires_grad for parameter in memory.parameters())


def test_prequential_margin_and_exact_state_restoration():
    first = CA3GoalQuality(3, 2)
    first.prototypes[:2] = torch.tensor([[0.5, 0.0], [0.0, 0.5]])
    ca3, option, valid = _rollout([[1, 0], [1, 0]], [0, 1], [True])
    checkpoint = {key: value.clone() for key, value in first.state_dict().items()}
    resumed = CA3GoalQuality(3, 2)
    resumed.load_state_dict(checkpoint)
    assert first.update_from_rollout(ca3, option, valid) == 1
    assert resumed.update_from_rollout(ca3, option, valid) == 1
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(first.quality[0], torch.tensor(0.01))
    torch.testing.assert_close(first.scores[0], torch.tensor(0.01 / 101))
    torch.testing.assert_close(first.prototypes[1], checkpoint["prototypes"][1])


def test_candidate_eligibility_capacity_and_uniform_ties():
    eligible = torch.tensor([[True, False, True, True], [False, True, False, False]])
    scores = torch.tensor([0.0, 99.0, 1.0, 2.0])
    hebb = candidate_mask(eligible, scores, "hebb", 2)
    assert hebb.tolist() == [[False, False, True, True], [False, True, False, False]]
    assert torch.equal(candidate_mask(eligible, scores, "random", 4), eligible)
    torch.manual_seed(11)
    draws = torch.stack([candidate_mask(torch.ones((1, 4), dtype=torch.bool), torch.zeros(4), "hebb", 1)[0]
                         for _ in range(400)])
    frequencies = draws.sum(0)
    assert torch.all((frequencies > 65) & (frequencies < 135))


def test_manager_scores_only_current_eligible_candidates():
    graph = PolicyControllableGraph(4)
    graph.node_visits[:] = torch.tensor([1.0, 5.0, 3.0, 2.0])
    option = torch.zeros(1, hrl_option_state_size(4))
    topology = torch.zeros(1, topological_state_size(4))
    activity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    def choice(scores):
        result, _, _ = advance_topological_manager(
            option.clone(), topology.clone(), activity,
            reduced_action_features(torch.tensor([5])), graph,
            fallback_horizon=64, margin_ratio=0.2, margin_steps=2,
            confidence_threshold=0.5, passive_threshold=2.0,
            passive_min_displacement=2.0, passive_max_length=64,
            use_motion_filter=True, frontier_uncertainty_weight=1.0,
            waypoint_planning=False, exploration_horizon=64,
            edge_exploration=False, target_timing="immediate",
            direct_target_selection="frontier", goal_candidate_mode="hebb",
            goal_candidate_k=1, goal_quality_scores=torch.tensor(scores),
        )
        return int(result[0, HRLStateLayout(4).target])

    assert choice([100.0, 2.0, 1.0, 0.0]) == 2  # source ID 1 cannot be offered
    graph.node_visits[1] = 0.0
    assert choice([100.0, 2.0, 1.0, 0.0]) == 3  # unseen ID 2 cannot be offered


def test_replay_teacher_forces_behavior_goal_after_candidate_scores_change():
    class ReplayHarness:
        cfg = SimpleNamespace(intrinsic_goal_mode="none", hrl_behavior_mode_condition=False)
        actor_critic = SimpleNamespace(core=SimpleNamespace(target_condition_start=2))

        def _uses_policy_graph(self):
            return True

        def _behavior_condition_from_states(self, states):
            return states[:, :4], None, None

    replay = ReplayHarness()
    behavior = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    newly_selected = torch.tensor([[8.0, 8.0, 0.0, 0.0, 1.0, 0.0]])
    output = DistanceLearnerReward._override_core_outputs_for_replay(
        replay, newly_selected, AttrDict(rnn_states=behavior)
    )
    torch.testing.assert_close(output[:, 2:], behavior)
    assert float(replay._last_behavior_replay_mismatch) == 0.0


def test_learner_applies_quality_once_at_accepted_rollout_boundary():
    class QualitySpy:
        def __init__(self):
            self.calls = []
            self.quality_counts = torch.zeros(3, dtype=torch.long)

        def update_from_rollout(self, states, options, valid):
            self.calls.append((states, options, valid.clone()))
            return int(valid.sum())

    class GraphSpy:
        def update_from_option_rollout(self, *args):
            return {"completion_count": torch.tensor(0.0)}

    quality = QualitySpy()
    options = torch.zeros(1, 3, HRLStateLayout(3).persistent_start)
    learner = SimpleNamespace(
        actor_critic=SimpleNamespace(core=SimpleNamespace(goal_quality=quality)),
        cfg=SimpleNamespace(hrl_fast_weight_half_life_options=5000, hrl_edge_confidence_threshold=0.5),
        _uses_policy_graph=lambda: True,
        _hrl_state_from_rnn=lambda _: options,
        _policy_graph=lambda: GraphSpy(),
        _uses_graph_recruitment=lambda: False,
        _uses_topological_manager=lambda: False,
    )
    states = torch.zeros(1, 3, 2)
    accepted = torch.tensor([[True, False]])
    stats = DistanceLearnerReward._update_policy_graph_from_rollout(learner, states, accepted)
    assert len(quality.calls) == 1
    torch.testing.assert_close(quality.calls[0][2], accepted)
    assert stats["goal_quality_events"] == 1


def test_odor_shape_centers_noise_and_zero_mode():
    rng = np.random.default_rng(7)
    np.testing.assert_array_equal(odor_observation(None, "zero", rng, 0.15), np.zeros(4))
    for index, center in enumerate(ODOR_CENTERS):
        assert clean_odor(center)[index] == 1.0
    center = ODOR_CENTERS[0]
    clean = clean_odor(center)
    draws = np.stack([odor_observation(center, "gaussian4", rng, 0.15) for _ in range(20000)])
    assert draws.shape == (20000, 4)
    np.testing.assert_allclose(draws.mean(0), clean, atol=0.005)
    np.testing.assert_allclose(draws.std(0), 0.15, atol=0.005)
    assert (draws < 0).any()  # no clipping


def test_recruitment_uses_identical_dg_input_width_and_gain():
    visual = torch.tensor([[1.0, 2.0, 3.0]])
    odor = torch.tensor([[0.25, -0.1, 0.4, 1.0]])
    encoder = SimpleNamespace(dg_odor_mode="gaussian4", dg_odor_gain=42.0)
    merged = HipposlamEncoder.append_dg_odor(encoder, visual, {"dg_odor": odor})
    torch.testing.assert_close(merged, torch.cat((visual, 42.0 * odor), dim=-1))
    assert merged.shape == (1, 7)
    encoder.dg_odor_mode = "none"
    assert HipposlamEncoder.append_dg_odor(encoder, visual, {}) is visual
