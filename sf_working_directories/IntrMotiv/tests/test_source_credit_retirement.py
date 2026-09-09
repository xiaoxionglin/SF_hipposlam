from types import SimpleNamespace

import torch

from sf_working_directories.IntrMotiv.dmlab.custom_core import SimpleSequenceWithBypassCore
from sf_working_directories.IntrMotiv.dmlab.custom_learner import (
    DistanceLearnerReward,
    build_matched_encoder_credit,
    retirement_endpoint_allowed,
)
from sf_working_directories.IntrMotiv.dmlab.dg_recruitment_graph import (
    PersistentPredictiveRecruitmentEvidence,
)


def _credit_inputs():
    baseline = 7
    progression = torch.full((1, 6, 4), baseline, dtype=torch.long)
    candidates = torch.zeros_like(progression, dtype=torch.bool)
    dominant = torch.zeros_like(candidates)
    valid = torch.ones(1, 6, dtype=torch.bool)
    # Verified source onset row 1 at t=1, followed by row 3 at t=4.
    progression[0, 1, 1] = 0
    candidates[0, 1, 1] = True
    dominant[0, 1, 1] = True
    progression[0, 4, 1] = 3
    progression[0, 4, 3] = 0
    candidates[0, 4, 3] = True
    dominant[0, 4, 3] = True
    return progression, candidates, dominant, valid, baseline


def test_arrival_and_source_credit_share_events_and_reward_mass_but_not_recipient():
    args = _credit_inputs()
    arrival_reward, arrival_mask, arrival_stats = build_matched_encoder_credit(*args, 0.1, "arrival")
    source_reward, source_mask, source_stats = build_matched_encoder_credit(*args, 0.1, "source")
    assert arrival_mask[0, 4, 3]
    assert source_mask[0, 1, 1]
    assert arrival_reward[0, 4, 3].item() == source_reward[0, 1, 1].item()
    assert torch.allclose(arrival_reward.sum(), source_reward.sum())
    assert arrival_stats["credited"].item() == source_stats["credited"].item() == 1
    assert arrival_stats["reward_mass"].item() == source_stats["reward_mass"].item()


def test_source_credit_drops_boundary_invalid_and_unverified_alignment():
    progression, candidates, dominant, valid, baseline = _credit_inputs()
    progression[0, 4, 1] = 5  # resolves before this actor rollout
    _, _, stats = build_matched_encoder_credit(
        progression, candidates, dominant, valid, baseline, 0.1, "source"
    )
    assert stats["boundary_dropped"].item() == 1

    progression, candidates, dominant, valid, baseline = _credit_inputs()
    valid[0, 2] = False
    _, _, stats = build_matched_encoder_credit(
        progression, candidates, dominant, valid, baseline, 0.1, "source"
    )
    assert stats["invalid_interval"].item() == 1

    progression, candidates, dominant, valid, baseline = _credit_inputs()
    dominant[0, 1, 1] = False
    _, _, stats = build_matched_encoder_credit(
        progression, candidates, dominant, valid, baseline, 0.1, "source"
    )
    assert stats["alignment_failure"].item() == 1


def test_source_credit_collisions_accumulate_deterministically():
    progression, candidates, dominant, valid, baseline = _credit_inputs()
    progression[0, 5, 1] = 4
    progression[0, 5, 2] = 0
    candidates[0, 5, 2] = True
    dominant[0, 5, 2] = True
    reward, mask, stats = build_matched_encoder_credit(
        progression, candidates, dominant, valid, baseline, 0.1, "source"
    )
    assert mask[0, 1, 1]
    assert torch.isclose(reward[0, 1, 1], torch.tensor(0.7))
    assert stats["collisions"].item() == 1


def test_predecessor_row_uses_behavior_dominant_label_within_nearest_lag_tie():
    progression, candidates, dominant, valid, baseline = _credit_inputs()
    # A lower-index row starts simultaneously with source row 1. At the later
    # arrival both rows have the same nearest lag; row 1 must still be selected
    # because it was the behavior-labeled dominant onset.
    progression[0, 1, 0] = 0
    progression[0, 4, 0] = 3
    candidates[0, 1, 0] = True
    reward, mask, stats = build_matched_encoder_credit(
        progression, candidates, dominant, valid, baseline, 0.1, "source"
    )
    assert mask[0, 1, 1]
    assert not mask[0, 1, 0]
    assert torch.isclose(reward[0, 1, 1], torch.tensor(0.3))
    assert stats["credited"].item() == 1
    # The earlier source onset itself has no predecessor in this synthetic
    # rollout, so it contributes the one remaining alignment failure.
    assert stats["alignment_failure"].item() == 1


def test_predecessor_excludes_ongoing_zero_age_rows_not_just_new_candidates():
    progression, candidates, dominant, valid, baseline = _credit_inputs()
    # Row 0 is active again at the arrival decision but did not newly onset;
    # it is simultaneous (age zero), not a predecessor. The prior row-1 onset
    # at lag three remains the source.
    progression[0, 4, 0] = 0
    candidates[0, 4, 0] = False
    reward, mask, stats = build_matched_encoder_credit(
        progression, candidates, dominant, valid, baseline, 0.1, "source"
    )
    assert mask[0, 1, 1]
    assert not mask[0, 4, 0]
    assert torch.isclose(reward[0, 1, 1], torch.tensor(0.3))
    assert stats["credited"].item() == 1


def test_encoder_loss_gradient_is_confined_to_behavior_labeled_credit_row():
    learner = DistanceLearnerReward.__new__(DistanceLearnerReward)
    learner.cfg = SimpleNamespace(Hippo_n_feature=4)
    head = torch.ones(2, 4, requires_grad=True)
    rewards = torch.zeros(2, 4)
    rewards[0, 2] = 0.4
    mask = rewards > 0
    loss = learner._encoder_loss(head, rewards, mask, torch.ones(2, dtype=torch.bool), 0)
    loss.backward()
    assert head.grad[0, 2] != 0
    assert torch.count_nonzero(head.grad) == 1


def test_silent_and_open_endpoint_gates_differ_only_for_active_endpoints():
    silent = torch.zeros(4, dtype=torch.bool)
    active = silent.clone()
    active[2] = True
    assert retirement_endpoint_allowed("silent", silent)
    assert retirement_endpoint_allowed("open", silent)
    assert not retirement_endpoint_allowed("silent", active)
    assert retirement_endpoint_allowed("open", active)


def test_persistent_pred_decays_in_event_order_and_requires_four_attempts_per_context():
    evidence = PersistentPredictiveRecruitmentEvidence(4)
    source = torch.tensor([0] * 10)
    goal = torch.tensor([3] * 10)
    context = torch.tensor([1] * 5 + [2] * 5)
    success = torch.tensor([True] * 5 + [False] * 5)
    evidence.update(source, goal, context, success, half_life_events=5000)
    result = evidence.eligibility(torch.zeros(4), min_context_attempts=4.0)
    assert result.eligible.tolist() == [True, False, False, False]
    assert result.victim == 0
    evidence.update(torch.tensor([1]), torch.tensor([3]), torch.tensor([2]), torch.tensor([True]), 1.0)
    dropped = evidence.eligibility(torch.zeros(4), min_context_attempts=4.0)
    assert dropped.victim is None


def test_persistent_pred_checkpoint_defaults_roundtrip_and_three_axis_invalidation():
    evidence = PersistentPredictiveRecruitmentEvidence(4)
    evidence.attempts[0, 1, 2] = 5
    evidence.successes[0, 1, 2] = 3
    state = evidence.state_dict()
    restored = PersistentPredictiveRecruitmentEvidence(4)
    restored.load_state_dict(state, strict=True)
    assert torch.equal(restored.attempts, evidence.attempts)
    restored.invalidate_node(1)
    assert restored.attempts[1, :, :].eq(0).all()
    assert restored.attempts[:, 1, :].eq(0).all()
    assert restored.attempts[:, :, 1].eq(0).all()
    legacy = PersistentPredictiveRecruitmentEvidence(4)
    legacy.load_state_dict({}, strict=True)
    assert legacy.attempts.eq(0).all()


def test_core_checkpoint_contains_persistent_pred_without_changing_rnn_state_shape():
    cfg = SimpleNamespace(
        Hippo_R=2,
        Hippo_L=4,
        Hippo_n_feature=4,
        hrl_controllable_graph=True,
        hrl_graph_memory="policy_buffer",
        dg_orthogonal_recruitment=True,
        dg_orthogonal_recruitment_mode="graph",
    )
    core = SimpleSequenceWithBypassCore(cfg, 16)
    assert core.predictive_recruitment_evidence is not None
    assert core.predictive_recruitment_evidence.attempts.shape == (4, 4, 4)
    recurrent_state_size = core.total_state_size
    state = core.state_dict()
    restored = SimpleSequenceWithBypassCore(cfg, 16)
    restored.load_state_dict(state, strict=True)
    assert restored.total_state_size == recurrent_state_size

    # Checkpoints written before persistent PRED have no evidence buffers. The
    # child module supplies zero defaults so strict model restoration remains
    # backward compatible.
    legacy_state = {
        key: value
        for key, value in state.items()
        if not key.startswith("predictive_recruitment_evidence.")
    }
    legacy_restored = SimpleSequenceWithBypassCore(cfg, 16)
    legacy_restored.load_state_dict(legacy_state, strict=True)
    assert legacy_restored.predictive_recruitment_evidence.attempts.eq(0).all()
