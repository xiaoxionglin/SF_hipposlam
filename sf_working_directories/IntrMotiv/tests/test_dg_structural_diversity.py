import torch

from sf_working_directories.IntrMotiv.dmlab.custom_encoder import DGProjection_batchnorm_relu
from sf_working_directories.IntrMotiv.dmlab.custom_learner import (
    DistanceLearnerReward,
    dg_ca3_temporal_exclusion_loss,
    dg_recruitment_candidate_mask,
    orthogonal_feature_residual,
)


def test_ca3_temporal_exclusion_is_a_dominant_event_margin_and_keeps_conflict_diagnostics():
    n_features, R, L = 3, 2, 4
    states = torch.zeros(2, n_features * (R + L - 1))
    ca3 = states.view(2, n_features, R + L - 1)
    ca3[:, 0, R - 1] = 1.0
    activity = torch.tensor([[1.5, 2.0, 0.0], [1.0, 0.0, 0.0]], requires_grad=True)
    dominant = torch.tensor([[False, True, False], [True, False, False]])
    valids = torch.ones(2, dtype=torch.bool)

    loss, conflict_fraction, conflicting_activation_fraction, conflict_activity = (
        dg_ca3_temporal_exclusion_loss(
            activity,
            states,
            dominant,
            n_features,
            R,
            L,
            coefficient=0.5,
            reward_scale=0.1,
            valids=valids,
            num_invalids=0,
        )
    )

    assert torch.isclose(loss, torch.tensor(0.15))
    assert torch.isclose(conflict_fraction, torch.tensor(2.0 / 3.0))
    assert torch.isclose(conflicting_activation_fraction, torch.tensor(1.0 / 3.0))
    assert torch.isclose(conflict_activity, torch.tensor(0.5))
    loss.backward()
    assert activity.grad[0, 1] > 0
    assert activity.grad[1, 0] > 0
    assert activity.grad[0, 0] == 0


def test_ca3_temporal_exclusion_centers_encourage_feedback_at_r():
    n_features, R, L = 2, 3, 5
    reward_scale = 0.1
    distances = torch.tensor([R - 1.0, float(R), R + 1.0])
    activity = torch.ones(3, n_features, requires_grad=True)
    dominant = torch.tensor([[True, False], [True, False], [True, False]])
    states = torch.zeros(3, n_features * (R + L - 1))
    valids = torch.ones(3, dtype=torch.bool)

    base_encourage_loss = -(reward_scale * distances * activity[:, 0]).mean()
    margin_loss, *_ = dg_ca3_temporal_exclusion_loss(
        activity,
        states,
        dominant,
        n_features,
        R,
        L,
        coefficient=1.0,
        reward_scale=reward_scale,
        valids=valids,
        num_invalids=0,
    )
    (base_encourage_loss + margin_loss).backward()

    assert activity.grad[0, 0] > 0
    assert torch.isclose(activity.grad[1, 0], torch.tensor(0.0), atol=1e-7)
    assert activity.grad[2, 0] < 0
    assert activity.grad[:, 1].eq(0).all()


def test_ca3_conflicting_activation_fraction_is_zero_without_activity():
    n_features, R, L = 3, 2, 4
    states = torch.zeros(1, n_features * (R + L - 1))
    states.view(1, n_features, R + L - 1)[0, 0, R - 1] = 1.0

    _, conflict_fraction, conflicting_activation_fraction, conflict_activity = (
        dg_ca3_temporal_exclusion_loss(
            torch.zeros(1, n_features),
            states,
            torch.zeros(1, n_features, dtype=torch.bool),
            n_features,
            R,
            L,
            coefficient=1.0,
            reward_scale=0.1,
            valids=torch.ones(1, dtype=torch.bool),
            num_invalids=0,
        )
    )

    assert torch.isclose(conflict_fraction, torch.tensor(2.0 / 3.0))
    assert conflicting_activation_fraction.item() == 0.0
    assert conflict_activity.item() == 0.0


def test_recruitment_candidate_is_one_shot_at_age_l_and_rejects_other_dg_history():
    n_features, R, L = 3, 2, 4
    expanded = R + L - 1
    states = torch.zeros(4, n_features * expanded)
    ca3 = states.view(4, n_features, expanded)
    ca3[1, 1, L - 1 :] = 1.0
    ca3[1, 1, 0] = 1.0  # same-source reactivation does not restart the clock
    ca3[2, 1, L:] = 1.0
    ca3[3, 1, L - 1 :] = 1.0
    ca3[3, 2, 0] = 1.0

    candidate, source, _ = dg_recruitment_candidate_mask(
        states, n_features, R, L, torch.ones(4, dtype=torch.bool)
    )

    assert candidate.tolist() == [False, True, False, False]
    assert source[1].item() == 1


def test_recruitment_ignores_a_tail_already_present_at_rollout_start():
    n_features, R, L = 2, 2, 4
    states = torch.zeros(2, n_features * (R + L - 1))
    ca3 = states.view(2, n_features, R + L - 1)
    ca3[:, 0, -1] = 1.0

    candidate, _, _ = dg_recruitment_candidate_mask(
        states, n_features, R, L, torch.ones(2, dtype=torch.bool)
    )

    assert candidate.tolist() == [False, False]


def test_orthogonal_feature_residual_has_unit_norm_and_leaves_existing_span():
    rows = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    residual = orthogonal_feature_residual(torch.tensor([2.0, 3.0, 4.0]), rows)

    assert residual is not None
    assert torch.allclose(residual, torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)
    assert torch.allclose(rows @ residual, torch.zeros(2), atol=1e-6)


def test_old_projection_checkpoint_loads_without_recruitment_buffers():
    original = DGProjection_batchnorm_relu(5, 3, intercept=2.43)
    old_state = {
        key: value
        for key, value in original.state_dict().items()
        if not key.startswith("recruitment_")
    }
    restored = DGProjection_batchnorm_relu(5, 3, intercept=2.43)

    restored.load_state_dict(old_state, strict=True)

    assert restored.recruitment_committed.eq(False).all()
    assert restored.recruitment_row_counts.eq(0).all()
    assert restored.recruitment_count.item() == 0


def test_recruitment_buffers_round_trip_in_state_dict():
    original = DGProjection_batchnorm_relu(5, 3, intercept=2.43)
    original.recruitment_committed[1] = True
    original.recruitment_activation_counts.copy_(torch.tensor([2.0, 7.0, 1.0]))
    original.recruitment_row_counts[1] = 1
    original.recruitment_count.fill_(1)
    restored = DGProjection_batchnorm_relu(5, 3, intercept=2.43)

    restored.load_state_dict(original.state_dict(), strict=True)

    assert restored.recruitment_committed.tolist() == [False, True, False]
    assert restored.recruitment_activation_counts.tolist() == [2.0, 7.0, 1.0]
    assert restored.recruitment_row_counts.tolist() == [0, 1, 0]
    assert restored.recruitment_count.item() == 1


def test_legacy_recruited_checkpoint_seeds_row_assignment_counts():
    original = DGProjection_batchnorm_relu(5, 3, intercept=2.43)
    original.recruitment_committed[2] = True
    legacy_state = {
        key: value
        for key, value in original.state_dict().items()
        if key not in ("recruitment_row_counts", "recruitment_repeat_count")
    }
    restored = DGProjection_batchnorm_relu(5, 3, intercept=2.43)

    restored.load_state_dict(legacy_state, strict=True)

    assert restored.recruitment_row_counts.tolist() == [0, 0, 1]
    assert restored.recruitment_repeat_count.item() == 0


def test_optimizer_row_reset_preserves_other_rows():
    parameter = torch.nn.Parameter(torch.ones(3, 2))
    optimizer = torch.optim.Adam([parameter], lr=0.01)
    parameter.sum().backward()
    optimizer.step()
    before = optimizer.state[parameter]["exp_avg"].clone()

    DistanceLearnerReward._reset_optimizer_row(optimizer, parameter, row=1)

    moment = optimizer.state[parameter]["exp_avg"]
    assert moment[1].eq(0).all()
    assert torch.equal(moment[0], before[0])
    assert torch.equal(moment[2], before[2])
