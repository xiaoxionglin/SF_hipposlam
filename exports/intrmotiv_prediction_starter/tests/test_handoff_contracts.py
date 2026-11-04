import pytest
import torch

from intrmotiv_transfer import (
    CA3TargetPredictor, advance_action_history, advance_ca3,
    future_target_labels, previous_action_onehot, shadow_prediction_loss,
    skip_silent_ca3,
)


@pytest.mark.parametrize("delay", [1, 2, 5, 12])
def test_silent_skip_matches_stepwise_updates_without_mutating_real_state(delay):
    state = torch.arange(20, dtype=torch.float32).reshape(2, 2, 5)
    original = state.clone()
    next_dg = torch.tensor([[0., 2.], [1., 0.]])
    expected = state.clone()
    for _ in range(delay - 1):
        expected = advance_ca3(expected, torch.zeros_like(next_dg), 2)
    expected = advance_ca3(expected, next_dg, 2)
    actual = skip_silent_ca3(state, next_dg, delay, 2)
    assert torch.equal(actual, expected)
    assert torch.equal(state, original)
    actual.zero_()
    assert torch.equal(state, original)


def test_skipping_a_nonzero_intermediate_input_is_not_exact():
    state, event = torch.zeros(1, 2, 5), torch.tensor([[0., 1.]])
    middle = advance_ca3(state, torch.tensor([[2., 0.]]), 2)
    actual = advance_ca3(middle, event, 2)
    assert not torch.equal(actual, skip_silent_ca3(state, event, 2, 2))


def test_action_history_reset_sentinel_and_causal_order():
    history = torch.ones(2, 3, 3)
    original = history.clone()
    updated = advance_action_history(history, torch.tensor([1, 2]), torch.tensor([True, False]))
    assert updated[0].eq(0).all()
    assert updated[1, -1].tolist() == [0., 0., 1.]
    assert torch.equal(updated[1, :-1], history[1, 1:])
    assert torch.equal(history, original)
    assert previous_action_onehot(torch.tensor([3]), 3).eq(0).all()
    with pytest.raises(ValueError):
        previous_action_onehot(torch.tensor([4]), 3)


def fixture():
    targets = torch.ones(1, 5, 1)
    dg = torch.zeros_like(targets)
    dones = torch.zeros(1, 5, dtype=torch.bool)
    return targets, dg, dones


def test_full_window_negative_but_trailing_windows_are_censored():
    targets, dg, dones = fixture()
    hit, delay, valid = future_target_labels(targets, dg, dones, 3)
    assert hit.eq(0).all()
    assert valid.tolist() == [[True, True, False, False, False]]
    assert not future_target_labels(targets, dg, dones, 10)[2].any()


def test_observed_positive_is_usable_even_when_remaining_window_is_short():
    targets, dg, dones = fixture()
    dg[:, 4] = 1
    hit, delay, valid = future_target_labels(targets, dg, dones, 3)
    assert hit.tolist() == [[0., 1., 1., 1., 0.]]
    assert delay.tolist() == [[0., 3., 2., 1., 0.]]
    assert valid.tolist() == [[True, True, True, True, False]]


def test_episode_reset_cannot_create_a_hit_in_previous_episode():
    targets, dg, dones = fixture()
    dones[:, 1] = True
    dg[:, 2] = 1  # observation after reset
    hit, _, valid = future_target_labels(targets, dg, dones, 3)
    assert not hit[:, :2].any()
    assert not valid[:, :2].any()


def test_padding_cannot_create_negative_labels_or_bridge_a_gap():
    targets, dg, dones = fixture()
    dg[:, 3] = 1
    valids = torch.tensor([[True, False, True, True, True]])
    hit, _, usable = future_target_labels(targets, dg, dones, 3, valids)
    assert not hit[0, 0]
    assert not usable[0, 0]
    assert not usable[0, 1]
    assert usable[0, 2]


def test_first_hit_delay_is_not_overwritten_and_zero_target_is_ignored():
    targets, dg, dones = fixture()
    dg[:, 1:3] = 1
    targets[:, 3] = 0
    hit, delay, usable = future_target_labels(targets, dg, dones, 3)
    assert delay[0, 0] == 1
    assert hit[0, 0] == 1
    assert not usable[0, 3]


def test_head_only_update_cannot_change_encoder_or_actor():
    torch.manual_seed(1)
    encoder = torch.nn.Linear(3, 4)
    actor = torch.nn.Linear(4, 2)
    head = CA3TargetPredictor(4, 2, 8)
    x = torch.randn(6, 3)
    ca3 = encoder(x)
    before_logits = actor(ca3).detach().clone()
    before_head = [p.detach().clone() for p in head.parameters()]
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
    targets = torch.eye(2).repeat(3, 1)
    loss, _ = shadow_prediction_loss(head, ca3, targets, torch.tensor([1., 0., 1., 0., 1., 0.]),
                                     torch.tensor([1., 0., 2., 0., 3., 0.]), torch.ones(6, dtype=torch.bool), 3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert all(p.grad is None for p in encoder.parameters())
    assert all(p.grad is None for p in actor.parameters())
    assert torch.equal(before_logits, actor(encoder(x)).detach())
    assert any(not torch.equal(old, new) for old, new in zip(before_head, head.parameters()))


def test_all_censored_batch_has_finite_zero_loss_and_zero_head_gradient():
    head = CA3TargetPredictor(4, 2, 8)
    loss, stats = shadow_prediction_loss(head, torch.zeros(3, 4), torch.zeros(3, 2),
                                        torch.zeros(3), torch.zeros(3), torch.zeros(3, dtype=torch.bool), 4)
    assert loss.item() == 0
    assert stats["usable_count"].item() == 0
    loss.backward()
    assert all(p.grad is not None and p.grad.eq(0).all() for p in head.parameters())


def test_positive_time_error_is_reported_in_decisions():
    head = CA3TargetPredictor(4, 2, 8)
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()  # sigmoid(0) * horizon = 4 decisions
    _, stats = shadow_prediction_loss(head, torch.zeros(2, 4), torch.eye(2),
                                     torch.tensor([1., 0.]), torch.tensor([1., 0.]),
                                     torch.ones(2, dtype=torch.bool), 8)
    assert stats["time_mae_decisions"].item() == 3
