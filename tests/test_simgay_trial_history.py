"""Completed-trial replay against independent row-wise reference equations."""

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from sf_examples.simgay.manager import HighLevelContextRNN_Learner, TrialHistory


def reference_append(layout, history, h, z, reward, mask, trigger, outcome):
    """Row-wise specification of the flat records used before extraction."""
    rows = []
    for b in range(history.size(0)):
        if mask[b] and z[b].sum() > 0.5:
            fields = [h[b].detach(), z[b].detach(), reward[b : b + 1]]
            if layout.representations:
                fields += [trigger[b].detach(), outcome[b].detach()]
            fields.append(reward.new_ones(1))
            rows.append(torch.cat([history[b, layout.event_width :], *fields]))
        else:
            rows.append(history[b])
    return torch.stack(rows)


def reference_scores(layout, learner, history, fallback):
    """Explicit per-row unroll, independent of the helper's masking/update API."""
    scores = []
    for b, records in enumerate(history.reshape(-1, layout.history_len, layout.event_width)):
        completed = records[records[:, -1] > 0.5]
        if len(completed) == 0:
            h = fallback[b]
        else:
            h = completed[0, : layout.d_H]
            for record in completed:
                # z, reward, and optional representations occupy every field
                # between the saved hidden state and the validity flag.
                h = learner.rnn_cell(record[layout.d_H : -1], h)
        scores.append(learner.head(h))
    return torch.stack(scores)


def example_history(layout):
    events = torch.randn(3, layout.history_len, layout.event_width)
    events[:, :, layout.d_H : layout.d_H + layout.K] = F.one_hot(
        torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), layout.K
    ).float()
    events[:, :, -1] = torch.tensor([[0, 1, 1], [1, 0, 1], [0, 0, 0]])
    return events.flatten(1)


@pytest.mark.parametrize("representations", [False, True])
def test_record_layout_shift_masks_and_detachment(representations):
    torch.manual_seed(12)
    layout = TrialHistory(K=2, d_H=3, history_len=3, representations=representations, d_rep=4)
    assert layout.event_width == (15 if representations else 7)
    history = torch.zeros(3, layout.history_size)
    reference = history.clone()
    for t in range(5):
        h = torch.randn(3, 3, requires_grad=True)
        z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], requires_grad=True)
        reward = torch.randn(3, requires_grad=True)
        trigger = torch.randn(3, 4, requires_grad=True)
        outcome = torch.randn(3, 4, requires_grad=True)
        mask = torch.tensor([True, t % 2 == 0, True])
        history = layout.append_event(history, h, z, reward, mask, trigger, outcome)
        reference = reference_append(layout, reference, h, z, reward, mask, trigger, outcome)
        torch.testing.assert_close(history, reference, rtol=0, atol=0)
    assert history[2].count_nonzero() == 0  # Initial trigger has no previous trial.
    history.sum().backward()
    assert h.grad is None
    assert z.grad is None
    assert trigger.grad is None
    assert outcome.grad is None
    torch.testing.assert_close(reward.grad, torch.tensor([1.0, 1.0, 0.0]))


@pytest.mark.parametrize("representations", [False, True])
@pytest.mark.parametrize("is_policy", [False, True])
def test_replay_matches_reference_scores_and_parameter_gradients(representations, is_policy, monkeypatch):
    torch.manual_seed(23)
    layout = TrialHistory(K=2, d_H=3, history_len=3, representations=representations, d_rep=4)
    learner = HighLevelContextRNN_Learner(K=2, d_H=3, is_policy=is_policy, representations=representations, d_rep=4)
    reference = deepcopy(learner)
    history = example_history(layout)
    fallback = torch.randn(3, layout.d_H, requires_grad=True)
    reference_fallback = fallback.detach().clone().requires_grad_()

    def fail_sampling(*args, **kwargs):
        raise AssertionError("Trial replay must use saved modes, never sample new modes")

    monkeypatch.setattr(learner, "sample_mode", fail_sampling)
    actual = layout.scores_from_history(learner, history, fallback)
    expected = reference_scores(layout, reference, history, reference_fallback)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for parameter, reference_parameter in zip(learner.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad)
    torch.testing.assert_close(fallback.grad, reference_fallback.grad)
    assert fallback.grad[:2].count_nonzero() == 0
    assert fallback.grad[2].abs().sum() > 0  # Empty history uses the caller's state.


@pytest.mark.parametrize("representations", [False, True])
def test_replay_gradient_reaches_oldest_completed_trial(representations):
    torch.manual_seed(34)
    layout = TrialHistory(K=2, d_H=3, history_len=3, representations=representations, d_rep=4)
    learner = HighLevelContextRNN_Learner(K=2, d_H=3, representations=representations, d_rep=4)
    history = example_history(layout)[1:2].detach().requires_grad_()
    scores = layout.scores_from_history(learner, history, torch.zeros(1, 3))
    scores.square().sum().backward()
    gradients = history.grad.reshape(1, layout.history_len, layout.event_width)
    assert gradients[0, 0, layout.d_H + layout.K].abs() > 0  # Reward two records back.
    assert gradients[0, 1].count_nonzero() == 0  # Invalid record is skipped.
    assert learner.rnn_cell.weight_hh.grad.abs().sum() > 0


def test_representation_history_requires_both_representations():
    layout = TrialHistory(K=2, d_H=3, history_len=3, representations=True, d_rep=4)
    with pytest.raises(ValueError, match="e_trigger and e_outcome"):
        layout.append_event(
            torch.zeros(1, layout.history_size),
            torch.zeros(1, 3),
            torch.tensor([[1.0, 0.0]]),
            torch.ones(1),
            torch.ones(1, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="at least one"):
        TrialHistory(K=2, d_H=3, history_len=0)
