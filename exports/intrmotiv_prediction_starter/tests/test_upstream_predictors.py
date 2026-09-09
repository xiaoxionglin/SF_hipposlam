import torch
from intrmotiv_transfer.contextual_dg import (ContextualDGFeedback, DGTransitionPredictor, build_transition_prediction_batch, transition_prediction_losses)

def test_feedback_zero_initialization_matches_visual_baseline():
    visual = torch.tensor([[3.0, 2.2, -1.0]])
    ca3 = torch.randn(1, 3, 6)
    expected = torch.relu(visual - 2.0)
    for mode in ("gate", "additive"):
        feedback = ContextualDGFeedback(3, 2, 2.0, mode)
        actual, stats = feedback(visual, ca3)
        assert torch.equal(actual, expected)
        assert stats["created_fraction"].item() == 0.0
        assert stats["suppressed_fraction"].item() == 0.0


def test_additive_feedback_uses_bounded_residual():
    feedback = ContextualDGFeedback(2, 1, 2.0, "additive")
    with torch.no_grad():
        feedback.adapter.bias.copy_(torch.tensor([100.0, -100.0]))
    activity, _ = feedback(torch.tensor([[1.5, 3.0]]), torch.zeros(1, 2, 2))
    assert torch.allclose(activity, torch.tensor([[0.5, 0.0]]), atol=1e-6)


def test_direct_detaches_context_but_bptt_reaches_previous_ca3():
    visual = torch.tensor([[3.0, 3.0]], requires_grad=True)
    for gradient_mode, expects_ca3_gradient in (("direct", False), ("bptt", True)):
        ca3 = torch.randn(1, 2, 2, requires_grad=True)
        feedback = ContextualDGFeedback(2, 2, 2.0, "additive", gradient_mode=gradient_mode)
        with torch.no_grad():
            feedback.adapter.weight.copy_(torch.tensor([[0.1, -0.2, 0.3, 0.4], [-0.3, 0.2, 0.1, -0.4]]))
        feedback(visual, ca3)[0].sum().backward(retain_graph=True)
        reached = ca3.grad is not None and ca3.grad.abs().sum().item() > 0
        assert reached is expects_ca3_gradient


def test_action_context_requires_exact_causal_history_shape():
    feedback = ContextualDGFeedback(2, 2, 2.0, "gate", action_count=3)
    visual = torch.ones(1, 2)
    ca3 = torch.zeros(1, 2, 3)
    actions = torch.zeros(1, 2, 3)
    feedback(visual, ca3, actions)
    try:
        feedback(visual, ca3, torch.zeros(1, 1, 3))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong action-history shape must be rejected")


def _prediction_batch():
    dg = torch.zeros(8, 3, requires_grad=True)
    with torch.no_grad():
        dg[1, 0] = 1.0
        dg[5, 1] = 1.0
    completed = torch.zeros(8, dtype=torch.bool)
    completed[3] = True  # elapsed=3 -> source timestep 1, within segment
    completed[4] = True  # elapsed=3 -> crosses the recurrence boundary
    timeout = torch.zeros(8, dtype=torch.bool)
    timeout[4] = True
    source = torch.full((8,), -1)
    source[3] = 0
    source[4] = 1
    goal = torch.full((8,), -1)
    goal[3] = 2
    goal[4] = 0
    outcome = torch.full((8,), -1)
    outcome[3] = 2
    elapsed = torch.zeros(8)
    elapsed[3] = 3
    elapsed[4] = 3
    batch = build_transition_prediction_batch(
        dg,
        completed,
        timeout,
        source,
        goal,
        outcome,
        elapsed,
        torch.ones(8, dtype=torch.bool),
        recurrence=4,
        n_features=3,
    )
    return dg, batch


def test_prediction_batch_recovers_source_and_drops_boundaries():
    _, batch = _prediction_batch()
    assert batch.scheduled_count.item() == 2
    assert batch.boundary_drop_count.item() == 1
    assert batch.applied_count.item() == 1
    assert batch.outcome.tolist() == [2]
    assert batch.goal.argmax(dim=-1).tolist() == [2]


def test_predictor_loss_reaches_source_dg_but_control_is_detached():
    dg, batch = _prediction_batch()
    # Ensure this single synthetic event participates in training.
    batch.validation.zero_()
    predictor = DGTransitionPredictor(3, "goal", hidden_size=8)
    loss, stats = transition_prediction_losses(predictor, batch)
    loss.backward()
    assert dg.grad is not None and dg.grad[1].abs().sum() > 0
    assert stats["main_loss"].isfinite()


def test_passive_and_goal_predictors_have_timeout_class():
    for mode in ("passive", "goal"):
        predictor = DGTransitionPredictor(3, mode, hidden_size=8)
        source = torch.randn(2, 3)
        goal = torch.eye(3)[:2]
        main, control = predictor(source, goal)
        assert main.shape == (2, 4)
        assert control.shape == (2, 4)
