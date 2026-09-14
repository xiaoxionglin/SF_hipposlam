from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ContextualDGFeedback(nn.Module):
    """Causal, identity-initialized CA3 feedback for DG activation."""

    def __init__(
        self,
        n_features: int,
        recent_steps: int,
        intercept: float,
        mode: str,
        action_count: int = 0,
        gradient_mode: str = "direct",
    ):
        super().__init__()
        if mode not in ("gate", "additive"):
            raise ValueError(f"Unknown DG context feedback mode: {mode}")
        if gradient_mode not in ("direct", "bptt"):
            raise ValueError(f"Unknown DG context gradient mode: {gradient_mode}")
        self.n_features = int(n_features)
        self.recent_steps = int(recent_steps)
        self.intercept = float(intercept)
        self.mode = mode
        self.action_count = int(action_count)
        self.gradient_mode = gradient_mode
        context_size = self.n_features * self.recent_steps
        context_size += self.action_count * self.recent_steps
        self.normalizer = nn.LayerNorm(context_size, elementwise_affine=False)
        self.adapter = nn.Linear(context_size, self.n_features)
        # ActorCritic recursively initializes Linear modules. Preserve the exact
        # identity initialization required by the experimental contract.
        self.adapter._intrmotiv_preserve_initialization = True
        nn.init.zeros_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)

    def forward(
        self,
        visual_evidence: Tensor,
        previous_ca3: Tensor,
        action_history: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if previous_ca3.ndim != 3:
            raise ValueError("previous_ca3 must have shape [batch, feature, history]")
        ca3 = previous_ca3[:, :, : self.recent_steps]
        if self.gradient_mode == "direct":
            ca3 = ca3.detach()
        pieces = [ca3.reshape(ca3.size(0), -1)]
        if self.action_count:
            if action_history is None:
                raise ValueError("action_history is required for CA3+ACTION feedback")
            expected = (visual_evidence.size(0), self.recent_steps, self.action_count)
            if tuple(action_history.shape) != expected:
                raise ValueError(f"Expected action history {expected}, got {tuple(action_history.shape)}")
            pieces.append(action_history.reshape(action_history.size(0), -1))
        context = self.normalizer(torch.cat(pieces, dim=-1))
        context_logits = self.adapter(context)
        baseline = F.relu(visual_evidence - self.intercept)
        if self.mode == "gate":
            modulation = 2.0 * torch.sigmoid(context_logits)
            activity = F.relu(modulation * visual_evidence - self.intercept)
            magnitude = (modulation - 1.0).abs()
            saturation = ((modulation < 0.05) | (modulation > 1.95)).to(visual_evidence.dtype)
        else:
            modulation = torch.tanh(context_logits)
            activity = F.relu(visual_evidence + modulation - self.intercept)
            magnitude = modulation.abs()
            saturation = (modulation.abs() > 0.95).to(visual_evidence.dtype)
        baseline_active = baseline > 0
        active = activity > 0
        stats = {
            "created_fraction": (active & ~baseline_active).float().mean(),
            "suppressed_fraction": (~active & baseline_active).float().mean(),
            "unchanged_fraction": (active == baseline_active).float().mean(),
            "modulation_abs_mean": magnitude.mean(),
            "modulation_saturation_fraction": saturation.mean(),
        }
        return activity, stats


class DGTransitionPredictor(nn.Module):
    """Training-only next-landmark predictor and intervention-only control."""

    def __init__(self, n_features: int, mode: str, hidden_size: int = 128):
        super().__init__()
        if mode not in ("passive", "goal"):
            raise ValueError(f"Unknown DG transition prediction mode: {mode}")
        self.n_features = int(n_features)
        self.mode = mode
        condition_size = self.n_features if mode == "goal" else 0
        self.predictor = nn.Sequential(
            nn.Linear(self.n_features + condition_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.n_features + 1),
        )
        if mode == "goal":
            self.control = nn.Sequential(
                nn.Linear(condition_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.n_features + 1),
            )
        else:
            self.control_logits = nn.Parameter(torch.zeros(self.n_features + 1))

    def forward(self, source_dg: Tensor, goal: Tensor | None) -> tuple[Tensor, Tensor]:
        if self.mode == "goal":
            if goal is None:
                raise ValueError("Goal-conditioned prediction requires a goal one-hot")
            main = self.predictor(torch.cat((source_dg, goal), dim=-1))
            control = self.control(goal.detach())
        else:
            main = self.predictor(source_dg)
            control = self.control_logits.unsqueeze(0).expand(source_dg.size(0), -1)
        return main, control


@dataclass
class TransitionPredictionBatch:
    source_dg: Tensor
    goal: Tensor
    outcome: Tensor
    validation: Tensor
    scheduled_count: Tensor
    applied_count: Tensor
    boundary_drop_count: Tensor


def build_transition_prediction_batch(
    dg_activity: Tensor,
    completed: Tensor,
    timeout: Tensor,
    source: Tensor,
    goal: Tensor,
    outcome: Tensor,
    elapsed: Tensor,
    valids: Tensor,
    recurrence: int,
    n_features: int,
) -> TransitionPredictionBatch:
    """Recover differentiable source activations for completed behavior options."""
    if dg_activity.ndim != 2 or dg_activity.size(1) != n_features:
        raise ValueError("dg_activity must have shape [samples, n_features]")
    samples = dg_activity.size(0)
    if samples % int(recurrence) != 0:
        raise ValueError("prediction samples must divide evenly into recurrence segments")
    segments = samples // int(recurrence)
    shape = (segments, int(recurrence))
    completed = completed.reshape(shape).bool()
    timeout = timeout.reshape(shape).bool()
    source = source.reshape(shape).long()
    goal = goal.reshape(shape).long()
    outcome = outcome.reshape(shape).long()
    elapsed = elapsed.reshape(shape).long()
    valids = valids.reshape(shape).bool()
    dg = dg_activity.reshape(segments, int(recurrence), n_features)

    event = completed & valids & (source >= 0) & (source < n_features)
    event &= (goal >= 0) & (goal < n_features) & (elapsed > 0)
    batch_idx, completion_t = torch.where(event)
    scheduled = event.sum()
    if batch_idx.numel() == 0:
        empty_dg = dg_activity[:0]
        empty_long = source.reshape(-1)[:0]
        empty_bool = event.reshape(-1)[:0]
        return TransitionPredictionBatch(
            empty_dg,
            F.one_hot(empty_long, n_features).to(dg_activity.dtype),
            empty_long,
            empty_bool,
            scheduled,
            scheduled,
            scheduled,
        )

    event_elapsed = elapsed[batch_idx, completion_t]
    source_t = completion_t - event_elapsed + 1
    within = source_t >= 0
    boundary_drops = (~within).sum()
    batch_idx = batch_idx[within]
    completion_t = completion_t[within]
    source_t = source_t[within]
    if batch_idx.numel() == 0:
        empty_dg = dg_activity[:0]
        empty_long = source.reshape(-1)[:0]
        empty_bool = event.reshape(-1)[:0]
        return TransitionPredictionBatch(
            empty_dg,
            F.one_hot(empty_long, n_features).to(dg_activity.dtype),
            empty_long,
            empty_bool,
            scheduled,
            scheduled * 0,
            boundary_drops,
        )

    source_id = source[batch_idx, completion_t]
    source_vectors = dg[batch_idx, source_t]
    replay_active = source_vectors.gather(1, source_id.unsqueeze(1)).squeeze(1) > 0
    batch_idx = batch_idx[replay_active]
    completion_t = completion_t[replay_active]
    source_vectors = source_vectors[replay_active]
    goal_id = goal[batch_idx, completion_t]
    labels = outcome[batch_idx, completion_t]
    labels = torch.where(timeout[batch_idx, completion_t], torch.full_like(labels, n_features), labels)
    label_valid = (labels >= 0) & (labels <= n_features)
    source_vectors = source_vectors[label_valid]
    goal_id = goal_id[label_valid]
    labels = labels[label_valid]
    batch_idx = batch_idx[label_valid]
    completion_t = completion_t[label_valid]

    # Stable, data-derived validation split. It does not require a new rollout
    # field and gives the same event to PASS and GOAL conditions.
    validation = ((batch_idx * 131 + completion_t * 17 + goal_id * 7 + labels) % 10) == 0
    return TransitionPredictionBatch(
        source_vectors,
        F.one_hot(goal_id, n_features).to(dg_activity.dtype),
        labels,
        validation,
        scheduled,
        torch.as_tensor(source_vectors.size(0), device=dg_activity.device),
        boundary_drops,
    )


def transition_prediction_losses(
    predictor: DGTransitionPredictor,
    batch: TransitionPredictionBatch,
) -> tuple[Tensor, dict[str, Tensor]]:
    zero = batch.source_dg.sum() * 0.0
    if batch.source_dg.size(0) == 0:
        return zero, {
            "main_loss": zero,
            "control_loss": zero,
            "validation_main_ce": zero,
            "validation_control_ce": zero,
            "validation_state_gain": zero,
            "validation_accuracy": zero,
            "validation_count": zero,
        }
    main_logits, control_logits = predictor(batch.source_dg, batch.goal)
    train = ~batch.validation
    if train.any():
        main_loss = F.cross_entropy(main_logits[train], batch.outcome[train])
        control_loss = F.cross_entropy(control_logits[train], batch.outcome[train])
    else:
        main_loss = zero
        control_loss = zero
    validation = batch.validation
    if validation.any():
        main_ce = F.cross_entropy(main_logits[validation], batch.outcome[validation]).detach()
        control_ce = F.cross_entropy(control_logits[validation], batch.outcome[validation]).detach()
        accuracy = (main_logits[validation].argmax(dim=-1) == batch.outcome[validation]).float().mean().detach()
    else:
        main_ce = zero.detach()
        control_ce = zero.detach()
        accuracy = zero.detach()
    stats = {
        "main_loss": main_loss.detach(),
        "control_loss": control_loss.detach(),
        "validation_main_ce": main_ce,
        "validation_control_ce": control_ce,
        "validation_state_gain": control_ce - main_ce,
        "validation_accuracy": accuracy,
        "validation_count": validation.sum().detach().float(),
    }
    return main_loss + control_loss, stats
