"""Editable action-conditioned next-DG-event model. No policy or state mutation.

An event is the next future timestep with ANY positive DG input, including a
continuing burst. It is not the next distinct landmark. Labels describe the
executed first action followed by the behavior in that rollout.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class EventLabels:
    dg: torch.Tensor
    delay: torch.Tensor
    hit: torch.Tensor
    usable: torch.Tensor


@torch.no_grad()
def next_dg_event_labels(dg, dones_after, valids, horizon):
    """[B,T,F] DG and [B,T] masks, chronological within each recurrence.

    A negative requires H observed event-free steps. Truncated windows, reset
    boundaries and padding are censored. A positive needs only the first hit.
    Terminal observations are not supplied by this adapter, so done censors.
    """
    if dg.ndim != 3 or dones_after.shape != dg.shape[:2] or valids.shape != dg.shape[:2]:
        raise ValueError("Expected DG [B,T,F], dones_after/valids [B,T]")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not torch.isfinite(dg).all() or (dg < 0).any():
        raise ValueError("DG labels must be finite nonnegative post-threshold activations")
    hit = torch.zeros_like(valids, dtype=torch.bool)
    delay = torch.zeros_like(valids, dtype=dg.dtype)
    outcome = torch.zeros_like(dg)
    observed = torch.zeros_like(valids, dtype=torch.long)
    alive = valids.bool().clone()
    steps = dg.size(1)
    for delta in range(1, min(horizon, steps - 1) + 1):
        end = steps - delta
        alive[:, :end] &= ~dones_after[:, delta - 1:steps - 1].bool() & valids[:, delta:].bool()
        observed[:, :end] += alive[:, :end].long()
        first = alive[:, :end] & ~hit[:, :end] & dg[:, delta:].gt(0).any(-1)
        outcome[:, :end] = torch.where(first[..., None], dg[:, delta:], outcome[:, :end])
        delay[:, :end] = torch.where(first, float(delta), delay[:, :end])
        hit[:, :end] |= first
    usable = valids.bool() & (hit | observed.eq(horizon))
    return EventLabels(outcome, delay, hit, usable)


class DGEventWorldModel(nn.Module):
    """Current CA3 + candidate first action -> event, multi-DG vector and delay.

    Future action is separate from action history in the state. Simultaneous
    DG activations are represented with independent logits, not argmax labels.
    Arrival time and amplitude are point estimates conditional on an event;
    this is a shadow predictor, not a calibrated stochastic rollout model.
    """

    def __init__(self, ca3_size, n_features, action_count, hidden_size=128):
        super().__init__()
        self.ca3_size = ca3_size
        self.n_features = n_features
        self.action_count = action_count
        self.network = nn.Sequential(
            nn.Linear(ca3_size + action_count, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 2 * n_features + 2),
        )

    def forward(self, ca3, actions):
        actions = actions.reshape(-1).long()
        if ca3.ndim != 2 or ca3.shape != (actions.numel(), self.ca3_size):
            raise ValueError("Expected CA3 [N,ca3_size] and discrete actions [N] or [N,1]")
        if ((actions < 0) | (actions >= self.action_count)).any():
            raise ValueError("Candidate actions must be in [0, action_count)")
        onehot = F.one_hot(actions, self.action_count).to(ca3)
        raw = self.network(torch.cat((ca3, onehot), -1))
        n = self.n_features
        return {
            "hit_logit": raw[:, 0],
            "dg_logits": raw[:, 1:1 + n],
            "dg_amplitude": F.softplus(raw[:, 1 + n:1 + 2 * n]),
            "delay_fraction": raw[:, -1].sigmoid(),
        }

    @torch.no_grad()
    def predict_actions(self, ca3, horizon):
        """Optional diagnostic API: [B,A,...] predictions, no action selection."""
        batch = ca3.size(0)
        states = ca3[:, None].expand(-1, self.action_count, -1).reshape(-1, self.ca3_size)
        actions = torch.arange(self.action_count, device=ca3.device).repeat(batch)
        raw = self(states, actions)
        result = {
            "hit_probability": raw["hit_logit"].sigmoid(),
            "dg_probability": raw["dg_logits"].sigmoid(),
            "dg_amplitude": raw["dg_amplitude"],
            "delay_decisions": 1 + (horizon - 1) * raw["delay_fraction"],
        }
        return {key: value.reshape(batch, self.action_count, *value.shape[1:]) for key, value in result.items()}


def event_prediction_loss(model, ca3, actions, labels, horizon):
    """Head-only training: all representation and label tensors are detached."""
    raw = model(ca3.detach(), actions.detach())
    usable = labels.usable.flatten().detach()
    hit = labels.hit.flatten().detach()
    positive = usable & hit
    target = labels.dg.reshape(-1, model.n_features).detach()
    delay = labels.delay.flatten().detach()
    zero = sum(x.sum() for x in raw.values()) * 0.0
    loss, hit_accuracy, dg_accuracy, time_mae = zero, zero.detach(), zero.detach(), zero.detach()
    if usable.any():
        loss = F.binary_cross_entropy_with_logits(raw["hit_logit"][usable], hit[usable].float())
        hit_accuracy = ((raw["hit_logit"][usable] >= 0) == hit[usable]).float().mean()
    if positive.any():
        active = target[positive] > 0
        loss = loss + F.binary_cross_entropy_with_logits(raw["dg_logits"][positive], active.float())
        loss = loss + F.smooth_l1_loss(
            raw["dg_amplitude"][positive][active].log1p(), target[positive][active].log1p()
        )
        predicted_delay = 1 + (horizon - 1) * raw["delay_fraction"][positive]
        loss = loss + F.smooth_l1_loss(predicted_delay / horizon, delay[positive] / horizon)
        time_mae = (predicted_delay - delay[positive]).abs().mean()
        dg_accuracy = ((raw["dg_logits"][positive] >= 0) == active).all(-1).float().mean()
    return loss, {
        "loss": loss.detach(), "usable_count": usable.sum(), "positive_count": positive.sum(),
        "hit_accuracy": hit_accuracy.detach(), "dg_exact_match": dg_accuracy.detach(),
        "time_mae_decisions": time_mae.detach(),
        "positive_fraction": positive.sum().float() / usable.sum().clamp_min(1),
    }
