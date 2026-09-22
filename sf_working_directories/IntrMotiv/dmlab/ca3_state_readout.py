"""Predictive CA3 readout and bounded contextual-anchor maintenance.

The canonical CA3 tensor remains the source of truth.  This module only owns a
compact worker view and the auxiliary future-DG objective; callers decide which
policy losses may consume the detached view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CA3StateReadout(nn.Module):
    """Bias-free linear state readout ``z = W S``."""

    def __init__(self, ca3_size: int, state_dim: int, lr_scale: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(int(ca3_size), int(state_dim), bias=False)
        nn.init.orthogonal_(self.linear.weight)
        self.linear._intrmotiv_preserve_initialization = True
        for parameter in self.parameters():
            parameter._intrmotiv_lr_scale = float(lr_scale)

    def forward(self, ca3: Tensor) -> Tensor:
        return self.linear(ca3)


class CausalDGInnovationPredictor(nn.Module):
    """Feed-forward multi-horizon predictor with structural prefix masking."""

    def __init__(
        self,
        state_dim: int,
        action_count: int,
        horizon: int,
        n_dg: int,
        hidden_size: int = 128,
        action_conditioned: bool = True,
    ):
        super().__init__()
        self.action_count = int(action_count)
        self.horizon = int(horizon)
        self.action_conditioned = bool(action_conditioned)
        self.horizon_embedding = nn.Embedding(self.horizon + 1, int(state_dim))
        self.network = nn.Sequential(
            nn.Linear(int(state_dim) * 2 + self.horizon * self.action_count, int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), int(n_dg)),
            nn.Softplus(),
        )

    def forward(self, state: Tensor, actions: Tensor, horizon: Tensor) -> Tensor:
        if actions.ndim != 2 or actions.size(1) != self.horizon:
            raise ValueError("actions must have shape [batch, configured_horizon]")
        horizon = horizon.long()
        positions = torch.arange(self.horizon, device=actions.device).unsqueeze(0)
        prefix = positions < horizon.unsqueeze(1)
        encoded = F.one_hot(actions.long().clamp(0, self.action_count - 1), self.action_count).to(state.dtype)
        encoded = encoded * prefix.unsqueeze(-1).to(state.dtype)
        if not self.action_conditioned:
            encoded = encoded * 0.0
        features = torch.cat((state, encoded.flatten(1), self.horizon_embedding(horizon)), dim=-1)
        return self.network(features)


@dataclass(frozen=True)
class ReadoutPrediction:
    loss: Tensor
    active_loss: Tensor
    zero_loss: Tensor
    valid_targets: Tensor
    active_fraction: Tensor
    state_shuffle_delta: Tensor
    action_shuffle_delta: Tensor


def _stratified_smooth_l1(prediction: Tensor, target: Tensor, active_coeff: float, zero_coeff: float):
    element = F.smooth_l1_loss(prediction, target, reduction="none")
    active = target > 0
    zero = ~active
    active_loss = element[active].mean() if active.any() else element.sum() * 0.0
    zero_loss = element[zero].mean() if zero.any() else element.sum() * 0.0
    return float(active_coeff) * active_loss + float(zero_coeff) * zero_loss, active_loss, zero_loss


def prediction_windows(
    core_outputs: Tensor,
    actions: Tensor,
    valids: Tensor,
    dones: Tensor,
    recurrence: int,
    ca3_size: int,
    n_dg: int,
    expanded: int,
    horizon: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return causal ``(S_t, action_prefix, h, u_{t+h})`` rows."""
    if horizon >= recurrence:
        raise ValueError("ca3_state_readout_horizon must be smaller than recurrence")
    streams = core_outputs.size(0) // int(recurrence)
    states = core_outputs[:, :ca3_size].reshape(streams, recurrence, ca3_size)
    action_ids = actions.reshape(streams, recurrence, -1)[..., 0].long()
    valid = valids.reshape(streams, recurrence).bool()
    done = dones.reshape(streams, recurrence).bool()
    dg = states.reshape(streams, recurrence, n_dg, expanded)[..., 0]
    starts, prefixes, horizons, targets = [], [], [], []
    for h in range(1, horizon + 1):
        width = recurrence - h
        if width <= 0:
            continue
        window_valid = valid[:, :width] & valid[:, h : h + width]
        for offset in range(h):
            window_valid &= valid[:, offset : offset + width]
            window_valid &= ~done[:, offset : offset + width]
        if not window_valid.any():
            continue
        s, t = torch.where(window_valid)
        padded = action_ids.new_zeros((s.numel(), horizon))
        for offset in range(h):
            padded[:, offset] = action_ids[s, t + offset]
        starts.append(states[s, t])
        prefixes.append(padded)
        horizons.append(torch.full((s.numel(),), h, device=states.device, dtype=torch.long))
        targets.append(dg[s, t + h])
    if not starts:
        empty_state = states.new_zeros((0, ca3_size))
        return (
            empty_state,
            action_ids.new_zeros((0, horizon)),
            action_ids.new_zeros((0,)),
            states.new_zeros((0, n_dg)),
        )
    return tuple(torch.cat(items, dim=0) for items in (starts, prefixes, horizons, targets))


def predictive_readout_loss(
    readout: CA3StateReadout,
    predictor: CausalDGInnovationPredictor,
    core_outputs: Tensor,
    actions: Tensor,
    valids: Tensor,
    dones: Tensor,
    recurrence: int,
    n_dg: int,
    expanded: int,
    horizon: int,
    active_coeff: float = 1.0,
    zero_coeff: float = 0.1,
) -> ReadoutPrediction:
    ca3_size = n_dg * expanded
    states, prefixes, horizons, targets = prediction_windows(
        core_outputs.detach(), actions, valids, dones, recurrence, ca3_size, n_dg, expanded, horizon
    )
    if not states.numel():
        zero = readout.linear.weight.sum() * 0.0
        return ReadoutPrediction(zero, zero, zero, zero.detach(), zero.detach(), zero.detach(), zero.detach())
    latent = readout(states)
    prediction = predictor(latent, prefixes, horizons)
    loss, active, zero = _stratified_smooth_l1(prediction, targets.detach(), active_coeff, zero_coeff)
    with torch.no_grad():
        base = F.smooth_l1_loss(prediction.detach(), targets, reduction="mean")
        shuffled_state = predictor(latent.detach().roll(1, 0), prefixes, horizons)
        shuffled_action = predictor(latent.detach(), prefixes.roll(1, 0), horizons)
        state_delta = F.smooth_l1_loss(shuffled_state, targets, reduction="mean") - base
        action_delta = F.smooth_l1_loss(shuffled_action, targets, reduction="mean") - base
    return ReadoutPrediction(
        loss,
        active,
        zero,
        targets.new_tensor(float(targets.size(0))),
        (targets > 0).float().mean(),
        state_delta,
        action_delta,
    )


def paired_anchor_improvement(
    readout: CA3StateReadout,
    predictor: CausalDGInnovationPredictor,
    incumbent: Tensor,
    candidate: Tensor,
    actions: Tensor,
    targets: Tensor,
    active_coeff: float,
    zero_coeff: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return paired mean improvement and its one-sided 95% lower bound."""
    count = targets.size(0)
    horizons = torch.arange(1, count + 1, device=targets.device)
    padded = actions.new_zeros((count, predictor.horizon))
    for h in range(1, count + 1):
        padded[h - 1, :h] = actions[:h]
    incumbent_z = readout(incumbent.unsqueeze(0)).expand(count, -1)
    candidate_z = readout(candidate.unsqueeze(0)).expand(count, -1)
    incumbent_pred = predictor(incumbent_z, padded, horizons)
    candidate_pred = predictor(candidate_z, padded, horizons)
    incumbent_element = F.smooth_l1_loss(incumbent_pred, targets, reduction="none")
    candidate_element = F.smooth_l1_loss(candidate_pred, targets, reduction="none")
    weights = torch.where(targets > 0, float(active_coeff), float(zero_coeff))
    differences = ((incumbent_element - candidate_element) * weights).mean(dim=-1)
    mean = differences.mean()
    if differences.numel() < 2:
        lower = mean.new_tensor(-math.inf)
    else:
        lower = mean - 1.645 * differences.std(unbiased=True) / math.sqrt(differences.numel())
    return mean, lower, candidate_element.mean()


def predictive_window_consistency(
    readout: CA3StateReadout,
    predictor: CausalDGInnovationPredictor,
    anchor: Tensor,
    candidate: Tensor,
    actions: Tensor,
    targets: Tensor,
    active_coeff: float,
    zero_coeff: float,
) -> tuple[Tensor, Tensor]:
    """Absolute anchor loss and anchor-minus-candidate excess on one window."""
    single = anchor.ndim == 1
    if single:
        anchor, candidate, actions, targets = (
            anchor.unsqueeze(0),
            candidate.unsqueeze(0),
            actions.unsqueeze(0),
            targets.unsqueeze(0),
        )
    batch, count = targets.shape[:2]
    horizons = torch.arange(1, count + 1, device=targets.device)
    padded = actions.new_zeros((batch, count, predictor.horizon))
    for h in range(1, count + 1):
        padded[:, h - 1, :h] = actions[:, :h]
    flat_horizons = horizons.unsqueeze(0).expand(batch, -1).reshape(-1)
    anchor_latent = readout(anchor).unsqueeze(1).expand(-1, count, -1).reshape(-1, readout.linear.out_features)
    candidate_latent = readout(candidate).unsqueeze(1).expand(-1, count, -1).reshape(-1, readout.linear.out_features)
    anchor_prediction = predictor(anchor_latent, padded.flatten(0, 1), flat_horizons).reshape_as(targets)
    candidate_prediction = predictor(candidate_latent, padded.flatten(0, 1), flat_horizons).reshape_as(targets)
    weights = torch.where(targets > 0, float(active_coeff), float(zero_coeff))
    anchor_loss = (F.smooth_l1_loss(anchor_prediction, targets, reduction="none") * weights).mean(-1)
    candidate_loss = (F.smooth_l1_loss(candidate_prediction, targets, reduction="none") * weights).mean(-1)
    absolute = anchor_loss.mean(-1)
    excess = (anchor_loss - candidate_loss).mean(-1)
    return (absolute[0], excess[0]) if single else (absolute, excess)


def action_probe_signature(
    readout: CA3StateReadout,
    predictor: CausalDGInnovationPredictor,
    ca3: Tensor,
) -> Tensor:
    """Causal predicted-future signature under a fixed, domain-neutral probe bank.

    Every discrete action is repeated under prefixes ending at horizons
    ``1, H/2, H``. The signature contains predictions only: it never reads a
    privileged coordinate or an unavailable realized future.
    """
    if ca3.ndim == 1:
        ca3 = ca3.unsqueeze(0)
    horizons = sorted({1, max(1, predictor.horizon // 2), predictor.horizon})
    batch = ca3.size(0)
    action_ids = torch.arange(predictor.action_count, device=ca3.device)
    probes = []
    probe_horizons = []
    for horizon in horizons:
        repeated = action_ids[:, None].expand(-1, predictor.horizon).clone()
        probes.append(repeated)
        probe_horizons.append(torch.full((predictor.action_count,), horizon, device=ca3.device, dtype=torch.long))
    actions = torch.cat(probes, dim=0)
    horizon_tensor = torch.cat(probe_horizons, dim=0)
    actions = actions.unsqueeze(0).expand(batch, -1, -1).reshape(-1, predictor.horizon)
    horizon_tensor = horizon_tensor.unsqueeze(0).expand(batch, -1).reshape(-1)
    latent = readout(ca3).unsqueeze(1).expand(-1, len(horizons) * predictor.action_count, -1).reshape(
        -1, readout.linear.out_features
    )
    prediction = predictor(latent, actions, horizon_tensor)
    return prediction.reshape(batch, -1)
