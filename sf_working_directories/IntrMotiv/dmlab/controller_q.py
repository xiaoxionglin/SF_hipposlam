"""Opt-in action-value heads at the original IntrMotiv controller boundary.

The main head estimates the parent's continuing return. Hindsight is a separate
bounded worker auxiliary; its predictions never select the acting controller.
No encoder, core, goal selector, reward, or graph implementation lives here.
"""

from __future__ import annotations

import torch
from torch import nn


class ControllerQHeads(nn.Module):
    def __init__(self, hidden_size: int, actions: int, auxiliary: bool):
        super().__init__()
        self.main = nn.Linear(hidden_size, actions)
        # A deadline-conditioned auxiliary needs state/deadline interactions.
        # It does not change the original decoder or the main return interface.
        self.auxiliary = nn.Linear(2 * hidden_size + 1, actions) if auxiliary else None
        self.register_buffer("fresh_version", torch.zeros((), dtype=torch.int64))
        self.register_buffer("publication_version", torch.zeros((), dtype=torch.int64))
        self.register_buffer("environment_decisions", torch.zeros((), dtype=torch.int64))

    def forward(self, hidden):
        return self.main(hidden)

    def hindsight(self, hidden, remaining, time_scale):
        if self.auxiliary is None:
            raise RuntimeError("Hindsight head is not enabled")
        if time_scale <= 0 or torch.any(remaining < 0):
            raise ValueError("Invalid auxiliary deadline")
        clock = remaining.to(hidden.dtype).reshape(-1, 1) / time_scale
        return self.auxiliary(torch.cat((hidden, hidden * clock, clock), dim=-1))


def epsilon_distribution(q, epsilon, action_mask=None):
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0,1]")
    allowed = torch.ones_like(q, dtype=torch.bool) if action_mask is None else action_mask.bool()
    if not allowed.any(-1).all():
        raise ValueError("No valid action")
    greedy = q.masked_fill(~allowed, -torch.inf).argmax(-1)
    probabilities = epsilon * allowed.to(q.dtype) / allowed.sum(-1, keepdim=True)
    probabilities = probabilities.scatter_add(-1, greedy[:, None], q.new_full((len(q), 1), 1 - epsilon))
    return probabilities


def continuing_double_q_target(reward, physical_done, next_online, next_target, gamma):
    """Retargeting/waypoints/options are successor context, not episode dones."""
    if not 0 <= gamma <= 1:
        raise ValueError("Invalid discount")
    if next_online.shape != next_target.shape or next_online.shape[:-1] != reward.shape:
        raise ValueError("Incompatible Bellman arrays")
    with torch.no_grad():
        selected = next_online.argmax(-1, keepdim=True)
        continuation = next_target.gather(-1, selected).squeeze(-1)
        return reward + gamma * torch.where(physical_done.bool(), torch.zeros_like(continuation), continuation)


def bounded_auxiliary_target(reward, task_ended, remaining, next_online, next_target, gamma):
    """Explicit auxiliary boundary; never substitute this for main-return done."""
    ended = task_ended.bool() | (remaining <= 1)
    return continuing_double_q_target(reward, ended, next_online, next_target, gamma)


def exploration_epsilon(decisions, cfg):
    if not hasattr(cfg, "controller_learning_starts"):
        return float(cfg.controller_epsilon)
    elapsed = max(0, decisions - int(cfg.controller_learning_starts))
    decay = int(cfg.controller_epsilon_decay_decisions)
    if decay <= 0:
        raise ValueError("Epsilon decay decisions must be positive")
    return 1.0 - (1.0 - float(cfg.controller_epsilon)) * min(1.0, elapsed / decay)
