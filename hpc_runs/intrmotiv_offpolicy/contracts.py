"""Pure training contracts; no environment, graph, or telemetry dependencies."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


def canonical_events(activity, exclusive=False):
    """Source recognition: dominant positive ID, optionally exclusive (waypoint)."""
    if activity.ndim != 2 or not torch.isfinite(activity).all():
        raise ValueError("expected finite [batch, landmarks] canonical activity")
    active = activity > 0
    valid = active.sum(-1) == 1 if exclusive else active.any(-1)
    return F.one_hot(activity.argmax(-1), activity.shape[-1]).bool() & valid[:, None]


def advance_memory(memory, activity, repeat_width):
    """Canonical shift then R-slot injection; W preceding steps erase all state."""
    if memory.shape[:-1] != activity.shape or not 0 < repeat_width <= memory.shape[-1]:
        raise ValueError("incompatible CA3 layout")
    shifted = F.pad(memory[..., :-1], (1, 0))
    injected = F.pad(activity[..., None].expand(*activity.shape, repeat_width), (0, memory.shape[-1] - repeat_width))
    return shifted + injected


def first_arrival(events, goal, budget, terminated, valid):
    """T+1 recognition sequence. A missing successor invalidates its loss."""
    t = len(terminated)
    if events.ndim != 2 or len(events) != t + 1 or len(valid) != t or budget < 1:
        raise ValueError("invalid first-arrival sequence")
    if not 0 <= goal < events.shape[1]:
        raise ValueError("goal outside registry")
    reward = torch.zeros(t, device=events.device)
    done = torch.zeros(t, dtype=torch.bool, device=events.device)
    mask = torch.zeros_like(done)
    alive = not bool(events[0, goal])
    for i in range(t):
        if not alive:
            break
        if not bool(valid[i]):
            break  # Never jump across an unknown physical successor.
        hit = bool(events[i + 1, goal])
        mask[i] = True
        reward[i] = float(hit)
        done[i] = hit or i + 1 >= budget or bool(terminated[i])
        alive = not bool(done[i])
    return reward, done, mask


def double_dqn_target(reward, done, next_online, next_target, gamma=0.99):
    if next_online.shape != next_target.shape or next_online.shape[:-1] != reward.shape:
        raise ValueError("incompatible target shapes")
    with torch.no_grad():
        action = next_online.argmax(-1, keepdim=True)
        value = next_target.gather(-1, action).squeeze(-1)
        return reward + gamma * torch.where(done, torch.zeros_like(value), value)


@dataclass(frozen=True)
class UpdateSchedule:
    decisions_per_update: int = 64
    target_updates: int = 1000

    def due(self, accepted_decisions, completed_updates, learning_start):
        if self.decisions_per_update < 1 or self.target_updates < 1:
            raise ValueError("update periods must be positive")
        return max(0, (accepted_decisions - learning_start) // self.decisions_per_update - completed_updates)
