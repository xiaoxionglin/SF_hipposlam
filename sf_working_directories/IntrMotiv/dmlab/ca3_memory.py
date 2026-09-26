"""Finite CA3 novelty, re-entry inhibition, and graph-free intrinsic goals."""

from __future__ import annotations

import torch
from torch.nn import functional as F

# Episode-local target, elapsed decisions, arrival-reward pulse, ambiguous-arrival
# pulse, latched success, and the replay-only target descriptor.
GOAL_STATE_SIZE = 6


def inhibit_reentry(activity, previous_ca3, mode):
    memory = previous_ca3.detach().amax(dim=-1)
    continuing = previous_ca3[..., 0].gt(0)
    if mode == "none":
        return activity
    if mode == "hard":
        return activity * (continuing | memory.eq(0))
    if mode == "trace_subtractive":
        return torch.where(continuing, activity, (activity - memory).clamp_min(0))
    raise ValueError(f"Unknown re-entry inhibition: {mode}")


def absent_event_gate(progression, dominant, empty_distance):
    previous = progression.roll(1, dims=1)
    gate = (dominant & previous.eq(empty_distance)).any(dim=-1)
    gate[:, 0] = False
    return gate


def advance_goal(state, previous_ca3, updated_ca3, horizon, reward_max, refractory):
    """Advance one episode-persistent intrinsic command.

    A target is sampled once and survives until the recurrent state is reset by
    the environment.  Arrival pays once.  Wrong or ambiguous landmark events do
    not cancel the command.  ``reward_max`` is independent of ``horizon`` so a
    longer episode cannot silently change the reward scale.
    """
    result = torch.zeros_like(state)
    target = state[:, 0].long()
    elapsed = state[:, 1] + 1
    active = target.gt(0)
    achieved = state[:, 4].gt(0)
    n = updated_ca3.size(-2)
    safe = (target - 1).clamp(0, n - 1)
    current = updated_ca3[..., 0]
    # Match the established dominant-onset refractory event definition.
    candidates = current.gt(0) & ~previous_ca3[..., :refractory].gt(0).any(-1)
    dominant = current.masked_fill(~candidates, -torch.inf).argmax(-1)
    hit = active & ~achieved & candidates.any(-1) & dominant.eq(safe)
    entered = active & current.gather(1, safe[:, None]).squeeze(1).gt(0)
    ambiguous = entered & ~hit & ~achieved
    hit &= elapsed.le(horizon)
    proximity = (horizon + 1 - elapsed).clamp_min(0) / float(horizon)
    result[:, 2] = hit * proximity * float(reward_max)
    result[:, 3] = ambiguous
    result[:, 4] = achieved | hit
    result[:, 0] = target
    result[:, 1] = torch.where(active, elapsed.clamp_max(horizon), torch.zeros_like(elapsed))
    absent = updated_ca3.amax(-1).eq(0)
    choose = ~active & absent.any(-1)
    if choose.any():
        # Uniform categorical sampling; never use a historical cumulative graph.
        sampled = torch.multinomial(absent[choose].float(), 1).squeeze(1)
        result[choose, 0] = (sampled + 1).to(result.dtype)
    result[:, -1] = result[:, 0]
    goal = F.one_hot((result[:, 0].long() - 1).clamp_min(0), n)
    goal = goal.to(current.dtype) * result[:, 0:1].gt(0)
    return result, goal


def advance_reference_goal(state, previous_activity, current_activity, horizon, reward_max):
    """Persistent goal whose arrival definition comes from a frozen detector."""
    result = torch.zeros_like(state)
    target = state[:, 0].long()
    elapsed = state[:, 1] + 1
    active = target.gt(0)
    achieved = state[:, 4].gt(0)
    n = current_activity.size(-1)
    safe = (target - 1).clamp(0, n - 1)
    onset = current_activity.gt(0) & ~previous_activity.gt(0)
    dominant = current_activity.masked_fill(~onset, -torch.inf).argmax(-1)
    entered = active & current_activity.gather(1, safe[:, None]).squeeze(1).gt(0)
    hit = active & ~achieved & onset.any(-1) & dominant.eq(safe) & elapsed.le(horizon)
    ambiguous = entered & ~hit & ~achieved
    proximity = (horizon + 1 - elapsed).clamp_min(0) / float(horizon)
    result[:, 0] = target
    result[:, 1] = torch.where(active, elapsed.clamp_max(horizon), torch.zeros_like(elapsed))
    result[:, 2] = hit * proximity * float(reward_max)
    result[:, 3] = ambiguous
    result[:, 4] = achieved | hit
    choose = ~active
    if choose.any():
        # Goal sampling is identity-uniform and never depends on physical pose.
        result[choose, 0] = torch.randint(n, (int(choose.sum()),), device=state.device).to(state.dtype) + 1
    result[:, -1] = result[:, 0]
    goal = F.one_hot((result[:, 0].long() - 1).clamp_min(0), n)
    goal = goal.to(current_activity.dtype) * result[:, 0:1].gt(0)
    return result, goal


def finite_shift_history(activity, injection_width, memory_length, initial_state=None):
    """All additive CA3 states, using the same finite impulse.

    ``activity`` is [time,batch,unit]. Each observation is injected into the
    first R slots, then shifted one slot per decision. No subtraction is used:
    zero/nonzero event identities are preserved for nonnegative DG activities.
    This is the packed replay equivalent of the existing single-step update.
    ``initial_state`` can provide the CA3 state at the start of each packed
    sequence; it is shifted once before the first returned state.
    """
    if activity.ndim != 3 or not 1 <= injection_width <= memory_length:
        raise ValueError("Invalid finite shift-register dimensions")
    steps, batch, units = activity.shape
    stream = activity.permute(1, 2, 0).reshape(batch * units, 1, steps)
    kernel = activity.new_ones(1, 1, injection_width)
    full = F.conv1d(F.pad(stream, (injection_width - 1, 0)), kernel).squeeze(1)
    older = (
        F.pad(full, (memory_length - injection_width, 0)).unfold(-1, memory_length - injection_width + 1, 1).flip(-1)
    )
    if injection_width > 1:
        recent = (
            F.pad(stream.squeeze(1), (injection_width - 2, 0)).unfold(-1, injection_width - 1, 1).flip(-1).cumsum(-1)
        )
        states = torch.cat((recent, older), -1)
    else:
        states = older
    states = states.reshape(batch, units, steps, memory_length).permute(2, 0, 1, 3)
    if initial_state is None:
        return states
    if initial_state.shape != (batch, units, memory_length):
        raise ValueError("Initial CA3 state must have shape [batch,unit,memory]")
    offsets = torch.arange(1, steps + 1, device=activity.device).unsqueeze(1)
    source = torch.arange(memory_length, device=activity.device).unsqueeze(0) - offsets
    valid = source.ge(0)
    carried = initial_state[:, :, source.clamp_min(0)].permute(2, 0, 1, 3)
    return states + carried * valid[:, None, None, :]
