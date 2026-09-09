"""Pure adapters of IntrMotiv state conventions. Inputs are never mutated."""

import torch
from torch.nn import functional as F


def advance_ca3(state, dg, repeat):
    """[B,F,E] memory BEFORE observation -> memory AFTER its [B,F] DG input."""
    if state.ndim != 3 or dg.shape != state.shape[:2]:
        raise ValueError("Expected state [B,F,E] and DG [B,F]")
    if not isinstance(repeat, int) or not 1 <= repeat <= state.size(-1):
        raise ValueError("repeat must be an integer in [1,E]")
    shifted = F.pad(state[..., :-1], (1, 0))
    injection = F.pad(dg.unsqueeze(-1).expand(-1, -1, repeat), (0, state.size(-1) - repeat))
    return shifted + injection


def skip_silent_ca3(state, next_dg, delay, repeat):
    """State AFTER current input -> AFTER next input, at integer delay >= 1.

    EXACT only if all delay-1 intermediate DG vectors are zero. Delay is a
    Python integer shared by this batch. No claim is made about intervening
    action histories, bypass observations, rewards, or termination.
    """
    if not isinstance(delay, int) or isinstance(delay, bool) or delay < 1:
        raise ValueError("delay must be a positive Python integer")
    # advance_ca3 performs the final shift and injection.
    gap = delay - 1
    if gap >= state.size(-1):
        before_event = torch.zeros_like(state)
    elif gap:
        before_event = F.pad(state[..., :-gap], (gap, 0))
    else:
        before_event = state
    return advance_ca3(before_event, next_dg, repeat)


def previous_action_onehot(previous_action, action_count):
    """Actual action that produced this observation; sentinel action_count -> 0."""
    if action_count < 1:
        raise ValueError("action_count must be positive")
    if previous_action.is_floating_point() or previous_action.dtype == torch.bool:
        raise ValueError("previous_action must use an integer dtype")
    if ((previous_action < 0) | (previous_action > action_count)).any():
        raise ValueError("Use actions 0..A-1 and reset sentinel A")
    encoded = F.one_hot(previous_action.long().clamp_max(action_count - 1), action_count).float()
    return encoded * previous_action.ne(action_count).unsqueeze(-1)


def advance_action_history(history, previous_action, reset=None):
    """[B,K,A], oldest first. Insert action producing CURRENT observation.

    On reset use sentinel A and clear the complete history for that row.
    Candidate FUTURE actions never belong in this real history.
    """
    if history.ndim != 3 or history.size(1) < 1 or history.size(2) < 1:
        raise ValueError("history must have shape [B,K,A], K,A >= 1")
    if previous_action.shape != history.shape[:1]:
        raise ValueError("previous_action must have shape [B]")
    if reset is not None:
        if reset.shape != history.shape[:1]:
            raise ValueError("reset must have shape [B]")
        history = torch.where(reset.bool()[:, None, None], torch.zeros_like(history), history)
        previous_action = torch.where(reset.bool(), history.size(-1), previous_action)
    encoded = previous_action_onehot(previous_action, history.size(-1)).to(history)
    return torch.cat((history[:, 1:], encoded[:, None]), dim=1)
