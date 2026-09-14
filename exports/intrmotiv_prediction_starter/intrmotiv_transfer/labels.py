"""Censor-aware adaptation; intentionally NOT the upstream label helper."""

import torch


@torch.no_grad()
def future_target_labels(target_onehot, dg_activity, dones_after, horizon, valids=None):
    """First future activation of a specified target, within H decisions.

    Inputs: targets and DG [B,T,F]; dones_after [B,T] means the action at t
    ended/reset the episode before observation t+1. Reset observations are
    never future targets for the previous episode. `valids` excludes padding.

    Returns hit [B,T], delay [B,T], usable [B,T]. Positives are usable as soon
    as observed; negatives require the ENTIRE requested horizon. Unfinished,
    padded, or episode-boundary windows are censored, not negative examples.
    This conservative helper does not consume terminal observations separately.
    No current-step hits, next-event identity, onset, or distinctness semantics.
    """
    if target_onehot.ndim != 3 or dg_activity.shape != target_onehot.shape:
        raise ValueError("targets and DG must have matching shape [B,T,F]")
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    shape = target_onehot.shape[:2]
    if dones_after.shape != shape or (valids is not None and valids.shape != shape):
        raise ValueError("dones_after and valids must have shape [B,T]")
    is_binary = (target_onehot == 0) | (target_onehot == 1)
    if not is_binary.all() or (target_onehot.sum(-1) > 1).any():
        raise ValueError("targets must be one-hot or all-zero")
    valids = torch.ones(shape, dtype=torch.bool, device=dg_activity.device) if valids is None else valids.bool()
    target = target_onehot.argmax(-1)
    eligible = valids & target_onehot.sum(-1).gt(0)
    hit = torch.zeros_like(eligible)
    delay = torch.zeros(shape, dtype=dg_activity.dtype, device=dg_activity.device)
    observed = torch.zeros(shape, dtype=torch.long, device=dg_activity.device)
    batch, steps = shape
    for t in range(steps):
        alive = eligible[:, t].clone()
        for delta in range(1, min(horizon, steps - t - 1) + 1):
            future = t + delta
            alive = alive & ~dones_after[:, future - 1].bool() & valids[:, future]
            observed[:, t] += alive.long()
            active = dg_activity[:, future].gather(1, target[:, t, None]).squeeze(1).gt(0)
            first = alive & active & ~hit[:, t]
            delay[:, t] = torch.where(first, float(delta), delay[:, t])
            hit[:, t] |= first
    usable = eligible & (hit | observed.eq(horizon))
    return hit.to(dg_activity.dtype), delay, usable
