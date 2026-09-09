"""Episode-contained DG hindsight labels for the empirical PPO-HER auxiliary."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class EmpiricalHERBatch:
    targets: Tensor
    returns: Tensor
    valid: Tensor
    terminal_reward: Tensor
    accepted_segments: Tensor
    skipped_no_endpoint: Tensor
    skipped_same_source: Tensor


def normalize_empirical_her_advantage(advantage: Tensor, valid: Tensor, enabled: bool) -> Tensor:
    """Normalize only when the hindsight slice has a defined standard deviation."""
    valid_advantage = advantage[valid]
    if not enabled or valid_advantage.numel() < 2:
        return advantage
    std, mean = torch.std_mean(valid_advantage, correction=0)
    return (advantage - mean) / std.clamp_min(1e-7)


def build_empirical_her_batch(
    active_ids: Tensor,
    dones: Tensor,
    valids: Tensor,
    terminal_rewards: Tensor,
    n_nodes: int,
    horizon: int,
    gamma: float,
) -> EmpiricalHERBatch:
    """Build one uniformly sampled fixed-goal future segment per rollout stream.

    ``active_ids[:, t]`` is the exclusive DG identity observed at decision t,
    encoded as 1..F, with zero denoting no exclusive landmark.  A segment ends
    on action ``e`` when the next observation (``e + 1``) activates its goal.
    """
    if active_ids.ndim != 2 or dones.shape != active_ids.shape or valids.shape != active_ids.shape:
        raise ValueError("active_ids, dones, and valids must have shape [batch, decisions]")
    if terminal_rewards.shape != active_ids.shape:
        raise ValueError("terminal_rewards must match active_ids")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    batch, steps = active_ids.shape
    targets = terminal_rewards.new_zeros((batch, steps, n_nodes))
    returns = terminal_rewards.new_zeros((batch, steps))
    valid = torch.zeros((batch, steps), dtype=torch.bool, device=active_ids.device)
    terminal = terminal_rewards.new_zeros((batch, steps))
    accepted = terminal_rewards.new_zeros(())
    skipped_no_endpoint = terminal_rewards.new_zeros(())
    skipped_same_source = terminal_rewards.new_zeros(())

    for row in range(batch):
        episode_start = 0
        candidates: list[tuple[int, int, int]] = []
        for end in range(max(steps - 1, 0)):
            if bool(dones[row, end]) or not bool(valids[row, end]):
                candidates.clear()
                episode_start = end + 1
                continue
            destination = int(active_ids[row, end + 1].item())
            if destination <= 0 or destination > n_nodes:
                continue
            low = max(episode_start, end - horizon + 1)
            for start in range(low, end + 1):
                source = int(active_ids[row, start].item())
                if source <= 0 or source > n_nodes:
                    continue
                if source == destination:
                    continue
                if not bool(valids[row, start : end + 1].all()):
                    continue
                candidates.append((start, end, destination))

        if not candidates:
            skipped_no_endpoint += 1
            continue
        # The random global Torch generator is already seeded by Sample Factory.
        start, end, destination = candidates[torch.randint(len(candidates), (), device=active_ids.device).item()]
        if int(active_ids[row, start].item()) == destination:
            skipped_same_source += 1
            continue
        targets[row, start : end + 1, destination - 1] = 1.0
        valid[row, start : end + 1] = True
        terminal[row, end] = terminal_rewards[row, end]
        discounts = torch.pow(
            terminal_rewards.new_tensor(float(gamma)),
            torch.arange(end - start, -1, -1, device=active_ids.device, dtype=terminal_rewards.dtype),
        )
        returns[row, start : end + 1] = discounts * terminal_rewards[row, end]
        accepted += 1

    return EmpiricalHERBatch(
        targets=targets,
        returns=returns,
        valid=valid,
        terminal_reward=terminal,
        accepted_segments=accepted,
        skipped_no_endpoint=skipped_no_endpoint,
        skipped_same_source=skipped_same_source,
    )
