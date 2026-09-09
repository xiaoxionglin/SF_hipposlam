from __future__ import annotations

import torch

from sf_working_directories.IntrMotiv.dmlab.empirical_her import (
    build_empirical_her_batch,
    normalize_empirical_her_advantage,
)


def test_hindsight_segment_has_static_goal_and_terminal_return():
    torch.manual_seed(0)
    active = torch.tensor([[1, 1, 2, 2, 0]], dtype=torch.long)
    dones = torch.zeros_like(active, dtype=torch.bool)
    valids = torch.ones_like(active, dtype=torch.bool)
    rewards = torch.full_like(active, 2.0, dtype=torch.float32)

    result = build_empirical_her_batch(active, dones, valids, rewards, n_nodes=2, horizon=4, gamma=0.5)

    assert result.accepted_segments.item() == 1
    assert result.valid.any()
    target_ids = result.targets[result.valid].argmax(dim=-1)
    assert target_ids.unique().numel() == 1
    terminal = result.terminal_reward.nonzero(as_tuple=False)
    assert terminal.shape[0] == 1
    terminal_step = terminal[0, 1].item()
    assert result.returns[0, terminal_step].item() == 2.0


def test_hindsight_does_not_cross_episode_boundary():
    torch.manual_seed(0)
    active = torch.tensor([[1, 1, 2, 0, 3, 3, 1]], dtype=torch.long)
    dones = torch.tensor([[False, True, False, False, False, False, False]])
    valids = torch.ones_like(dones)
    rewards = torch.ones_like(active, dtype=torch.float32)

    result = build_empirical_her_batch(active, dones, valids, rewards, n_nodes=3, horizon=8, gamma=0.9)

    # Any accepted segment must lie entirely after the boundary: the only
    # pre-boundary destination is excluded because action 1 terminates.
    if result.valid.any():
        assert not result.valid[0, :2].any()


def test_hindsight_requires_distinct_exclusive_source_and_destination():
    active = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
    dones = torch.zeros_like(active, dtype=torch.bool)
    valids = torch.ones_like(dones)
    rewards = torch.ones_like(active, dtype=torch.float32)

    result = build_empirical_her_batch(active, dones, valids, rewards, n_nodes=1, horizon=4, gamma=0.9)

    assert result.accepted_segments.item() == 0
    assert not result.valid.any()


def test_singleton_hindsight_advantage_is_not_normalized_to_nan():
    advantage = torch.tensor([2.0, -3.0])
    valid = torch.tensor([True, False])

    normalized = normalize_empirical_her_advantage(advantage, valid, enabled=True)

    assert torch.equal(normalized, advantage)
    assert torch.isfinite(normalized).all()
