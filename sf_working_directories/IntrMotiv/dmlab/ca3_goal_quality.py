"""Non-gradient, prequential CA3-to-DG goal-quality memory.

The learner owns these buffers. Actor model forwards only read ``scores``;
accepted rollout transitions are applied once, in rollout order, by the learner.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .hrl_controllable_graph import HRLStateLayout


class CA3GoalQuality(nn.Module):
    def __init__(self, n_goals: int, ca3_size: int, alpha: float = 0.01, support_prior: int = 100):
        super().__init__()
        if n_goals < 2 or ca3_size <= 0 or not 0 < alpha <= 1 or support_prior <= 0:
            raise ValueError("Invalid CA3 goal-quality dimensions or fixed coefficients")
        self.n_goals = n_goals
        self.ca3_size = ca3_size
        self.alpha = float(alpha)
        self.support_prior = int(support_prior)
        self.register_buffer("prototypes", torch.zeros(n_goals, ca3_size))
        self.register_buffer("quality", torch.zeros(n_goals))
        self.register_buffer("event_counts", torch.zeros(n_goals, dtype=torch.int64))
        self.register_buffer("quality_counts", torch.zeros(n_goals, dtype=torch.int64))

    @property
    def scores(self) -> Tensor:
        count = self.quality_counts.to(self.quality.dtype)
        return self.quality * (count / (count + self.support_prior))

    @torch.no_grad()
    def update_from_rollout(self, rnn_states: Tensor, option_states: Tensor, valid_steps: Tensor) -> int:
        """Pair pre-action CA3 at t with a new exclusive event in state t+1.

        Both state tensors include the terminal state and have a T+1 time axis.
        Invalid transitions and repeats of the same exclusive event are ignored.
        This function must never be called from an actor or a replay forward.
        """
        if rnn_states.ndim != 3 or option_states.ndim != 3 or valid_steps.ndim != 2:
            raise ValueError("Expected rollout tensors with [batch, time, features] axes")
        batch, time, width = rnn_states.shape
        if width < self.ca3_size or option_states.shape[:2] != (batch, time):
            raise ValueError("CA3 and option histories are not aligned")
        if valid_steps.shape != (batch, time - 1):
            raise ValueError("Validity mask must cover each rollout transition")
        layout = HRLStateLayout(self.n_goals)
        previous_id = option_states[:, :-1, layout.active_dg].long() - 1
        event_id = option_states[:, 1:, layout.active_dg].long() - 1
        exclusive = option_states[:, 1:, layout.multi_activation] == 0
        novel = (previous_id != event_id) | (option_states[:, :-1, layout.multi_activation] > 0)
        selected = (
            valid_steps.bool()
            & exclusive
            & novel
            & (event_id >= 0)
            & (event_id < self.n_goals)
        )
        contexts = rnn_states[:, :-1, : self.ca3_size].reshape(-1, self.ca3_size)
        ids = event_id.reshape(-1)
        indices = torch.nonzero(selected.reshape(-1), as_tuple=False).flatten()
        applied = 0
        for index in indices.tolist():
            context = contexts[index].detach().to(self.prototypes)
            norm = torch.linalg.vector_norm(context)
            if not torch.isfinite(norm) or norm <= 0:
                continue
            context = context / norm
            goal = int(ids[index])
            lengths = torch.linalg.vector_norm(self.prototypes, dim=-1)
            alternatives = (lengths > 0)
            alternatives[goal] = False
            if lengths[goal] > 0 and alternatives.any():
                cosine = F.cosine_similarity(context[None], self.prototypes, dim=-1)
                margin = cosine[goal] - cosine.masked_fill(~alternatives, -torch.inf).max()
                self.quality[goal].mul_(1 - self.alpha).add_(self.alpha * margin)
                self.quality_counts[goal] += 1
            self.prototypes[goal].mul_(1 - self.alpha).add_(self.alpha * context)
            self.event_counts[goal] += 1
            applied += 1
        return applied


def candidate_mask(eligible: Tensor, scores: Tensor, mode: str, k: int) -> Tensor:
    """Choose up to k currently selectable destinations for each manager choice."""
    if eligible.ndim != 2 or scores.shape != (eligible.size(1),):
        raise ValueError("Candidate eligibility and quality scores are misaligned")
    if mode not in ("all", "random", "hebb") or k <= 0:
        raise ValueError("Invalid goal candidate selector")
    if mode == "all" or k >= eligible.size(1):
        return eligible.clone()
    if not torch.isfinite(scores).all():
        raise ValueError("CA3 goal-quality scores must be finite")
    # Random order first, then stable score order: exact Hebbian ties are
    # uniform without adding jitter that could reorder unequal scores.
    random_order = torch.argsort(torch.rand(eligible.shape, device=eligible.device), dim=-1)
    if mode == "hebb":
        ordered_scores = scores[None, :].expand_as(eligible).gather(1, random_order)
        rank = torch.argsort(ordered_scores, dim=-1, descending=True, stable=True)
        order = random_order.gather(1, rank)
    else:
        order = random_order
    ranked_eligible = eligible.gather(1, order)
    ranks = ranked_eligible.long().cumsum(dim=-1)
    selected_order = ranked_eligible & (ranks <= k)
    selected = torch.zeros_like(eligible)
    selected.scatter_(1, order, selected_order)
    return selected
