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
        # rnn_states[:, t + 1] was produced while observing frame t, before
        # action t. Its new DG event is the outcome of action t - 1, whose
        # accepted flag is valid_steps[:, t - 1]. The preceding action is not
        # present for t == 0 (the rollout boundary), so exclude that event.
        accepted_preceding_action = torch.zeros_like(valid_steps, dtype=torch.bool)
        accepted_preceding_action[:, 1:] = valid_steps[:, :-1].bool()
        selected = (
            accepted_preceding_action
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


def candidate_mask(
    eligible: Tensor, scores: Tensor, mode: str, k: int, *, draw_keys: Tensor | None = None
) -> Tensor:
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
    if draw_keys is None:
        random_values = torch.rand(eligible.shape, device=eligible.device)
    else:
        if draw_keys.shape != eligible.shape[:1]:
            raise ValueError("One deterministic candidate draw key is required per stream")
        modulus = 2147483647
        goals = torch.arange(eligible.size(1), device=eligible.device, dtype=torch.int64)
        values = (draw_keys.long()[:, None] + (goals[None, :] + 1) * 2654435761).remainder(modulus)
        values = ((values ^ (values >> 16)) * 2246822519).remainder(modulus)
        values = ((values ^ (values >> 13)) * 3266489917).remainder(modulus)
        random_values = values
    random_order = torch.argsort(random_values, dim=-1, stable=True)
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


@torch.no_grad()
def candidate_rollout_stats(candidate_masks: Tensor, choice_flags: Tensor, valid_steps: Tensor) -> dict[str, Tensor]:
    """Summarize the candidate sets actually offered on accepted actor steps.

    The masks are captured from actor policy outputs before episode resets;
    commanded-goal counts alone cannot recover them. Turnover compares each
    accepted choice with the preceding accepted choice in the same rollout.
    """
    if candidate_masks.ndim != 3 or choice_flags.shape != candidate_masks.shape[:2] or valid_steps.shape != choice_flags.shape:
        raise ValueError("Candidate telemetry must have aligned [batch, time] axes")
    chosen = choice_flags.bool() & valid_steps.bool()
    masks = candidate_masks.bool()
    selected = masks & chosen.unsqueeze(-1)
    per_goal = selected.sum(dim=(0, 1))
    count = chosen.sum()
    size_sum = selected.sum()
    empty = chosen & ~masks.any(dim=-1)

    prior = torch.zeros_like(masks[:, 0])
    has_prior = torch.zeros(masks.size(0), dtype=torch.bool, device=masks.device)
    turnover_sum = masks.new_zeros((), dtype=torch.float32)
    turnover_pairs = masks.new_zeros((), dtype=torch.int64)
    for step in range(masks.size(1)):
        current = masks[:, step]
        paired = chosen[:, step] & has_prior
        union = (current | prior).sum(dim=-1).clamp_min(1)
        changed = 1.0 - (current & prior).sum(dim=-1).float() / union.float()
        turnover_sum += (changed * paired).sum()
        turnover_pairs += paired.sum()
        prior = torch.where(chosen[:, step, None], current, prior)
        has_prior |= chosen[:, step]

    return {
        "choice_count": count,
        "candidate_count_mean": size_sum.float() / count.clamp_min(1).float(),
        "candidate_empty_count": empty.sum(),
        "candidate_distinct_goals": (per_goal > 0).sum(),
        "candidate_turnover": turnover_sum / turnover_pairs.clamp_min(1).float(),
        "candidate_turnover_pairs": turnover_pairs,
        "candidate_per_goal": per_goal,
    }
