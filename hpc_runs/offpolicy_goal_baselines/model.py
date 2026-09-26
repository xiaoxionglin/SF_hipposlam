"""Discrete-action CRL networks adapted from the official CRL objective.

The visual trunk is deliberately outside this module. Training consumes stable
features produced by IntrMotiv's frozen ImageNet ResNet-18 layer-2 trunk.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


def contrastive_loss(scores: torch.Tensor, logsumexp_coeff: float = 0.1) -> tuple[torch.Tensor, dict[str, float]]:
    """Symmetric InfoNCE with the JaxGCRL log-sum-exp stabilizer."""
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be a square [batch, batch] matrix")
    labels = torch.arange(scores.shape[0], device=scores.device)
    forward = F.cross_entropy(scores, labels)
    backward = F.cross_entropy(scores.T, labels)
    regularizer = 0.5 * (
        torch.logsumexp(scores, dim=1).square().mean() + torch.logsumexp(scores, dim=0).square().mean()
    )
    # JaxGCRL's symmetric objective is the sum of the two directions.
    loss = forward + backward + float(logsumexp_coeff) * regularizer
    with torch.no_grad():
        accuracy = (scores.argmax(dim=1) == labels).float().mean()
        positive = scores.diagonal().mean()
        negative = (scores.sum() - scores.diagonal().sum()) / max(scores.numel() - len(labels), 1)
    return loss, {
        "critic_accuracy": float(accuracy),
        "critic_positive": float(positive),
        "critic_negative": float(negative),
        "critic_logsumexp": float(regularizer.detach()),
    }


class ContrastiveGoalAgent(nn.Module):
    """Categorical actor and action-conditioned L2-energy critic."""

    def __init__(
        self,
        feature_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
        repr_dim: int = 64,
        action_dim: int = 32,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_actions = int(num_actions)
        self.temperature = float(temperature)
        self.action_embedding = nn.Embedding(num_actions, action_dim)
        self.state_encoder = _mlp(feature_dim, hidden_dim, hidden_dim)
        self.goal_encoder = _mlp(feature_dim, hidden_dim, repr_dim)
        self.sa_encoder = _mlp(hidden_dim + action_dim, hidden_dim, repr_dim)
        self.actor = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        # L3P's graph requires a directed, action-independent temporal cost.
        self.distance_state = _mlp(feature_dim, hidden_dim, repr_dim)
        self.distance_goal = _mlp(feature_dim, hidden_dim, repr_dim)
        self.distance_head = nn.Sequential(
            nn.Linear(3 * repr_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Softplus()
        )
        # L3P requires a symmetric landmark space whose geometry reflects
        # temporal reachability. Medoids make a decoder unnecessary: every
        # selected node remains an actually achieved frozen-visual feature.
        self.landmark_encoder = _mlp(feature_dim, hidden_dim, repr_dim)

    def policy_logits(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.actor(torch.cat((state, goal), dim=-1))

    def policy(self, state: torch.Tensor, goal: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        logits = self.policy_logits(state, goal)
        return logits.argmax(dim=-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()

    def _sa_repr(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state = self.state_encoder(state)
        action = self.action_embedding(action.long())
        return self.sa_encoder(torch.cat((state, action), dim=-1))

    def goal_repr(self, goal: torch.Tensor) -> torch.Tensor:
        return self.goal_encoder(goal)

    def score_matrix(self, state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        sa = self._sa_repr(state, action)
        g = self.goal_repr(goal)
        return -torch.cdist(sa, g).square() / self.temperature

    def paired_scores(self, state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        sa = self._sa_repr(state, action)
        g = self.goal_repr(goal)
        return -(sa - g).square().sum(dim=-1) / self.temperature

    def all_action_scores(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        states = state[:, None].expand(batch, self.num_actions, self.feature_dim).reshape(-1, self.feature_dim)
        goals = goal[:, None].expand(batch, self.num_actions, self.feature_dim).reshape(-1, self.feature_dim)
        actions = torch.arange(self.num_actions, device=state.device).repeat(batch)
        return self.paired_scores(states, actions, goals).view(batch, self.num_actions)

    def temporal_distance(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        s = self.distance_state(state)
        g = self.distance_goal(goal)
        return self.distance_head(torch.cat((s, g, s - g), dim=-1)).squeeze(-1)

    def landmark_repr(self, goal: torch.Tensor) -> torch.Tensor:
        return self.landmark_encoder(goal)

    def losses(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        future_goal: torch.Tensor,
        offset: torch.Tensor,
        random_goal: torch.Tensor,
        entropy_coeff: float = 0.01,
        logsumexp_coeff: float = 0.1,
        landmark_loss_coeff: float = 1.0,
        max_future: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        critic, metrics = contrastive_loss(self.score_matrix(state, action, future_goal), logsumexp_coeff)
        positive_distance = self.temporal_distance(state, future_goal)
        distance_loss = F.smooth_l1_loss(positive_distance, torch.log1p(offset.float()))
        negative_distance = self.temporal_distance(state, random_goal)
        negative_floor = math.log1p(max_future)
        distance_loss = distance_loss + 0.5 * F.relu(negative_floor - negative_distance).square().mean()

        if landmark_loss_coeff > 0.0:
            landmark_state = self.landmark_repr(state)
            landmark_goal = self.landmark_repr(future_goal)
            landmark_distance = (landmark_state - landmark_goal).square().mean(dim=-1)
            landmark_target = torch.log1p(offset.float())
            landmark_loss = F.smooth_l1_loss(landmark_distance, landmark_target)
        else:
            # Keep the CRL cell free of L3P-only forward/backward work.
            landmark_loss = state.new_zeros(())
            landmark_distance = state.new_zeros(state.shape[0])

        logits = self.policy_logits(state, future_goal)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        with torch.no_grad():
            q = self.all_action_scores(state, future_goal)
        actor = (probs * (float(entropy_coeff) * log_probs - q)).sum(dim=-1).mean()
        metrics.update(
            actor_loss=float(actor.detach()),
            critic_loss=float(critic.detach()),
            distance_loss=float(distance_loss.detach()),
            distance_positive=float(torch.expm1(positive_distance.detach()).mean()),
            landmark_loss=float(landmark_loss.detach()),
            landmark_distance=float(landmark_distance.detach().mean()),
            policy_entropy=float((-(probs * log_probs).sum(dim=-1)).mean().detach()),
        )
        return critic + distance_loss + float(landmark_loss_coeff) * landmark_loss, actor, metrics
