"""L3P-style landmark medoids and directed shortest-path planning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


def farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or not 0 < count <= len(points):
        raise ValueError("points must be [N,D] and 0 < count <= N")
    chosen = [0]
    distances = np.square(points - points[0]).sum(axis=1)
    while len(chosen) < count:
        index = int(np.argmax(distances))
        chosen.append(index)
        distances = np.minimum(distances, np.square(points - points[index]).sum(axis=1))
    return np.asarray(chosen, dtype=np.int64)


def floyd_warshall_next(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = np.asarray(cost, dtype=np.float64).copy()
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("cost must be square")
    n = distance.shape[0]
    next_hop = np.full((n, n), -1, dtype=np.int64)
    rows, cols = np.where(np.isfinite(distance))
    next_hop[rows, cols] = cols
    for k in range(n):
        candidate = distance[:, k, None] + distance[k, None, :]
        better = candidate < distance
        distance[better] = candidate[better]
        proposed = np.broadcast_to(next_hop[:, k, None], (n, n))
        next_hop[better] = proposed[better]
    return distance, next_hop


@dataclass(frozen=True)
class PlanDecision:
    goal: np.ndarray
    steps: int
    landmark_index: int | None
    estimated_steps: float


class LandmarkPlanner:
    def __init__(
        self,
        landmark_count: int = 50,
        candidates: int = 1000,
        neighbors: int = 8,
        local_horizon: float = 16.0,
        edge_horizon: float = 64.0,
    ) -> None:
        self.landmark_count = int(landmark_count)
        self.candidates = int(candidates)
        self.neighbors = int(neighbors)
        self.local_horizon = float(local_horizon)
        self.edge_horizon = float(edge_horizon)
        self.features: np.ndarray | None = None
        self.poses: np.ndarray | None = None
        self.graph_cost: np.ndarray | None = None
        self.graph_distance: np.ndarray | None = None
        self.rebuild_count = 0
        self.subgoal_queries = 0
        self.landmark_subgoals = 0
        self.finite_edges = 0
        self.reachable_pair_fraction = 0.0
        self.unreachable_queries = 0
        self.repeat_avoided = 0
        self.commitment_steps = 0

    @staticmethod
    def _distances(agent, states: np.ndarray, goals: np.ndarray, device: torch.device, block: int = 4096) -> np.ndarray:
        pairs = [(i, j) for i in range(len(states)) for j in range(len(goals))]
        result = np.empty(len(pairs), dtype=np.float32)
        agent.eval()
        with torch.no_grad():
            for start in range(0, len(pairs), block):
                chunk = pairs[start : start + block]
                s = torch.as_tensor(np.stack([states[i] for i, _ in chunk]), device=device)
                g = torch.as_tensor(np.stack([goals[j] for _, j in chunk]), device=device)
                result[start : start + len(chunk)] = torch.expm1(agent.temporal_distance(s, g)).cpu().numpy()
        return result.reshape(len(states), len(goals))

    def rebuild(self, replay, agent, device: torch.device) -> None:
        features, poses = replay.sample_candidates(self.candidates)
        with torch.no_grad():
            embedding = agent.landmark_repr(torch.as_tensor(features, device=device)).cpu().numpy()
        indices = farthest_point_indices(embedding, min(self.landmark_count, len(features)))
        self.features, self.poses = features[indices], poses[indices]
        dense = self._distances(agent, self.features, self.features, device)
        cost = np.full_like(dense, np.inf, dtype=np.float64)
        np.fill_diagonal(cost, 0.0)
        for row in range(len(cost)):
            order = np.argsort(dense[row])
            order = order[(order != row) & np.isfinite(dense[row, order])]
            order = order[dense[row, order] <= self.edge_horizon][: self.neighbors]
            cost[row, order] = dense[row, order]
        self.graph_cost = cost
        self.graph_distance, _ = floyd_warshall_next(cost)
        self.rebuild_count += 1
        self.finite_edges = int(np.isfinite(cost).sum() - len(cost))
        off_diagonal = ~np.eye(len(cost), dtype=bool)
        self.reachable_pair_fraction = float(np.isfinite(self.graph_distance[off_diagonal]).mean())

    @property
    def mean_commitment(self) -> float:
        return self.commitment_steps / max(self.landmark_subgoals, 1)

    def plan(
        self,
        state: np.ndarray,
        final_goal: np.ndarray,
        agent,
        device: torch.device,
        previous_landmark: int | None = None,
        max_horizon: int = 64,
    ) -> PlanDecision:
        """Choose an L3P subgoal and persist it for predicted travel time.

        The immediate previously attempted landmark is excluded, matching the
        paper's anti-sticking mechanism. A direct goal is used whenever the
        learned temporal model says it is locally reachable.
        """
        self.subgoal_queries += 1
        if self.features is None or self.graph_cost is None:
            direct_cost = float(self._distances(agent, state[None], final_goal[None], device)[0, 0])
            steps = (
                max(1, min(int(max_horizon), int(math.ceil(direct_cost))))
                if math.isfinite(direct_cost)
                else int(max_horizon)
            )
            return PlanDecision(final_goal, steps, None, direct_cost)
        n = len(self.features)
        augmented = np.full((n + 2, n + 2), np.inf, dtype=np.float64)
        augmented[:n, :n] = self.graph_cost
        np.fill_diagonal(augmented, 0.0)
        start, goal = n, n + 1
        start_cost = self._distances(agent, state[None], self.features, device)[0]
        valid_start = np.flatnonzero(np.isfinite(start_cost) & (start_cost <= self.edge_horizon))
        valid_start = valid_start[np.argsort(start_cost[valid_start])[: self.neighbors]]
        if previous_landmark is not None and previous_landmark in valid_start:
            valid_start = valid_start[valid_start != previous_landmark]
            self.repeat_avoided += 1
        for index in valid_start:
            augmented[start, index] = start_cost[index]
        finish_cost = self._distances(agent, self.features, final_goal[None], device)[:, 0]
        valid_finish = np.flatnonzero(np.isfinite(finish_cost) & (finish_cost <= self.edge_horizon))
        valid_finish = valid_finish[np.argsort(finish_cost[valid_finish])[: self.neighbors]]
        for index in valid_finish:
            augmented[index, goal] = finish_cost[index]
        direct_cost = float(self._distances(agent, state[None], final_goal[None], device)[0, 0])
        if direct_cost <= self.local_horizon:
            augmented[start, goal] = direct_cost
        _, next_hop = floyd_warshall_next(augmented)
        hop = int(next_hop[start, goal])
        if hop in (-1, goal):
            if hop == -1:
                self.unreachable_queries += 1
            steps = max(1, min(int(max_horizon), int(math.ceil(direct_cost)))) if math.isfinite(direct_cost) else 1
            return PlanDecision(final_goal, steps, None, direct_cost)
        estimated = float(start_cost[hop])
        steps = max(1, min(int(max_horizon), int(math.ceil(estimated))))
        self.landmark_subgoals += 1
        self.commitment_steps += steps
        return PlanDecision(self.features[hop], steps, hop, estimated)

    def subgoal(self, state: np.ndarray, final_goal: np.ndarray, agent, device: torch.device) -> np.ndarray:
        """Compatibility wrapper for callers that only need the goal feature."""
        return self.plan(state, final_goal, agent, device).goal
