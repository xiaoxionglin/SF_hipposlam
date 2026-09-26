"""Episode replay with future-state relabeling and no raw-pixel duplication."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Episode:
    features: np.ndarray
    actions: np.ndarray
    poses: np.ndarray

    @property
    def transitions(self) -> int:
        return int(self.actions.shape[0])


class EpisodeReplay:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.episodes: deque[Episode] = deque()
        self.size = 0
        self.rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)

    def add(self, features: np.ndarray, actions: np.ndarray, poses: np.ndarray) -> None:
        features = np.asarray(features, dtype=np.float16)
        actions = np.asarray(actions, dtype=np.int16)
        poses = np.asarray(poses, dtype=np.float32)
        if features.ndim != 2 or actions.ndim != 1 or poses.ndim != 2:
            raise ValueError("features/actions/poses must have ranks 2/1/2")
        if features.shape[0] != actions.shape[0] + 1 or poses.shape[0] != features.shape[0]:
            raise ValueError("an episode needs T+1 features and poses for T actions")
        episode = Episode(features, actions, poses)
        self.episodes.append(episode)
        self.size += episode.transitions
        while self.size > self.capacity and len(self.episodes) > 1:
            self.size -= self.episodes.popleft().transitions

    def _episode(self) -> Episode:
        if not self.episodes:
            raise RuntimeError("cannot sample from empty replay")
        weights = np.asarray([ep.transitions for ep in self.episodes], dtype=np.float64)
        index = int(self.rng.choice(len(weights), p=weights / weights.sum()))
        return self.episodes[index]

    def sample_goal(self) -> tuple[np.ndarray, np.ndarray]:
        episode = self._episode()
        index = int(self.rng.integers(0, episode.features.shape[0]))
        return episode.features[index].astype(np.float32), episode.poses[index].copy()

    def sample_candidates(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        sampled = [self.sample_goal() for _ in range(int(count))]
        return np.stack([item[0] for item in sampled]), np.stack([item[1] for item in sampled])

    def sample(self, batch_size: int, max_future: int, discount: float = 0.99) -> dict[str, np.ndarray]:
        if not 0.0 < discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        states, actions, goals, offsets, random_goals = [], [], [], [], []
        for _ in range(int(batch_size)):
            episode = self._episode()
            t = int(self.rng.integers(0, episode.transitions))
            high = min(episode.transitions, t + int(max_future))
            possible = np.arange(t + 1, high + 1)
            weights = np.power(float(discount), possible - t)
            future = int(self.rng.choice(possible, p=weights / weights.sum()))
            random_goal, _ = self.sample_goal()
            states.append(episode.features[t])
            actions.append(episode.actions[t])
            goals.append(episode.features[future])
            offsets.append(future - t)
            random_goals.append(random_goal)
        return {
            "state": np.asarray(states, dtype=np.float32),
            "action": np.asarray(actions, dtype=np.int64),
            "future_goal": np.asarray(goals, dtype=np.float32),
            "offset": np.asarray(offsets, dtype=np.float32),
            "random_goal": np.asarray(random_goals, dtype=np.float32),
        }
