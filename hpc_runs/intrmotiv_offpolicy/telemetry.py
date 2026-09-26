"""Bounded counters for coverage, replay supervision and finite-budget support."""

from collections import Counter, defaultdict


class Coverage:
    def __init__(self, registry):
        self.registry = registry
        self.counts = Counter()
        self.episodes = defaultdict(set)

    def collect(self, stream, episode, successor, goal, finished):
        if finished:
            self.counts[f"goal_{goal}/attempts"] += 1
        if successor is None:
            return
        for g in self.registry:
            if successor.events[g]:
                self.counts[f"goal_{g}/achieved_events"] += 1
                self.episodes[f"goal_{g}/achieved_episodes"].add((stream, episode))
        if successor.events[goal]:
            self.counts[f"goal_{goal}/commanded_arrivals"] += 1

    def replay(self, samples):
        for s in samples:
            g = s["goal"]
            kind = "her" if s["relabeled"] else "original"
            self.counts[f"goal_{g}/requested_her_segments"] += int(s["requested_her"])
            self.counts[f"goal_{g}/realized_her_segments"] += int(s["relabeled"])
            self.counts[f"goal_{g}/{kind}/segments"] += 1
            self.counts[f"goal_{g}/{kind}/reward_segments"] += int(s["reward"].sum() > 0)
            self.episodes[f"goal_{g}/{kind}/contributing_episodes"].add(
                (s["segment"][0].stream, s["segment"][0].episode)
            )
            if s["selected_goal_offset"] is not None:
                self.counts[f'goal_{g}/her/selected_offset_{s["selected_goal_offset"]}'] += 1
            for i, valid in enumerate(s["mask"]):
                if valid:
                    budget = s["budget"] - i
                    self.counts[f"goal_{g}/{kind}/budget_{(budget-1)//8*8+1}_{(budget-1)//8*8+8}"] += 1

    def metrics(self):
        return {**self.counts, **{k: len(v) for k, v in self.episodes.items()}}
