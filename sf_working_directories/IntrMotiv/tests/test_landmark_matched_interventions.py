from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from sf_working_directories.IntrMotiv.evaluation import target_control_interventions as ev


class Graph(nn.Module):
    def __init__(self):
        super().__init__()
        for key, val in [
            ("passive_confidence", torch.ones(3, 3)),
            ("edge_confidence", torch.ones(3, 3)),
            ("control_attempts", torch.ones(3, 3)),
            ("tctrl", torch.ones(3, 3)),
        ]:
            self.register_buffer(key, val)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Module()
        self.core.policy_graph = Graph()
        self.core.Hippo_n_feature = 3
        self.core.target_condition_start = 3
        self.action_space = SimpleNamespace(n=3)

    def forward_head(self, obs):
        return obs["dg"]

    def forward_core(self, h, s):
        goal = h[:, 3:] if h.shape[-1] == 6 else torch.zeros_like(h)
        return torch.cat((h[:, :3], goal), -1), s + 1

    def forward_tail(self, o, **kwargs):
        return {"action_logits": o[:, 3:] * 100}


class Env:
    num_agents = 1

    def __init__(self, offset=0):
        self.unwrapped = self
        self.node = 0
        self.offset = offset

    def close(self):
        pass

    def seed(self, seed):
        self.node = 0

    def obs(self):
        return {
            "dg": torch.nn.functional.one_hot(torch.tensor([self.node]), 3).float(),
            "pos": torch.tensor([[float(self.node) + self.offset, 0, 0]]),
            "rot": torch.zeros(1, 3),
        }

    def reset(self):
        self.node = 0
        return self.obs(), {}

    def step(self, a):
        self.node = int(torch.as_tensor(a).flatten()[0])
        return self.obs(), torch.zeros(1), torch.tensor([False]), torch.tensor([False]), {}


def configure_fake_runtime(monkeypatch, unstable=False):
    import sample_factory.algo.utils.make_env as maker

    count = [0]

    def create(*a, **k):
        count[0] += 1
        return Env(count[0] if unstable else 0)

    monkeypatch.setattr(maker, "make_env_func_batched", create)
    monkeypatch.setattr(ev, "prepare_and_normalize_obs", lambda actor, obs: obs)
    monkeypatch.setattr(ev, "preprocess_actions", lambda info, actions: actions)
    monkeypatch.setattr(ev, "get_rnn_size", lambda cfg: 3)
    return ev.AttrDict(Hippo_n_feature=3, dg_goal_input="write")


def test_executes_alternative_commands_and_scores_same_destination(monkeypatch):
    cfg = configure_fake_runtime(monkeypatch)
    frame, summary = ev.run_landmark_matched_interventions(
        cfg,
        Env(),
        None,
        Actor(),
        "test",
        torch.device("cpu"),
        1000,
        True,
        max_sources=1,
        targets_per_source=2,
        repeats=1,
        prefix_cap=4,
    )
    assert summary["exact_start_verified"]
    assert summary["paired_arrival_lift"] == 1
    assert set(frame.command) == {1, 2}
    assert frame[frame.commanded].hit.all()
    assert not frame[~frame.commanded].hit.any()


def test_rejects_unmatched_engine_starts(monkeypatch):
    cfg = configure_fake_runtime(monkeypatch, unstable=True)
    with pytest.raises(RuntimeError, match="Nonreproducible"):
        ev.run_landmark_matched_interventions(
            cfg,
            Env(),
            None,
            Actor(),
            "test",
            torch.device("cpu"),
            1000,
            True,
            max_sources=1,
            targets_per_source=2,
            repeats=1,
            prefix_cap=4,
        )


def test_late_arrival_is_a_timeout(monkeypatch):
    cfg = configure_fake_runtime(monkeypatch)
    import sample_factory.algo.utils.make_env as maker

    class Delayed(Env):
        def reset(self):
            self.steps = 0
            return super().reset()

        def step(self, a):
            self.steps += 1
            return super().step(a if self.steps >= 2 else 0)

    monkeypatch.setattr(maker, "make_env_func_batched", lambda *a, **k: Delayed())
    monkeypatch.setattr(ev, "pair_deadline", lambda graph, source, target: 1 if target == 1 else 3)
    rows, _ = ev.run_landmark_matched_interventions(
        cfg,
        Env(),
        None,
        Actor(),
        "test",
        torch.device("cpu"),
        1000,
        True,
        max_sources=1,
        targets_per_source=2,
        repeats=1,
        prefix_cap=4,
    )
    late = rows[(rows.command == 1) & (rows.target == 1)].iloc[0]
    assert late.hit_time == 2 and not late.hit and late.timeout and not late.censored


def test_observed_success_is_not_censored_by_later_termination(monkeypatch):
    cfg = configure_fake_runtime(monkeypatch)
    import sample_factory.algo.utils.make_env as maker

    class EarlyTerminal(Env):
        def reset(self):
            self.steps = 0
            return super().reset()

        def step(self, a):
            self.steps += 1
            obs, rew, term, trunc, info = super().step(a)
            return obs, rew, torch.tensor([self.steps == 2]), trunc, info

    monkeypatch.setattr(maker, "make_env_func_batched", lambda *a, **k: EarlyTerminal())
    monkeypatch.setattr(ev, "pair_deadline", lambda *a: 3)
    rows, _ = ev.run_landmark_matched_interventions(
        cfg,
        Env(),
        None,
        Actor(),
        "test",
        torch.device("cpu"),
        1000,
        True,
        max_sources=1,
        targets_per_source=2,
        repeats=1,
        prefix_cap=4,
    )
    assert rows[rows.commanded].hit.all()
    assert not rows[rows.commanded].censored.any()
    assert rows[~rows.commanded].censored.all()
