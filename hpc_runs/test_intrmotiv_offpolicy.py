import copy
import unittest

import torch
from torch import nn

from hpc_runs.intrmotiv_offpolicy.contracts import (
    UpdateSchedule,
    advance_memory,
    canonical_events,
    double_dqn_target,
    first_arrival,
)
from hpc_runs.intrmotiv_offpolicy.replay import Observation, SequenceReplay, Transition
from hpc_runs.intrmotiv_offpolicy.worker import DoubleDQNLearner, QWorker


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2 * 3 + 1 + 2, 8)

    def forward(self, x):
        return torch.tanh(self.layer(x))

    def get_out_size(self):
        return 8


def observation(i):
    return Observation(torch.tensor([float(i), 0.0]), torch.tensor([0.0]), torch.tensor([False, i >= 3]))


def fill(replay, count=8):
    for i in range(count):
        replay.append(Transition(0, 0, i, observation(i), observation(i + 1), 0, 0, 0, 64 - i))


class ContractsTest(unittest.TestCase):
    def test_double_dqn_disagreeing_heads(self):
        online = torch.tensor([[3.0, 2.0]], requires_grad=True)
        target = torch.tensor([[1.0, 8.0]], requires_grad=True)
        actual = double_dqn_target(torch.tensor([0.0]), torch.tensor([False]), online, target)
        self.assertAlmostEqual(actual.item(), 0.99, places=6)
        self.assertFalse(actual.requires_grad)
        self.assertEqual(double_dqn_target(torch.tensor([1.0]), torch.tensor([True]), online, target).item(), 1.0)

    def test_source_recognition_semantics(self):
        a = torch.tensor([[0.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
        self.assertEqual(canonical_events(a).tolist(), [[False, False], [True, False], [False, True]])
        self.assertEqual(canonical_events(a, True).tolist(), [[False, False], [False, False], [False, True]])

    def test_first_arrival_and_budget(self):
        events = torch.tensor([[False], [False], [True], [False], [True]])
        r, d, m = first_arrival(events, 0, 4, torch.zeros(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool))
        self.assertEqual(r.tolist(), [0, 1, 0, 0])
        self.assertEqual(m.tolist(), [True, True, False, False])
        self.assertTrue(d[1])
        r, d, m = first_arrival(events, 0, 1, torch.zeros(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool))
        self.assertTrue(d[0])
        self.assertEqual(r.sum(), 0)
        events[0] = True
        self.assertFalse(
            first_arrival(events, 0, 4, torch.zeros(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool))[2].any()
        )

    def test_washout_width_not_recurrence(self):
        memory = torch.ones(1, 2, 71)
        for _ in range(70):
            memory = advance_memory(memory, torch.zeros(1, 2), 8)
        self.assertGreater(memory.sum(), 0)
        self.assertEqual(advance_memory(memory, torch.zeros(1, 2), 8).sum(), 0)

    def test_invalid_final_is_not_training_data(self):
        _, _, mask = first_arrival(torch.tensor([[False], [True]]), 0, 64, torch.tensor([False]), torch.tensor([False]))
        self.assertFalse(mask.any())

    def test_aggregate_update_intensity(self):
        schedule = UpdateSchedule()
        self.assertEqual(schedule.due(16384 + 64 * 3, 1, 16384), 2)

    def test_replay_her_and_restore(self):
        replay = SequenceReplay(capacity=20, width=3, seed=11, reference_hash="test")
        fill(replay)
        state = replay.state_dict()
        a = replay.sample([1], her_fraction=1, horizon=4)
        self.assertTrue(a["relabeled"])
        self.assertEqual(a["budget"], 4)
        self.assertEqual(a["goal"], 1)
        self.assertEqual(a["reward"].sum(), 1)
        replay.load_state_dict(state)
        b = replay.sample([1], her_fraction=1, horizon=4)
        self.assertEqual(a["segment"][0].index, b["segment"][0].index)
        self.assertTrue(torch.equal(a["reward"], b["reward"]))
        self.assertEqual(a["segment"][0].goal, 0)

    def test_eviction_and_cross_stream(self):
        replay = SequenceReplay(capacity=5, width=3, reference_hash="test")
        fill(replay)
        with self.assertRaisesRegex(ValueError, "prefix"):
            replay.segment((0, 0, 3))
        with self.assertRaises(ValueError):
            replay.append(Transition(1, 0, 4, observation(4), observation(5), 0, 0, 0, 4))
        with self.assertRaises(ValueError):
            replay.append(Transition(0, 1, 0, observation(0), observation(1), 0, 0, 0, 4))

    def test_write_rebuild_target_and_freeze(self):
        torch.manual_seed(1)
        worker = QWorker(
            Decoder(), 2, 1, repeat_width=1, length=3, write_modulation=torch.ones(2, 4) * 0.1, n_actions=2
        )
        learner = DoubleDQNLearner(worker, target_period=2)
        pre, bypass = torch.ones(4, 1, 2) * 3, torch.zeros(4, 1, 1)
        goal, budget = torch.zeros(4, 1, dtype=torch.long), torch.arange(64, 60, -1)[:, None]
        original = copy.deepcopy(learner.target.state_dict())
        learner.update(
            pre,
            bypass,
            goal,
            budget,
            torch.zeros(3, 1, dtype=torch.long),
            torch.ones(3, 1),
            torch.ones(3, 1, dtype=torch.bool),
            torch.ones(3, 1, dtype=torch.bool),
            worker.initial(1),
            worker.initial(1),
        )
        self.assertTrue(all(p.grad is None for p in learner.target.parameters()))
        self.assertTrue(all(torch.equal(v, learner.target.state_dict()[k]) for k, v in original.items()))
        self.assertFalse(torch.equal(worker.write_modulation, learner.target.write_modulation))
        with self.assertRaisesRegex(ValueError, "prefix"):
            worker.rebuild(pre[:2], bypass[:2], goal[:2], budget[:2])
        self.assertFalse(
            torch.equal(worker.rebuild(pre, bypass, goal, budget), learner.target.rebuild(pre, bypass, goal, budget))
        )


if __name__ == "__main__":
    unittest.main()


class IntegrationContractsTest(unittest.TestCase):
    def test_terminal_reset_does_not_alias_observation(self):
        from hpc_runs.intrmotiv_offpolicy.terminal import reset_with_final

        class Env:
            def __init__(self):
                self.image = {"obs": torch.tensor([3.0])}

            def reset(self):
                self.image["obs"].fill_(9)
                return self.image, {}

        env = Env()
        reset, info = reset_with_final(env, env.image, {"intrmotiv_final_observation_valid": True})
        self.assertEqual(reset["obs"].item(), 9)
        self.assertEqual(info["final_observation"]["obs"].item(), 3)
        _, info = reset_with_final(env, env.image, {})
        self.assertIsNone(info["final_observation"])
        self.assertFalse(info["final_observation_valid"])

    def test_invalid_terminal_row_has_no_payload(self):
        replay = SequenceReplay(20, 3, reference_hash="x")
        replay.append(Transition(0, 0, 0, observation(0), None, 0, 0, 0, 64, True, False, False))
        with self.assertRaises(ValueError):
            replay.segment((0, 0, 0))
        replay.append(Transition(0, 1, 0, observation(0), observation(1), 0, 0, 1, 64))
        self.assertEqual(len(replay.segment((0, 1, 0))[0]), 0)

    def test_prefix_rebuild_matches_full_history(self):
        worker = QWorker(Decoder(), 2, 1, repeat_width=1, length=3, n_actions=2)
        pre = torch.randn(15, 1, 2) + 3
        bypass = torch.zeros(15, 1, 1)
        goals = torch.zeros(15, 1, dtype=torch.long)
        budgets = torch.ones(15, 1) * 64
        full = worker.rebuild(pre, bypass, goals, budgets, episode_start=True)
        tail = worker.rebuild(pre[-3:], bypass[-3:], goals[-3:], budgets[-3:])
        self.assertTrue(torch.equal(full, tail))

    def test_two_destination_navigation_learns_goal_specific_actions(self):
        torch.manual_seed(17)
        worker = QWorker(Decoder(), 2, 1, repeat_width=1, length=3, n_actions=2)
        learner = DoubleDQNLearner(worker, learning_rate=0.002, target_period=10)
        # Two directed one-step exits: action 0 reaches goal 0, action 1 goal 1.
        # Off-diagonal rows are failed original commands; their actual actions
        # also supply genuine successes when relabeled to the reached exit.
        goals = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
        actions = torch.tensor([[0, 1, 0, 1]])
        reward = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
        for _ in range(500):
            learner.update(
                torch.ones(2, 4, 2) * 3,
                torch.zeros(2, 4, 1),
                goals,
                torch.ones(2, 4),
                actions,
                reward,
                torch.ones(1, 4, dtype=torch.bool),
                torch.ones(1, 4, dtype=torch.bool),
                worker.initial(4),
                worker.initial(4),
            )
        with torch.no_grad():
            q, _ = worker.step(
                worker.initial(2), torch.ones(2, 2) * 3, torch.zeros(2, 1), torch.tensor([0, 1]), torch.ones(2)
            )
        self.assertEqual(q.argmax(-1).tolist(), [0, 1])
        self.assertLess(torch.max(torch.abs(q - torch.eye(2))).item(), 0.1)
        self.assertEqual(learner.updates // learner.target_period, 50)


class GoalWriteTest(unittest.TestCase):
    def test_goal_changes_memory_but_not_achievement(self):
        modulation = torch.zeros(2, 4)
        modulation[1, :2] = 1
        worker = QWorker(Decoder(), 2, 1, repeat_width=1, length=3, write_modulation=modulation, n_actions=2)
        pre = torch.tensor([[3.0, 2.0]])
        canonical = canonical_events(torch.relu(pre - 2.43), exclusive=True)
        _, a = worker.step(worker.initial(1), pre, torch.zeros(1, 1), torch.tensor([0]), torch.tensor([64]))
        _, b = worker.step(worker.initial(1), pre, torch.zeros(1, 1), torch.tensor([1]), torch.tensor([64]))
        self.assertFalse(torch.equal(a, b))
        self.assertTrue(torch.equal(canonical, canonical_events(torch.relu(pre - 2.43), exclusive=True)))


class RuntimeAuditTest(unittest.TestCase):
    def test_completion_gate_rejects_missing_target_copy(self):
        import json
        import tempfile
        from pathlib import Path

        from hpc_runs.intrmotiv_offpolicy.audit_runtime import audit
        from hpc_runs.intrmotiv_study import load_study

        study = load_study(Path(__file__).with_name("studies") / "intrmotiv_ddqn_her_preflight2.study.json")
        with tempfile.TemporaryDirectory() as tmp:
            for run in study.expand_runs():
                path = Path(tmp) / run.name
                path.mkdir()
                (path / "runtime_gate.json").write_text(
                    json.dumps(
                        dict(
                            frames=500096,
                            updates=1696,
                            target_copies=1,
                            invalid_final_exclusions=64,
                            frozen_reference_unchanged=True,
                            her_samples=100,
                        )
                    )
                )
                (path / "conversion.json").write_text(json.dumps(dict(worker_hash="same")))
                (path / "metrics.jsonl").write_text(
                    "\n".join(
                        json.dumps(dict(frames=f, updates=u, throughput_fps=1000.0, realized_her_fraction=0.3))
                        for f, u in [(100000, 100), (500096, 1696)]
                    )
                )
                (path / "checkpoint_p0").mkdir()
                for frames in (0, 500096):
                    (path / "checkpoint_p0" / f"checkpoint_000000001_{frames}.pth").touch()
            self.assertTrue(audit(study, tmp)["runtime_passed"])
            gate_path = path / "runtime_gate.json"
            gate = json.loads(gate_path.read_text())
            gate["target_copies"] = 0
            gate_path.write_text(json.dumps(gate))
            with self.assertRaisesRegex(ValueError, "target-copy"):
                audit(study, tmp)


class FeatureIsolationTest(unittest.TestCase):
    def test_pose_sidecar_cannot_change_source_features(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from hpc_runs.intrmotiv_offpolicy.features import FrozenParentFeatures
        from sf_working_directories.IntrMotiv.dmlab.dmlab30 import DMLAB_INSTRUCTIONS

        class Projection(nn.Module):
            intercept = 2.43

            def preactivation(self, x):
                return x[:, :2]

            def activation(self, x):
                return torch.relu(x)

        class Encoder(nn.Module):
            depth_sensor = True
            bypass = True
            goal_reference_projection = None
            context_feedback = "none"
            action_path_integration = False
            context_action_count = 0
            instructions_lstm_units = 1
            dg_goal_write = False

            def __init__(self):
                super().__init__()
                self.DG_projection = Projection()
                self.depth_encoder = nn.Identity()

            def projection_input(self, obs):
                assert set(obs) == {"obs", DMLAB_INSTRUCTIONS}
                return torch.cat((obs["obs"][:, :2].flatten(1), obs[DMLAB_INSTRUCTIONS].float()), -1)

        class Actor(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder()
                self.core = SimpleNamespace(Hippo_n_feature=2)

            def forward_head(self, obs):
                x = self.encoder.projection_input(obs)
                return torch.cat((torch.relu(x[:, :2] - 2.43), obs["obs"][:, -1:].flatten(1), x[:, -1:]), -1)

        actor = Actor()
        extract = FrozenParentFeatures(actor, True)
        obs = {"obs": torch.tensor([3.0, 0.0, 0.0, 0.5]).reshape(1, 4, 1, 1), DMLAB_INSTRUCTIONS: torch.ones(1, 1)}
        with patch("sample_factory.algo.utils.rl_utils.prepare_and_normalize_obs", lambda actor, value: value):
            a = extract({**obs, "telemetry_pose": torch.zeros(1, 3)})[0]
            b = extract({**obs, "telemetry_pose": torch.ones(1, 3) * 1e9, "pos": torch.randn(1, 3)})[0]
        for name in vars(a):
            self.assertTrue(torch.equal(getattr(a, name), getattr(b, name)))
