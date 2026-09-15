import unittest

import torch

from hpc_runs.intrmotiv_offpolicy.batch import PositionBatcher, slice_attempt
from hpc_runs.intrmotiv_offpolicy.qualify import FixtureDecoder, expressivity
from hpc_runs.intrmotiv_offpolicy.replay import Observation, SequenceReplay, Transition
from hpc_runs.intrmotiv_offpolicy.terminal import certified_successor
from hpc_runs.intrmotiv_offpolicy.worker import QWorker


def replay_fixture(hit=40, end=64, invalid=False, truncated=False):
    replay = SequenceReplay(200, 3, seed=7, reference_hash="fixture")

    def obs(i):
        return Observation(torch.zeros(2), torch.tensor([float(i)]), torch.tensor([False, i == hit]))

    for i in range(end):
        valid = not (i == end - 1 and invalid)
        replay.append(
            Transition(
                0,
                0,
                i,
                obs(i),
                obs(i + 1) if valid else None,
                0,
                0,
                0,
                64 - i,
                terminated=i == end - 1 and not truncated,
                truncated=i == end - 1 and truncated,
                successor_valid=valid,
            )
        )
    return replay


class RepairTest(unittest.TestCase):
    def test_her_hit40_and_suffix_deadline(self):
        r = replay_fixture()
        s = r.attempt((0, 0, 0), [1], True, unroll=16)
        self.assertEqual(len(s["segment"]), 40)
        self.assertEqual(s["reward"].sum(), 1)
        self.assertEqual(s["reward"][-1], 1)
        self.assertTrue(s["done"][-1])
        self.assertTrue(s["mask"].all())
        suffix = slice_attempt(s, 24, 40, r.width)
        self.assertEqual(suffix["budget"], 40)
        self.assertEqual(suffix["virtual_attempt"]["deadline"], 64)
        self.assertEqual(suffix["virtual_attempt"]["first_hit_index"], 40)
        self.assertEqual(suffix["segment"][0].index, 24)
        self.assertEqual(suffix["prefix"][-1].index, 23)
        self.assertEqual(suffix["reward"].sum(), 1)

    def test_observed_hit_before_invalid_final_is_eligible(self):
        r = replay_fixture(5, 20, True)
        s = r.attempt((0, 0, 0), [1], True)
        self.assertTrue(s["relabeled"])
        self.assertTrue(s["episode_complete"])
        self.assertEqual(s["reward"].sum(), 1)
        self.assertIsNone(r.rows[(0, 0, 19)].environment_remaining)
        self.assertIsNone(r.rows[(0, 0, 19)].successor)

    def test_boundary_bootstrapping(self):
        for truncated in (False, True):
            s = replay_fixture(hit=99, end=8, truncated=truncated).attempt((0, 0, 0), [1], True)
            self.assertFalse(s["relabeled"])
            self.assertEqual(s["reward"].sum(), 0)
            self.assertEqual(bool(s["done"][-1]), not truncated)
        s = replay_fixture(hit=99, end=64).attempt((0, 0, 60), [1], False)
        self.assertEqual(s["budget"], 4)
        self.assertTrue(s["done"][-1])
        r = replay_fixture(hit=99, end=8, invalid=True)
        s = r.attempt((0, 0, 0), [1], True)
        self.assertEqual(len(s["segment"]), 7)
        self.assertFalse(s["done"][-1])
        with self.assertRaises(ValueError):
            r.attempt((0, 0, 7), [1], True)

    def test_incomplete_future_observed_hit_and_failure(self):
        r = replay_fixture(hit=5, end=8)
        from dataclasses import replace

        r.rows[(0, 0, 7)] = replace(r.rows[(0, 0, 7)], terminated=False)
        s = r.attempt((0, 0, 0), [1], True)
        self.assertTrue(s["relabeled"])
        self.assertFalse(s["episode_complete"])
        f = r.attempt((0, 0, 6), [1], True)
        self.assertFalse(f["relabeled"])
        self.assertFalse(f["done"][-1])

    def test_certified_final_and_autoreset(self):
        reset = {"obs": torch.tensor([99.0])}
        final = {"obs": torch.tensor([3.0])}
        self.assertIsNone(certified_successor(reset, {}, True))
        self.assertIsNone(certified_successor(reset, {"final_observation": final}, True))
        self.assertIs(
            certified_successor(reset, {"final_observation": final, "final_observation_valid": True}, True), final
        )
        self.assertIs(certified_successor(reset, {}, False), reset)

    def test_readout_parity_and_no_double_memory_advance(self):
        torch.manual_seed(5)
        w = QWorker(FixtureDecoder(9), 2, 1, repeat_width=1, length=3, n_actions=2)
        g = torch.tensor([0, 1])
        b = torch.tensor([1, 40])
        clock = torch.tensor([12, 31])
        bypass = torch.randn(2, 1)
        q, m = w.step(w.initial(2), torch.ones(2, 2) * 3, bypass, g, b, clock)
        saved = m.clone()
        r = w.readout(m, bypass, g, b, clock)
        state = torch.cat((m.flatten(1), bypass, torch.nn.functional.one_hot(g, 2)), -1)
        torch.testing.assert_close(q, r, rtol=0, atol=0)
        torch.testing.assert_close(q, w.readout_state(state, b, clock), rtol=0, atol=0)
        self.assertTrue(torch.equal(m, saved))
        self.assertEqual(w.schema, "intrmotiv/ddqn-worker/v2")

    def test_equal_position_budget_and_reward_carry(self):
        r = replay_fixture()
        s = r.attempt((0, 0, 0), [1], True)
        batcher = PositionBatcher()
        batcher.pending = s
        first = batcher.sample(r, [1], 1.0, 24)
        second = batcher.sample(r, [1], 1.0, 16)
        self.assertEqual(sum(int(s["mask"].sum()) for s in first), 24)
        self.assertEqual(sum(int(s["mask"].sum()) for s in second), 16)
        self.assertEqual(second[0]["budget"], 40)
        self.assertEqual(sum(float(s["reward"].sum()) for s in first + second), 1)
        for fraction in (0.0, 0.8):
            self.assertEqual(sum(int(s["mask"].sum()) for s in batcher.sample(r, [1], fraction, 256)), 256)

    def test_nonlinear_finite_horizon_values(self):
        torch.set_num_threads(1)
        self.assertTrue(expressivity()["passed"])


if __name__ == "__main__":
    unittest.main()


class AdditionalRepairTest(unittest.TestCase):
    def test_vector_final_fields_survive_final_info_selection(self):
        import numpy as np

        from hpc_runs.intrmotiv_offpolicy.terminal import vector_final_info

        infos = {
            "final_info": {"num_frames": np.array([4, 4])},
            "_final_info": np.array([True, True]),
            "final_observation": {"obs": np.array([[3.0], [9.0]])},
            "final_observation_valid": np.array([True, False]),
            "_final_observation": np.array([True, True]),
        }
        a = vector_final_info(infos, 0, 2)
        b = vector_final_info(infos, 1, 2)
        self.assertEqual(certified_successor(None, a, True)["obs"][0], 3.0)
        self.assertIsNone(certified_successor(None, b, True))

    def test_checkpoint_schema_rejects_v1(self):
        from hpc_runs.intrmotiv_offpolicy.evaluate import validate_child_schema

        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_child_schema({"schema": "intrmotiv/ddqn-worker/v1"})
        validate_child_schema({"schema": "intrmotiv/ddqn-worker/v2"})

    def test_original_her_budget_histograms(self):
        from hpc_runs.intrmotiv_offpolicy.telemetry import Coverage

        r = replay_fixture(hit=40)
        original = r.attempt((0, 0, 0), [1], False)
        her = r.attempt((0, 0, 0), [1], True)
        coverage = Coverage([0, 1])
        coverage.replay([original, her])
        m = coverage.metrics()
        self.assertEqual(sum(v for k, v in m.items() if "/original/budget_" in k), 64)
        self.assertEqual(sum(v for k, v in m.items() if "/her/budget_" in k), 40)
        self.assertEqual(m["goal_1/her/budget_25_32"], 8)
        self.assertEqual(m["goal_0/original/budget_1_8"], 8)
        self.assertEqual(m["goal_1/her/reward_segments"], 1)

    def test_failed_batch_preserves_previously_sampled_losses(self):
        from unittest.mock import patch

        r = replay_fixture()
        s = r.attempt((0, 0, 0), [1], True)
        batcher = PositionBatcher()
        batcher.pending = s
        with patch.object(r, "sample", side_effect=ValueError("not ready")):
            with self.assertRaises(ValueError):
                batcher.sample(r, [1], 1.0, 64)
        restored = batcher.sample(r, [1], 1.0, 40)
        self.assertEqual(sum(float(s["reward"].sum()) for s in restored), 1)
        self.assertEqual(restored[0]["budget"], 64)
