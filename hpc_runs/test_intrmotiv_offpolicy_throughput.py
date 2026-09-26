import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hpc_runs.intrmotiv_offpolicy.batch import batched_prefix_memory, learn_batch, prefix_memory, slice_attempt
from hpc_runs.intrmotiv_offpolicy.qualify import FixtureDecoder
from hpc_runs.intrmotiv_offpolicy.worker import DoubleDQNLearner, QWorker
from hpc_runs.test_intrmotiv_offpolicy_v2 import replay_fixture


class ThroughputParityTest(unittest.TestCase):
    def worker(self):
        torch.manual_seed(123)
        return QWorker(FixtureDecoder(9, 16), 2, 1, repeat_width=1, length=3, n_actions=2, batch_independent=True)

    def test_mixed_prefixes_equal_reference_exactly(self):
        worker = self.worker()
        replay = replay_fixture()
        prefixes = [
            [],
            list(replay.rows.values())[:1],
            list(replay.rows.values())[:3],
            list(replay.rows.values())[13:16],
        ]
        # Use nonzero preactivations so this tests memory contents, not zeros.
        from dataclasses import replace

        prefixes = [
            [
                replace(r, observation=replace(r.observation, preactivation=torch.tensor([3.0 + r.index, 4.0])))
                for r in p
            ]
            for p in prefixes
        ]
        expected = torch.cat([prefix_memory(worker, p, "cpu") for p in prefixes])
        actual = batched_prefix_memory(worker, prefixes, "cpu")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "prefix"):
            batched_prefix_memory(worker, [prefixes[-1][:1]], "cpu")

    def test_sequence_q_gradient_action_parity(self):
        a = self.worker()
        b = copy.deepcopy(a)
        pre = torch.randn(7, 4, 2) + 3.0
        by = torch.randn(7, 4, 1)
        goals = torch.randint(0, 2, (7, 4))
        budgets = torch.arange(7, 0, -1)[:, None].expand(7, 4)
        clocks = budgets + 100
        state = a.initial(4)
        reference = []
        for i in range(7):
            q, state = a.step(state, pre[i], by[i], goals[i], budgets[i], clocks[i])
            reference.append(q)
        reference = torch.stack(reference)
        actual = b.sequence(b.initial(4), pre, by, goals, budgets, clocks)
        torch.testing.assert_close(reference, actual, rtol=1e-5, atol=2e-6)
        self.assertTrue(torch.equal(reference.argmax(-1), actual.argmax(-1)))
        reference.square().sum().backward()
        actual.square().sum().backward()
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=2e-5)

    def test_mixed_length_her_optimizer_and_target_parity(self):
        worker = self.worker()
        a = DoubleDQNLearner(copy.deepcopy(worker), target_period=2)
        b = DoubleDQNLearner(copy.deepcopy(worker), target_period=2, execution="batched")
        replay = replay_fixture()
        her = replay.attempt((0, 0, 0), [1], True)
        samples = [her, slice_attempt(her, 24, 40, 3), replay.attempt((0, 0, 40), [1], False)]
        for _ in range(3):
            x = learn_batch(a, samples, "cpu")
            y = learn_batch(b, samples, "cpu")
            self.assertAlmostEqual(x["td_loss"], y["td_loss"], places=6)
            self.assertEqual(x["valid_loss_positions"], y["valid_loss_positions"])
            self.assertEqual(x["target_copies"], y["target_copies"])
            for p, q in zip(a.online.parameters(), b.online.parameters()):
                torch.testing.assert_close(p, q, rtol=1e-5, atol=2e-6)
        self.assertTrue(all(p.grad is None for p in b.target.parameters()))
        self.assertEqual(set(a.online.state_dict()), set(b.online.state_dict()))

    def test_decoder_calls_collapse_to_one_per_network(self):
        w = self.worker()
        learner = DoubleDQNLearner(w, execution="batched")
        sample = replay_fixture().attempt((0, 0, 0), [1], True)
        with patch.object(
            learner.online.decoder, "forward", wraps=learner.online.decoder.forward
        ) as online, patch.object(learner.target.decoder, "forward", wraps=learner.target.decoder.forward) as target:
            learn_batch(learner, [sample], "cpu")
        self.assertEqual(online.call_count, 1)
        self.assertEqual(target.call_count, 1)

    def test_unqualified_and_goal_write_fail_closed(self):
        w = self.worker()
        w.batch_independent = False
        with self.assertRaisesRegex(ValueError, "qualified"):
            DoubleDQNLearner(w, execution="batched")
        w.batch_independent = True
        w.write_modulation = nn.Parameter(torch.zeros(2, 4))
        with self.assertRaisesRegex(ValueError, "qualified"):
            DoubleDQNLearner(w, execution="batched")

    def test_training_batchnorm_rejected(self):
        w = self.worker()
        w.decoder.layer = nn.Sequential(w.decoder.layer, nn.BatchNorm1d(16))
        with self.assertRaisesRegex(ValueError, "batch-dependent"):
            w.sequence(
                w.initial(1),
                torch.ones(2, 1, 2),
                torch.zeros(2, 1, 1),
                torch.zeros(2, 1, dtype=torch.long),
                torch.ones(2, 1),
                torch.zeros(2, 1),
            )


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
