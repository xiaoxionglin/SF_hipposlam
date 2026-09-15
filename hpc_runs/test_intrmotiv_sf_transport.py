import unittest
from dataclasses import replace

import torch

from hpc_runs.intrmotiv_offpolicy.replay import Observation, SequenceReplay, Transition
from hpc_runs.intrmotiv_offpolicy.sf_transport import OrderedIngress, updates_due


def row(stream, index, episode=0, done=False):
    obs = Observation(torch.tensor([float(index), 3.0]), torch.zeros(1), torch.tensor([False, True]))
    return Transition(stream, episode, index, obs, None, 0, 0, 1, 64, terminated=done, successor_valid=False)


class TransportTest(unittest.TestCase):
    def test_reordered_rollouts_are_reassembled_without_missing_successors(self):
        ingress = OrderedIngress(32)
        replay = SequenceReplay(32, 3, 0, "frozen")
        for i in (3, 4, 5, 0, 1, 2):
            ingress.add(0, i, row(0, i, done=i == 5))
            for r in ingress.drain():
                replay.append(r)
        self.assertEqual([r.index for r in replay.rows.values()], list(range(6)))
        self.assertEqual(ingress.pending, 0)
        self.assertIsNone(replay.last[0].successor)

    def test_rollout_tail_waits_for_actual_next_actor_features(self):
        ingress = OrderedIngress(8)
        ingress.add(0, 0, row(0, 0))
        self.assertEqual(list(ingress.drain()), [])
        next_row = row(0, 1)
        ingress.add(0, 1, next_row)
        result = list(ingress.drain())
        self.assertEqual(len(result), 1)
        self.assertIs(result[0].successor, next_row.observation)
        self.assertEqual(ingress.pending, 1)

    def test_reset_is_never_used_as_terminal_successor(self):
        ingress = OrderedIngress(8)
        ingress.add(0, 0, row(0, 0, done=True))
        ingress.add(0, 1, row(0, 0, episode=1))
        result = list(ingress.drain())
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0].successor_valid)
        self.assertIsNone(result[0].successor)

    def test_duplicate_gap_and_bound_fail_closed(self):
        ingress = OrderedIngress(4)
        ingress.add(0, 0, row(0, 0))
        with self.assertRaises(ValueError):
            ingress.add(0, 0, row(0, 0))
        ingress.add(0, 1, row(0, 2))
        with self.assertRaises(ValueError):
            list(ingress.drain())
        ingress = OrderedIngress(1)
        ingress.add(0, 0, row(0, 0))
        with self.assertRaises(ValueError):
            ingress.add(0, 1, row(0, 1))

    def test_runtime_gate_rejects_dropped_update_debt(self):
        from hpc_runs.intrmotiv_offpolicy.audit_runtime import validate_native_accounting

        gate = dict(
            execution="sample_factory_native",
            accepted=980,
            decisions=1024,
            invalid_final_exclusions=44,
            updates=14,
            update_debt=0,
            transport_emitted=1024,
            transport_received=1056,
            transport_pending=32,
        )
        arguments = {"learning-start": "32", "decisions-per-update": "64"}
        validate_native_accounting(gate, gate, arguments)
        gate["updates"] = 13
        with self.assertRaisesRegex(ValueError, "budget"):
            validate_native_accounting(gate, gate, arguments)
        gate["updates"] = 14
        gate["transport_pending"] = 31
        with self.assertRaisesRegex(ValueError, "transport"):
            validate_native_accounting(gate, gate, arguments)

    def test_update_debt_survives_delayed_learning(self):
        self.assertEqual(updates_due(16384 + 640, 16384, 64, 2), 8)
        self.assertEqual(updates_due(16384 + 640, 16384, 64, 3), 7)
        self.assertEqual(updates_due(32, 16384, 64, 0), 0)


if __name__ == "__main__":
    unittest.main()
