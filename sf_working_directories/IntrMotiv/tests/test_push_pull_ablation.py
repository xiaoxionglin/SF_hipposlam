"""Focused checks for the push–pull ablation's term boundaries."""

import unittest

import torch

from sf_working_directories.IntrMotiv.dmlab.custom_learner import (
    ablation_distance_weight,
    build_matched_encoder_credit,
    legacy_reward_streams,
    target_reward_magnitude,
    target_success_worker_reward,
)


class PushPullAblationTests(unittest.TestCase):
    def test_distance_weights_keep_temporal_default_and_reject_invalid_constant(self):
        distance = torch.tensor([2.0, 5.0])
        torch.testing.assert_close(ablation_distance_weight(distance, "temporal", 0.0), distance)
        torch.testing.assert_close(ablation_distance_weight(distance, "constant", 3.0), torch.tensor([3.0, 3.0]))
        torch.testing.assert_close(ablation_distance_weight(distance, "none", 0.0), torch.zeros_like(distance))
        with self.assertRaises(ValueError):
            ablation_distance_weight(distance, "constant", 0.0)

    def test_legacy_encoder_credit_changes_weight_only(self):
        internal = torch.tensor([[7.0, 2.0, 5.0, 7.0]])
        _, temporal = legacy_reward_streams(internal, 7.0, 0.1, "encourage")
        _, constant = legacy_reward_streams(internal, 7.0, 0.1, "encourage", "constant", 3.0)
        _, removed = legacy_reward_streams(internal, 7.0, 0.1, "encourage", "none")
        torch.testing.assert_close(temporal, torch.tensor([[0.2, 0.5]]))
        torch.testing.assert_close(constant, torch.tensor([[0.3, 0.3]]))
        torch.testing.assert_close(removed, torch.zeros_like(temporal))

    def test_matched_credit_keeps_the_event_mask(self):
        progression = torch.tensor([[[0, 7], [1, 7], [2, 0]]])
        candidate = torch.tensor([[[True, False], [False, False], [False, True]]])
        dominant = candidate.clone()
        valids = torch.ones((1, 3), dtype=torch.bool)
        outputs = [
            build_matched_encoder_credit(
                progression, candidate, dominant, valids, 7, 0.1, "arrival", mode, constant
            )
            for mode, constant in (("temporal", 0.0), ("constant", 3.0), ("none", 0.0))
        ]
        self.assertTrue(all(torch.equal(mask, outputs[0][1]) for _, mask, _ in outputs))
        self.assertTrue(all(output[2]["credited"] == outputs[0][2]["credited"] for output in outputs))
        self.assertAlmostEqual(outputs[0][0].sum().item(), 0.2)
        self.assertAlmostEqual(outputs[1][0].sum().item(), 0.3)
        self.assertAlmostEqual(outputs[2][0].sum().item(), 0.0)

    def test_worker_constant_and_none_preserve_hit_reward(self):
        internal = torch.tensor([[7.0, 7.0, 2.0, 5.0]])
        hit = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        temporal = target_reward_magnitude(internal, 7.0, 0.1, "hit_distance", 1.0, 0.1)
        constant = target_reward_magnitude(internal, 7.0, 0.1, "hit_distance", 1.0, 0.1, "constant", 3.0)
        removed = target_reward_magnitude(internal, 7.0, 0.1, "hit_distance", 1.0, 0.1, "none")
        torch.testing.assert_close(temporal, torch.tensor([[1.05, 1.02]]))
        torch.testing.assert_close(constant, torch.tensor([[1.03, 1.03]]))
        torch.testing.assert_close(removed, torch.ones_like(removed))
        actual = target_success_worker_reward(
            internal, hit, 7.0, 0.1, "hit_distance", 1.0, 0.1,
            bonus_mode="constant", bonus_constant=3.0,
        )
        torch.testing.assert_close(actual, torch.tensor([[1.03, 1.03]]))


if __name__ == "__main__":
    unittest.main()
