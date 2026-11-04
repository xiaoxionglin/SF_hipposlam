"""Exact comparison must reject small parameter and replay-state changes."""

import copy
import unittest

import numpy as np
import torch

from hpc_runs.intrmotiv_study.checkpoint_reload import assert_exact, assert_optimizer_restore


class ReloadEqualityTests(unittest.TestCase):
    def test_nested_checkpoint_and_mutations(self):
        saved = {"weight": torch.tensor([1.0]), "replay": [{"id": 3, "pose": np.array([2, 4])}]}
        assert_exact(saved, copy.deepcopy(saved))
        for key in ("weight", "id", "pose"):
            changed = copy.deepcopy(saved)
            if key == "weight":
                changed[key][0] += 1e-6
            else:
                changed["replay"][0][key] += 1
            with self.assertRaises(AssertionError):
                assert_exact(saved, changed)

    def optimizer(self, lr=0.000196875):
        return {
            "state": {
                0: {"step": torch.tensor(132.0), "exp_avg": torch.tensor([0.1]), "exp_avg_sq": torch.tensor([0.2])}
            },
            "param_groups": [
                {"params": [0], "lr": lr, "intrmotiv_lr_scale": 1.0, "weight_decay": 0, "betas": (0.9, 0.999)}
            ],
        }

    def test_scheduled_lr_transition_preserves_all_optimizer_state(self):
        saved = self.optimizer()
        restored = copy.deepcopy(saved)
        restored["param_groups"][0]["lr"] = 0.0002
        result = assert_optimizer_restore(saved, restored, 0.0002, 0.0002)
        self.assertFalse(result["optimizer_exact"])
        self.assertTrue(result["optimizer_state_exact"])
        self.assertTrue(result["optimizer_lr_reset_verified"])
        self.assertTrue(result["optimizer_lr_reset_applied"])
        self.assertEqual(saved["param_groups"][0]["lr"], 0.000196875)

    def test_unchanged_lr_reports_raw_exactness(self):
        saved = self.optimizer(0.0002)
        result = assert_optimizer_restore(saved, copy.deepcopy(saved), 0.0002, 0.0002)
        self.assertTrue(result["optimizer_exact"])
        self.assertFalse(result["optimizer_lr_reset_applied"])

    def test_group_scale_is_applied_exactly(self):
        saved = self.optimizer()
        saved["param_groups"].append({"params": [1], "lr": 0.0000984375, "intrmotiv_lr_scale": 0.5})
        restored = copy.deepcopy(saved)
        restored["param_groups"][0]["lr"] = 0.0002
        restored["param_groups"][1]["lr"] = 0.0001
        self.assertTrue(assert_optimizer_restore(saved, restored, 0.0002, 0.0002)["optimizer_state_exact"])

    def test_first_group_guard_preserves_other_groups_when_no_reset_occurs(self):
        saved = self.optimizer(0.0002)
        saved["param_groups"].append({"params": [1], "lr": 0.00009, "intrmotiv_lr_scale": 0.5})
        result = assert_optimizer_restore(saved, copy.deepcopy(saved), 0.0002, 0.0002)
        self.assertTrue(result["optimizer_exact"])
        changed = copy.deepcopy(saved)
        changed["param_groups"][1]["lr"] = 0.0001
        with self.assertRaises(AssertionError):
            assert_optimizer_restore(saved, changed, 0.0002, 0.0002)

    def test_unexpected_state_or_runtime_transition_is_rejected(self):
        for mutation in ("moment", "step", "decay", "lr", "scale", "curr_lr"):
            with self.subTest(mutation=mutation):
                saved = self.optimizer()
                restored = copy.deepcopy(saved)
                restored["param_groups"][0]["lr"] = 0.0002
                restored_curr_lr = 0.0002
                if mutation == "moment":
                    restored["state"][0]["exp_avg"][0] += 1e-6
                elif mutation == "step":
                    restored["state"][0]["step"] += 1
                elif mutation == "decay":
                    restored["param_groups"][0]["weight_decay"] = 1e-12
                elif mutation == "lr":
                    restored["param_groups"][0]["lr"] += 1e-12
                elif mutation == "scale":
                    restored["param_groups"][0]["intrmotiv_lr_scale"] += 1e-12
                else:
                    restored_curr_lr += 1e-12
                with self.assertRaises(AssertionError):
                    assert_optimizer_restore(saved, restored, 0.0002, restored_curr_lr)


if __name__ == "__main__":
    unittest.main()
