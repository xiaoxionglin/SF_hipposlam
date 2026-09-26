"""Exact comparison must reject small parameter and replay-state changes."""

import copy
import unittest

import numpy as np
import torch

from hpc_runs.intrmotiv_study.checkpoint_reload import assert_exact


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


if __name__ == "__main__":
    unittest.main()
