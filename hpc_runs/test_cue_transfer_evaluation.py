"""Keep nearest-checkpoint fallback from becoming a false matched-age result."""

from __future__ import annotations

import copy
from pathlib import Path
import unittest

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.telemetry import CheckpointRecord, build_place_field_manifests
from hpc_runs.post_cue_transfer_20261104 import audit_manifest


class CueTransferEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.study = load_study(
            Path(__file__).with_name("studies") / "cue_transfer_20261104.study.json"
        )
        # Real checkpoints cross requested targets at learner update boundaries.
        # The evaluator must retain that actual age instead of rewriting it.
        self.ages = {
            target: ((target + 8191) // 8192) * 8192
            for target in self.study.telemetry["target_frames"]
        }
        inventory = []
        for run in self.study.expand_runs():
            run_dir = Path(self.study.output_root) / run.name
            for target, actual in self.ages.items():
                inventory.append(CheckpointRecord(
                    run.name, target, actual,
                    run_dir / "checkpoint_p0/milestones" / f"checkpoint_000001_{actual}.pth",
                    run_dir,
                ))
        self.rows, _ = build_place_field_manifests(
            self.study, inventory, require_checkpoint_files=False
        )

    def test_complete_canonical_panel_preserves_actual_ages(self):
        audit = audit_manifest(self.study, self.rows)
        self.assertEqual((audit["field_rows"], audit["terminal_rows"]), (28, 12))
        self.assertEqual(
            audit["actual_frames_by_requested_target"],
            {str(target): actual for target, actual in self.ages.items()},
        )
        self.assertEqual(audit["heldout"]["trials_per_run"], 100)
        self.assertFalse(audit["heldout"]["cue_shuffle"])

    def test_missing_or_duplicate_condition_seed_target_fails(self):
        for rows in (self.rows[:-1], self.rows + [self.rows[0]]):
            with self.subTest(rows=len(rows)), self.assertRaisesRegex(ValueError, "complete declared"):
                audit_manifest(self.study, rows)

    def test_nearest_earlier_checkpoint_cannot_stand_in_for_terminal(self):
        rows = copy.deepcopy(self.rows)
        terminal = next(row for row in rows if row["target_frames"] == "25000000")
        terminal["checkpoint_frames"] = str(self.ages[15000000])
        with self.assertRaisesRegex(ValueError, "Missing completed milestone"):
            audit_manifest(self.study, rows)

    def test_unequal_saved_ages_across_terminal_seeds_fails(self):
        rows = copy.deepcopy(self.rows)
        terminal = next(row for row in rows if row["seed"] == "1234")
        terminal["checkpoint_frames"] = str(self.ages[25000000] + 8192)
        with self.assertRaisesRegex(ValueError, "shared actual checkpoint ages"):
            audit_manifest(self.study, rows)

    def test_one_later_checkpoint_cannot_cover_two_milestones(self):
        rows = copy.deepcopy(self.rows)
        first = next(row for row in rows if row["target_frames"] == "1000000")
        later = next(
            row for row in rows
            if row["condition"] == first["condition"]
            and row["seed"] == first["seed"]
            and row["target_frames"] == "5000000"
        )
        first["checkpoint"] = later["checkpoint"]
        first["checkpoint_frames"] = later["checkpoint_frames"]
        with self.assertRaisesRegex(ValueError, "reused for different requested milestones"):
            audit_manifest(self.study, rows)


if __name__ == "__main__":
    unittest.main()
