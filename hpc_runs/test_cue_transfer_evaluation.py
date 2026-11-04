"""Keep nearest-checkpoint fallback from becoming a false matched-age result."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.telemetry import CheckpointRecord, build_place_field_manifests
from hpc_runs.post_cue_transfer_20261104 import audit_manifest, heldout_settings


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

    def test_protocol_controls_preserve_default_and_reject_invalid_panels(self):
        defaults = heldout_settings()
        self.assertEqual(defaults["protocols"], ["correct"])
        interventions = heldout_settings("shuffled_cue,zero_ca3,zero_depth", 30, 71000, 900)
        self.assertTrue(interventions["cue_shuffle"])
        self.assertEqual(interventions["trials_per_run"], 30)
        self.assertEqual(interventions["seed_base"], 71000)
        self.assertEqual(interventions["horizon_decisions"], 900)
        for protocols in ("", "correct,correct", "other", "correct,"):
            with self.subTest(protocols=protocols), self.assertRaises(ValueError):
                heldout_settings(protocols)
        for kwargs in ({"trials": 0}, {"seed_base": -1}, {"horizon": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                heldout_settings(**kwargs)

    def test_manifest_counts_come_from_selected_study(self):
        study = load_study(Path(__file__).with_name("studies") / "cue_transfer_qualification_20261104.study.json")
        inventory = []
        for run in study.expand_runs():
            run_dir = Path(study.output_root) / run.name
            for target in study.telemetry["target_frames"]:
                inventory.append(CheckpointRecord(
                    run.name, target, target,
                    run_dir / "checkpoint_p0/milestones" / f"checkpoint_000001_{target}.pth", run_dir,
                ))
        rows, _ = build_place_field_manifests(study, inventory, require_checkpoint_files=False)
        audit = audit_manifest(study, rows)
        self.assertEqual((audit["field_rows"], audit["terminal_rows"]), (8, 4))

    def test_runner_recovers_persisted_protocol_and_rejects_changed_manifest(self):
        with tempfile.TemporaryDirectory(prefix="cue-heldout-runner-") as temporary:
            workspace = Path(temporary)
            source, run_dir, runfiles = (workspace / name for name in ("source", "run", "runfiles"))
            for path in (source, run_dir, runfiles):
                path.mkdir()
            checkpoint = run_dir / "checkpoint.pth"
            checkpoint.touch()
            (run_dir / "config.json").write_text(json.dumps({"dmlab_runfiles_path": str(runfiles)}))
            row = dict(self.rows[0], checkpoint=str(checkpoint), run_dir=str(run_dir), label_suffix="example")
            manifest = workspace / "terminal_heldout_manifest.tsv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
                writer.writeheader()
                writer.writerow(row)
            settings = heldout_settings("shuffled_cue,zero_ca3,zero_depth", 2, 71000, 7)
            (workspace / "evaluation_audit.json").write_text(json.dumps({
                "heldout": settings,
                "terminal_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            }))
            # Delegate the real manifest reader to Python, but capture evaluator
            # invocations instead of starting a native environment in this test.
            python = workspace / "capture_python"
            python.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "if sys.argv[1] == '-':\n"
                "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
                "with open(os.environ['TEST_INVOCATION_LOG'], 'a') as stream:\n"
                "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            )
            python.chmod(0o755)
            invocation_log = workspace / "calls.jsonl"
            env = dict(os.environ, INTRMOTIV_WORKSPACE_ROOT=str(workspace),
                       INTRMOTIV_RUNTIME_SOURCE=str(source), INTRMOTIV_EVAL_PYTHON=str(python),
                       INTRMOTIV_HELDOUT_PROTOCOLS="correct", TEST_INVOCATION_LOG=str(invocation_log))
            command = ["bash", str(Path(__file__).with_name("run_cue_transfer_heldout_single.sh")),
                       str(manifest), "0", str(workspace / "output")]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in invocation_log.read_text().splitlines()]
            self.assertEqual([call[call.index("--protocol") + 1] for call in calls], settings["protocols"])
            for call in calls:
                for flag, expected in (("--trials", "2"), ("--horizon", "7"), ("--seed-base", "71000")):
                    self.assertEqual(call[call.index(flag) + 1], expected)
            manifest.write_text(manifest.read_text() + "\n")
            failed = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("differs from its audited protocol", failed.stderr)
            self.assertEqual(len(invocation_log.read_text().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
