from pathlib import Path
import unittest

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.analysis import linear_contrasts
from hpc_runs.intrmotiv_study.telemetry import (
    CheckpointRecord,
    build_intervention_manifest,
    build_place_field_manifests,
)


SPEC = Path(__file__).with_name("studies") / "source_credit_retirement.study.json"


class SourceCreditRetirementStudyTests(unittest.TestCase):
    def setUp(self):
        self.study = load_study(SPEC)

    def test_complete_cross_has_ten_cells_and_thirty_unique_fresh_runs(self):
        runs = self.study.expand_runs()
        self.assertEqual(self.study.provenance()["study_workflow_version"], "1.4.1")
        self.assertEqual(len(runs), 30)
        self.assertEqual(len({run.name for run in runs}), 30)
        self.assertEqual(len({run.condition for run in runs}), 10)
        self.assertEqual({run.seed for run in runs}, {8, 99, 123})
        self.assertEqual({run.base for run in runs}, {"C15"})
        self.assertEqual({run.factors["encoder_credit"] for run in runs}, {"arrival", "source"})
        self.assertEqual(
            {run.factors["retirement"] for run in runs},
            {"monitor", "dir_silent", "dir_open", "pred_silent", "pred_open"},
        )
        self.assertTrue(all("--train_for_env_steps=75000000" in run.args for run in runs))

    def test_c15_film_credit_retirement_and_deadline_contract(self):
        fixed = {
            "--hrl_goal_conditioning=target_id_film",
            "--hrl_target_timing=immediate",
            "--hrl_manager_mode=frontier_direct",
            "--hrl_timeout_margin_ratio=0.20",
            "--hrl_timeout_margin_steps=2",
            "--dg_recruitment_pred_min_context_attempts=4.0",
            "--dg_recruitment_pred_half_life_options=5000",
            "--dg_recruitment_attempt_threshold=0.5",
            "--dg_recruitment_redundancy_max_steps=4",
            "--encoder_reward_require_local_predecessor=True",
        }
        for run in self.study.expand_runs():
            self.assertTrue(fixed.issubset(set(run.args)))
            flags = [arg.split("=", 1)[0] for arg in run.args]
            self.assertEqual(len(flags), len(set(flags)))
            self.assertIn(f"--encoder_reward_recipient={run.factors['encoder_credit']}", run.args)
            retirement = run.factors["retirement"]
            gate = "open" if retirement.endswith("open") else "silent"
            rule = "monitor" if retirement == "monitor" else retirement.split("_", 1)[0]
            rule = {"dir": "directional", "pred": "predictive"}.get(rule, rule)
            self.assertIn(f"--dg_recruitment_endpoint_gate={gate}", run.args)
            self.assertIn(f"--dg_recruitment_victim_rule={rule}", run.args)
            expected_max = 0 if retirement == "monitor" else 1
            self.assertIn(f"--dg_orthogonal_recruitment_max_per_rollout={expected_max}", run.args)

    def test_all_seventeen_declared_contrasts_are_exact_and_seed_paired(self):
        records = []
        for run in self.study.expand_runs():
            credit = run.factors["encoder_credit"] == "source"
            retirement = run.factors["retirement"]
            score = (
                credit
                + 2 * (retirement == "dir_silent")
                + 4 * (retirement == "dir_open")
                + 8 * (retirement == "pred_silent")
                + 16 * (retirement == "pred_open")
                + 32 * credit * (retirement == "dir_open")
                + 64 * credit * (retirement == "pred_open")
            )
            records.append({"seed": run.seed, **run.factors, "score": score})
        _, summary = linear_contrasts(
            records, ["score"], [], ["seed"], self.study.analysis["contrasts"]
        )
        expected = {
            "SRC_minus_ARR_MON": 1.0,
            "SRC_minus_ARR_DIRS": 1.0,
            "SRC_minus_ARR_DIRO": 33.0,
            "SRC_minus_ARR_PREDS": 1.0,
            "SRC_minus_ARR_PREDO": 65.0,
            "DIRO_minus_DIRS_ARR": 2.0,
            "DIRO_minus_DIRS_SRC": 34.0,
            "PREDO_minus_PREDS_ARR": 8.0,
            "PREDO_minus_PREDS_SRC": 72.0,
            "credit_x_DIR_gate": 32.0,
            "credit_x_PRED_gate": 64.0,
            "DIRO_minus_MON_ARR": 4.0,
            "DIRO_minus_MON_SRC": 36.0,
            "PREDO_minus_MON_ARR": 16.0,
            "PREDO_minus_MON_SRC": 80.0,
            "PREDO_minus_DIRO_ARR": 12.0,
            "PREDO_minus_DIRO_SRC": 44.0,
        }
        self.assertEqual(len(summary), len(expected))
        for row in summary:
            self.assertEqual(row["mean"], expected[row["contrast"]])

    def test_telemetry_declares_seventy_jobs_and_thirty_interventions(self):
        inventory = []
        batch_root = Path(self.study.output_root)
        for run in self.study.expand_runs():
            targets = [5_000_000, 15_000_000, 30_000_000, 50_000_000, 75_000_000]
            if run.seed != 99:
                targets = [75_000_000]
            for target in targets:
                inventory.append(
                    CheckpointRecord(
                        run_name=run.name,
                        target_frames=target,
                        checkpoint_frames=target,
                        checkpoint=batch_root / run.name / f"checkpoint_{target}.pth",
                        run_dir=batch_root / run.name,
                    )
                )
        rows, trajectory = build_place_field_manifests(
            self.study, inventory, require_checkpoint_files=False
        )
        self.assertEqual(len(rows), 70)
        self.assertEqual(len(trajectory), 50)
        self.assertEqual(len(build_intervention_manifest(self.study, rows)), 30)


if __name__ == "__main__":
    unittest.main()
