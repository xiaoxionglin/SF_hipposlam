from pathlib import Path
import unittest

from hpc_runs.intrmotiv_study import load_study


STUDIES = Path(__file__).with_name("studies")


class CA3FeedbackPredictiveDGStudyTests(unittest.TestCase):
    def setUp(self):
        self.production = load_study(STUDIES / "ca3_feedback_predictive_dg.study.json")
        self.preflight = load_study(
            STUDIES / "ca3_feedback_predictive_dg_preflight.study.json"
        )

    def test_complete_structural_matrix(self):
        production = self.production.expand_runs()
        preflight = self.preflight.expand_runs()
        self.assertEqual(len(production), 81)
        self.assertEqual(len(preflight), 27)
        self.assertEqual(len({run.condition for run in production}), 27)
        self.assertEqual({run.seed for run in production}, {8, 99, 123})
        self.assertEqual({run.seed for run in preflight}, {99})
        self.assertEqual(len({run.name for run in production}), 81)

    def test_fixed_minimal_baseline(self):
        fixed = {
            "--encoder_reward_recipient=arrival",
            "--encoder_reward_require_local_predecessor=True",
            "--dg_batchnorm_semantics=legacy_batch",
            "--hrl_control_outcome=first_distinct",
            "--ppo_dg_gradient=joint",
            "--hrl_goal_conditioning=target_id_film",
            "--hrl_direct_target_selection=least_tested",
            "--dg_orthogonal_recruitment=False",
            "--dg_orthogonal_recruitment_max_per_rollout=0",
            "--iterative_update=False",
        }
        for study in (self.production, self.preflight):
            for run in study.expand_runs():
                self.assertTrue(fixed.issubset(set(run.args)), run.name)
                flags = [arg.split("=", 1)[0] for arg in run.args]
                self.assertEqual(len(flags), len(set(flags)), run.name)

    def test_cells_have_expected_factorization(self):
        runs = self.preflight.expand_runs()
        cells = {run.factors["cell"]: run for run in runs}
        self.assertEqual(len(cells), 27)
        self.assertIn("base", cells)
        self.assertIn("predictor_passive", cells)
        self.assertIn("predictor_goal", cells)
        feedback = [r for r in runs if r.metadata["cell_feedback"] != "none"]
        self.assertEqual(len(feedback), 24)
        self.assertEqual({r.metadata["cell_feedback"] for r in feedback}, {"gate", "additive"})
        self.assertEqual({r.metadata["cell_history"] for r in feedback}, {"ca3", "ca3_action"})
        self.assertEqual({r.metadata["cell_gradient"] for r in feedback}, {"direct", "bptt"})
        self.assertEqual({r.metadata["cell_predictor"] for r in feedback}, {"none", "passive", "goal"})

    def test_metrics_contrasts_and_workspace_contract(self):
        metrics = self.production.analysis["window_metrics"]
        for metric in (
            "context_created_fraction",
            "context_suppressed_fraction",
            "context_modulation_abs_mean",
            "context_adapter_gradient_norm",
            "prediction_validation_state_gain",
            "prediction_replay_match",
            "target_action_sensitivity",
        ):
            self.assertIn(metric, metrics)
        self.assertEqual(len(self.production.analysis["contrasts"]), 54)
        self.assertEqual(
            self.production.analysis["synchronized_steps"],
            [5_000_000, 25_000_000, 50_000_000, 75_000_000],
        )
        for study in (self.production, self.preflight):
            self.assertEqual(study.provenance()["study_workflow_version"], "1.4.1")
            self.assertTrue(study.output_root.startswith("/work/classic/fr_xl1014-train/"))


if __name__ == "__main__":
    unittest.main()
