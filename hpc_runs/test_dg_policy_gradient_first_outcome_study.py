from pathlib import Path
import unittest

from hpc_runs.intrmotiv_study import load_study


STUDIES = Path(__file__).with_name("studies")


class DGPolicyGradientFirstOutcomeStudyTests(unittest.TestCase):
    def setUp(self):
        self.production = load_study(STUDIES / "dg_policy_gradient_first_outcome.study.json")
        self.preflight = load_study(
            STUDIES / "dg_policy_gradient_first_outcome_preflight.study.json"
        )

    def test_complete_crosses(self):
        production = self.production.expand_runs()
        preflight = self.preflight.expand_runs()
        self.assertEqual(len(production), 24)
        self.assertEqual(len(preflight), 8)
        self.assertEqual(len({run.name for run in production}), 24)
        self.assertEqual({run.seed for run in production}, {8, 99, 123})
        self.assertEqual({run.seed for run in preflight}, {99})
        for runs in (production, preflight):
            self.assertEqual({run.base for run in runs}, {"C15"})
            self.assertEqual(
                {run.factors["worker_outcome"] for run in runs},
                {"target_hit", "first_distinct"},
            )
            self.assertEqual(
                {run.factors["ppo_dg_gradient"] for run in runs}, {"stop", "joint"}
            )
            self.assertEqual(
                {run.factors["goal_conditioning"] for run in runs},
                {"legacy", "target_id_film"},
            )

    def test_fixed_contract_and_unique_flags(self):
        fixed = {
            "--encoder_reward_recipient=arrival",
            "--encoder_reward_require_local_predecessor=True",
            "--dg_batchnorm_semantics=legacy_batch",
            "--hrl_target_timing=immediate",
            "--dg_recruitment_victim_rule=monitor",
            "--dg_orthogonal_recruitment_max_per_rollout=0",
            "--iterative_update=False",
        }
        for study in (self.production, self.preflight):
            for run in study.expand_runs():
                self.assertTrue(fixed.issubset(set(run.args)))
                flags = [arg.split("=", 1)[0] for arg in run.args]
                self.assertEqual(len(flags), len(set(flags)), run.name)
                self.assertIn(
                    f"--hrl_control_outcome={run.factors['worker_outcome']}", run.args
                )
                self.assertIn(
                    f"--ppo_dg_gradient={run.factors['ppo_dg_gradient']}", run.args
                )
                self.assertIn(
                    f"--hrl_goal_conditioning={run.factors['goal_conditioning']}", run.args
                )
        self.assertTrue(
            all(
                "--hrl_direct_target_selection=local_successor" in run.args
                for run in self.production.expand_runs()
            )
        )
        self.assertTrue(
            all(
                "--hrl_direct_target_selection=least_tested" in run.args
                for run in self.preflight.expand_runs()
            )
        )

    def test_steps_names_metrics_contrasts_and_telemetry(self):
        self.assertTrue(
            all("--train_for_env_steps=5000000" in r.args for r in self.preflight.expand_runs())
        )
        self.assertTrue(
            all("--train_for_env_steps=75000000" in r.args for r in self.production.expand_runs())
        )
        self.assertTrue(all(r.name.startswith("DGPF_C15_") for r in self.preflight.expand_runs()))
        self.assertTrue(all(r.name.startswith("DGP_C15_") for r in self.production.expand_runs()))
        metrics = self.production.analysis["window_metrics"]
        for metric in (
            "control_correct_count", "control_wrong_count", "control_timeout_count",
            "control_command_entropy", "control_observed_pair_coverage",
            "ppo_to_dg_gradient_norm", "encoder_to_dg_gradient_norm",
            "ppo_encoder_dg_gradient_cosine", "ppo_encoder_dg_row_conflict_fraction",
            "local_candidate_pair_count", "local_candidate_source_fraction",
            "local_candidate_count_mean", "behavior_candidate_count_mean",
        ):
            self.assertIn(metric, metrics)
        self.assertEqual(self.production.analysis["synchronized_steps"], [5_000_000, 25_000_000, 50_000_000, 75_000_000])
        self.assertEqual(len(self.production.analysis["contrasts"]), 15)
        self.assertEqual(self.production.telemetry["target_frames"], [5_000_000, 25_000_000, 50_000_000, 75_000_000])
        self.assertEqual(self.production.telemetry["terminal_seeds"], [8, 123])
        self.assertTrue(self.production.telemetry["intervention"]["terminate_on_first_distinct_exclusive_outcome"])
        self.assertTrue(self.production.telemetry["intervention"]["balanced_local_successor_targets"])

    def test_workspace_paths_and_provenance(self):
        for study in (self.production, self.preflight):
            self.assertEqual(study.provenance()["study_workflow_version"], "1.4.1")
            self.assertTrue(study.output_root.startswith("/work/classic/fr_xl1014-train/"))


if __name__ == "__main__":
    unittest.main()
