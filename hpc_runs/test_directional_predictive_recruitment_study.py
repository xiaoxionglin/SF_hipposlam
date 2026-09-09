from pathlib import Path
import unittest

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.analysis import linear_contrasts
from hpc_runs.intrmotiv_study.telemetry import build_intervention_manifest


SPEC = Path(__file__).with_name("studies") / "directional_predictive_recruitment.study.json"


class DirectionalPredictiveRecruitmentStudyTests(unittest.TestCase):
    def setUp(self):
        self.study = load_study(SPEC)

    def test_complete_cross_has_18_cells_and_54_unique_fresh_runs(self):
        runs = self.study.expand_runs()
        self.assertEqual(self.study.training_mode, "sample_factory")
        self.assertEqual(len(runs), 54)
        self.assertEqual(len({run.name for run in runs}), 54)
        self.assertEqual(len({run.condition for run in runs}), 18)
        self.assertEqual({run.seed for run in runs}, {8, 99, 123})
        self.assertEqual({run.base for run in runs}, {"C05", "C13", "C15"})
        self.assertEqual({run.factors["recruitment"] for run in runs}, {"monitor", "directional", "predictive"})
        self.assertTrue(all("--train_for_env_steps=75000000" in run.args for run in runs))

    def test_fixed_recruitment_goal_and_deadline_contract(self):
        fixed = {
            "--dg_orthogonal_recruitment=True",
            "--dg_orthogonal_recruitment_mode=graph",
            "--dg_recruitment_connectivity_threshold=0.25",
            "--dg_recruitment_redundancy_max_steps=4",
            "--dg_recruitment_passive_half_life_events=5000",
            "--dg_recruitment_attempt_threshold=0.5",
            "--dg_recruitment_pred_min_context_attempts=2",
            "--hrl_fast_weight_half_life_options=5000",
            "--hrl_target_timing=immediate",
            "--hrl_timeout_margin_ratio=0.20",
            "--hrl_timeout_margin_steps=2",
            "--hrl_bootstrap_horizon=64",
            "--hrl_edge_confidence_threshold=0.5",
            "--hrl_edge_reliability_threshold=0.5",
            "--hrl_behavior_mode_condition=False",
            "--hrl_exploration_policy=shared",
        }
        for run in self.study.expand_runs():
            self.assertTrue(fixed.issubset(set(run.args)))
            flags = [arg.split("=", 1)[0] for arg in run.args]
            self.assertEqual(len(flags), len(set(flags)))
            expected_max = 0 if run.factors["recruitment"] == "monitor" else 1
            self.assertIn(f"--dg_orthogonal_recruitment_max_per_rollout={expected_max}", run.args)
            self.assertIn(f"--dg_recruitment_victim_rule={run.factors['recruitment']}", run.args)
            self.assertIn(f"--hrl_goal_conditioning={run.factors['goal_conditioning']}", run.args)

    def test_base_definitions_are_exact(self):
        for run in self.study.expand_runs():
            args = set(run.args)
            if run.base == "C05":
                self.assertIn("--dg_global_punishment_coeff=0.01", args)
                self.assertIn("--dg_row_repulsion_coeff=1.0", args)
                self.assertIn("--dg_ca3_temporal_exclusion_coeff=0.0", args)
                self.assertIn("--hrl_manager_mode=visit_direct", args)
            elif run.base == "C13":
                self.assertIn("--dg_ca3_temporal_exclusion_coeff=1.0", args)
                self.assertIn("--hrl_manager_exploration_probability=0.10", args)
                self.assertIn("--hrl_manager_mode=visit_direct", args)
            else:
                self.assertIn("--hrl_manager_mode=frontier_direct", args)
                self.assertIn("--hrl_landmark_geometry=none", args)
                self.assertIn("--hrl_action_path_integration=False", args)

    def test_all_declared_contrasts_are_exact_and_seed_paired_within_base(self):
        records = []
        for run in self.study.expand_runs():
            recruitment = run.factors["recruitment"]
            film = run.factors["goal_conditioning"] == "target_id_film"
            directional = recruitment == "directional"
            predictive = recruitment == "predictive"
            score = directional + 2 * predictive + 4 * film + 8 * directional * film + 16 * predictive * film
            records.append({"base": run.base, "seed": run.seed, **run.factors, "score": score})
        _, summary = linear_contrasts(
            records,
            ["score"],
            ["base"],
            ["seed"],
            self.study.analysis["contrasts"],
        )
        expected = {
            "DIR_minus_MON_LEG": 1.0,
            "DIR_minus_MON_FILM": 9.0,
            "PRED_minus_MON_LEG": 2.0,
            "PRED_minus_MON_FILM": 18.0,
            "DIR_minus_PRED_LEG": -1.0,
            "DIR_minus_PRED_FILM": -9.0,
            "FILM_minus_LEG_MON": 4.0,
            "FILM_minus_LEG_DIR": 12.0,
            "FILM_minus_LEG_PRED": 20.0,
            "DIR_x_goal": 8.0,
            "PRED_x_goal": 16.0,
        }
        self.assertEqual(len(summary), len(expected) * 3)
        for row in summary:
            self.assertEqual(row["mean"], expected[row["contrast"]])

    def test_intervention_manifest_selects_every_terminal_run(self):
        rows = []
        for run in self.study.expand_runs():
            targets = [5_000_000, 25_000_000, 50_000_000, 75_000_000] if run.seed == 99 else [75_000_000]
            for target in targets:
                rows.append(
                    {
                        "condition": run.condition,
                        "seed": str(run.seed),
                        "target_frames": str(target),
                        "checkpoint": f"/work/checkpoint_{target}.pth",
                    }
                )
        selected = build_intervention_manifest(self.study, rows)
        self.assertEqual(len(selected), 54)
        self.assertEqual({row["target_frames"] for row in selected}, {"75000000"})


if __name__ == "__main__":
    unittest.main()
