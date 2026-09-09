from pathlib import Path
import unittest

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.analysis import linear_contrasts
from hpc_runs.intrmotiv_study.telemetry import build_intervention_manifest


SPEC = Path(__file__).with_name("studies") / "controllability_edge_exploration.study.json"


class ControllabilityStudyTests(unittest.TestCase):
    def setUp(self):
        self.study = load_study(SPEC)

    def test_complete_cross_has_16_cells_and_48_unique_fresh_runs(self):
        runs = self.study.expand_runs()
        self.assertEqual(self.study.training_mode, "sample_factory")
        self.assertEqual(len(runs), 48)
        self.assertEqual(len({run.name for run in runs}), 48)
        self.assertEqual(len({run.condition for run in runs}), 16)
        self.assertEqual({run.seed for run in runs}, {8, 99, 123})
        self.assertTrue(all("--train_for_env_steps=75000000" in run.args for run in runs))

    def test_fixed_settings_and_factor_flags_are_present_once(self):
        fixed = {
            "--hrl_target_timing=immediate",
            "--hrl_manager_mode=control_graph",
            "--hrl_goal_conditioning=target_trace",
            "--hrl_behavior_mode_condition=True",
            "--hrl_timeout_margin_ratio=0.20",
            "--hrl_timeout_margin_steps=0",
            "--hrl_bootstrap_horizon=64",
            "--hrl_exploration_horizon=64",
            "--hrl_worker_reward_mode=hit",
            "--hrl_distance_bonus_coeff=0.0",
            "--hrl_empirical_her=False",
            "--dg_path_scatter_coeff=0.0",
            "--dg_recruitment_redundancy_max_steps=4",
            "--dg_recruitment_passive_half_life_events=5000",
            "--hrl_fast_weight_half_life_options=5000",
            "--dg_recruitment_connectivity_threshold=0.25",
        }
        for run in self.study.expand_runs():
            self.assertTrue(fixed.issubset(set(run.args)))
            flags = [arg.split("=", 1)[0] for arg in run.args]
            self.assertEqual(len(flags), len(set(flags)))

    def test_all_declared_contrasts_are_exact_and_seed_paired(self):
        records = []
        for run in self.study.expand_runs():
            edge = run.factors["manager_objective"] == "edge_ucb"
            head = run.factors["exploration_policy"] == "separate"
            geometry = run.factors["geometry"] == "se2"
            temporal = run.factors["representation"] == "X1"
            score = temporal + 2 * head + 4 * edge + 8 * geometry
            score += 16 * edge * head + 32 * edge * geometry + 64 * edge * temporal
            records.append({"seed": run.seed, **run.factors, "score": score})
        _, summary = linear_contrasts(
            records,
            ["score"],
            [],
            ["seed"],
            self.study.analysis["contrasts"],
        )
        observed = {row["contrast"]: row["mean"] for row in summary}
        self.assertEqual(observed["representation_X1_minus_X0"], 33.0)
        self.assertEqual(observed["separate_minus_shared"], 10.0)
        self.assertEqual(observed["edge_ucb_minus_node"], 60.0)
        self.assertEqual(observed["se2_minus_none"], 24.0)
        self.assertEqual(observed["edge_x_head"], 16.0)
        self.assertEqual(observed["edge_x_geometry"], 32.0)
        self.assertEqual(observed["edge_x_temporal_exclusion"], 64.0)

    def test_intervention_manifest_selects_terminal_checkpoint_for_every_run(self):
        rows = []
        for run in self.study.expand_runs():
            targets = [5_000_000, 25_000_000, 50_000_000, 75_000_000]
            if run.seed != 99:
                targets = [75_000_000]
            for target in targets:
                rows.append({
                    "condition": run.condition,
                    "seed": str(run.seed),
                    "target_frames": str(target),
                    "checkpoint": f"/work/checkpoint_{target}.pth",
                })
        selected = build_intervention_manifest(self.study, rows)
        self.assertEqual(len(selected), 48)
        self.assertEqual({row["target_frames"] for row in selected}, {"75000000"})
        self.assertEqual(
            {(row["condition"], row["seed"]) for row in selected},
            {(run.condition, str(run.seed)) for run in self.study.expand_runs()},
        )


if __name__ == "__main__":
    unittest.main()
