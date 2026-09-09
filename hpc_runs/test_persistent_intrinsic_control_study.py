from pathlib import Path
import unittest

from hpc_runs.intrmotiv_study import load_study


ROOT = Path(__file__).resolve().parent


class PersistentIntrinsicControlStudyTests(unittest.TestCase):
    def test_main_matrix_and_scientific_boundaries(self):
        study = load_study(ROOT / "studies/persistent_intrinsic_control.study.json")
        runs = study.expand_runs()
        self.assertEqual(len(runs), 33)
        self.assertEqual({run.seed for run in runs}, {8, 99, 123})
        self.assertEqual(sum(run.metadata["family"] != "warm_reference" for run in runs), 27)
        forbidden = ("--with_pos_obs=True", "--hrl_landmark_geometry=se2")
        self.assertFalse(any(arg in forbidden for run in runs for arg in run.args))
        goal_runs = [run for run in runs if run.metadata["control"] == "goal"]
        self.assertEqual(len(goal_runs), 21)
        for run in goal_runs:
            self.assertIn("--intrinsic_goal_horizon=900", run.args)
            self.assertIn("--intrinsic_goal_reward_max=6.4", run.args)
            self.assertIn("--hrl_goal_conditioning=target_id_additive", run.args)
        references = [run for run in runs if run.metadata["family"] == "warm_reference"]
        self.assertEqual(len(references), 6)
        self.assertEqual({run.metadata["gradient"] for run in references}, {"stop", "joint"})
        self.assertTrue(all(any(arg.startswith("--intrinsic_goal_reference_checkpoint=") for arg in run.args) for run in references))

    def test_c15_continuations_are_exact_three_seed_supplements(self):
        study = load_study(ROOT / "studies/persistent_intrinsic_control_c15.study.json")
        runs = study.expand_runs()
        self.assertEqual(len(runs), 3)
        self.assertEqual(study.training_mode, "supplemental_args")
        for run in runs:
            self.assertIn("--train_for_env_steps=600000000", run.args)
            checkpoint = next(arg for arg in run.args if arg.startswith("--load_model_path="))
            self.assertIn(f"_S{run.seed}", checkpoint)
            self.assertIn("100040704.pth", checkpoint)


if __name__ == "__main__":
    unittest.main()
