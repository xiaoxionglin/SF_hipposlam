import unittest
from pathlib import Path

from hpc_runs.intrmotiv_study import load_study

ROOT = Path(__file__).parent
PRODUCTION = ROOT / "studies" / "fixed_reward_transfer.study.json"
PREFLIGHT = ROOT / "studies" / "fixed_reward_transfer_preflight3.study.json"


class FixedRewardTransferStudyTests(unittest.TestCase):
    def test_production_matrix_and_matched_training_contract(self):
        study = load_study(PRODUCTION)
        runs = study.expand_runs()
        self.assertEqual(len(runs), 21)
        self.assertEqual({run.seed for run in runs}, {42, 1234, 9999})
        self.assertEqual(len({run.condition for run in runs}), 7)
        for run in runs:
            args = set(run.args)
            self.assertIn("--env=openfield_map2_fixed_loc3", args)
            self.assertIn("--train_for_seconds=108000", args)
            self.assertIn("--num_workers=32", args)
            self.assertIn("--num_envs_per_worker=8", args)
            self.assertIn("--worker_num_splits=8", args)
            self.assertIn("--env_frameskip=4", args)
            self.assertIn("--dmlab_reduced_action_set=True", args)
            self.assertIn("--dmlab_navigation_action_set=False", args)
            self.assertIn("--num_policies=1", args)
            self.assertIn("--with_pbt=False", args)
            self.assertIn("--advantage_reward_source=external", args)
            self.assertIn("--extra_encoder_losses=False", args)
            self.assertIn("--reward_scale=1.0", args)
            self.assertIn("--fixed_task_conditioning=True", args)
            self.assertIn("--hrl_controllable_graph=False", args)
            self.assertIn("--hrl_goal_conditioning=target_id_film", args)
            self.assertIn("--encoder_conv_architecture=layer2_resnet18", args)
            self.assertIn("--with_pos_obs=False", args)

    def test_transfer_scopes_and_sources_are_exact(self):
        runs = {run.base: run for run in load_study(PRODUCTION).expand_runs() if run.seed == 42}
        self.assertEqual(
            set(runs),
            {
                "SCRATCH",
                "SCR_DG_FROZEN",
                "SCR_DG_TUNE",
                "SCR_POLICY_TUNE",
                "SAT_DG_FROZEN",
                "SAT_DG_TUNE",
                "SAT_POLICY_TUNE",
            },
        )
        self.assertIn("--transfer_scope=none", runs["SCRATCH"].args)
        self.assertFalse(any(a.startswith("--transfer_model_path=") for a in runs["SCRATCH"].args))
        for name, run in runs.items():
            if name == "SCRATCH":
                continue
            expected_scope = "policy" if "POLICY" in name else "dg"
            self.assertIn(f"--transfer_scope={expected_scope}", run.args)
            checkpoint = next(a for a in run.args if a.startswith("--transfer_model_path="))
            self.assertIn("checkpoint_000004580_75038720.pth", checkpoint)
            self.assertIn("source_credit_retirement" if name.startswith("SCR") else "saturday_batch", checkpoint)
            frozen = "FROZEN" in name
            self.assertIn(f"--transfer_freeze_dg={str(frozen)}", run.args)
            self.assertIn("--ppo_dg_gradient=stop" if frozen else "--ppo_dg_gradient=joint", run.args)

    def test_preflight_is_same_matrix_with_bounded_gate(self):
        production = load_study(PRODUCTION)
        preflight = load_study(PREFLIGHT)
        self.assertEqual(len(preflight.expand_runs()), 7)
        self.assertEqual({run.seed for run in preflight.expand_runs()}, {9999})
        for run in preflight.expand_runs():
            self.assertIn("--train_for_env_steps=2000000", run.args)
            self.assertIn("--train_for_seconds=3600", run.args)
        self.assertEqual(
            {run.base for run in production.expand_runs()},
            {run.base for run in preflight.expand_runs()},
        )


if __name__ == "__main__":
    unittest.main()
