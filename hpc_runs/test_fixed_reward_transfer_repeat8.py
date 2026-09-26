import unittest
from pathlib import Path

from hpc_runs.audit_fixed_reward_transfer_preflight import declared_frameskip, source_interface_errors
from hpc_runs.intrmotiv_study import load_study

ROOT = Path(__file__).parent / "studies"


class Repeat8TransferTests(unittest.TestCase):
    def test_only_timing_and_namespace_change_in_production(self):
        old = load_study(ROOT / "fixed_reward_transfer.study.json")
        new = load_study(ROOT / "fixed_reward_transfer_repeat8.study.json")
        self.assertEqual(len(new.expand_runs()), 21)
        ignored = ("--env_frameskip=", "--wandb_group=")
        for before, after in zip(old.expand_runs(), new.expand_runs()):
            self.assertEqual((before.base, before.seed), (after.base, after.seed))
            self.assertEqual(
                [a for a in before.args if not a.startswith(ignored)],
                [a for a in after.args if not a.startswith(ignored)],
            )
            self.assertEqual(declared_frameskip(after), 8)
        self.assertNotEqual(old.output_root, new.output_root)
        self.assertTrue(new.study_metadata["require_source_interface_match"])

    def test_preflight_preserves_all_conditions_and_initial_checkpoints(self):
        study = load_study(ROOT / "fixed_reward_transfer_repeat8_preflight.study.json")
        self.assertEqual(len(study.expand_runs()), 7)
        for run in study.expand_runs():
            self.assertEqual(declared_frameskip(run), 8)
            self.assertIn("--keep_checkpoints=100", run.args)
            self.assertIn("--train_for_env_steps=2000000", run.args)
            self.assertEqual(run.seed, 9999)

    def test_interface_audit_rejects_timing_and_action_mismatches(self):
        config = dict(
            env_frameskip=8,
            core_name="BypassSS",
            Hippo_n_feature=16,
            Hippo_R=8,
            Hippo_L=64,
            encoder_conv_architecture="layer2_resnet18",
            depth_sensor=True,
            normalize_input=False,
            hrl_goal_conditioning="target_id_film",
            dmlab_reduced_action_set=True,
            dmlab_extended_action_set=False,
        )
        self.assertEqual(source_interface_errors(config, {**config, "dmlab_navigation_action_set": False}), [])
        self.assertTrue(source_interface_errors(config, {**config, "env_frameskip": 4}))
        self.assertTrue(source_interface_errors(config, {**config, "dmlab_navigation_action_set": True}))
        self.assertTrue(source_interface_errors(config, {**config, "Hippo_L": 32}))


if __name__ == "__main__":
    unittest.main()
