"""Canonical preflight matrix and preservation checks for controller migration."""

import unittest
from pathlib import Path

from hpc_runs.intrmotiv_study import load_study


class FullSystemControllerStudy(unittest.TestCase):
    def test_six_bounded_from_scratch_preflights(self):
        study = load_study(Path(__file__).with_name("studies") / "full_system_controller_preflight.study.json")
        runs = study.expand_runs()
        self.assertEqual(len(runs), 6)
        for run in runs:
            args = dict(item.removeprefix("--").split("=", 1) for item in run.args)
            self.assertEqual(args["seed"], "99")
            self.assertEqual(args["decorrelate_envs_on_one_worker"], "False")
            self.assertEqual(args["train_for_env_steps"], "2000000")
            self.assertEqual(args["controller_preflight"], "True")
            self.assertEqual(args["controller_td_positions"], "256")
            self.assertEqual(args["controller_decisions_per_update"], "64")
            self.assertEqual(args["controller_her_positions"], "256")
            self.assertEqual(args["encoder_conv_architecture"], "layer2_resnet18")
            self.assertEqual(args["depth_sensor"], "True")
            self.assertEqual(args["with_wandb"], "True")
            self.assertEqual(args["online_spatial_telemetry"], "True")
            self.assertFalse(any("transfer_checkpoint" in k and v not in ("None", "") for k, v in args.items()))
            self.assertNotEqual(args.get("transfer_freeze_dg"), "True")
            self.assertNotIn("controller_her_fraction", args)
            self.assertEqual(
                (args["Hippo_n_feature"], args["dg_goal_input"]),
                ("16", "none") if run.base == "DIRECT_F16" else ("64", "write"),
            )


if __name__ == "__main__":
    unittest.main()
