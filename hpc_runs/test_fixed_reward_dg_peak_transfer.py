"""Study and resource-shard checks for the fixed-reward transfer campaign."""

from pathlib import Path
import unittest

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description


SPEC = Path(__file__).with_name("studies") / "fixed_reward_dg_peak_transfer_20260924.study.json"


class FixedRewardDGPeakTransferTests(unittest.TestCase):
    def test_matrix_and_matched_reward_task(self):
        study = load_study(SPEC)
        runs = study.expand_runs()
        self.assertEqual(len(runs), 42)
        self.assertEqual({r.seed for r in runs}, {42, 1234, 9999})
        self.assertEqual({r.metadata["site"] for r in runs}, {"dg50", "dg51"})
        self.assertEqual(len({r.condition for r in runs}), 14)
        for run in runs:
            args = set(run.args)
            self.assertIn("--train_for_env_steps=100000000", args)
            self.assertIn("--env_frameskip=4", args)
            self.assertIn("--dmlab_navigation_action_set=True", args)
            self.assertIn("--advantage_reward_source=external", args)
            self.assertIn("--use_internal=False", args)
            self.assertIn("--with_pos_obs=False", args)
            self.assertIn("--encoder_conv_architecture=layer2_resnet18", args)
            self.assertTrue(all("/work/classic/fr_xl1014-train" not in a for a in args))
            device = "gpu" if run.metadata["site"] == "dg51" else "cpu"
            self.assertIn(f"--device={device}", args)

    def test_resource_shards_partition_one_study(self):
        study = load_study(SPEC)
        cpu = build_run_description(
            study, run_filter=lambda run: run.metadata["site"] == "dg50", batch_name="cpu"
        )
        gpu = build_run_description(
            study, run_filter=lambda run: run.metadata["site"] == "dg51", batch_name="gpu"
        )
        self.assertEqual(len(cpu.experiments), 21)
        self.assertEqual(len(gpu.experiments), 21)
        self.assertEqual(
            {experiment.base_name for experiment in cpu.experiments + gpu.experiments},
            {run.name for run in study.expand_runs()},
        )


if __name__ == "__main__":
    unittest.main()
