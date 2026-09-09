from pathlib import Path
import unittest

from hpc_runs.intrmotiv_study import load_study


SPEC = Path(__file__).with_name("studies") / "encoder_decoder_update_contract_preflight.study.json"


class EncoderDecoderUpdateContractStudyTests(unittest.TestCase):
    def setUp(self):
        self.study = load_study(SPEC)

    def test_complete_ten_run_seed_99_cross_at_75m(self):
        runs = self.study.expand_runs()
        self.assertEqual(self.study.provenance()["study_workflow_version"], "1.4.1")
        self.assertEqual(len(runs), 10)
        self.assertEqual(len({run.name for run in runs}), 10)
        self.assertEqual({run.seed for run in runs}, {99})
        self.assertEqual({run.base for run in runs}, {"C15"})
        self.assertEqual({run.factors["encoder_credit"] for run in runs}, {"arrival", "source"})
        self.assertEqual(
            {run.factors["retirement"] for run in runs},
            {"monitor", "dir_silent", "dir_open", "pred_silent", "pred_open"},
        )
        self.assertTrue(all("--train_for_env_steps=75000000" in run.args for run in runs))

    def test_every_run_enables_the_corrected_contract(self):
        fixed = {
            "--dg_batchnorm_semantics=running_consistent",
            "--dg_recruitment_reset_goal_adapter=True",
            "--hrl_goal_conditioning=target_id_film",
            "--hrl_target_timing=immediate",
            "--encoder_reward_require_local_predecessor=True",
            "--iterative_update=False",
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

    def test_contract_metrics_and_early_diagnostic_steps_are_declared(self):
        metrics = self.study.analysis["window_metrics"]
        for name in (
            "encoder_credit_scheduled_count",
            "encoder_credit_applied_count",
            "encoder_credit_replay_match",
            "dg_forward_count",
            "dg_running_stats_update_count",
            "stale_generation_rejected_fraction",
            "goal_adapter_resets",
        ):
            self.assertIn(name, metrics)
        self.assertEqual(
            self.study.analysis["synchronized_steps"],
            [5_000_000, 25_000_000, 50_000_000, 75_000_000],
        )
        self.assertIn("goal_adapter_reset_total", self.study.analysis["cumulative_metrics"])


if __name__ == "__main__":
    unittest.main()
