"""Telemetry must preserve environment measurements and remain outside learning."""

import unittest

import numpy as np
import torch

from hpc_runs.intrmotiv_offpolicy.sf_telemetry import extra_reports


class ReportTests(unittest.TestCase):
    def test_periodic_and_finished_episode_only(self):
        from sample_factory.algo.utils.misc import EPISODIC

        infos = [
            {
                "periodic_stats": {"intrmotiv/exploration/window/coverage_auc": 3.0},
                "episode_extra_stats": {"coverage_auc": 7.0},
            },
            {"episode_extra_stats": {"coverage_auc": 9.0, "invalid": float("nan")}},
        ]
        report = extra_reports(infos, [False, True], 0)[0][EPISODIC]
        np.testing.assert_array_equal(report["coverage_auc"], [9.0])
        np.testing.assert_array_equal(report["intrmotiv/exploration/window/coverage_auc"], [3.0])
        self.assertNotIn("invalid", report)
        self.assertIn("periodic_stats", infos[0])

    def test_hook_preserves_existing_reports_and_is_opt_in(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from hpc_runs.intrmotiv_offpolicy.sf_telemetry import install_sampling_reports
        from sample_factory.algo.sampling.batched_sampling import BatchedVectorEnvRunner

        def upstream(self, rewards, dones, infos):
            # Deployed SF already handles periodic stats and removes them.
            infos[0].pop("periodic_stats", None)
            return [{"upstream": True}]

        with patch.object(BatchedVectorEnvRunner, "_process_env_step", upstream):
            install_sampling_reports()
            installed = BatchedVectorEnvRunner._process_env_step
            install_sampling_reports()
            self.assertIs(installed, BatchedVectorEnvRunner._process_env_step)
            obj = SimpleNamespace(cfg=SimpleNamespace(ddqn_telemetry=True), policy_id=0)
            info = [{"periodic_stats": {"periodic": 1}, "episode_extra_stats": {"coverage": 2}}]
            reports = installed(obj, torch.zeros(1), torch.ones(1, dtype=torch.bool), info)
            self.assertEqual(len(reports), 2)
            self.assertEqual(reports[0], {"upstream": True})
            obj.cfg.ddqn_telemetry = False
            self.assertEqual(installed(obj, torch.zeros(1), torch.ones(1), info), [{"upstream": True}])


class SpatialTests(unittest.TestCase):
    def test_reconstructs_activity_not_exclusive_events_and_omits_graph_zeros(self):
        try:
            from sf_working_directories.IntrMotiv.dmlab.custom_learner import dg_usage_metrics
        except ImportError:
            self.skipTest("Full IntrMotiv runtime required")
        from types import SimpleNamespace

        from hpc_runs.intrmotiv_offpolicy.sf_telemetry import NativeTelemetry

        captured = {}

        class Spatial:
            def append_batch(self, buff, valids):
                captured.update(buff)
                captured["valids"] = valids

            def on_env_steps(self, frames):
                return {
                    "online_spatial_graph_grounded_controllability": 0.0,
                    "online_spatial_place_active_unit_fraction": 1.0,
                }

        telemetry = NativeTelemetry.__new__(NativeTelemetry)
        telemetry.spatial = Spatial()
        telemetry.n = 2
        telemetry.projection = SimpleNamespace(intercept=2.0, activation=torch.relu)
        packet = torch.tensor([[[3.0, 4.0, 0.0], [0.0, 1.0, 0.0]]])
        original = packet.clone()
        stats = telemetry.capture({"ddqn_packet": packet, "policy_id": torch.zeros(1, 2)}, 0, 8)
        torch.testing.assert_close(captured["dg_activity"], torch.tensor([[[1.0, 2.0], [0.0, 0.0]]]))
        torch.testing.assert_close(packet, original)
        self.assertEqual(stats["dg_density"], 0.5)
        self.assertEqual(stats["dg_silent_unit_frac"], 0.0)
        self.assertAlmostEqual(stats["dg_usage_entropy"], 1.0)
        self.assertFalse(any(k.startswith("online_spatial_graph_") for k in stats))
