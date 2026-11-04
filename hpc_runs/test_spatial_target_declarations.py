"""Regression: a study's late milestones must not disappear in discovery."""
import unittest
from types import SimpleNamespace
from hpc_runs.intrmotiv_study.spatial import expected_spatial_targets
from hpc_runs.intrmotiv_study.spec import SpecError


class SpatialTargetsTest(unittest.TestCase):
    def test_late_declared_targets(self):
        spec = SimpleNamespace(telemetry={'target_frames':[5000000,150000000,300000000]})
        self.assertEqual(expected_spatial_targets(spec),(5000000,150000000,300000000))

    def test_explicit_online_override(self):
        spec = SimpleNamespace(telemetry={'target_frames':[300000000], 'online_spatial_target_frames':[2000000]})
        self.assertEqual(expected_spatial_targets(spec),(2000000,))

    def test_invalid_declarations(self):
        for targets in ([], [0], [-1], [5000000,5000000]):
            with self.assertRaises(SpecError):
                expected_spatial_targets(SimpleNamespace(telemetry={'target_frames':targets}))


if __name__ == '__main__': unittest.main()
