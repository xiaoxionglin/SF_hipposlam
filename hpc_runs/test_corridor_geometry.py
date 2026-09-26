"""Geometry contract, backward compatibility and study inheritance checks."""

import unittest
from pathlib import Path

import numpy as np

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.geometry import (
    AccessibleCoverage,
    accessible_cell,
    geometry_payload,
    load_geometry,
    traversable_field_components,
    verify_entity,
)

ROOT = Path(__file__).parent
ARCHIVE = ROOT / "studies/assets/corridor_geometry/maps.json"


class GeometryTests(unittest.TestCase):
    def test_archived_maps_connected_nested_and_shared_spawns(self):
        for seed in (1001, 1002, 1003):
            records = [load_geometry(str(ARCHIVE), seed, q) for q in (0, 0.35, 0.75)]
            for a, b in zip(records, records[1:]):
                self.assertTrue(np.all(np.array(a["accessible_mask"]) <= np.array(b["accessible_mask"])))
                self.assertEqual(a["spawn_cells_rc"], b["spawn_cells_rc"])
            self.assertEqual(records[0]["accessible_cells"], 199)
            self.assertGreater(records[0]["corridor_fraction"], records[2]["corridor_fraction"])
            self.assertEqual(len(records[0]["spawn_cells_rc"]), 100)

    def test_world_coordinate_orientation(self):
        record = load_geometry(str(ARCHIVE), 1001, 0)
        for row in range(1, 20):
            for col in range(1, 20):
                expected = (19 - row, col - 1) if record["entity_layer"].splitlines()[row][col] != "*" else None
                self.assertEqual(accessible_cell((col * 100 + 50, (20 - row) * 100 + 50), record), expected)
        self.assertIsNone(accessible_cell((0, 0), record))

    def test_coverage_denominator_and_missing_terminal(self):
        record = {"accessible_mask": [[1, 1], [0, 1]], "accessible_cells": 3, "cell_size": 100}
        tracker = AccessibleCoverage(record)
        for pos in [(150, 150), (250, 150), (250, 150), None]:
            tracker.step(pos)
        self.assertAlmostEqual(tracker.metrics()["accessible_coverage_auc"], 7 / 12)
        self.assertAlmostEqual(tracker.metrics()["accessible_coverage_fraction"], 2 / 3)
        self.assertEqual(tracker.metrics()["geometry_invalid_pose_steps"], 1)

    def test_hash_mismatch_fails_and_legacy_payload_empty(self):
        record = load_geometry(str(ARCHIVE), 1001, 0)
        verify_entity(record["entity_layer"], record)
        with self.assertRaises(ValueError):
            verify_entity(record["entity_layer"] + "\n", record)
        self.assertEqual(geometry_payload(None), {})
        self.assertEqual(geometry_payload(record)["geometry_accessible_mask"].shape, (19, 19))

    def test_components_never_bridge_walls_or_diagonals(self):
        maps = np.ones((1, 3, 3))
        mask = np.ones((3, 3), bool)
        mask[:, 1] = False
        result = traversable_field_components(maps, np.ones((3, 3)), mask)
        self.assertEqual(result["geometry_field_components_half_peak"].tolist(), [2])
        result = traversable_field_components(maps, np.eye(3), np.ones((3, 3), bool))
        self.assertEqual(result["geometry_field_components_half_peak"].tolist(), [3])

    def test_study_matrix_and_preflight_resume_identities(self):
        study = load_study(ROOT / "studies/corridor_geometry.study.json")
        pre = load_study(ROOT / "studies/corridor_geometry_preflight.study.json")
        runs = study.expand_runs()
        self.assertEqual(len(runs), 27)
        self.assertEqual({r.seed for r in runs}, {99})
        self.assertEqual({r.factors["map_seed"] for r in runs}, {1001, 1002, 1003})
        self.assertEqual({r.name for r in pre.expand_runs()}, {r.name for r in runs if r.factors["map_seed"] == 1001})
        self.assertEqual(study.output_root, pre.output_root)
        for r in runs:
            args = dict(a[2:].split("=", 1) for a in r.args if a.startswith("--"))
            self.assertEqual(args["train_for_env_steps"], "100000000")
            self.assertEqual(args["env_frameskip"], "4")
            self.assertEqual(args["dmlab_navigation_action_set"], "True")
            self.assertEqual(args["with_pos_obs"], "False")
            if r.base == "WAYPOINT_HER":
                self.assertEqual(args["controller_decisions_per_update"], "2048")
                self.assertEqual(args["controller_her"], "True")
            parent = load_study(ROOT / "studies" / f"{r.metadata['parent_study']}.study.json")
            self.assertEqual(parent.fingerprint, r.metadata["parent_sha256"])


class GeometryManifestTests(unittest.TestCase):
    def test_five_checkpoints_and_two_interventions_for_every_run(self):
        from hpc_runs.intrmotiv_study.telemetry import (
            CheckpointRecord,
            build_intervention_manifest,
            build_place_field_manifests,
        )

        study = load_study(ROOT / "studies/corridor_geometry.study.json")
        inventory = [
            CheckpointRecord(
                run.name,
                target,
                target,
                Path(study.output_root) / run.name / f"{target}.pth",
                Path(study.output_root) / run.name,
            )
            for run in study.expand_runs()
            for target in study.telemetry["target_frames"]
        ]
        rows, _ = build_place_field_manifests(study, inventory, require_checkpoint_files=False)
        self.assertEqual(len(rows), 135)
        self.assertEqual(len(build_intervention_manifest(study, rows)), 54)


if __name__ == "__main__":
    unittest.main()
