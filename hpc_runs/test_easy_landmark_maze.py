"""Easy landmark-maze geometry, cue metrics, and study contracts."""

import json
import unittest
from collections import deque
from pathlib import Path

import numpy as np

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.geometry import (
    GEOMETRY_SCHEMA_V1,
    GEOMETRY_SCHEMA_V2,
    _minimum_cost_assignment,
    cue_sites,
    cue_spatial_metrics,
    geometry_payload,
    landmark_entity_record,
    load_geometry,
    load_landmark_geometry,
    validate_geometry_payload,
)

ROOT = Path(__file__).parent
CORRIDOR_ARCHIVE = ROOT / "studies/assets/corridor_geometry/maps.json"
LANDMARK_ARCHIVE = ROOT / "studies/assets/easy_landmark_maze/maps.json"


class EasyLandmarkGeometryTests(unittest.TestCase):
    def test_rectangular_assignment_is_exact_and_transpose_safe(self):
        cost = np.asarray([[4, 1, 3], [2, 0, 5]], dtype=float)
        rows, columns = _minimum_cost_assignment(cost)
        self.assertEqual(list(zip(rows, columns)), [(0, 1), (1, 0)])
        transposed_rows, transposed_columns = _minimum_cost_assignment(cost.T)
        self.assertEqual(list(zip(transposed_rows, transposed_columns)), [(0, 1), (1, 0)])

    def test_fixed_layout_has_twenty_disjoint_deterministic_sites(self):
        source = load_landmark_geometry(str(LANDMARK_ARCHIVE), 1001, 0.85, 11, 11, 20260923, "rich")
        first = cue_sites(source["entity_layer"], cue_layout_seed=20260923)
        second = cue_sites(source["entity_layer"], cue_layout_seed=20260923)
        self.assertEqual(first, second)
        self.assertEqual([site["cue_type"] for site in first].count("decal"), 10)
        self.assertEqual([site["cue_type"] for site in first].count("color"), 10)
        self.assertEqual(len({tuple(site["wall_rc"]) for site in first}), 20)
        self.assertEqual(len({tuple(site["floor_rc"]) for site in first}), 20)
        self.assertEqual(first[0]["cue_id"], "D01")
        self.assertEqual(first[-1]["cue_id"], "C10")
        self.assertEqual(source["entity_shape"], [11, 11])
        self.assertEqual(np.asarray(source["accessible_mask"]).shape, (9, 9))
        self.assertEqual(source["bounds"], [100.0, 1000.0, 100.0, 1000.0])
        self.assertTrue(any(0 in site["wall_rc"] or 10 in site["wall_rc"] for site in first))

    def test_archive_modes_share_sites_and_v2_payload_round_trips(self):
        rich = load_landmark_geometry(str(LANDMARK_ARCHIVE), 1001, 0.85, 11, 11, 20260923, "rich")
        control = load_landmark_geometry(str(LANDMARK_ARCHIVE), 1001, 0.85, 11, 11, 20260923, "none")
        self.assertEqual(rich["schema"], GEOMETRY_SCHEMA_V2)
        self.assertEqual(rich["cue_sites"], control["cue_sites"])
        self.assertNotEqual(rich["cue_layout_sha256"], control["cue_layout_sha256"])
        rich_payload = geometry_payload(rich)
        control_payload = geometry_payload(control)
        validate_geometry_payload(rich_payload)
        validate_geometry_payload(control_payload)
        self.assertTrue(rich_payload["geometry_cue_rendered"].all())
        self.assertFalse(control_payload["geometry_cue_rendered"].any())

    def test_v1_corridor_payload_remains_valid(self):
        record = load_geometry(str(CORRIDOR_ARCHIVE), 1001, 0)
        payload = geometry_payload(record)
        self.assertEqual(str(payload["geometry_schema"]), GEOMETRY_SCHEMA_V1)
        validate_geometry_payload(payload)
        self.assertNotIn("geometry_cue_ids", payload)

    def test_capacity_normalized_assignment_preserves_f16_limit(self):
        record = load_landmark_geometry(str(LANDMARK_ARCHIVE), 1001, 0.85, 11, 11, 20260923, "rich")
        mask = np.asarray(record["accessible_mask"], dtype=bool)
        occupancy = mask.astype(np.int64)
        maps = np.zeros((16, *mask.shape), dtype=np.float32)
        for unit, site in enumerate(record["cue_sites"][:16]):
            y, x = site["floor_yx"]
            maps[unit, y, x] = 1.0
        summary, assignments = cue_spatial_metrics(maps, occupancy, record)
        self.assertEqual(summary["cue_peak_match_count"], 16)
        self.assertAlmostEqual(summary["cue_peak_coverage_fraction"], 0.8)
        self.assertAlmostEqual(summary["cue_peak_capacity_normalized_coverage"], 1.0)
        self.assertEqual(sum(row["unit_id"] == -1 for row in assignments), 4)

    def test_invalid_mode_is_rejected(self):
        source = load_geometry(str(CORRIDOR_ARCHIVE), 1001, 0)
        with self.assertRaises(ValueError):
            landmark_entity_record(
                source["entity_layer"],
                map_seed=1001,
                wall_removal_probability=0.0,
                cue_layout_seed=20260923,
                cue_mode="random",
            )

    def test_original_flood_distance_spawn_rule(self):
        record = load_landmark_geometry(str(LANDMARK_ARCHIVE), 1001, 0.85, 11, 11, 20260923, "rich")
        grid = np.asarray([list(row) for row in record["entity_layer"].splitlines()])
        floor = grid != "*"
        observed_spawns = {tuple(cell) for cell in np.argwhere(grid == "P")}
        matching_anchors = []
        for anchor in map(tuple, np.argwhere(floor)):
            distance = {anchor: 0}
            queue = deque([anchor])
            while queue:
                row, column = queue.popleft()
                for cell in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
                    if floor[cell] and cell not in distance:
                        distance[cell] = distance[(row, column)] + 1
                        queue.append(cell)
            expected = {cell for cell, steps in distance.items() if steps > 5}
            if expected == observed_spawns:
                matching_anchors.append(anchor)
        self.assertTrue(matching_anchors)
        self.assertNotIn("G", record["entity_layer"])
        self.assertNotIn("A", record["entity_layer"])


class EasyLandmarkStudyTests(unittest.TestCase):
    def setUp(self):
        self.rich = load_study(ROOT / "studies/easy_landmark_maze_rich.study.json")
        self.control = load_study(ROOT / "studies/easy_landmark_maze_control.study.json")
        self.preflight = load_study(ROOT / "studies/easy_landmark_maze_preflight.study.json")

    def test_declared_run_matrix(self):
        self.assertEqual(len(self.rich.expand_runs()), 9)
        self.assertEqual(len(self.control.expand_runs()), 3)
        self.assertEqual(len(self.preflight.expand_runs()), 6)
        self.assertEqual({run.seed for run in self.rich.expand_runs()}, {8, 99, 123})
        self.assertEqual({run.seed for run in self.control.expand_runs()}, {99})
        self.assertEqual(
            {run.factors["cue_mode"] for run in self.preflight.expand_runs()},
            {"rich", "none"},
        )

    def test_runtime_boundaries_and_architecture_specific_controller(self):
        for study in (self.rich, self.control, self.preflight):
            for run in study.expand_runs():
                args = dict(arg[2:].split("=", 1) for arg in run.args if arg.startswith("--"))
                self.assertEqual(args["env"], "easy_landmark_maze_noreward")
                self.assertEqual(args["dmlab_map_seed"], "1001")
                self.assertEqual(args["dmlab_wall_removal_probability"], "0.85")
                self.assertEqual(args["dmlab_map_rows"], "11")
                self.assertEqual(args["dmlab_map_cols"], "11")
                self.assertEqual(args["dmlab_cue_layout_seed"], "20260923")
                self.assertEqual(args["env_frameskip"], "4")
                self.assertEqual(args["dmlab_navigation_action_set"], "True")
                self.assertEqual(args["with_pos_obs"], "False")
                if run.base == "WAYPOINT_F64_DDQN_HER":
                    self.assertEqual(args["Hippo_n_feature"], "64")
                    self.assertEqual(args["controller_learning"], "ddqn")
                    self.assertEqual(args["controller_her"], "True")
                    self.assertEqual(args["controller_decisions_per_update"], "2048")
                    self.assertEqual(args["controller_td_positions"], "1024")
                    self.assertEqual(args["controller_her_positions"], "1024")
                else:
                    self.assertEqual(args["Hippo_n_feature"], "16")
                overrides = json.loads(run.metadata["overrides_json"])
                self.assertEqual(overrides["dmlab_map_seed"]["study"], "1001")
                self.assertEqual(overrides["dmlab_wall_removal_probability"]["study"], "0.85")
                self.assertEqual(overrides["dmlab_map_rows"]["study"], "11")
                self.assertEqual(overrides["dmlab_map_cols"]["study"], "11")
                self.assertEqual(overrides["dmlab_cue_layout_seed"]["study"], "20260923")
                expected_cues = "<factor:cue_mode>" if study is self.preflight else args["dmlab_landmark_cues"]
                self.assertEqual(overrides["dmlab_landmark_cues"]["study"], expected_cues)

    def test_parent_fingerprints_resolve(self):
        for run in self.rich.expand_runs():
            parent = load_study(ROOT / "studies" / f"{run.metadata['parent_study']}.study.json")
            self.assertEqual(parent.fingerprint, run.metadata["parent_sha256"])

    def test_telemetry_counts(self):
        from hpc_runs.intrmotiv_study.telemetry import (
            CheckpointRecord,
            build_intervention_manifest,
            build_place_field_manifests,
            select_standard_place_field_rows,
        )

        for study, expected_fields, expected_interventions in (
            (self.rich, 21, 18),
            (self.control, 15, 6),
            (self.preflight, 12, 6),
        ):
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
            all_rows, _ = build_place_field_manifests(study, inventory, require_checkpoint_files=False)
            rows = select_standard_place_field_rows(study, all_rows)
            self.assertEqual(len(rows), expected_fields)
            self.assertEqual(len(build_intervention_manifest(study, all_rows)), expected_interventions)


if __name__ == "__main__":
    unittest.main()
