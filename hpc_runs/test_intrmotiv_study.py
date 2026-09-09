from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hpc_runs.graph_stabilized_recruitment_manifest import rows as legacy_rows
from hpc_runs.intrmotiv_study import SCHEMA_ID, SpecError, WORKFLOW_VERSION, load_study
from hpc_runs.intrmotiv_study.analysis import linear_contrasts, summarize_records
from hpc_runs.intrmotiv_study.sample_factory import build_run_description
from hpc_runs.intrmotiv_study.submission import audit_submission
from hpc_runs.intrmotiv_study.telemetry import (
    MANIFEST_COLUMNS,
    CheckpointRecord,
    build_place_field_manifests,
    build_intervention_manifest,
    selected_intervention_runs,
)
from hpc_runs.intrmotiv_study.tensorboard import latest_at_or_before, mean_in_window
from hpc_runs.intrmotiv_study.spatial import (
    collect_spatial_detail_records,
    collect_spatial_records,
    summarize_spatial_records,
)
from hpc_runs.intrmotiv_study.spatial_contract import (
    SNAPSHOT_SCHEMA,
    OnlineSpatialWindow,
    SpatialBounds,
    SpatialContractError,
    calculate_spatial_metrics,
    calculate_graph_diagnostics,
    calculate_place_field_details,
    load_spatial_snapshot,
    spatial_rate_maps,
    write_spatial_snapshot_atomic,
)


SPEC_PATH = (
    Path(__file__).with_name("studies")
    / "graph_stabilized_recruitment.study.json"
)


@dataclass(frozen=True)
class Event:
    step: int
    value: float
    wall_time: float = 0.0


class StudySpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.study = load_study(SPEC_PATH)

    def test_real_factorial_study_expands_to_unique_runs(self):
        runs = self.study.expand_runs()
        self.assertEqual(self.study.expected_runs, 36)
        self.assertEqual(len({run.name for run in runs}), 36)
        self.assertEqual(runs[0].name, "GSR_C05_D4_H5K_S8")
        self.assertEqual(runs[-1].name, "GSR_C15_D8_H10K_S123")
        self.assertIn("--dg_recruitment_redundancy_max_steps=4", runs[0].args)
        self.assertIn("--seed=8", runs[0].args)
        self.assertEqual(self.study.raw["schema"], SCHEMA_ID)
        self.assertEqual(self.study.declared_workflow_version, "1.0.0")
        self.assertEqual(WORKFLOW_VERSION, "1.5.0")
        self.assertEqual(len(self.study.fingerprint), 64)

    def test_machine_readable_schema_is_valid_json(self):
        schema_path = Path(__file__).with_name("intrmotiv_study") / "study.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["$id"], SCHEMA_ID)
        self.assertIn("training", schema["properties"])

    def test_goal_subset_interventions_include_all_five_seeds(self):
        study = load_study(SPEC_PATH.with_name("ca3_memory_novelty_goal.study.json"))
        selected = selected_intervention_runs(study)
        self.assertEqual(len(selected), 10)
        inventory = [CheckpointRecord(run.name, t, t, Path(study.workspace_root) / (run.name + ".pth"),
                     Path(study.workspace_root) / run.name) for run in study.expand_runs()
                     for t in study.telemetry["target_frames"]]
        rows, _ = build_place_field_manifests(study, inventory, require_checkpoint_files=False)
        interventions = build_intervention_manifest(study, rows)
        self.assertEqual(len(interventions), 10)
        self.assertEqual({int(row['seed']) for row in interventions}, set(study.seeds))
        with self.assertRaises(SpecError):
            build_intervention_manifest(study, [row for row in rows if row != interventions[0]])
        study.telemetry['intervention']['where'] = {'misspelled': 'goal'}
        with self.assertRaises(SpecError):
            selected_intervention_runs(study)

    def test_historical_manifest_is_now_a_thin_compatibility_adapter(self):
        rows = legacy_rows()
        self.assertEqual(len(rows), 36)
        self.assertEqual(rows[0]["name"], "GSR_C05_D4_H5K_S8")
        self.assertNotIn("--seed=8", rows[0]["args"])
        self.assertEqual(rows[0]["context_controls"], [
            "original_C05_seed8",
            "original_C05_seed99",
            "original_C05_seed123",
        ])

    def test_incorrect_expected_product_is_rejected(self):
        raw = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        raw["expected_runs"] = 35
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(SpecError, "Cartesian product contains 36"):
                load_study(path)

    def test_duplicate_cli_flags_are_rejected(self):
        raw = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        raw["training"]["common_args"].append("--seed=456")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(SpecError, "duplicate flags"):
                load_study(path)

    def test_supplemental_study_cannot_be_submitted_as_complete(self):
        with self.assertRaisesRegex(RuntimeError, "not a complete Sample Factory"):
            build_run_description(self.study)


class AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.study = load_study(SPEC_PATH)
        self.records = []
        for run in self.study.expand_runs():
            distance = int(run.factors["redundancy_max_steps"])
            half_life = int(run.factors["half_life_events"])
            self.records.append({
                "run_name": run.name,
                "base": run.base,
                "seed": run.seed,
                **run.factors,
                "score": distance + half_life / 10_000 + run.seed / 1_000,
            })

    def test_group_summary_has_mean_sd_and_count(self):
        summary = summarize_records(
            self.records,
            ["base", "redundancy_max_steps", "half_life_events"],
            ["score"],
        )
        self.assertEqual(len(summary), 12)
        self.assertEqual({row["score__n"] for row in summary}, {3})
        self.assertTrue(all(row["score__sd"] > 0 for row in summary))

    def test_explicit_factorial_contrasts_are_seed_paired(self):
        detailed, summary = linear_contrasts(
            self.records,
            ["score"],
            ["base"],
            ["seed"],
            self.study.analysis["contrasts"],
        )
        self.assertEqual(len(detailed), 27)
        by_name = {row["contrast"]: row for row in summary if row["base"] == "C05"}
        self.assertAlmostEqual(by_name["D8_minus_D4"]["mean"], 4.0)
        self.assertAlmostEqual(by_name["H10k_minus_H5k"]["mean"], 0.5)
        self.assertAlmostEqual(by_name["interaction"]["mean"], 0.0)
        self.assertEqual(by_name["interaction"]["n"], 3)

    def test_contrast_refuses_ambiguous_terms(self):
        bad = [{
            "name": "ambiguous",
            "terms": [{"weight": 1, "where": {"redundancy_max_steps": 8}}],
        }]
        with self.assertRaisesRegex(SpecError, "selected 2 rows"):
            linear_contrasts(self.records, ["score"], ["base"], ["seed"], bad)


class TelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.study = load_study(SPEC_PATH)

    def test_standard_protocol_builds_five_plus_two_rows_per_condition(self):
        inventory = []
        targets = self.study.telemetry["target_frames"]
        for run in self.study.expand_runs():
            for target in targets:
                inventory.append(CheckpointRecord(
                    run_name=run.name,
                    target_frames=target,
                    checkpoint_frames=target + run.seed,
                    checkpoint=Path(
                        f"/work/classic/fr_xl1014-train/checkpoints/{run.name}/{target}.pth"
                    ),
                    run_dir=Path(f"/work/classic/fr_xl1014-train/runs/{run.name}"),
                ))
        rows, trajectory = build_place_field_manifests(
            self.study, inventory, require_checkpoint_files=False
        )
        self.assertEqual(len(rows), 84)
        self.assertEqual(len(trajectory), 60)
        self.assertEqual(tuple(rows[0]), MANIFEST_COLUMNS)
        self.assertEqual(len({row["label_suffix"] for row in rows}), 84)
        self.assertEqual({row["seed"] for row in trajectory}, {"99"})

    def test_nonworkspace_checkpoint_is_rejected(self):
        run = self.study.expand_runs()[0]
        inventory = [CheckpointRecord(
            run_name=run.name,
            target_frames=target,
            checkpoint_frames=target,
            checkpoint=Path(f"/home/fr/fr_xl1014/{target}.pth"),
            run_dir=Path("/home/fr/fr_xl1014/run"),
        ) for target in self.study.telemetry["target_frames"]]
        with self.assertRaisesRegex(SpecError, "outside the workspace"):
            build_place_field_manifests(
                self.study, inventory, require_checkpoint_files=False
            )


class SubmissionAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.study = load_study(SPEC_PATH)

    def _write_jobs(self, path: Path, remove_first_arg: bool = False) -> None:
        fields = [
            "job_id", "status", "experiment", "train_root", "sbatch_file",
            "stdout", "stderr", "command",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for index, run in enumerate(self.study.expand_runs(), start=1000):
                args = list(run.args)
                if remove_first_arg and index == 1000:
                    args.pop(0)
                root = "/work/classic/fr_xl1014-train/study"
                writer.writerow({
                    "job_id": str(index),
                    "status": "submitted",
                    "experiment": f"00_{run.name}",
                    "train_root": f"batch/{run.name}",
                    "sbatch_file": f"{root}/sbatch_{run.name}.sh",
                    "stdout": f"{root}/{run.name}-%j.out",
                    "stderr": f"{root}/{run.name}-%j.err",
                    "command": " ".join([
                        *args,
                        f"--experiment=00_{run.name}",
                        f"--train_dir={root}/{run.name}",
                    ]),
                })

    def test_submission_audit_matches_matrix_commands_and_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jobs.tsv"
            self._write_jobs(path)
            result = audit_submission(self.study, path, require_submitted=True)
        self.assertTrue(result["submitted_complete"])
        self.assertTrue(result["commands_match_study"])
        self.assertEqual(result["status_counts"], {"submitted": 36})

    def test_submission_audit_rejects_missing_study_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jobs.tsv"
            self._write_jobs(path, remove_first_arg=True)
            with self.assertRaisesRegex(SpecError, "missing study arguments"):
                audit_submission(self.study, path, require_submitted=True)


class TensorBoardPrimitiveTests(unittest.TestCase):
    def test_window_mean_and_latest_are_shared_primitives(self):
        events = [Event(1, 1.0, 1.0), Event(2, 3.0, 2.0), Event(3, 8.0, 3.0)]
        self.assertEqual(mean_in_window(events, 1, 2), (2.0, 2))
        self.assertEqual(latest_at_or_before(events, 2), (3.0, 2))
        missing, count = mean_in_window(events, 10, 20)
        self.assertTrue(math.isnan(missing))
        self.assertEqual(count, 0)


class SpatialContractTests(unittest.TestCase):
    @staticmethod
    def _payload(run_name: str = "GSR_C05_D4_H5K_S8", target: int = 25_000_000):
        pose = np.asarray(
            ((100, 100, 359), (200, 100, 1), (300, 100, 91), (400, 100, 89)),
            dtype=np.float32,
        )
        activity = np.asarray(((1, 0), (1, 0), (0, 2), (0, 2)), dtype=np.float32)
        return {
            "schema": np.asarray(SNAPSHOT_SCHEMA),
            "schema_version": np.asarray(1, dtype=np.int16),
            "pose": pose,
            "dg_activity": activity,
            "actions": np.asarray((0, 1, 2, 3), dtype=np.int16),
            "dones": np.asarray((False, True, False, False)),
            "segment_id": np.asarray((0, 0, 1, 1), dtype=np.int32),
            "policy_version": np.asarray((3, 3, 4, 4), dtype=np.int64),
            "target_env_steps": np.asarray(target, dtype=np.int64),
            "actual_env_steps": np.asarray(target + 128, dtype=np.int64),
            "window_limit": np.asarray(4, dtype=np.int32),
            "window_start_env_steps": np.asarray(target - 16, dtype=np.int64),
            "window_end_env_steps": np.asarray(target + 128, dtype=np.int64),
            "policy_id": np.asarray(0, dtype=np.int16),
            "run_name": np.asarray(run_name),
            "experiment_identity": np.asarray(f"batch/{run_name}"),
            "environment": np.asarray("dmlab_openfield_map2_fixed_loc3_fixedlength_noreward"),
            "frameskip": np.asarray(4, dtype=np.int16),
            "grain": np.asarray(19, dtype=np.int16),
            "bounds": np.asarray((100, 2000, 100, 2000), dtype=np.float32),
            "stationary_distance": np.asarray(1.0, dtype=np.float32),
        }

    def test_online_maps_match_offline_spatial_information_formula(self):
        payload = self._payload()
        bounds = SpatialBounds()
        maps, occupancy, _ = spatial_rate_maps(payload["pose"], payload["dg_activity"], bounds, 19)
        metrics = calculate_spatial_metrics(
            payload["pose"], payload["dg_activity"], payload["dones"], payload["segment_id"], bounds, 19
        )
        probability = occupancy / occupancy.sum()
        reference = []
        for rate_map in maps:
            mean_rate = (rate_map * probability).sum()
            positive = (rate_map > 0) & (probability > 0)
            reference.append(float((rate_map[positive] * probability[positive] * np.log2(
                rate_map[positive] / mean_rate
            )).sum()))
        self.assertAlmostEqual(metrics["active_unit_mean_spatial_information"], np.mean(reference))
        self.assertEqual(metrics["active_unit_fraction"], 1.0)
        self.assertEqual(metrics["unique_active_peak_bins"], 2.0)

    def test_multilevel_fields_distinguish_mono_and_multifield_units(self):
        bins_by_unit = (
            ((2, 2), (3, 2), (2, 3)),
            ((2, 12), (3, 12), (2, 13), (14, 3), (15, 3), (14, 4)),
        )
        pose_rows = []
        activity_rows = []
        for unit, bins in enumerate(bins_by_unit):
            for x_bin, y_bin in bins:
                x = 100 + (x_bin + 0.5) * 100
                y = 100 + (y_bin + 0.5) * 100
                for _ in range(10):
                    pose_rows.append((x, y, 0.0))
                    activity_rows.append((1.0 if unit == 0 else 0.0, 1.0 if unit == 1 else 0.0))
        details = calculate_place_field_details(
            np.asarray(pose_rows, dtype=np.float32),
            np.asarray(activity_rows, dtype=np.float32),
        )

        self.assertEqual(details["rate_maps"].shape, (19, 19, 2))
        self.assertEqual(details["field_component_labels"].shape, (3, 19, 19, 2))
        self.assertEqual(details["field_eligible"].tolist(), [True, True])
        self.assertEqual(details["field_mono"].tolist(), [True, False])
        self.assertTrue(np.isnan(details["field_primary_secondary_peak_distance"][0]))
        self.assertGreater(details["field_primary_secondary_peak_distance"][1], 500.0)

    def test_directed_global_efficiency_exact_graphs(self):
        graphs = {
            "complete": (np.ones((4, 4), dtype=bool) ^ np.eye(4, dtype=bool), 1.0),
            "chain": (np.asarray(((0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (0, 0, 0, 0))), 13.0 / 36.0),
            "cycle": (np.asarray(((0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (1, 0, 0, 0))), 11.0 / 18.0),
            "disconnected": (np.asarray(((0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))), 1.0 / 12.0),
        }
        for name, (adjacency, expected) in graphs.items():
            adjacency = np.asarray(adjacency, dtype=np.float32)
            diagnostics = calculate_graph_diagnostics(
                adjacency, adjacency, adjacency,
                confidence_threshold=0.5, reliability_threshold=0.5,
            )
            self.assertAlmostEqual(
                float(diagnostics["graph_reliable_global_efficiency"]), expected, msg=name
            )

    def test_grounded_controllability_multiplies_prospective_and_endpoint_factors(self):
        adjacency = np.asarray(((0, 1, 0), (0, 0, 1), (0, 0, 0)), dtype=np.float32)
        prospective_attempts = np.zeros((3, 3), dtype=np.float32)
        prospective_successes = np.zeros((3, 3), dtype=np.float32)
        prospective_attempts[0, 1] = 2
        prospective_attempts[1, 2] = 2
        prospective_successes[0, 1] = 2
        diagnostics = calculate_graph_diagnostics(
            adjacency,
            adjacency,
            adjacency,
            prospective_attempts,
            prospective_successes,
            np.asarray((True, True, False)),
            np.asarray(((0, 0), (100, 0), (200, 0)), dtype=np.float32),
        )
        self.assertAlmostEqual(float(diagnostics["graph_prospective_success_fraction"]), 0.5)
        self.assertAlmostEqual(float(diagnostics["graph_spatial_endpoint_valid_fraction"]), 0.5)
        self.assertAlmostEqual(float(diagnostics["graph_grounded_controllability"]), 0.25)

    def test_trajectory_metrics_respect_terminals_and_circular_yaw(self):
        payload = self._payload()
        metrics = calculate_spatial_metrics(
            payload["pose"], payload["dg_activity"], payload["dones"], payload["segment_id"]
        )
        self.assertAlmostEqual(metrics["mean_absolute_circular_yaw_change"], 2.0)
        self.assertAlmostEqual(metrics["mean_physical_step_distance"], 100.0)
        self.assertAlmostEqual(metrics["path_efficiency"], 1.0)

    def test_window_filters_invalid_policy_lag_and_preserves_episode_segments(self):
        window = OnlineSpatialWindow(3)
        pose = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
        activity = np.ones((2, 3, 2), dtype=np.float32)
        appended = window.append_rollouts(
            pose,
            activity,
            np.arange(6).reshape(2, 3),
            np.asarray(((False, True, False), (False, False, False))),
            np.asarray(((1, 1, 1), (2, 2, 2))),
            np.asarray(((True, True, True), (False, True, True))),
        )
        self.assertEqual(appended, 5)
        self.assertEqual(len(window), 3)
        arrays = window.arrays()
        self.assertEqual(arrays["actions"].tolist(), [2, 4, 5])
        self.assertNotEqual(arrays["segment_id"][0], arrays["segment_id"][1])
        self.assertEqual(arrays["segment_id"][1], arrays["segment_id"][2])

    def test_window_splits_unmarked_large_pose_relocations(self):
        window = OnlineSpatialWindow(4)
        window.append_rollouts(
            np.asarray((((100, 100, 0), (140, 100, 0), (1500, 1600, 0), (1540, 1600, 0)),), dtype=np.float32),
            np.ones((1, 4, 2), dtype=np.float32),
            np.zeros((1, 4), dtype=np.int16),
            np.zeros((1, 4), dtype=bool),
            np.zeros((1, 4), dtype=np.int64),
            np.ones((1, 4), dtype=bool),
            max_segment_jump_distance=250.0,
        )
        segments = window.arrays()["segment_id"]
        self.assertEqual(segments[0], segments[1])
        self.assertNotEqual(segments[1], segments[2])
        self.assertEqual(segments[2], segments[3])

    def test_window_ring_wrap_and_latest_scalar_tail_are_chronological(self):
        window = OnlineSpatialWindow(5)

        def append(start, count):
            values = np.arange(start, start + count, dtype=np.float32)
            pose = np.stack((values, values + 100, values + 200), axis=-1)[None, ...]
            window.append_rollouts(
                pose,
                values.reshape(1, count, 1),
                values.astype(np.int16).reshape(1, count),
                np.zeros((1, count), dtype=bool),
                values.astype(np.int64).reshape(1, count),
                np.ones((1, count), dtype=bool),
            )

        append(0, 3)
        append(3, 4)
        self.assertEqual(window.arrays()["pose"][:, 0].tolist(), [2, 3, 4, 5, 6])
        self.assertEqual(window.arrays(3)["pose"][:, 0].tolist(), [4, 5, 6])
        with self.assertRaisesRegex(SpatialContractError, "maximum returned samples"):
            window.arrays(0)

    def test_atomic_snapshot_keeps_existing_valid_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path, created = write_spatial_snapshot_atomic(Path(directory), self._payload())
            self.assertTrue(created)
            self.assertEqual(load_spatial_snapshot(path)["pose"].shape, (4, 3))
            same, created = write_spatial_snapshot_atomic(Path(directory), self._payload())
            self.assertFalse(created)
            self.assertEqual(same, path)
            self.assertEqual(len(list(Path(directory).glob("*.npz"))), 1)

    def test_invalid_bounds_and_alignment_are_rejected(self):
        with self.assertRaises(SpatialContractError):
            SpatialBounds(2, 1, 0, 1)
        payload = self._payload()
        payload["dones"] = payload["dones"][:-1]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SpatialContractError, "expected 4"):
                write_spatial_snapshot_atomic(Path(directory), payload)

        payload = self._payload()
        payload["scalar_window_limit"] = np.asarray(5, dtype=np.int32)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SpatialContractError, "within the snapshot window"):
                write_spatial_snapshot_atomic(Path(directory), payload)

    def test_collector_propagates_study_fingerprint_and_exact_metadata(self):
        study = load_study(SPEC_PATH)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_spatial_snapshot_atomic(root / "arbitrary" / "policy_00", self._payload())
            records, inventory, manifest = collect_spatial_records(
                study, root, require_workspace=False
            )
        self.assertEqual(records[0]["seed"], 8)
        self.assertEqual(records[0]["base"], "C05")
        self.assertEqual(len(inventory), 1)
        self.assertEqual(manifest["study_sha256"], study.fingerprint)
        self.assertFalse(manifest["complete"])
        condition, seed = summarize_spatial_records(records, ["base"])
        self.assertEqual(condition[0]["valid_sample_count__n"], 1)
        self.assertEqual(seed[0]["seed"], 8)

    def test_detail_collector_uses_cached_arrays_and_graph_edges(self):
        study = load_study(SPEC_PATH)
        payload = self._payload()
        details = calculate_place_field_details(payload["pose"], payload["dg_activity"])
        payload.update(details)
        matrix = np.zeros((2, 2), dtype=np.float32)
        payload.update({
            "control_node_visits": np.ones(2, dtype=np.float32),
            "control_tctrl": matrix.copy(),
            "control_edge_confidence": matrix.copy(),
            "control_attempts": matrix.copy(),
            "control_prospective_attempts": matrix.copy(),
            "control_prospective_successes": matrix.copy(),
            "control_prospective_probability_sum": matrix.copy(),
            "control_prospective_brier_sum": matrix.copy(),
            "control_prospective_timing_count": matrix.copy(),
            "control_prospective_timing_sum": matrix.copy(),
            "control_prospective_predicted_timing_sum": matrix.copy(),
            "control_prospective_timing_absolute_error_sum": matrix.copy(),
            "control_passive_confidence": matrix.copy(),
            "control_passive_time": matrix.copy(),
            "control_passive_path_length": matrix.copy(),
            "control_passive_dx": matrix.copy(),
            "control_passive_dy": matrix.copy(),
            "control_passive_dtheta_sin": matrix.copy(),
            "control_passive_dtheta_cos": matrix.copy(),
            "control_frontier_attempts": np.zeros(2, dtype=np.float32),
            "control_frontier_discoveries": np.zeros(2, dtype=np.float32),
            "control_landmark_pose": np.zeros((2, 3), dtype=np.float32),
            "control_pose_valid": np.zeros(2, dtype=bool),
            "control_pose_stress": np.asarray(0.0, dtype=np.float32),
            "control_representation_generation": np.asarray(0, dtype=np.int64),
            "control_confidence_threshold": np.asarray(0.5, dtype=np.float32),
            "control_reliability_threshold": np.asarray(0.5, dtype=np.float32),
        })
        payload.update(calculate_graph_diagnostics(
            matrix, matrix, matrix, matrix, matrix,
            details["field_mono"], details["field_dominant_peak_xy"],
        ))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_spatial_snapshot_atomic(root / "policy_00", payload)
            units, fields, edges = collect_spatial_detail_records(
                study, root, require_workspace=False
            )
        self.assertEqual(len(units), 2)
        self.assertEqual(fields, [])
        self.assertEqual(len(edges), 2)
        self.assertEqual(edges[0]["reliable"], 0)


if __name__ == "__main__":
    unittest.main()
