"""Training-side coordinator for compact online spatial telemetry."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from hpc_runs.intrmotiv_study.spatial_contract import (
    SNAPSHOT_SCHEMA,
    OnlineSpatialWindow,
    SpatialBounds,
    SpatialContractError,
    calculate_graph_diagnostics,
    calculate_place_field_details,
    calculate_spatial_metrics,
    write_spatial_snapshot_atomic,
)
from sample_factory.utils.utils import log


PLACE_FIELD_KEYS = {
    "valid_sample_count": "online_spatial_place_valid_sample_count",
    "in_bounds_fraction": "online_spatial_place_in_bounds_fraction",
    "visited_cell_fraction": "online_spatial_place_visited_cell_fraction",
    "active_unit_fraction": "online_spatial_place_active_unit_fraction",
    "silent_unit_fraction": "online_spatial_place_silent_unit_fraction",
    "active_unit_mean_spatial_information": "online_spatial_place_active_unit_mean_spatial_information",
    "active_only_map_cosine": "online_spatial_place_active_only_map_cosine",
    "unique_active_peak_bins": "online_spatial_place_unique_active_peak_bins",
    "mono_field_unit_fraction": "online_spatial_place_mono_field_unit_fraction",
    "mean_primary_secondary_peak_distance": "online_spatial_place_mean_primary_secondary_peak_distance",
    "median_dominant_peak_nearest_neighbor_distance": (
        "online_spatial_place_median_dominant_peak_nearest_neighbor_distance"
    ),
}
GRAPH_KEYS = {
    "graph_reliable_global_efficiency": "online_spatial_graph_reliable_global_efficiency",
    "graph_grounded_controllability": "online_spatial_graph_grounded_controllability",
}
DEFAULT_SNAPSHOT_TARGETS = (5_000_000, 25_000_000, 50_000_000, 75_000_000, 100_000_000)
TRAJECTORY_KEYS = {
    "mean_physical_step_distance": "online_spatial_trajectory_mean_physical_step_distance",
    "stationary_step_fraction": "online_spatial_trajectory_stationary_step_fraction",
    "path_efficiency": "online_spatial_trajectory_path_efficiency",
    "mean_absolute_circular_yaw_change": "online_spatial_trajectory_mean_absolute_circular_yaw_change",
}


def _strict_next(value: int, interval: int) -> int:
    return (int(value) // int(interval) + 1) * int(interval)


def _contained(candidate: Path, root: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


class TrainingSpatialTelemetry:
    """Accumulate policy-local behavior samples and emit cadence-bound artifacts."""

    def __init__(
        self,
        cfg: Any,
        env_info: Any,
        policy_id: int,
        restored_env_steps: int,
        actor_critic: Any | None = None,
    ):
        self.cfg = cfg
        self.policy_id = int(policy_id)
        self.frameskip = int(env_info.frameskip)
        self.window_limit = int(cfg.online_spatial_window_observations)
        configured_scalar_window = int(
            getattr(cfg, "online_spatial_scalar_window_observations", 10_000)
        )
        self.scalar_window_limit = min(configured_scalar_window, self.window_limit)
        self.scalar_interval = int(cfg.online_spatial_scalar_interval_frames)
        self.snapshot_interval = int(cfg.online_spatial_snapshot_interval_frames)
        self.snapshot_max = int(cfg.online_spatial_snapshot_max_frames)
        self.grain = int(cfg.online_spatial_grid_grain)
        self.stationary_distance = float(cfg.online_spatial_stationary_distance)
        self.max_segment_jump_distance = float(
            getattr(cfg, "online_spatial_max_segment_jump_distance", 250.0)
        )
        self.bounds = SpatialBounds(
            float(cfg.online_spatial_x_min),
            float(cfg.online_spatial_x_max),
            float(cfg.online_spatial_y_min),
            float(cfg.online_spatial_y_max),
        )
        if min(
            self.window_limit,
            configured_scalar_window,
            self.scalar_interval,
            self.snapshot_interval,
            self.snapshot_max,
            self.grain,
        ) <= 0:
            raise SpatialContractError("online spatial window, cadence, maximum, and grain must be positive")
        if self.snapshot_max < self.snapshot_interval:
            raise SpatialContractError("online spatial snapshot maximum must be at least one interval")
        if self.snapshot_max % self.snapshot_interval:
            raise SpatialContractError("online spatial snapshot maximum must be divisible by its interval")
        if self.frameskip <= 0:
            raise SpatialContractError("environment frameskip must be positive")
        if not np.isfinite(self.max_segment_jump_distance) or self.max_segment_jump_distance <= 0:
            raise SpatialContractError("online spatial maximum segment jump distance must be finite and positive")
        target_setting = str(getattr(cfg, "online_spatial_snapshot_targets", "auto") or "auto").strip()
        if target_setting.lower() == "auto":
            if self.snapshot_interval == 25_000_000 and self.snapshot_max == 100_000_000:
                self.snapshot_targets = DEFAULT_SNAPSHOT_TARGETS
            else:
                self.snapshot_targets = tuple(range(self.snapshot_interval, self.snapshot_max + 1, self.snapshot_interval))
        else:
            try:
                self.snapshot_targets = tuple(int(value.strip()) for value in target_setting.split(","))
            except ValueError as error:
                raise SpatialContractError("online spatial snapshot targets must be comma-separated integers") from error
            if (
                not self.snapshot_targets
                or tuple(sorted(set(self.snapshot_targets))) != self.snapshot_targets
                or any(value <= 0 for value in self.snapshot_targets)
            ):
                raise SpatialContractError("online spatial snapshot targets must be unique increasing positive integers")

        workspace = Path(cfg.online_spatial_workspace_root).expanduser().resolve()
        configured_root = str(getattr(cfg, "online_spatial_output_root", "") or "").strip()
        output_root = Path(configured_root).expanduser() if configured_root else Path(cfg.train_dir) / "analysis" / "online_spatial"
        output_root = output_root.resolve()
        if not workspace.is_absolute() or not _contained(output_root, workspace):
            raise SpatialContractError(
                f"online spatial output root {output_root} must be inside workspace {workspace}"
            )
        experiment = PurePosixPath(str(cfg.experiment))
        if experiment.is_absolute() or ".." in experiment.parts or not experiment.name:
            raise SpatialContractError(f"invalid experiment identity for spatial telemetry: {cfg.experiment!r}")
        # Sample Factory's Slurm launcher prefixes each StudySpec run with its
        # numeric row (for example ``00_GSR_BASELINE_S8``).  Preserve that full
        # execution identity for provenance, but expose the exact StudySpec run
        # name to the standardized collector.
        launcher_name = experiment.name
        self.run_name = re.sub(r"^\d+_", "", launcher_name)
        if not self.run_name:
            raise SpatialContractError(f"invalid run name for spatial telemetry: {cfg.experiment!r}")

        # The canonical output root is <workspace train_dir>/analysis/online_spatial.
        # Keep batch identity in the path so repeated StudySpec run names cannot
        # collide across submitted batches, while retaining the actual launcher
        # path in snapshot metadata.
        canonical_train_tree = output_root.parents[1]
        train_root = Path(cfg.train_dir).expanduser().resolve()
        try:
            training_relative = train_root.relative_to(canonical_train_tree)
        except ValueError:
            training_relative = Path()
        relative_parts = tuple(part for part in training_relative.parts if part not in ("", "."))
        batch_name = relative_parts[0] if relative_parts else (
            experiment.parts[-2] if len(experiment.parts) > 1 else "unbatched"
        )
        self.batch_name = batch_name
        identity_parts = (*relative_parts, *experiment.parts)
        self.experiment_identity = str(PurePosixPath(*identity_parts))
        self.output_dir = output_root / batch_name / self.run_name / f"policy_{self.policy_id:02d}"
        if not _contained(self.output_dir, workspace):
            raise SpatialContractError("resolved policy telemetry path escaped the workspace")

        self.environment = str(cfg.env)
        self.actor_critic = actor_critic
        self.window = OnlineSpatialWindow(self.window_limit)
        self.next_scalar_target = _strict_next(restored_env_steps, self.scalar_interval)
        self._snapshot_index = next(
            (index for index, target in enumerate(self.snapshot_targets) if target > int(restored_env_steps)),
            len(self.snapshot_targets),
        )
        self.next_snapshot_target = (
            self.snapshot_targets[self._snapshot_index] if self._snapshot_index < len(self.snapshot_targets) else None
        )
        self.pending_snapshot_targets: list[int] = []
        log.info(
            "Online spatial telemetry policy %d resumes at %d frames; snapshot window=%d, "
            "scalar window=%d, next scalar=%d, next snapshot=%s",
            self.policy_id,
            restored_env_steps,
            self.window_limit,
            self.scalar_window_limit,
            self.next_scalar_target,
            self.next_snapshot_target,
        )

    def append_batch(self, buff: Any, valids: Any) -> int:
        obs = buff.get("obs")
        if obs is None or "telemetry_pose" not in obs:
            raise SpatialContractError("online spatial telemetry requires obs.telemetry_pose")
        if "dg_activity" not in buff:
            raise SpatialContractError("online spatial telemetry requires behavior-time dg_activity")

        def cpu(value: Any) -> np.ndarray:
            if hasattr(value, "detach"):
                value = value.detach().to("cpu")
            return np.asarray(value)

        return self.window.append_rollouts(
            cpu(obs["telemetry_pose"][:, :-1]),
            cpu(buff["dg_activity"]),
            cpu(buff["actions"]),
            cpu(buff["dones"]),
            cpu(buff["policy_version"]),
            cpu(valids),
            self.max_segment_jump_distance,
        )

    @staticmethod
    def _cpu(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().to("cpu")
        return np.asarray(value)

    def _graph_payload(self) -> dict[str, Any]:
        actor_critic = self.actor_critic
        core = getattr(actor_critic, "core", None)
        graph = getattr(core, "policy_graph", None)
        result: dict[str, Any] = {}
        if graph is not None:
            control_names = (
                "node_visits", "tctrl", "edge_confidence", "control_attempts",
                "passive_confidence", "passive_time", "passive_path_length", "passive_dx", "passive_dy",
                "passive_dtheta_sin", "passive_dtheta_cos", "frontier_attempts", "frontier_discoveries",
                "landmark_pose", "pose_valid", "pose_stress", "representation_generation",
                "prospective_attempts", "prospective_successes", "prospective_probability_sum",
                "prospective_brier_sum", "prospective_timing_count", "prospective_timing_sum",
                "prospective_predicted_timing_sum", "prospective_timing_absolute_error_sum",
            )
            result.update({
                ("control_attempts" if name == "control_attempts" else f"control_{name}"): self._cpu(
                    getattr(graph, name)
                )
                for name in control_names
            })
            result["control_confidence_threshold"] = np.asarray(
                float(getattr(self.cfg, "hrl_edge_confidence_threshold", 0.5)), dtype=np.float32
            )
            result["control_reliability_threshold"] = np.asarray(
                float(getattr(self.cfg, "hrl_edge_reliability_threshold", 0.5)), dtype=np.float32
            )
        passive = getattr(core, "passive_recruitment_graph", None)
        if passive is not None:
            for name in ("confidence", "elapsed", "birth_support", "representation_generation"):
                destination = "passive_recruitment_generation" if name == "representation_generation" else f"passive_recruitment_{name}"
                result[destination] = self._cpu(getattr(passive, name))
        projection = getattr(getattr(actor_critic, "encoder", None), "DG_projection", None)
        if projection is not None:
            for name in (
                "recruitment_committed", "recruitment_activation_counts", "recruitment_row_counts",
                "recruitment_count", "recruitment_repeat_count", "recruitment_tiny_residual_count",
            ):
                if hasattr(projection, name):
                    result[f"dg_{name}"] = self._cpu(getattr(projection, name))
        return result

    @staticmethod
    def _graph_diagnostics(
        graph_payload: dict[str, Any], spatial_details: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        if "control_tctrl" not in graph_payload:
            return {}
        return calculate_graph_diagnostics(
            graph_payload["control_tctrl"],
            graph_payload["control_edge_confidence"],
            graph_payload["control_attempts"],
            graph_payload["control_prospective_attempts"],
            graph_payload["control_prospective_successes"],
            spatial_details["field_mono"],
            spatial_details["field_dominant_peak_xy"],
            float(graph_payload["control_confidence_threshold"]),
            float(graph_payload["control_reliability_threshold"]),
        )

    def _snapshot_payload(
        self,
        target: int,
        actual: int,
        spatial_details: dict[str, np.ndarray],
        graph_payload: dict[str, Any],
        graph_diagnostics: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        arrays = self.window.arrays()
        if len(self.window) != self.window_limit:
            raise SpatialContractError("snapshot requested before the latest-N window was full")
        arrays["actions"] = arrays["actions"].astype(np.int16, copy=False)
        arrays["segment_id"] = arrays["segment_id"].astype(np.int32, copy=False)
        arrays["policy_version"] = arrays["policy_version"].astype(np.int64, copy=False)
        return {
            "schema": np.asarray(SNAPSHOT_SCHEMA),
            "schema_version": np.asarray(1, dtype=np.int16),
            **arrays,
            **spatial_details,
            **graph_payload,
            **graph_diagnostics,
            "target_env_steps": np.asarray(target, dtype=np.int64),
            "actual_env_steps": np.asarray(actual, dtype=np.int64),
            "window_limit": np.asarray(self.window_limit, dtype=np.int32),
            "scalar_window_limit": np.asarray(self.scalar_window_limit, dtype=np.int32),
            "window_start_env_steps": np.asarray(max(0, actual - self.window_limit * self.frameskip), dtype=np.int64),
            "window_end_env_steps": np.asarray(actual, dtype=np.int64),
            "policy_id": np.asarray(self.policy_id, dtype=np.int16),
            "run_name": np.asarray(self.run_name),
            "batch_name": np.asarray(self.batch_name),
            "experiment_identity": np.asarray(self.experiment_identity),
            "environment": np.asarray(self.environment),
            "frameskip": np.asarray(self.frameskip, dtype=np.int16),
            "grain": np.asarray(self.grain, dtype=np.int16),
            "bounds": self.bounds.as_array(),
            "stationary_distance": np.asarray(self.stationary_distance, dtype=np.float32),
            "max_segment_jump_distance": np.asarray(self.max_segment_jump_distance, dtype=np.float32),
        }

    def on_env_steps(self, actual_env_steps: int) -> dict[str, float]:
        actual = int(actual_env_steps)
        scalar_target: int | None = None
        while self.next_scalar_target <= actual:
            scalar_target = self.next_scalar_target
            self.next_scalar_target += self.scalar_interval
        while self.next_snapshot_target is not None and self.next_snapshot_target <= actual:
            self.pending_snapshot_targets.append(self.next_snapshot_target)
            self._snapshot_index += 1
            self.next_snapshot_target = (
                self.snapshot_targets[self._snapshot_index]
                if self._snapshot_index < len(self.snapshot_targets)
                else None
            )

        has_samples = len(self.window) > 0
        scalar_due = scalar_target is not None and has_samples
        snapshot_ready = len(self.window) == self.window_limit and bool(self.pending_snapshot_targets)
        if not scalar_due and not snapshot_ready:
            return {}

        graph_payload = self._graph_payload()
        if snapshot_ready:
            snapshot_arrays = self.window.arrays()
            snapshot_details = calculate_place_field_details(
                snapshot_arrays["pose"], snapshot_arrays["dg_activity"], self.bounds, self.grain
            )
            snapshot_graph_diagnostics = self._graph_diagnostics(graph_payload, snapshot_details)
            for target in self.pending_snapshot_targets:
                path, created = write_spatial_snapshot_atomic(
                    self.output_dir,
                    self._snapshot_payload(
                        target, actual, snapshot_details, graph_payload, snapshot_graph_diagnostics
                    ),
                )
                log.info("%s online spatial snapshot %s", "Wrote" if created else "Kept", path)
            self.pending_snapshot_targets.clear()

        if not scalar_due:
            return {}
        arrays = self.window.arrays(self.scalar_window_limit)
        spatial_details = calculate_place_field_details(
            arrays["pose"], arrays["dg_activity"], self.bounds, self.grain
        )
        graph_diagnostics = self._graph_diagnostics(graph_payload, spatial_details)
        values = calculate_spatial_metrics(
            arrays["pose"],
            arrays["dg_activity"],
            arrays["dones"],
            arrays["segment_id"],
            self.bounds,
            self.grain,
            self.stationary_distance,
        )
        result = {destination: values[source] for source, destination in PLACE_FIELD_KEYS.items()}
        result.update({destination: values[source] for source, destination in TRAJECTORY_KEYS.items()})
        result.update({destination: float(graph_diagnostics.get(source, 0.0)) for source, destination in GRAPH_KEYS.items()})
        result["online_spatial_scalar_target_env_steps"] = float(scalar_target)
        return result
