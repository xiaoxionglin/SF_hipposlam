from math import cos, hypot, radians, sin, tanh

import gymnasium as gym
import numpy as np

from sample_factory.utils.utils import log

RAW_SCORE_SUMMARY_KEY_SUFFIX = "dmlab_raw_score"

REDUCED_ACTION_COMMANDS = (
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, -radians(20.0)),
    (1.0, 0.0, radians(20.0)),
)


def repeated_command_transform(action: int, action_repeat: int) -> tuple[float, float, float, float]:
    """Return a policy action's net midpoint-frame transform and path length."""
    if action_repeat <= 0:
        raise ValueError("action_repeat must be positive")
    forward, strafe, yaw_step = REDUCED_ACTION_COMMANDS[int(action)]
    repeats = float(action_repeat)
    half_yaw = 0.5 * yaw_step
    scale = repeats if abs(half_yaw) < 1e-6 else sin(repeats * half_yaw) / sin(half_yaw)
    return forward * scale, strafe * scale, yaw_step * repeats, hypot(forward, strafe) * repeats


def similarity_trajectory_error(predicted, actual):
    """Scale/rotation-aligned trajectory RMSE, normalized by actual RMS extent."""
    predicted = np.asarray(predicted, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if predicted.shape != actual.shape or predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError("predicted and actual trajectories must both have shape [steps, 2]")
    if predicted.shape[0] < 2:
        return 0.0
    predicted = predicted - predicted.mean(axis=0, keepdims=True)
    actual = actual - actual.mean(axis=0, keepdims=True)
    actual_energy = float(np.square(actual).sum())
    predicted_energy = float(np.square(predicted).sum())
    if actual_energy <= 1e-12 or predicted_energy <= 1e-12:
        return 0.0 if actual_energy <= 1e-12 and predicted_energy <= 1e-12 else 1.0
    u, singular_values, vt = np.linalg.svd(predicted.T @ actual)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
        singular_values[-1] *= -1
    scale = float(singular_values.sum() / predicted_energy)
    residual = scale * predicted @ rotation - actual
    return float(np.sqrt(np.square(residual).sum() / actual_energy))


class DmlabRewardShapingWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        coverage_telemetry=False,
        coverage_grid_size=100.0,
        coverage_heading_bin_degrees=15.0,
        exploration_window_steps=0,
        action_path_integration=False,
    ):
        super().__init__(env)
        self.raw_episode_return = self.episode_length = 0
        self.coverage_telemetry = coverage_telemetry
        self.coverage_grid_size = float(coverage_grid_size)
        self.coverage_heading_bin_degrees = float(coverage_heading_bin_degrees)
        if not 0.0 < self.coverage_heading_bin_degrees <= 360.0:
            raise ValueError("coverage_heading_bin_degrees must be in (0, 360]")
        self.coverage_cells = {}
        self.coverage_auc_sum = 0.0
        self.coverage_steps = 0
        self.pose_cells = {}
        self.pose_auc_sum = 0.0
        self.pose_steps = 0
        self.exploration_window_steps = int(exploration_window_steps)
        self.window_return = 0.0
        self.window_frames = 0
        self.window_steps = 0
        self.window_cells = {}
        self.window_auc_sum = 0.0
        self.window_pose_cells = {}
        self.window_pose_auc_sum = 0.0
        self.window_pose_steps = 0
        self.action_path_integration = bool(action_path_integration)
        self.action_repeat = int(getattr(env.unwrapped, "action_repeat", 1))
        self.command_x = self.command_y = self.command_heading = 0.0
        self.command_path_length = 0.0
        self.command_positions = []
        self.actual_positions = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.raw_episode_return = self.episode_length = 0
        self.coverage_cells = {}
        self.coverage_auc_sum = 0.0
        self.coverage_steps = 0
        self.pose_cells = {}
        self.pose_auc_sum = 0.0
        self.pose_steps = 0
        self.window_return = 0.0
        self.window_frames = 0
        self.window_steps = 0
        self.window_cells = {}
        self.window_auc_sum = 0.0
        self.window_pose_cells = {}
        self.window_pose_auc_sum = 0.0
        self.window_pose_steps = 0
        self.command_x = self.command_y = self.command_heading = 0.0
        self.command_path_length = 0.0
        self.command_positions = [(0.0, 0.0)]
        position = getattr(self.env.unwrapped, "last_debug_position", None)
        self.actual_origin = np.asarray(position, dtype=np.float64)[:2].copy() if position is not None else None
        self.actual_positions = [(0.0, 0.0)] if self.actual_origin is not None else []
        return obs, info

    def _coverage_stats(self, cells, auc_sum, steps):
        visits = np.asarray(list(cells.values()), dtype=np.float64)
        probabilities = visits / visits.sum() if visits.size and visits.sum() > 0 else visits
        entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum())
        return float(len(cells)), entropy, float(auc_sum / max(steps, 1))

    def step(self, action):
        if self.action_path_integration:
            forward, strafe, yaw, path_length = repeated_command_transform(int(action), self.action_repeat)
            midpoint = self.command_heading + 0.5 * yaw
            self.command_x += forward * cos(midpoint) - strafe * sin(midpoint)
            self.command_y += forward * sin(midpoint) + strafe * cos(midpoint)
            self.command_heading = np.arctan2(
                sin(self.command_heading + yaw), cos(self.command_heading + yaw)
            )
            self.command_path_length += path_length
            self.command_positions.append((self.command_x, self.command_y))
        obs, rew, terminated, truncated, info = self.env.step(action)
        done = terminated | truncated
        pose = info.pop("intrmotiv_pose", None)
        # Compatibility with synthetic tests and older custom environments.
        legacy_position = info.pop("intrmotiv_position", None)
        terminal_pose_fresh = bool(
            info.pop(
                "intrmotiv_terminal_pose_fresh",
                info.pop("intrmotiv_terminal_position_fresh", True),
            )
        )
        position = pose[:2] if pose is not None else legacy_position
        position_valid = position is not None and (not done or terminal_pose_fresh)
        heading_valid = (
            pose is not None
            and np.asarray(pose).size >= 3
            and np.isfinite(np.asarray(pose, dtype=np.float64)[:3]).all()
            and (not done or terminal_pose_fresh)
        )
        if self.action_path_integration and position_valid:
            horizontal = np.asarray(position, dtype=np.float64)[:2]
            if self.actual_origin is None:
                self.actual_origin = horizontal.copy()
            self.actual_positions.append(tuple(horizontal - self.actual_origin))
        if self.coverage_telemetry and position_valid:
            horizontal = np.asarray(position, dtype=np.float64)[:2]
            cell = tuple(np.floor(horizontal / self.coverage_grid_size).astype(np.int64))
            self.coverage_cells[cell] = self.coverage_cells.get(cell, 0) + 1
            self.coverage_steps += 1
            self.coverage_auc_sum += len(self.coverage_cells)
            if heading_valid:
                heading_bin = int(
                    np.floor((float(pose[2]) % 360.0) / self.coverage_heading_bin_degrees)
                )
                pose_cell = (*cell, heading_bin)
                self.pose_cells[pose_cell] = self.pose_cells.get(pose_cell, 0) + 1
                self.pose_steps += 1
                self.pose_auc_sum += len(self.pose_cells)
            if self.exploration_window_steps > 0:
                self.window_cells[cell] = self.window_cells.get(cell, 0) + 1
                self.window_auc_sum += len(self.window_cells)
                if heading_valid:
                    self.window_pose_cells[pose_cell] = self.window_pose_cells.get(pose_cell, 0) + 1
                    self.window_pose_steps += 1
                    self.window_pose_auc_sum += len(self.window_pose_cells)
        self.raw_episode_return += rew
        self.episode_length += info.get("num_frames", 1)
        self.window_return += rew
        self.window_frames += info.get("num_frames", 1)
        self.window_steps += 1

        # optimistic asymmetric clipping from IMPALA paper
        squeezed = tanh(rew / 5.0)
        clipped = 0.3 * squeezed if rew < 0.0 else squeezed
        rew = clipped * 5.0

        if self.exploration_window_steps > 0 and self.window_steps >= self.exploration_window_steps:
            unique_cells, entropy, coverage_auc = self._coverage_stats(
                self.window_cells, self.window_auc_sum, self.window_steps
            )
            info["intrmotiv_periodic_stats"] = {
                "intrmotiv/exploration/window/return": float(self.window_return),
                "intrmotiv/exploration/window/length_frames": float(self.window_frames),
                "intrmotiv/exploration/window/length_policy_steps": float(self.window_steps),
                "intrmotiv/exploration/window/coverage_unique_cells": unique_cells,
                "intrmotiv/exploration/window/coverage_entropy": entropy,
                "intrmotiv/exploration/window/coverage_auc": coverage_auc,
            }
            if self.window_pose_steps > 0:
                pose_bins, pose_entropy, pose_auc = self._coverage_stats(
                    self.window_pose_cells, self.window_pose_auc_sum, self.window_pose_steps
                )
                info["intrmotiv_periodic_stats"].update(
                    {
                        "intrmotiv/exploration/window/pose_unique_bins": pose_bins,
                        "intrmotiv/exploration/window/pose_entropy": pose_entropy,
                        "intrmotiv/exploration/window/pose_auc": pose_auc,
                    }
                )
            self.window_return = 0.0
            self.window_frames = 0
            self.window_steps = 0
            self.window_cells = {}
            self.window_auc_sum = 0.0
            self.window_pose_cells = {}
            self.window_pose_auc_sum = 0.0
            self.window_pose_steps = 0

        if done:
            score = self.raw_episode_return

            info["episode_extra_stats"] = dict()
            level_name = self.unwrapped.level_name

            # add extra 'z_' to the summary key to put them towards the end on tensorboard (just convenience)
            level_name_key = f"z_{self.unwrapped.task_id:02d}_{level_name}"
            info["episode_extra_stats"][f"{level_name_key}_{RAW_SCORE_SUMMARY_KEY_SUFFIX}"] = score
            info["episode_extra_stats"][f"{level_name_key}_len"] = self.episode_length
            info["episode_extra_stats"][f"{level_name_key}_lenweighted_score"] = (
                (10000 - self.episode_length) / 10000 * score
            )
            if self.coverage_telemetry:
                unique_cells, entropy, coverage_auc = self._coverage_stats(
                    self.coverage_cells, self.coverage_auc_sum, self.coverage_steps
                )
                info["episode_extra_stats"][f"{level_name_key}_coverage_unique_cells"] = unique_cells
                info["episode_extra_stats"][f"{level_name_key}_coverage_entropy"] = entropy
                info["episode_extra_stats"][f"{level_name_key}_coverage_auc"] = float(coverage_auc)
                if self.pose_steps > 0:
                    pose_bins, pose_entropy, pose_auc = self._coverage_stats(
                        self.pose_cells, self.pose_auc_sum, self.pose_steps
                    )
                    info["episode_extra_stats"][f"{level_name_key}_pose_unique_bins"] = pose_bins
                    info["episode_extra_stats"][f"{level_name_key}_pose_entropy"] = pose_entropy
                    info["episode_extra_stats"][f"{level_name_key}_pose_auc"] = pose_auc
            if self.action_path_integration and len(self.actual_positions) == len(self.command_positions):
                periodic = info.setdefault("intrmotiv_periodic_stats", {})
                periodic["intrmotiv/path/telemetry_error"] = similarity_trajectory_error(
                    self.command_positions, self.actual_positions
                )
            # log.info(f'Episode Extra Stats: {info["episode_extra_stats"]}')
        return obs, rew, terminated, truncated, info
