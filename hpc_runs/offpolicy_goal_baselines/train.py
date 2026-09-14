"""Standalone off-policy CRL+/L3P+ trainer for IntrMotiv DMLab.

This entry point intentionally reuses the authoritative DMLab environment and
frozen ``ResNet18Layer2`` implementation, but not Sample Factory's on-policy
learner. Bulk artifacts are required to live under the configured workspace.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from tensorboardX import SummaryWriter

from hpc_runs.offpolicy_goal_baselines.model import ContrastiveGoalAgent
from hpc_runs.offpolicy_goal_baselines.planner import LandmarkPlanner
from hpc_runs.offpolicy_goal_baselines.replay import EpisodeReplay

WORKSPACE_ROOT = Path("/work/classic/fr_xl1014-train")


@dataclass(frozen=True)
class BaselineConfig:
    method: str
    total_frames: int
    replay_capacity: int
    replay_min: int
    replay_segment_steps: int
    batch_size: int
    max_future: int
    discount: float
    goal_horizon: int
    planner_horizon: int
    updates_per_step: int
    update_every_steps: int
    learning_rate: float
    entropy_coeff: float
    target_entropy_fraction: float
    logsumexp_coeff: float
    landmark_loss_coeff: float
    hidden_dim: int
    repr_dim: int
    landmark_count: int
    landmark_candidates: int
    landmark_neighbors: int
    landmark_local_horizon: float
    landmark_edge_horizon: float
    planner_rebuild_frames: int
    checkpoint_frames: int
    torch_threads: int


def _baseline_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--baseline-method", choices=("crl", "l3p"), required=True)
    parser.add_argument("--baseline-total-frames", type=int, required=True)
    parser.add_argument("--baseline-replay-capacity", type=int, default=200_000)
    parser.add_argument("--baseline-replay-min", type=int, default=5_000)
    parser.add_argument("--baseline-replay-segment-steps", type=int, default=512)
    parser.add_argument("--baseline-batch-size", type=int, default=256)
    parser.add_argument("--baseline-max-future", type=int, default=64)
    parser.add_argument("--baseline-discount", type=float, default=0.99)
    parser.add_argument("--baseline-goal-horizon", type=int, default=64)
    parser.add_argument("--baseline-planner-horizon", type=int, default=16)
    parser.add_argument("--baseline-updates-per-step", type=int, default=1)
    parser.add_argument("--baseline-update-every-steps", type=int, default=16)
    parser.add_argument("--baseline-learning-rate", type=float, default=3e-4)
    parser.add_argument("--baseline-entropy-coeff", type=float, default=0.1)
    parser.add_argument("--baseline-target-entropy-fraction", type=float, default=0.5)
    parser.add_argument("--baseline-logsumexp-coeff", type=float, default=0.1)
    parser.add_argument("--baseline-landmark-loss-coeff", type=float, default=1.0)
    parser.add_argument("--baseline-hidden-dim", type=int, default=256)
    parser.add_argument("--baseline-repr-dim", type=int, default=64)
    parser.add_argument("--baseline-landmark-count", type=int, default=50)
    parser.add_argument("--baseline-landmark-candidates", type=int, default=1000)
    parser.add_argument("--baseline-landmark-neighbors", type=int, default=8)
    parser.add_argument("--baseline-landmark-local-horizon", type=float, default=16.0)
    parser.add_argument("--baseline-landmark-edge-horizon", type=float, default=64.0)
    parser.add_argument("--baseline-planner-rebuild-frames", type=int, default=100_000)
    parser.add_argument("--baseline-checkpoint-frames", type=int, default=1_000_000)
    parser.add_argument("--baseline-torch-threads", type=int, default=8)
    return parser


def parse_args(argv=None):
    baseline_ns, remaining = _baseline_parser().parse_known_args(argv)
    # Importing the runtime parser lazily keeps unit tests independent of the
    # NEMO2-only DMLab installation.
    from sf_working_directories.IntrMotiv.dmlab.train_hipposlam import parse_dmlab_args

    cfg = parse_dmlab_args(remaining)
    values = {key[len("baseline_") :]: value for key, value in vars(baseline_ns).items()}
    baseline = BaselineConfig(**values)
    _validate_baseline_config(baseline)
    if cfg.encoder_conv_architecture != "layer2_resnet18":
        raise ValueError("off-policy baselines require --encoder_conv_architecture=layer2_resnet18")
    if bool(cfg.with_pos_obs):
        raise ValueError("ground-truth pose must not be part of the policy observation")
    if not bool(cfg.online_spatial_telemetry) or not bool(cfg.exploration_coverage_telemetry):
        raise ValueError("pose and coverage telemetry are required for evaluation (but are excluded from policy input)")
    return baseline, cfg


def _validate_baseline_config(baseline: BaselineConfig) -> None:
    if (
        min(
            baseline.total_frames,
            baseline.replay_capacity,
            baseline.replay_min,
            baseline.replay_segment_steps,
            baseline.batch_size,
            baseline.max_future,
            baseline.goal_horizon,
            baseline.planner_horizon,
            baseline.update_every_steps,
            baseline.planner_rebuild_frames,
            baseline.checkpoint_frames,
        )
        <= 0
    ):
        raise ValueError("frame, replay, batch, horizon, and cadence settings must be positive")
    if baseline.replay_segment_steps < baseline.max_future:
        raise ValueError("replay_segment_steps must be at least max_future")
    if baseline.replay_min > baseline.replay_capacity:
        raise ValueError("replay_min must not exceed replay_capacity")
    if not 0.0 < baseline.target_entropy_fraction <= 1.0:
        raise ValueError("target_entropy_fraction must be in (0, 1]")
    if baseline.landmark_loss_coeff < 0.0:
        raise ValueError("landmark_loss_coeff must be nonnegative")
    if baseline.landmark_local_horizon <= 0.0 or baseline.landmark_edge_horizon <= 0.0:
        raise ValueError("landmark horizons must be positive")


class FrozenVisualFeatures:
    """Exact IntrMotiv frozen layer-2 trunk plus fixed map-number encoding."""

    def __init__(self, cfg, observation_space, device: torch.device) -> None:
        from sf_working_directories.IntrMotiv.dmlab.custom_encoder import ResNet18Layer2

        obs_space = observation_space.spaces["obs"]
        shape = tuple(obs_space.shape)
        if shape[0] not in (3, 4):
            if shape[-1] not in (3, 4):
                raise ValueError(f"cannot identify RGB channels in observation shape {shape}")
            shape = (shape[-1], shape[0], shape[1])
        rgb_space = gym.spaces.Box(0, 255, shape=(3, shape[1], shape[2]), dtype=np.uint8)
        self.trunk = ResNet18Layer2(cfg, rgb_space, pretrained=True, fixed=True).to(device).eval()
        self.device = device
        self.number_instruction_coef = float(cfg.number_instruction_coef)
        for parameter in self.trunk.parameters():
            if parameter.requires_grad:
                raise RuntimeError("visual trunk is not frozen")

    def __call__(self, observation: dict) -> np.ndarray:
        batched = {key: np.expand_dims(np.asarray(value), 0) for key, value in observation.items()}
        return self.batch(batched)[0]

    def batch(self, observations: dict) -> np.ndarray:
        """Encode a vector-environment observation batch in one trunk call."""
        image = np.asarray(observations["obs"])
        if image.ndim != 4:
            raise ValueError(f"expected batched images [N,C,H,W] or [N,H,W,C], got {image.shape}")
        if image.shape[1] not in (3, 4):
            if image.shape[-1] not in (3, 4):
                raise ValueError(f"cannot identify RGB channels in batched shape {image.shape}")
            image = np.transpose(image, (0, 3, 1, 2))
        tensor = torch.as_tensor(image[:, :3], device=self.device, dtype=torch.float32) / 255.0
        with torch.no_grad():
            visual = self.trunk(tensor).cpu().numpy()
        batch_size = image.shape[0]
        instruction = np.zeros((batch_size, 3), dtype=np.float32)
        for key, value in observations.items():
            array = np.asarray(value)
            if key not in ("obs", "telemetry_pose", "prev_action") and array.size == batch_size:
                indices = array.reshape(batch_size).astype(np.int64) - 1
                valid = (indices >= 0) & (indices < 3)
                instruction[np.arange(batch_size)[valid], indices[valid]] = self.number_instruction_coef
                break
        return np.concatenate((visual.astype(np.float32), instruction), axis=1)


def _pose(observation: dict) -> np.ndarray:
    pose = np.asarray(observation.get("telemetry_pose"), dtype=np.float32)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise RuntimeError("valid telemetry_pose is required for evaluation")
    return pose


def _tensor(batch: dict[str, np.ndarray], key: str, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(batch[key], device=device)


def _workspace_output(cfg) -> Path:
    # The established Slurm launcher already supplies a unique per-run
    # train_dir; appending the experiment again would create a misleading
    # second nesting level.
    output = Path(cfg.train_dir).resolve()
    try:
        output.relative_to(WORKSPACE_ROOT)
    except ValueError as error:
        raise ValueError(f"training output must be inside {WORKSPACE_ROOT}: {output}") from error
    output.mkdir(parents=True, exist_ok=True)
    return output


def _write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        )
        + "\n"
    )
    os.replace(temporary, path)


def _save_checkpoint(
    path: Path, frames: int, agent, critic_optimizer, actor_optimizer, alpha_optimizer, log_alpha, baseline, cfg
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "schema": "intrmotiv/offpolicy-goal-baseline/v1",
            "frames": int(frames),
            "model": agent.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "alpha_optimizer": alpha_optimizer.state_dict(),
            "log_alpha": log_alpha.detach().cpu(),
            "baseline": asdict(baseline),
            "seed": int(cfg.seed),
            "visual_encoder": "layer2_resnet18_imagenet_frozen",
        },
        temporary,
    )
    os.replace(temporary, path)


def train(argv=None) -> int:
    baseline, cfg = parse_args(argv)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(baseline.torch_threads)
    device = torch.device("cpu" if str(cfg.device).lower() == "cpu" else "cuda")

    from sf_working_directories.IntrMotiv.dmlab.dmlab_env import make_dmlab_env

    env_config = {"env_id": 0, "vector_index": 0, "worker_index": 0}
    env = make_dmlab_env(cfg.env, cfg, env_config, dmlab_level_caches_per_policy=None)
    observation, _ = env.reset()
    feature_extractor = FrozenVisualFeatures(cfg, env.observation_space, device)
    current_feature = feature_extractor(observation)
    feature_dim = len(current_feature)
    agent = ContrastiveGoalAgent(
        feature_dim,
        env.action_space.n,
        hidden_dim=baseline.hidden_dim,
        repr_dim=baseline.repr_dim,
    ).to(device)
    critic_parameters = [parameter for name, parameter in agent.named_parameters() if not name.startswith("actor.")]
    critic_optimizer = torch.optim.Adam(critic_parameters, lr=baseline.learning_rate)
    actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=baseline.learning_rate)
    log_alpha = torch.tensor(np.log(baseline.entropy_coeff), device=device, dtype=torch.float32, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=baseline.learning_rate)
    target_entropy = baseline.target_entropy_fraction * np.log(int(env.action_space.n))
    replay = EpisodeReplay(baseline.replay_capacity, seed=cfg.seed)
    planner = LandmarkPlanner(
        baseline.landmark_count,
        baseline.landmark_candidates,
        baseline.landmark_neighbors,
        baseline.landmark_local_horizon,
        baseline.landmark_edge_horizon,
    )

    output = _workspace_output(cfg)
    _write_json(
        output / "run_config.json",
        {
            "schema": "intrmotiv/offpolicy-goal-baseline/v1",
            "baseline": asdict(baseline),
            "seed": int(cfg.seed),
            "environment": cfg.env,
            "action_count": int(env.action_space.n),
            "action_repeat": int(cfg.env_frameskip),
            "visual_encoder": "layer2_resnet18_imagenet_frozen",
            "policy_pose_input": False,
            "feature_dim": feature_dim,
        },
    )
    # Match the canonical IntrMotiv/Sample Factory event-directory contract so
    # collect-online can consume these standalone off-policy runs unchanged.
    writer = SummaryWriter(str(output / ".summary" / "0"))
    metrics_file = (output / "metrics.jsonl").open("a", buffering=1)

    episode_features = [current_feature]
    episode_poses = [_pose(observation)]
    episode_actions: list[int] = []
    frames = decisions = updates = option_steps = 0
    planner_steps_remaining = 0
    previous_landmark = None
    goal = goal_pose = final_goal = None
    option_attempts = option_successes = 0
    next_checkpoint = baseline.checkpoint_frames
    next_planner_rebuild = baseline.planner_rebuild_frames
    last_log = time.monotonic()

    try:
        while frames < baseline.total_frames:
            ready = replay.size >= baseline.replay_min
            if not ready:
                action = env.action_space.sample()
            else:
                if final_goal is None or option_steps >= baseline.goal_horizon:
                    final_goal, goal_pose = replay.sample_goal()
                    goal = final_goal
                    option_steps = 0
                    planner_steps_remaining = 0
                    previous_landmark = None
                    option_attempts += 1
                if baseline.method == "l3p" and planner_steps_remaining <= 0:
                    decision = planner.plan(
                        current_feature,
                        final_goal,
                        agent,
                        device,
                        previous_landmark=previous_landmark,
                        max_horizon=baseline.planner_horizon,
                    )
                    goal = decision.goal
                    planner_steps_remaining = decision.steps
                    previous_landmark = decision.landmark_index
                state_tensor = torch.as_tensor(current_feature, device=device).unsqueeze(0)
                goal_tensor = torch.as_tensor(goal, device=device).unsqueeze(0)
                agent.eval()
                with torch.no_grad():
                    action = int(agent.policy(state_tensor, goal_tensor).item())

            next_observation, _, terminated, truncated, info = env.step(action)
            step_frames = int(info.get("num_frames", cfg.env_frameskip))
            frames += step_frames
            decisions += 1
            option_steps += 1
            if baseline.method == "l3p":
                planner_steps_remaining -= 1
            done = bool(terminated or truncated)
            # DMLab does not expose a fresh terminal image. Its compatibility
            # path returns the cached observation, which PixelFormatChwWrapper
            # would otherwise transpose a second time. Retain the last valid
            # feature/pose as the absorbing terminal sample.
            if done:
                next_feature = current_feature.copy()
                next_pose = episode_poses[-1].copy()
            else:
                next_feature = feature_extractor(next_observation)
                next_pose = _pose(next_observation)
            episode_actions.append(action)
            episode_features.append(next_feature)
            episode_poses.append(next_pose)
            current_feature = next_feature

            if ready and goal_pose is not None:
                distance = float(np.linalg.norm(episode_poses[-1][:2] - goal_pose[:2]))
                if distance <= float(cfg.exploration_coverage_grid_size):
                    option_successes += 1
                    final_goal = None

            if done:
                replay.add(np.asarray(episode_features), np.asarray(episode_actions), np.asarray(episode_poses))
                for key, value in info.get("episode_extra_stats", {}).items():
                    writer.add_scalar(key, float(value), frames)
                observation, _ = env.reset()
                current_feature = feature_extractor(observation)
                episode_features = [current_feature]
                episode_poses = [_pose(observation)]
                episode_actions = []
                final_goal = goal = goal_pose = None
                planner_steps_remaining = 0
                previous_landmark = None
            elif len(episode_actions) >= baseline.replay_segment_steps:
                # Long DMLab episodes otherwise delay all learning until the
                # first timeout. Bounded trajectory segments retain valid
                # future-goal ordering while making replay available online.
                replay.add(
                    np.asarray(episode_features),
                    np.asarray(episode_actions),
                    np.asarray(episode_poses),
                )
                episode_features = [current_feature.copy()]
                episode_poses = [next_pose.copy()]
                episode_actions = []

            for key, value in info.get("periodic_stats", {}).items():
                writer.add_scalar(key, float(value), frames)

            if replay.size >= baseline.replay_min and decisions % baseline.update_every_steps == 0:
                for _ in range(baseline.updates_per_step):
                    batch = replay.sample(baseline.batch_size, baseline.max_future, baseline.discount)
                    agent.train()
                    critic_optimizer.zero_grad(set_to_none=True)
                    actor_optimizer.zero_grad(set_to_none=True)
                    critic_loss, actor_loss, train_metrics = agent.losses(
                        _tensor(batch, "state", device),
                        _tensor(batch, "action", device),
                        _tensor(batch, "future_goal", device),
                        _tensor(batch, "offset", device),
                        _tensor(batch, "random_goal", device),
                        entropy_coeff=float(log_alpha.exp().detach()),
                        logsumexp_coeff=baseline.logsumexp_coeff,
                        landmark_loss_coeff=(baseline.landmark_loss_coeff if baseline.method == "l3p" else 0.0),
                        max_future=baseline.max_future,
                    )
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic_parameters, 10.0)
                    critic_optimizer.step()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 10.0)
                    actor_optimizer.step()
                    alpha_optimizer.zero_grad(set_to_none=True)
                    observed_entropy = torch.as_tensor(train_metrics["policy_entropy"], device=device)
                    alpha_loss = log_alpha.exp() * (observed_entropy - target_entropy)
                    alpha_loss.backward()
                    alpha_optimizer.step()
                    train_metrics["entropy_alpha"] = float(log_alpha.exp().detach())
                    train_metrics["alpha_loss"] = float(alpha_loss.detach())
                    updates += 1
            if replay.size >= baseline.replay_min and baseline.method == "l3p" and frames >= next_planner_rebuild:
                planner.rebuild(replay, agent, device)
                next_planner_rebuild += baseline.planner_rebuild_frames

            if time.monotonic() - last_log >= 30.0:
                record = {
                    "frames": frames,
                    "updates": updates,
                    "replay_size": replay.size,
                    "option_attempts": option_attempts,
                    "option_success_rate": option_successes / max(option_attempts, 1),
                }
                if replay.size >= baseline.replay_min:
                    record.update(train_metrics)
                if baseline.method == "l3p":
                    record.update(
                        planner_rebuilds=planner.rebuild_count,
                        planner_finite_edges=planner.finite_edges,
                        planner_queries=planner.subgoal_queries,
                        planner_landmark_fraction=(planner.landmark_subgoals / max(planner.subgoal_queries, 1)),
                        planner_reachable_pair_fraction=planner.reachable_pair_fraction,
                        planner_unreachable_queries=planner.unreachable_queries,
                        planner_repeat_avoided=planner.repeat_avoided,
                        planner_mean_commitment=planner.mean_commitment,
                    )
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                for key, value in record.items():
                    if key != "frames":
                        writer.add_scalar(f"offpolicy/{key}", float(value), frames)
                writer.add_scalar("train/env_steps", frames, frames)
                writer.flush()
                last_log = time.monotonic()

            if frames >= next_checkpoint:
                _save_checkpoint(
                    output / f"checkpoint_{frames:012d}.pt",
                    frames,
                    agent,
                    critic_optimizer,
                    actor_optimizer,
                    alpha_optimizer,
                    log_alpha,
                    baseline,
                    cfg,
                )
                next_checkpoint += baseline.checkpoint_frames
    finally:
        _save_checkpoint(
            output / f"checkpoint_{frames:012d}_final.pt",
            frames,
            agent,
            critic_optimizer,
            actor_optimizer,
            alpha_optimizer,
            log_alpha,
            baseline,
            cfg,
        )
        metrics_file.close()
        writer.close()
        env.close()
    return 0


def main() -> None:
    sys.exit(train())


if __name__ == "__main__":
    main()
