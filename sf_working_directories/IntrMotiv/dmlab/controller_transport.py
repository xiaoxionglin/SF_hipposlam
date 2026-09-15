"""Original-environment identity and explicitly certified terminal transport."""

import gymnasium as gym
import numpy as np

IDENTITY_KEY = "controller_identity"
FINAL_OBSERVATION = "controller_final_observation"
FINAL_VALID = "controller_final_observation_valid"


def enabled(cfg):
    return getattr(cfg, "controller_learning", "ppo") in ("ddqn", "shadow")


class ControllerIdentity(gym.Wrapper):
    """Do not alter environment seeding, rewards, actions, or reset semantics."""

    def __init__(self, env, stream):
        super().__init__(env)
        self.stream, self.episode, self.index, self.serial = int(stream), -1, 0, 0
        self.observation_space = gym.spaces.Dict(
            dict(
                env.observation_space.spaces,
                controller_identity=gym.spaces.Box(0, np.iinfo(np.int64).max, (4,), dtype=np.int64),
            )
        )

    def annotate(self, obs):
        return dict(
            obs, controller_identity=np.array([self.stream, self.episode, self.index, self.serial], dtype=np.int64)
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode += 1
        self.index = 0
        return self.annotate(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.index += 1
        self.serial += 1
        obs = self.annotate(obs)
        if terminated or truncated:
            # The underlying environment must explicitly certify freshness.
            valid = bool(info.get("intrmotiv_final_observation_valid", False))
            info = dict(info, controller_final_observation_valid=valid)
            if valid:
                info[FINAL_OBSERVATION] = {k: np.array(v, copy=True) for k, v in obs.items()}
        return obs, reward, terminated, truncated, info


def record_terminal(buffer, step, terminated, info):
    """Write all validity fields, including on reused nonterminal buffers."""
    buffer["controller_terminated"][step] = terminated
    valid = bool(info.get(FINAL_VALID, False))
    buffer["controller_final_valid"][step] = valid
    if valid:
        terminal = info[FINAL_OBSERVATION]
        for key in buffer["controller_final_obs"]:
            buffer["controller_final_obs"][key][step] = terminal[key]


def validate_controller_config(cfg):
    if getattr(cfg, "controller_replay_state", "reconstruct") == "stored":
        if getattr(cfg, "dg_goal_input", "none") != "none" or getattr(cfg, "ppo_dg_gradient", "stop") != "stop":
            raise ValueError("Stored-state replay requires goal-independent memory and STOP routing")
    required = {
        "core_name": "BypassSS",
        "dg_context_feedback": "none",
        "hrl_graph_memory": "policy_buffer",
        "hrl_target_timing": "immediate",
        "hrl_exploration_policy": "shared",
        "policy_workers_per_policy": 1,
        "batched_sampling": False,
        "serial_mode": False,
        "with_pbt": False,
        "decorrelate_envs_on_one_worker": False,
        "dg_orthogonal_recruitment": False,
        "hrl_behavior_mode_condition": False,
        "hrl_motion_policy_input": False,
        "hrl_landmark_geometry": "none",
        "hrl_action_path_integration": False,
        "value_bootstrap": False,
        "distance_learning": True,
        "double_value": False,
        "num_epochs": 1,
        "ca3_predictor_shadow": False,
        "hrl_empirical_her": False,
    }
    for key, expected in required.items():
        if getattr(cfg, key, expected) != expected:
            raise ValueError(f"Controller integration requires {key}={expected!r}")
    if cfg.hrl_manager_mode not in ("frontier_direct", "frontier_waypoint"):
        raise ValueError("Controller integration requires a preserved direct/waypoint parent")
    for key in (
        "controller_replay_capacity",
        "controller_td_positions",
        "controller_decisions_per_update",
        "controller_learning_starts",
        "controller_target_updates",
        "controller_epsilon_decay_decisions",
    ):
        if int(getattr(cfg, key)) <= 0:
            raise ValueError(f"{key} must be positive")
    if cfg.controller_her_positions < 0 or cfg.controller_her_loss_coeff < 0:
        raise ValueError("Invalid auxiliary HER budget or coefficient")
    if not 0 <= cfg.controller_epsilon <= 1:
        raise ValueError("Invalid epsilon")
    if cfg.controller_preflight and cfg.train_for_env_steps > 2_000_000:
        raise ValueError("Unqualified controller preflights are limited to 2M frames")


def cached_observation(observation, visual):
    """Exact FP32 frozen features plus raw trainable depth/instruction inputs.

    These selected parents have four-channel RGB+depth observations. RGB can be
    omitted only after the actor computed the actual original frozen trunk.
    """
    result = dict(observation)
    if result["obs"].shape[-3] != 4:
        raise ValueError("Expected original RGB+depth observations")
    result["obs"] = result["obs"][-1:].copy() if isinstance(result["obs"], np.ndarray) else result["obs"][-1:].clone()
    result["controller_visual"] = visual.copy() if isinstance(visual, np.ndarray) else visual.clone()
    return result
