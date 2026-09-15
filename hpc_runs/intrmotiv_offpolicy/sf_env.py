"""Physical stream identity outside policy inputs, before SF's autoreset."""

import gymnasium as gym
import numpy as np


class StreamIdentity(gym.Wrapper):
    def __init__(self, env, stream, seed, frameskip):
        super().__init__(env)
        self.stream, self.initial_seed, self.frameskip = stream, seed, frameskip
        self.episode = -1
        self.index = self.serial = 0
        self.observation_space = gym.spaces.Dict(
            dict(
                env.observation_space.spaces,
                ddqn_identity=gym.spaces.Box(0, np.iinfo(np.int64).max, (4,), dtype=np.int64),
            )
        )

    def annotate(self, obs):
        return dict(obs, ddqn_identity=np.array([self.stream, self.episode, self.index, self.serial], dtype=np.int64))

    def reset(self, **kwargs):
        if self.episode < 0:
            kwargs["seed"] = self.initial_seed
        obs, info = self.env.reset(**kwargs)
        self.episode += 1
        self.index = 0
        return self.annotate(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if int(info.get("num_frames", self.frameskip)) != self.frameskip:
            raise ValueError("native adapter requires fixed frames per decision")
        if (terminated or truncated) and info.get("intrmotiv_final_observation_valid", False):
            raise ValueError("certified terminal images require an extended SF transport contract")
        self.index += 1
        self.serial += 1
        return self.annotate(obs), reward, terminated, truncated, info


def make_native_env(env_name, cfg, env_config, render_mode):
    from sf_working_directories.IntrMotiv.dmlab.dmlab_env import make_dmlab_env

    from .sf_telemetry import install_sampling_reports

    install_sampling_reports()
    stream = int(env_config.get("env_id", 0)) if env_config else 0
    env = make_dmlab_env(env_name, cfg, env_config, render_mode)
    return StreamIdentity(env, stream, cfg.seed + stream, cfg.env_frameskip)
