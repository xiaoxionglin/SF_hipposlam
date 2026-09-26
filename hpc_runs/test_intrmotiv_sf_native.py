"""Actual SF runtime smoke fixture; no DMLab or production data required."""

import argparse
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from hpc_runs.intrmotiv_offpolicy.qualify import FixtureDecoder
from hpc_runs.intrmotiv_offpolicy.replay import Observation
from hpc_runs.intrmotiv_offpolicy.sf_buffers import install
from hpc_runs.intrmotiv_offpolicy.sf_env import StreamIdentity
from hpc_runs.intrmotiv_offpolicy.sf_learner import make_native_learner
from hpc_runs.intrmotiv_offpolicy.sf_model import NativeActor
from hpc_runs.intrmotiv_offpolicy.sf_native import build_cfg, ddqn_summary
from hpc_runs.intrmotiv_offpolicy.worker import QWorker


class FixtureParent(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)

    def model_to_device(self, device):
        self.to(device)

    def type_for_input_tensor(self, key):
        return torch.float32


class FixtureFeatures:
    def __call__(self, obs):
        pre = obs["obs"][:, :2]
        return [Observation(p, torch.zeros(1, device=p.device), p > 2.43) for p in pre]


class FixtureActor(NativeActor):
    def __init__(self, counter, lock):
        nn.Module.__init__(self)
        self.parent = FixtureParent().requires_grad_(False)
        self.worker = QWorker(
            FixtureDecoder(9, 16), 2, 1, repeat_width=1, length=3, n_actions=8, batch_independent=True
        )
        self.extractor = FixtureFeatures()
        self.registry = [0, 1]
        self.counter = counter
        self.counter_lock = lock
        self.report = {"fixture": True}
        self.device = torch.device("cpu")
        self.inference_threads = 1


class FixtureFactory:
    def __init__(self, counter, lock):
        self.counter = counter
        self.lock = lock

    def __call__(self, cfg, obs_space, action_space):
        torch.manual_seed(cfg.seed)
        return FixtureActor(self.counter, self.lock)


class FixtureEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Dict(obs=gym.spaces.Box(0, 4, (2,), dtype=np.float32))
        self.action_space = gym.spaces.Discrete(8)
        self.index = 0

    def observation(self):
        return {"obs": np.array([3.0 if self.index % 7 == 3 else 0.0, 3.0 if self.index % 7 == 5 else 0.0], np.float32)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return self.observation(), {}

    def step(self, action):
        self.index += 1
        return self.observation(), 0.0, False, self.index >= 23, {"num_frames": 4}


def fixture_env(name, cfg, env_config, render_mode):
    stream = int(env_config.get("env_id", 0)) if env_config else 0
    return StreamIdentity(FixtureEnv(), stream, cfg.seed + stream, 4)


class NativeContractTest(unittest.TestCase):
    def test_packet_matches_action_readout_and_shared_epsilon_clock(self):
        from sample_factory.algo.utils.multiprocessing_utils import FakeLock

        model = FixtureActor(torch.zeros((), dtype=torch.int64), FakeLock())
        obs = {
            "obs": torch.tensor([[0.0, 0.0], [3.0, 0.0]]),
            "ddqn_identity": torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0]]),
        }
        with torch.no_grad():
            result = model(obs, torch.zeros(2, 9))
            packet = result["ddqn_packet"]
            goals = packet[:, -3].long()
            budgets = packet[:, -2].long()
            expected, _ = model.worker.step(
                model.worker.initial(2), obs["obs"], torch.zeros(2, 1), goals, budgets, torch.zeros(2, dtype=torch.long)
            )
        torch.testing.assert_close(result["action_logits"], expected, rtol=0, atol=0)
        self.assertEqual(int(model.counter), 2)
        self.assertTrue(torch.all(packet[:, -2] == 64))
        self.assertEqual(int(goals[1]), 1)
        self.assertTrue(torch.all(result["new_rnn_states"][:, -2] == 63))

    def test_fresh_child_does_not_inherit_wandb_run_identity(self):
        from sample_factory.utils.attr_dict import AttrDict

        parent = AttrDict(
            Hippo_n_feature=2,
            Hippo_R=1,
            Hippo_L=3,
            wandb_unique_id="parent-id",
            wandb_group="parent-study",
            wandb_tags=["parent"],
        )
        args = SimpleNamespace(
            env="fixture",
            seed=99,
            experiment="child",
            train_dir=Path("/tmp/child"),
            device="cpu",
            num_envs=4,
            total_frames=4096,
            torch_threads=1,
            registry="0,1",
        )
        runtime = SimpleNamespace(
            sf_workers=2,
            sf_splits=2,
            sf_rollout=8,
            sf_batch_size=32,
            sf_async=True,
            sf_serial=False,
            sf_learner_threads=1,
            with_wandb=True,
            wandb_project="IntrMotiv",
        )
        cfg = build_cfg(args, runtime, parent)
        self.assertNotIn("wandb_unique_id", cfg)
        self.assertIsNone(cfg.wandb_group)
        self.assertEqual(cfg.wandb_tags, [])
        self.assertEqual(parent.wandb_unique_id, "parent-id")

    def test_existing_entrypoint_dispatches_to_sf(self):
        from unittest.mock import patch

        from hpc_runs.intrmotiv_offpolicy.train import train

        with patch("hpc_runs.intrmotiv_offpolicy.sf_native.main", return_value=0) as main:
            args = ["--execution-backend=sample_factory", "--sf-workers=16"]
            self.assertEqual(train(args), 0)
            main.assert_called_once_with(args)

    def test_sf_buffer_extension_is_opt_in(self):
        from sample_factory.algo.utils.shared_buffers import policy_output_shapes

        cfg = SimpleNamespace(double_value=False)
        standard = policy_output_shapes(cfg, 1, 8)
        install()
        from sample_factory.algo.utils.shared_buffers import policy_output_shapes

        self.assertEqual(standard, policy_output_shapes(cfg, 1, 8))
        cfg.ddqn_packet_width = 8
        self.assertEqual(policy_output_shapes(cfg, 1, 8)[-1], ("ddqn_packet", [8]))

    def test_environment_identity_survives_physical_reset(self):
        env = StreamIdentity(FixtureEnv(), 2, 99, 4)
        obs, _ = env.reset()
        self.assertEqual(obs["ddqn_identity"].tolist(), [2, 0, 0, 0])
        env.step(0)
        obs, _ = env.reset()
        self.assertEqual(obs["ddqn_identity"].tolist(), [2, 1, 0, 1])


def smoke(output, serial, with_wandb=False):
    from sample_factory.algo.utils.model_context import global_learner_factory, global_model_factory
    from sample_factory.algo.utils.multiprocessing_utils import get_mp_ctx
    from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
    from sample_factory.envs.env_utils import register_env
    from sample_factory.train import make_runner

    register_env("intrmotiv_sf_fixture", fixture_env)
    cli = ["--env=intrmotiv_sf_fixture", "--experiment=smoke", "--device=cpu"]
    parser, _ = parse_sf_args(cli)
    from sample_factory.utils.attr_dict import AttrDict

    parent_cfg = AttrDict(vars(parse_full_cfg(parser, cli)))
    parent_cfg.env_frameskip = 4
    parent_cfg.Hippo_n_feature = 2
    parent_cfg.Hippo_R = 1
    parent_cfg.Hippo_L = 3
    args = SimpleNamespace(
        env="intrmotiv_sf_fixture",
        seed=99,
        experiment=output.name,
        train_dir=output.parent,
        device="cpu",
        num_envs=4,
        total_frames=4096,
        torch_threads=1,
        registry="0,1",
        target_period=3,
        replay_capacity=2000,
        learning_start=32,
        learner_execution="batched",
        her_fraction=0.8,
        decisions_per_update=64,
        td_positions_per_update=64,
    )
    runtime = SimpleNamespace(
        sf_workers=2,
        sf_splits=1 if serial else 2,
        sf_rollout=8,
        sf_batch_size=32,
        sf_async=not serial,
        sf_serial=serial,
        sf_learner_threads=1,
        with_wandb=with_wandb,
        wandb_project="IntrMotiv",
    )
    cfg = build_cfg(args, runtime, parent_cfg)
    cfg.ddqn_packet_width = 8
    cfg.ddqn_parent = {"fixture": True}
    cfg.heartbeat_interval = 5
    cfg.heartbeat_reporting_interval = 10
    cfg.heartbeat_timeout = 30
    cfg.train_for_seconds = 60
    counter = torch.zeros((), dtype=torch.int64).share_memory_()
    global_model_factory().register_actor_critic_factory(FixtureFactory(counter, get_mp_ctx(False).Lock()))
    global_learner_factory().register_learner_factory(make_native_learner)
    cfg, runner = make_runner(cfg)
    runner.policy_msg_handlers["ddqn_metrics"] = [ddqn_summary]
    status = runner.init()
    if status == 0:
        status = runner.run()
    if status != 0:
        raise RuntimeError(f"SF smoke failed: {status}")
    gate = json.loads((output / "runtime_gate.json").read_text())
    assert gate["frames"] == 4096, gate
    assert gate["valid_loss_positions"] == gate["updates"] * 64, gate
    assert gate["update_debt"] == 0, gate
    assert gate["transport_received"] == gate["transport_emitted"] + gate["transport_pending"], gate
    assert gate["frozen_reference_unchanged"] and gate["invalid_final_exclusions"] > 0, gate
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    import sys

    if "--smoke-output" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--smoke-output", type=Path, required=True)
        parser.add_argument("--serial", action="store_true")
        parser.add_argument("--with-wandb", action="store_true")
        args = parser.parse_args()
        smoke(args.smoke_output, args.serial, args.with_wandb)
    else:
        unittest.main()
