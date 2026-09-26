"""Run v2 DDQN/HER through SF's native parallel Runner and learner hook."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from .sf_buffers import install
from .train import json_write, parse_args

install()


def ddqn_summary(runner, msg, policy_id):
    metrics = msg["ddqn_metrics"]
    for key, value in metrics.items():
        runner.writers[policy_id].add_scalar("ddqn/" + key, value, metrics["frames"])
    runner.writers[policy_id].add_scalar("train/env_steps", metrics["frames"], metrics["frames"])


def build_cfg(args, runtime, parent_cfg):
    from sample_factory.utils.attr_dict import AttrDict

    cfg = AttrDict(copy.deepcopy(dict(parent_cfg)))
    workers = runtime.sf_workers
    if args.num_envs % workers or args.num_envs // workers % runtime.sf_splits:
        raise ValueError("environment count must divide workers and splits")
    cfg.update(
        env=args.env,
        seed=args.seed,
        experiment=args.experiment,
        train_dir=str(args.train_dir),
        device="gpu" if args.device == "cuda" else "cpu",
        use_jit=False,
        with_pos_obs=False,
        async_rl=runtime.sf_async,
        serial_mode=runtime.sf_serial,
        num_workers=workers,
        num_envs_per_worker=args.num_envs // workers,
        worker_num_splits=runtime.sf_splits,
        policy_workers_per_policy=1,
        num_policies=1,
        batched_sampling=True,
        rollout=runtime.sf_rollout,
        batch_size=runtime.sf_batch_size,
        num_batches_per_epoch=1,
        num_epochs=1,
        num_batches_to_accumulate=2,
        train_for_env_steps=args.total_frames - 1,
        train_for_seconds=int(1e10),
        use_rnn=True,
        rnn_type="gru",
        rnn_num_layers=1,
        actor_critic_share_weights=True,
        rnn_size=int(parent_cfg.Hippo_n_feature) * (int(parent_cfg.Hippo_R) + int(parent_cfg.Hippo_L) - 1) + 3,
        recurrence=runtime.sf_rollout,
        env_gpu_observations=False,
        double_value=False,
        with_pbt=False,
        restart_behavior="restart",
        use_env_info_cache=False,
        decorrelate_envs_on_one_worker=False,
        decorrelate_experience_max_seconds=0,
        save_every_sec=120,
        save_best_every_sec=1000000,
        save_milestones_sec=-1,
        normalize_input=False,
        normalize_returns=False,
        reward_scale=1.0,
        reward_clip=1000.0,
        benchmark=False,
        default_niceness=0,
        ddqn_registry=[int(x) for x in args.registry.split(",")],
        ddqn_telemetry=True,
        online_spatial_telemetry=True,
        exploration_coverage_telemetry=True,
        extra_policy_output_shapes=(),
        ddqn_learner_threads=runtime.sf_learner_threads,
        ddqn_inference_threads=args.torch_threads,
        ddqn_args=json.loads(json.dumps(vars(args), default=str)),
        ddqn_max_pending=8 * max(args.num_envs * runtime.sf_rollout, runtime.sf_batch_size * 2),
        dmlab_level_cache_path="/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/dmlab_cache",
        dmlab_use_level_cache=False,
        with_wandb=runtime.with_wandb,
        wandb_project=runtime.wandb_project,
    )
    if runtime.sf_serial and runtime.sf_async:
        raise ValueError("serial qualification requires --sf-async=false")
    if cfg.batch_size % cfg.rollout:
        raise ValueError("SF batch size must contain complete rollouts")
    # A fresh child must never resume the parent's W&B identity or labels.
    cfg.pop("wandb_unique_id", None)
    if getattr(runtime, "sf_telemetry_snapshot_targets", None):
        cfg.online_spatial_snapshot_targets = runtime.sf_telemetry_snapshot_targets
    cfg.wandb_group = None
    cfg.wandb_tags = []
    cfg.cli_args = {}
    cfg.command_line = " ".join(__import__("sys").argv[1:])
    cfg.wandb_step_metric_namespaces = ("intrmotiv", "ddqn")
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--sf-workers", type=int, default=16)
    p.add_argument("--sf-splits", type=int, default=2)
    p.add_argument("--sf-rollout", type=int, default=32)
    p.add_argument("--sf-batch-size", type=int, default=256)
    p.add_argument("--sf-learner-threads", type=int, default=1)
    from sample_factory.utils.utils import str2bool

    p.add_argument("--sf-async", type=str2bool, default=True)
    p.add_argument("--sf-serial", action="store_true")
    p.add_argument("--sf-telemetry-snapshot-targets", default=None)
    p.add_argument("--with-wandb", action="store_true")
    p.add_argument("--wandb-project", default="IntrMotiv")
    runtime, remaining = p.parse_known_args(argv)
    if not any(x.startswith("--learner-execution") for x in remaining):
        remaining.append("--learner-execution=batched")
    args = parse_args(remaining)
    args.execution_backend = "sample_factory"
    torch.set_num_threads(args.torch_threads)
    if (
        min(
            runtime.sf_workers, runtime.sf_splits, runtime.sf_rollout, runtime.sf_batch_size, runtime.sf_learner_threads
        )
        < 1
    ):
        raise ValueError("positive SF resource counts required")
    output = (args.train_dir / args.experiment).resolve()
    output.relative_to("/work/classic/fr_xl1014-train")
    if output.exists():
        raise ValueError("refusing to resume or overwrite an existing run")
    parent = next(p for p in json.loads(args.parent_manifest.read_text())["parents"] if p["run"] == args.parent_run)
    checkpoint = Path(parent["checkpoint"])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != parent["sha256"]:
        raise ValueError("parent checksum mismatch")
    from sample_factory.algo.utils.model_context import global_learner_factory, global_model_factory
    from sample_factory.algo.utils.multiprocessing_utils import get_mp_ctx
    from sample_factory.envs.env_utils import register_env
    from sample_factory.train import make_runner
    from sample_factory.utils.attr_dict import AttrDict
    from sf_working_directories.IntrMotiv.dmlab.dmlab_env import make_dmlab_env
    from sf_working_directories.IntrMotiv.dmlab.train_hipposlam import register_dmlab_components

    from .sf_env import make_native_env
    from .sf_learner import make_native_learner
    from .sf_model import NativeModelFactory

    parent_cfg = AttrDict(json.loads((checkpoint.parent.parent.parent / "config.json").read_text()))
    if parent_cfg.env != args.env or parent_cfg.dg_goal_input != "none":
        raise ValueError("requires matching environment and frozen goal-independent writes")
    parent_cfg.use_jit = False
    parent_cfg.device = args.device
    parent_cfg.with_pos_obs = False
    cfg = build_cfg(args, runtime, parent_cfg)
    cfg.ddqn_parent = parent
    register_dmlab_components()
    parent_factory = global_model_factory().make_actor_critic_func
    probe = make_dmlab_env(cfg.env, cfg, dict(env_id=0, vector_index=0, worker_index=0), None)
    from .checkpoint import validate_parent

    validate_parent(parent_cfg, probe.unwrapped.action_list)
    vectors = probe.unwrapped.action_list
    # Decoder layout was qualified by conversion; determine bypass width from source core.
    source = parent_factory(parent_cfg, probe.observation_space, probe.action_space)
    cfg.ddqn_packet_width = 2 * int(parent_cfg.Hippo_n_feature) + source.core.bypass_size + 3
    if (
        not cfg.ddqn_registry
        or len(set(cfg.ddqn_registry)) != len(cfg.ddqn_registry)
        or any(g < 0 or g >= parent_cfg.Hippo_n_feature for g in cfg.ddqn_registry)
    ):
        raise ValueError("invalid command registry")
    probe.close()
    del source
    # Published weights never overwrite this independent shared exploration clock.
    counter = torch.zeros((), dtype=torch.int64).share_memory_()
    counter_lock = get_mp_ctx(False).Lock()
    global_model_factory().register_actor_critic_factory(
        NativeModelFactory(parent_factory, parent_cfg, str(checkpoint), vectors, counter, counter_lock)
    )
    global_learner_factory().register_learner_factory(make_native_learner)
    register_env(cfg.env, make_native_env)
    cfg, runner = make_runner(cfg)
    from sf_working_directories.IntrMotiv.dmlab.train_hipposlam import register_msg_handlers

    register_msg_handlers(cfg, runner)
    runner.policy_msg_handlers["ddqn_metrics"] = [ddqn_summary]
    status = runner.init()
    if status == 0:
        status = runner.run()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
