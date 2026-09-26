"""Standalone fixed-reference DDQN/HER diagnostic using the source environment.

The graph is deliberately absent from replay and the local diagnostic. Runtime
promotion requires independently commanded evaluation, not declining TD loss.
"""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .batch import PositionBatcher, learn_batch, prefix_memory
from .checkpoint import convert_actor, state_hash
from .features import FrozenParentFeatures
from .replay import SequenceReplay, Transition
from .telemetry import Coverage
from .terminal import certified_successor, vector_final_info
from .worker import DoubleDQNLearner, QWorker


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--parent-manifest", type=Path, required=True)
    p.add_argument("--parent-run", required=True)
    p.add_argument("--train_dir", type=Path, required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--her-fraction", type=float, choices=(0.0, 0.8), required=True)
    p.add_argument("--total-frames", type=int, default=5000000)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--torch-threads", type=int, default=8)
    p.add_argument("--registry", default="1,4,11")
    p.add_argument("--replay-capacity", type=int, default=200000)
    p.add_argument("--learning-start", type=int, default=16384)
    p.add_argument("--execution-backend", choices=("standalone", "sample_factory"), default="standalone")
    p.add_argument("--learner-execution", choices=("reference", "batched"), default="reference")
    p.add_argument("--target-period", type=int, default=100)  # optimizer updates
    p.add_argument("--decisions-per-update", type=int, default=64)  # accepted aggregate decisions
    p.add_argument("--td-positions-per-update", type=int, default=256)
    p.add_argument("--milestones", default="0,250000,1000000,2500000,5000000")
    p.add_argument("--train_for_env_steps", type=int)  # Sample Factory launcher compatibility
    p.add_argument("--env", default="openfield_map2_fixed_loc3_fixedlength_noreward")
    args = p.parse_args(argv)
    if (
        min(
            args.num_envs,
            args.total_frames,
            args.torch_threads,
            args.learning_start,
            args.target_period,
            args.decisions_per_update,
            args.td_positions_per_update,
        )
        < 1
    ):
        p.error("positive runtime budgets are required")
    return args


def json_write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n"
    )
    os.replace(tmp, path)


def save_checkpoint(output, learner, replay, frames, decisions, config, reference_hash):
    path = output / "checkpoint_p0" / f"checkpoint_{learner.updates:09d}_{frames}.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    # Replay is intentionally omitted here: this is explicitly a warm-restart
    # artifact, not an exact-resume checkpoint. Resume is not silently accepted.
    torch.save(
        dict(
            schema=QWorker.schema,
            model=learner.online.state_dict(),
            target=learner.target.state_dict(),
            optimizer=learner.optimizer.state_dict(),
            frames=frames,
            decisions=decisions,
            updates=learner.updates,
            config=json.loads(json.dumps(config, default=str)),
            reference_hash=reference_hash,
            replay_schema=replay.schema,
            replay_rng=replay.rng.bit_generator.state,
            replay_insertions=replay.insertions,
            restart_kind="warm_restart_requires_refill",
            environment_state_restored=False,
        ),
        temporary,
    )
    os.replace(temporary, path)


def train(argv=None):
    backend_parser = argparse.ArgumentParser(add_help=False)
    backend_parser.add_argument("--execution-backend", choices=("standalone", "sample_factory"), default="standalone")
    backend, _ = backend_parser.parse_known_args(argv)
    if backend.execution_backend == "sample_factory":
        from .sf_native import main

        return main(argv)
    args = parse_args(argv)
    output = args.train_dir.resolve()
    output.relative_to(Path("/work/classic/fr_xl1014-train"))
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "ddqn_started.json"
    with lock.open("x") as stream:
        json.dump(dict(seed=args.seed, experiment=args.experiment), stream)
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    parent = next(p for p in json.loads(args.parent_manifest.read_text())["parents"] if p["run"] == args.parent_run)
    path = Path(parent["checkpoint"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != parent["sha256"]:
        raise ValueError("parent checkpoint checksum mismatch")
    run_dir = path.parent.parent.parent
    import gymnasium as gym
    from tensorboardX import SummaryWriter

    from hpc_runs.offpolicy_goal_baselines.train_parallel import DmlabEnvFactory, _same_step_vector_worker, _select_info
    from sample_factory.model.actor_critic import create_actor_critic
    from sample_factory.utils.attr_dict import AttrDict
    from sf_working_directories.IntrMotiv.dmlab.dmlab_env import make_dmlab_env
    from sf_working_directories.IntrMotiv.dmlab.train_hipposlam import register_dmlab_components
    from sf_working_directories.IntrMotiv.evaluation.place_fields import load_checkpoint_dict

    cfg = AttrDict(json.loads((run_dir / "config.json").read_text()))
    if cfg.env != args.env:
        raise ValueError("parent environment differs from the declared study")
    cfg.device = args.device
    cfg.use_jit = False
    cfg.with_pos_obs = False
    cfg.seed = args.seed
    cfg.experiment = args.experiment
    cfg.train_dir = str(output)
    cfg.dmlab_level_cache_path = "/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/dmlab_cache"
    register_dmlab_components()
    # Construct and close a source environment for exact action-vector parity.
    probe = make_dmlab_env(cfg.env, cfg, dict(env_id=0, vector_index=0, worker_index=0), None)
    actor = create_actor_critic(cfg, probe.observation_space, probe.action_space)
    actor.model_to_device(torch.device(args.device))
    checkpoint = load_checkpoint_dict(path, torch.device(args.device))
    actor.load_state_dict(checkpoint["model"], strict=True)
    vectors = probe.unwrapped.action_list
    worker, report = convert_actor(actor, cfg, vectors, args.seed)
    probe.close()
    if cfg.dg_goal_input != "none":
        raise ValueError("write-conditioned collection requires actor rebuild integration; not production-qualified")
    learner = DoubleDQNLearner(
        worker.to(args.device), target_period=args.target_period, execution=args.learner_execution
    )
    extractor = FrozenParentFeatures(actor, exclusive=actor.core.topological_enabled)
    reference_hash = state_hash(actor.encoder)
    registry = [int(x) for x in args.registry.split(",")]
    if not registry or len(set(registry)) != len(registry) or any(g < 0 or g >= cfg.Hippo_n_feature for g in registry):
        raise ValueError("invalid diagnostic registry")
    batcher = PositionBatcher()
    coverage = Coverage(registry)
    replay = SequenceReplay(args.replay_capacity, actor.core.expanded_length, args.seed, reference_hash)
    json_write(output / "conversion.json", report)
    json_write(output / "config.json", dict(cfg))
    json_write(
        output / "run_config.json",
        dict(
            schema="intrmotiv/ddqn-run/v2",
            args=vars(args),
            parent=parent,
            representation="frozen_source",
            exploration="epsilon_greedy_fixed_command",
            control="first_arrival",
            development_registry=registry,
            graph_learning=False,
            pose_input=False,
            source_frames=parent["actual_frames"],
        ),
    )
    env = gym.vector.AsyncVectorEnv(
        [DmlabEnvFactory(cfg, i) for i in range(args.num_envs)],
        context="forkserver",
        shared_memory=True,
        worker=_same_step_vector_worker if int(gym.__version__.split(".")[0]) >= 1 else None,
    )
    writer = SummaryWriter(str(output / ".summary" / "0"))
    metrics_file = (output / "metrics.jsonl").open("w", buffering=1)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    def extract(obs):
        return extractor({k: torch.as_tensor(v, device=device) for k, v in obs.items() if k != "telemetry_pose"})

    def stack(rows, name):
        return torch.stack([getattr(r, name) for r in rows]).to(device)

    obs, _ = env.reset(seed=[args.seed + i for i in range(args.num_envs)])
    current = extract(obs)
    memory = worker.initial(args.num_envs, device)
    episodes = np.zeros(args.num_envs, dtype=int)
    indexes = episodes.copy()
    options = episodes.copy()
    goals = np.zeros(args.num_envs, dtype=int)
    budgets = np.zeros(args.num_envs, dtype=int)

    def select(i):
        candidates = [g for g in registry if not bool(current[i].events[g])]
        if not candidates:
            raise RuntimeError("no nontrivial diagnostic command available")
        goals[i] = rng.choice(candidates)
        budgets[i] = 64
        options[i] += 1

    for i in range(args.num_envs):
        select(i)
    frames = decisions = invalid_final = arrivals = attempts = realized_her = sample_count = 0
    accepted = 0
    milestones = sorted(set(int(x) for x in args.milestones.split(",")))
    valid_positions = original_positions = her_positions = 0
    collection_seconds = learning_seconds = 0.0
    started = time.monotonic()
    last_log = started
    last_metrics = {}
    save_checkpoint(output, learner, replay, 0, 0, vars(args), reference_hash)
    try:
        while frames < args.total_frames:
            collection_started = time.monotonic()
            epsilon = max(0.1, 1 - 0.9 * decisions / 250000.0)
            with torch.no_grad():
                q, memory = worker.step(
                    memory,
                    stack(current, "preactivation"),
                    stack(current, "bypass"),
                    torch.as_tensor(goals, device=device),
                    torch.as_tensor(budgets, device=device),
                    torch.as_tensor(indexes, device=device),
                )
            actions = q.argmax(-1).cpu().numpy()
            explore = rng.random(args.num_envs) < epsilon
            actions[explore] = rng.integers(0, 8, size=int(explore.sum()))
            next_obs, _, terminated, truncated, infos = env.step(actions)
            ended = np.asarray(terminated) | np.asarray(truncated)
            # Encode reset observations as reset observations only. A terminal
            # successor is excluded unless a separate, certified final image exists.
            next_rows = extract(next_obs)
            before = frames
            for i in range(args.num_envs):
                info = _select_info(infos, i, args.num_envs)
                info.update(vector_final_info(infos, i, args.num_envs))
                physical = certified_successor(None, info, bool(ended[i])) if ended[i] else True
                valid = physical is not None
                if ended[i] and not valid:
                    invalid_final += 1
                successor = (
                    extract({k: np.expand_dims(v, 0) for k, v in physical.items()})[0]
                    if ended[i] and valid
                    else next_rows[i] if valid else None
                )
                # Invalid final rows retain boundary metadata and no successor payload.
                replay.append(
                    Transition(
                        i,
                        int(episodes[i]),
                        int(indexes[i]),
                        current[i],
                        successor,
                        int(actions[i]),
                        int(goals[i]),
                        int(options[i]),
                        int(budgets[i]),
                        bool(terminated[i]),
                        bool(truncated[i]),
                        valid,
                        learner.updates,
                    )
                )
                accepted += int(valid)
                hit = valid and bool(successor.events[goals[i]])
                coverage.collect(
                    i, int(episodes[i]), successor, int(goals[i]), hit or budgets[i] <= 1 or bool(ended[i])
                )
                arrivals += int(hit)
                budgets[i] -= 1
                indexes[i] += 1
                current[i] = next_rows[i]
                if ended[i]:
                    episodes[i] += 1
                    indexes[i] = 0
                    memory[i] = 0
                if hit or budgets[i] <= 0 or ended[i]:
                    attempts += 1
                    select(i)
                frames += int(info.get("num_frames", cfg.env_frameskip))
                decisions += 1
            collection_seconds += time.monotonic() - collection_started
            learning_started = time.monotonic()
            due = max(0, (accepted - args.learning_start) // args.decisions_per_update - learner.updates)
            for _ in range(due):
                try:
                    samples = batcher.sample(replay, registry, args.her_fraction, args.td_positions_per_update)
                except ValueError:
                    break
                last_metrics = learn_batch(learner, samples, device)
                coverage.replay(samples)
                sample_count += len(samples)
                realized_her += sum(s["relabeled"] for s in samples)
                valid_positions += last_metrics["valid_loss_positions"]
                her_positions += sum(int(s["mask"].sum()) for s in samples if s["relabeled"])
                original_positions += sum(int(s["mask"].sum()) for s in samples if not s["relabeled"])
            learning_seconds += time.monotonic() - learning_started
            if any(before < m <= frames for m in milestones):
                save_checkpoint(output, learner, replay, frames, decisions, vars(args), reference_hash)
            now = time.monotonic()
            if now - last_log >= 10 or frames >= args.total_frames:
                metrics = dict(
                    frames=frames,
                    decisions=decisions,
                    updates=learner.updates,
                    accepted=accepted,
                    invalid_final=invalid_final,
                    attempts=attempts,
                    arrivals=arrivals,
                    epsilon=epsilon,
                    throughput_fps=frames / (now - started),
                    realized_her_fraction=realized_her / max(1, sample_count),
                    effective_loss_positions_per_decision=valid_positions / max(1, decisions),
                    valid_loss_positions_total=valid_positions,
                    original_loss_positions=original_positions,
                    her_loss_positions=her_positions,
                    collection_seconds=collection_seconds,
                    learning_seconds=learning_seconds,
                    replay_size=len(replay.rows),
                    requested_her_fraction=args.her_fraction,
                    target_period_updates=args.target_period,
                    decisions_per_update=args.decisions_per_update,
                    td_positions_per_update=args.td_positions_per_update,
                    **coverage.metrics(),
                    **last_metrics,
                )
                metrics_file.write(json.dumps(metrics) + "\n")
                for key, value in metrics.items():
                    writer.add_scalar("ddqn/" + key, value, frames)
                writer.add_scalar("train/env_steps", frames, frames)
                print(json.dumps(metrics), flush=True)
                last_log = now
        if state_hash(actor.encoder) != reference_hash:
            raise RuntimeError("frozen reference encoder state changed")
        save_checkpoint(output, learner, replay, frames, decisions, vars(args), reference_hash)
        json_write(
            output / "runtime_gate.json",
            dict(
                frames=frames,
                updates=learner.updates,
                target_copies=learner.updates // learner.target_period,
                invalid_final_exclusions=invalid_final,
                frozen_reference_unchanged=True,
                her_samples=realized_her,
                target_period_updates=args.target_period,
                valid_loss_positions=valid_positions,
                td_positions_per_update=args.td_positions_per_update,
                pending_loss_positions=len(batcher.pending["segment"]) if batcher.pending else 0,
                scientific_qualification="pending_independent_commanded_evaluation",
            ),
        )
    finally:
        env.close()
        writer.close()
        metrics_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(train())
