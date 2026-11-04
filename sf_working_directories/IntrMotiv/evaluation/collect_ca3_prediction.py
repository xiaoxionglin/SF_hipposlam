"""Collect frozen canonical CA3, executed actions and reset boundaries for offline prediction.

Reuses the manifest evaluator's policy/environment loader and forward-hook
contract. Raw images and privileged pose never enter the prediction heads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from sf_working_directories.IntrMotiv.evaluation.offline_ca3_prediction import SCHEMA, file_sha256, validate_dataset


def model_digest(actor):
    digest = hashlib.sha256()
    for name, tensor in sorted(actor.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def collect(args):
    # Keep DMLab and model registration out of the fit module and its CPU tests.
    import deepmind_lab

    from sample_factory.algo.sampling.batched_sampling import preprocess_actions
    from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
    from sample_factory.model.model_utils import get_rnn_size
    from sf_working_directories.IntrMotiv.evaluation.place_fields import evaluation_pose, load_policy_env

    output = args.output.resolve()
    workspace = Path(os.environ.get("INTRMOTIV_WORKSPACE_ROOT", "/work/classic/fr_xl1014-corridor-geometry")).resolve()
    if not output.is_relative_to(workspace):
        raise ValueError(f"Collection must be in the allocated workspace: {workspace}")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    source_config = args.run_dir / "config.json"
    config = json.loads(source_config.read_text())
    if config.get("core_name") != "BypassSS" or config.get("hrl_controllable_graph", False):
        raise ValueError("This probe requires graph-free canonical BypassSS")
    if not config.get("transfer_freeze_dg", False) or config.get("dg_context_feedback", "none") != "none":
        raise ValueError("Use the qualification donor with frozen, context-independent DG")
    # The shared loader opens an environment immediately. A private evaluation
    # config prevents even that bootstrap from consuming a training cache.
    output.parent.mkdir(parents=True, exist_ok=True)
    loader = output.parent / (output.stem + "_loader")
    loader.mkdir(exist_ok=False)
    overrides = dict(
        dmlab_use_level_cache=False,
        dmlab_level_cache_path=str(loader / "cache"),
        dmlab_runfiles_path=str(args.runfiles.resolve(strict=True)),
        online_spatial_telemetry=False,
        with_wandb=False,
    )
    config.update(overrides)
    (loader / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    deepmind_lab.set_runfiles_path(str(args.runfiles.resolve(strict=True)))
    cfg, env, info, actor, checkpoint, _ = load_policy_env(loader, args.decisions, False, 0, args.checkpoint)
    if env.num_agents != 1:
        env.close()
        raise ValueError("One environment stream is required for unambiguous temporal alignment")
    actor.eval().requires_grad_(False)
    device = torch.device(args.device)
    actor.model_to_device(device)
    n, expanded = int(cfg.Hippo_n_feature), int(cfg.Hippo_R + cfg.Hippo_L - 1)
    before = model_digest(actor)
    capture = []

    def capture_core(_module, _inputs, result):
        output_core = result[0] if isinstance(result, (tuple, list)) else result
        capture.append(output_core[0, : n * expanded].detach().cpu().numpy().copy())

    hook = actor.core.register_forward_hook(capture_core)
    data = dict(
        ca3=np.empty((args.decisions, n * expanded), np.float32),
        actions=np.empty(args.decisions, np.int64),
        dones_after=np.zeros(args.decisions, bool),
        episode_id=np.empty(args.decisions, np.int64),
        step_in_episode=np.empty(args.decisions, np.int64),
        random_action=np.zeros(args.decisions, bool),
        pose=np.empty((args.decisions, 3), np.float32),
    )
    random = np.random.default_rng(args.seed + 17011)  # independent of policy sampling
    torch.manual_seed(args.seed)
    episode, step, engine_seeds = 0, 0, []
    try:
        env.unwrapped.seed(args.seed)
        obs, _ = env.reset()
        engine_seeds.append(int(getattr(env.unwrapped, "last_reset_seed", args.seed)))
        state = torch.zeros(1, get_rnn_size(cfg), dtype=torch.float32, device=device)
        with torch.no_grad():
            for index in range(args.decisions):
                pose, _ = evaluation_pose(env, obs)
                capture.clear()
                result = actor(prepare_and_normalize_obs(actor, obs), state)
                if len(capture) != 1:
                    raise RuntimeError("Expected exactly one canonical core observation per action")
                data["ca3"][index] = capture[0]  # S_t, before a_t and before o_(t+1)
                data["pose"][index] = pose[0].cpu().numpy()
                action = result["actions"].reshape(1, 1)
                randomized = random.random() < args.random_action_probability
                if randomized:
                    action = torch.full_like(action, int(random.integers(actor.action_space.n)))
                data["actions"][index] = int(action.item())
                data["random_action"][index] = randomized
                data["episode_id"][index], data["step_in_episode"][index] = episode, step
                state = result["new_rnn_states"]
                obs, _, terminated, truncated, _ = env.step(preprocess_actions(info, action))
                done = bool(torch.as_tensor(make_dones(terminated, truncated)).reshape(-1)[0])
                data["dones_after"][index] = done
                if done:
                    state.zero_()
                    episode, step = episode + 1, 0
                    engine_seeds.append(int(getattr(env.unwrapped, "last_reset_seed", -1)))
                else:
                    step += 1
                if (index + 1) % 5000 == 0:
                    print(f"Collected {index + 1}/{args.decisions} decisions, {episode} completed episodes", flush=True)
    finally:
        hook.remove()
        env.close()
    if model_digest(actor) != before:
        raise RuntimeError("Frozen model parameters or buffers changed during collection")
    metadata = dict(
        schema=SCHEMA,
        source_run=str(args.run_dir.resolve()),
        checkpoint=str(checkpoint.resolve()),
        collection_code_sha256=file_sha256(__file__),
        checkpoint_sha256=file_sha256(checkpoint),
        source_config_sha256=file_sha256(source_config),
        evaluation_overrides=overrides,
        model_tensor_sha256=before,
        model_unchanged=True,
        env=cfg.env,
        runfiles=str(args.runfiles.resolve()),
        env_frameskip=int(cfg.env_frameskip),
        n_dg=n,
        expanded=expanded,
        action_count=int(actor.action_space.n),
        decisions=args.decisions,
        behavior="stochastic saved policy with independent uniform-action replacement",
        random_action_probability=args.random_action_probability,
        random_actions=int(data["random_action"].sum()),
        seed=args.seed,
        engine_seeds=engine_seeds,
        episodes=int(np.unique(data["episode_id"]).size),
        completed_episodes=int(data["dones_after"].sum()),
        final_episode_censored=not bool(data["dones_after"][-1]),
        action_counts=np.bincount(data["actions"], minlength=actor.action_space.n).tolist(),
        active_observations=int((data["ca3"][:, ::expanded] > 0).any(1).sum()),
        pose_min=data["pose"].min(0).tolist(),
        pose_max=data["pose"].max(0).tolist(),
        causal_alignment="S_t includes o_t only; action[t] is executed before done_after[t] and o_(t+1)",
    )
    validate_dataset(data, metadata)
    np.savez_compressed(output, **data)
    metadata["dataset_sha256"] = file_sha256(output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runfiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions", type=int, default=60000)
    parser.add_argument("--random-action-probability", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=73001)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.decisions < 2 or not 0 <= args.random_action_probability <= 1 or args.output.suffix != ".npz":
        parser.error("Require at least two decisions, mixture probability in [0,1], and an .npz output")
    collect(args)


if __name__ == "__main__":
    main()
