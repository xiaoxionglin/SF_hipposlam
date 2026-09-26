"""Evaluate physical reward success from matched, held-out map resets.

This uses the saved policy's own manager or flat controller. Position is read
only for evaluation output; it is never passed as a task cue to the policy.
"""

import argparse
import csv
import json
from pathlib import Path

import deepmind_lab
import numpy as np
import torch

from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sf_working_directories.IntrMotiv.evaluation.place_fields import load_policy_env

CUED_REWARD_SITES = ((350, 1950), (250, 1950), (1550, 1250), (650, 1450), (850, 250))


def start_region(x, y):
    """Coarse, prespecified map quadrant for stratified success reporting."""
    return ("west" if x < 1000 else "east") + "_" + ("south" if y < 1000 else "north")


def evaluate_episode(actor, cfg, env, env_info, seed, target, reward_cell, horizon):
    env.unwrapped.seed(seed)
    obs, _ = env.reset()
    start = np.asarray(obs["pos"][0], dtype=float).copy()
    instruction = int(np.asarray(obs["INSTR"]).reshape(-1)[0])
    reward_x, reward_y = CUED_REWARD_SITES[instruction - 1] if reward_cell is None else reward_cell
    assert not (reward_x - 50 <= start[0] < reward_x + 50 and reward_y - 50 <= start[1] < reward_y + 50)
    engine_seed = int(env.unwrapped.last_reset_seed)
    state = torch.zeros(1, get_rnn_size(cfg), dtype=torch.float32)
    policy_seed = seed + 17011
    torch.manual_seed(policy_seed)
    dg_hits = 0
    total_reward = 0.0
    terminal = False
    end = start.copy()
    with torch.no_grad():
        for decision in range(1, horizon + 1):
            normalized = prepare_and_normalize_obs(actor, obs)
            output = actor(normalized, state)
            activity = getattr(actor.core, "last_dg_activity", None)
            if activity is not None and target is not None:
                dg_hits += int(bool(activity[0, target] > 0))
            action = output["actions"]
            if cfg.eval_deterministic:
                action = argmax_actions(actor.action_distribution())
            if action.ndim == 1:
                action = action.unsqueeze(-1)
            state = output["new_rnn_states"]
            obs, reward, terminated, truncated, _ = env.step(preprocess_actions(env_info, action))
            total_reward += float(torch.as_tensor(reward).reshape(-1)[0])
            end = np.asarray(obs["pos"][0], dtype=float).copy()
            terminal = bool(make_dones(terminated, truncated)[0])
            if terminal:
                break
    return {
        "requested_seed": seed,
        "engine_seed": engine_seed,
        "number_instruction": instruction,
        "reward_center_x": reward_x,
        "reward_center_y": reward_y,
        "policy_seed": policy_seed,
        "start_x": float(start[0]),
        "start_y": float(start[1]),
        "start_region": start_region(*start[:2]),
        "terminal_x": float(end[0]),
        "terminal_y": float(end[1]),
        "physical_success": total_reward > 0,
        "external_reward": total_reward,
        "decisions_to_termination": decision,
        "time_to_reward_seconds": decision * 4 / 60 if total_reward > 0 else "",
        "target_dg_active_decisions": dg_hits,
        "terminal": terminal,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runfiles", type=Path, required=True)
    parser.add_argument("--site", choices=("dg50", "dg51", "cued5"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1800)
    parser.add_argument("--seed-base", type=int, default=61000)
    parser.add_argument("--stochastic", action="store_true", help="Sample reproducibly; default is deterministic")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    deepmind_lab.set_runfiles_path(str(args.runfiles.resolve(strict=True)))
    cfg, bootstrap_env, _, actor, checkpoint, _ = load_policy_env(
        args.run_dir, args.horizon, not args.stochastic, 0, args.checkpoint
    )
    bootstrap_env.close()
    cfg.env = "openfield_map2_cued_reward5" if args.site == "cued5" else "openfield_map2_fixed_reward_" + args.site
    cfg.dmlab_runfiles_path = str(args.runfiles.resolve(strict=True))
    cfg.dmlab_use_level_cache = False
    cfg.with_pos_obs = True
    env = make_env_func_batched(cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=None)
    env_info = extract_env_info(env, cfg)
    if hasattr(env.unwrapped, "reset_on_init"):
        env.unwrapped.reset_on_init = False
    target = None if args.site == "cued5" else (50 if args.site == "dg50" else 51)
    reward_cell = None if args.site == "cued5" else ((350, 1950) if args.site == "dg50" else (250, 1950))
    rows = []
    try:
        for index in range(args.trials):
            row = evaluate_episode(
                actor,
                cfg,
                env,
                env_info,
                args.seed_base + index,
                target,
                reward_cell,
                args.horizon,
            )
            row.update(site=args.site, checkpoint=str(checkpoint))
            rows.append(row)
            print(args.site, index + 1, row["physical_success"], flush=True)
    finally:
        env.close()
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "site": args.site,
        "checkpoint": str(checkpoint),
        "trials": len(rows),
        "successes": sum(row["physical_success"] for row in rows),
        "by_start_region": {
            region: {
                "trials": sum(row["start_region"] == region for row in rows),
                "successes": sum(row["start_region"] == region and row["physical_success"] for row in rows),
            }
            for region in sorted({row["start_region"] for row in rows})
        },
        "by_instruction": {
            str(number): {
                "trials": sum(row["number_instruction"] == number for row in rows),
                "successes": sum(row["number_instruction"] == number and row["physical_success"] for row in rows),
            }
            for number in sorted({row["number_instruction"] for row in rows})
        },
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
