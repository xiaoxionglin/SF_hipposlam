"""Frozen source-worker contact probe on a fixed invisible reward site.

Each command uses the same engine reset seed and policy sampling seed. Physical
reward contact, target-DG activation, and terminal pose are separate outcomes.
Debug pose is evaluation-only and is not supplied to the policy network.
"""

import argparse
import csv
import json
from pathlib import Path

import deepmind_lab
import numpy as np
import torch

from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sf_working_directories.IntrMotiv.evaluation.place_fields import load_policy_env
from sf_working_directories.IntrMotiv.evaluation.target_control_interventions import condition_for_target


def command_for_condition(condition, target, seed, capacity):
    if condition == "nominated":
        return target
    if condition == "wrong":
        return (target + 1) % capacity
    alternatives = [goal for goal in range(capacity) if goal != target]
    return alternatives[np.random.default_rng(seed + 901).integers(len(alternatives))]


def episode(actor, cfg, env, env_info, *, seed, command, target, cell_x, cell_y, horizon):
    base = env.unwrapped
    base.seed(seed)
    obs, _ = env.reset()
    start = np.asarray(obs["pos"][0], dtype=float)
    assert not (cell_x <= start[0] < cell_x + 100 and cell_y <= start[1] < cell_y + 100)
    actual_seed = int(base.last_reset_seed)
    state = torch.zeros(1, get_rnn_size(cfg), dtype=torch.float32)
    rng = torch.Generator().manual_seed(seed + 17011)
    dg_hits = 0
    reward = 0.0
    final = start.copy()
    elapsed = 0
    terminal = False
    with torch.no_grad():
        for elapsed in range(1, horizon + 1):
            head = actor.forward_head(prepare_and_normalize_obs(actor, obs))
            dg_hits += int(bool(head[0, target] > 0))
            if getattr(cfg, "dg_goal_input", "none") == "write":
                goal = head.new_zeros((1, int(cfg.Hippo_n_feature)))
                goal[:, command] = 1
                head = torch.cat((head, goal), dim=-1)
            core, state = actor.forward_core(head, state)
            conditioned = condition_for_target(actor, core, -1, command)
            logits = actor.forward_tail(conditioned, values_only=False, sample_actions=False)["action_logits"]
            action = torch.multinomial(logits.softmax(-1), 1, generator=rng)
            obs, step_reward, terminated, truncated, _ = env.step(preprocess_actions(env_info, action))
            reward += float(torch.as_tensor(step_reward).reshape(-1)[0])
            final = np.asarray(obs["pos"][0], dtype=float).copy()
            if bool(make_dones(terminated, truncated)[0]):
                terminal = True
                break
    return dict(
        requested_seed=seed, engine_seed=actual_seed, command=command,
        reward_contact=reward > 0, external_reward=reward,
        target_dg_active_decisions=dg_hits, elapsed_decisions=elapsed,
        terminal=terminal, start_x=float(start[0]), start_y=float(start[1]),
        terminal_x=float(final[0]), terminal_y=float(final[1]),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--runfiles", type=Path, required=True)
    p.add_argument("--site", choices=["dg50", "dg51"], required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--horizon", type=int, default=64, help="Source option timeout in decisions")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    deepmind_lab.set_runfiles_path(str(args.runfiles.resolve(strict=True)))
    cfg, original_env, _, actor, checkpoint, _ = load_policy_env(
        args.run_dir, args.horizon, False, 0, args.checkpoint
    )
    original_env.close()
    cfg.env = "openfield_map2_fixed_reward_" + args.site
    cfg.dmlab_runfiles_path = str(args.runfiles.resolve(strict=True))
    cfg.dmlab_use_level_cache = False
    cfg.with_pos_obs = True
    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=None
    )
    env_info = extract_env_info(env, cfg)
    if hasattr(env.unwrapped, "reset_on_init"):
        env.unwrapped.reset_on_init = False
    target = 50 if args.site == "dg50" else 51
    x = 300 if args.site == "dg50" else 200
    rows = []
    try:
        for trial in range(args.trials):
            seed = 51000 + trial
            for condition in ("nominated", "wrong", "shuffled"):
                command = command_for_condition(condition, target, seed, int(cfg.Hippo_n_feature))
                row = episode(
                    actor, cfg, env, env_info, seed=seed, command=command,
                    target=target, cell_x=x, cell_y=1900, horizon=args.horizon,
                )
                row.update(site=args.site, condition=condition, target_id=target, checkpoint=str(checkpoint))
                rows.append(row)
            assert len({row["engine_seed"] for row in rows[-3:]}) == 1
            print(args.site, trial + 1, "matched trials", flush=True)
    finally:
        env.close()
    output = args.out_dir / f"{args.site}_zero_shot.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(checkpoint), "site": args.site, "trials_per_condition": args.trials,
        "option_horizon_decisions": args.horizon,
        "successes": {name: sum(row["reward_contact"] for row in rows if row["condition"] == name)
                      for name in ("nominated", "wrong", "shuffled")},
    }
    (args.out_dir / f"{args.site}_zero_shot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
