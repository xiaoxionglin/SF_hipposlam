"""Matched complete-episode coverage for a frozen policy or uniform actions.

Called by the existing manifest-driven place-field worker. Reset and action RNG
seeds are explicit; episode metrics come from the training coverage wrapper.
"""

import copy
import json
from pathlib import Path

import numpy as np
import torch

from hpc_runs.intrmotiv_study.geometry import AccessibleCoverage, geometry_from_config
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict


def evaluate_episodes(
    cfg,
    actor,
    env_info,
    output: Path,
    *,
    episodes=100,
    random_actions=False,
    reset_seed_start=51000,
    action_seed_start=61000,
):
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    record = geometry_from_config(cfg)
    if record is None:
        raise ValueError("Accessible-space episode evaluation requires verified geometry")
    # The training cache chooses unused episode seeds. Matched evaluation must
    # use only the declared reset RNG, as the intervention evaluator does.
    cfg = copy.copy(cfg)
    cfg.dmlab_use_level_cache = False
    before = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    rows = []
    curves = []
    with torch.no_grad():
        for episode in range(episodes):
            # Recreate the native engine to eliminate renderer/episode-history
            # dependence, as in the matched-command evaluator.
            env = make_env_func_batched(
                cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=None
            )
            try:
                env.unwrapped.reset_on_init = False
                env.unwrapped.seed(reset_seed_start + episode)
                torch.manual_seed(action_seed_start + episode)
                rng = np.random.default_rng(action_seed_start + episode)
                obs, _ = env.reset()
                state = torch.zeros((1, get_rnn_size(cfg)), device=next(actor.parameters()).device)
                tracker = AccessibleCoverage(record)
                curve = []
                reward_sum = 0.0
                for decision in range(1802):
                    if random_actions:
                        action = torch.tensor([[rng.integers(actor.action_space.n)]])
                    else:
                        result = actor(prepare_and_normalize_obs(actor, obs), state)
                        state = result["new_rnn_states"]
                        action = result["actions"].reshape(1, -1)
                    obs, reward, term, trunc, infos = env.step(preprocess_actions(env_info, action))
                    reward_sum += float(reward[0])
                    done = bool(make_dones(term, trunc)[0])
                    info = infos[0] if isinstance(infos, (list, tuple)) else infos
                    if done:
                        stats = info.get("episode_extra_stats", {})
                        # Coverage stats are namespaced by the configured environment.
                        # Keep this evaluator reusable across every geometry-backed map.
                        prefix = f"z_00_{cfg.env}_"
                        metrics = {
                            key: float(stats[prefix + key])
                            for key in (
                                "accessible_coverage_auc",
                                "accessible_coverage_fraction",
                                "coverage_unique_cells",
                                "geometry_invalid_pose_steps",
                            )
                        }
                        curve.append(metrics["accessible_coverage_fraction"])
                        if reward_sum != 0:
                            raise RuntimeError("External reward in reward-free probe")
                        rows.append(
                            dict(
                                episode=episode,
                                reset_seed=reset_seed_start + episode,
                                action_seed=action_seed_start + episode,
                                decisions=decision + 1,
                                external_reward=reward_sum,
                                **metrics,
                            )
                        )
                        break
                    tracker.step(env.unwrapped.last_debug_position)
                    curve.append(tracker.metrics()["accessible_coverage_fraction"])
                else:
                    raise RuntimeError("Episode exceeded the 120-second protocol")
                curves.append(curve)
            finally:
                env.close()
    after = actor.state_dict()
    changed = [k for k, v in before.items() if not torch.equal(v, after[k].detach().cpu())]
    if changed:
        raise RuntimeError(f"Frozen evaluation changed model/graph state: {changed}")
    output.mkdir(parents=True, exist_ok=True)
    label = "uniform_random" if random_actions else "policy"
    payload = dict(
        schema="intrmotiv/episode-coverage/v1",
        geometry_sha256=record["sha256"],
        map_seed=record["map_seed"],
        wall_removal_probability=record["wall_removal_probability"],
        policy=label,
        episodes=rows,
        coverage_curves=curves,
        mean_accessible_coverage_auc=float(np.mean([r["accessible_coverage_auc"] for r in rows])),
        frozen_state_verified=True,
    )
    (output / f"{label}_episode_coverage.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
