"""Common observation trajectories for policy-independent representation tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.model_utils import get_rnn_size


def record_observation(records, obs, previous_actions):
    """Store an action-history superset without changing the policy's input."""
    for key, value in obs.items():
        if torch.is_tensor(value):
            records["obs_" + key].append(value.detach().cpu().numpy().copy())
    if "prev_action" not in obs:
        records["obs_prev_action"].append(np.asarray(previous_actions, dtype=np.int32).copy())


def previous_actions_after_step(actions, dones, n_actions):
    result = np.asarray(actions, dtype=np.int32).reshape(-1, 1).copy()
    result[np.asarray(dones, dtype=bool).reshape(-1)] = n_actions
    return result


def save_panel(path, records):
    path = Path(path).resolve()
    if not str(path).startswith("/work/classic/fr_xl1014-train/"):
        raise ValueError("Observation panels must live in the allocated workspace")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: np.stack(value) for key, value in records.items()})


def replay_observations(actor, cfg, path, device):
    # NPZ indexing decompresses the whole member: materialize once, not once
    # per observation (which becomes quadratic for a long image trajectory).
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    keys = [key for key in data if key.startswith("obs_")]
    if not keys or "obs_pos" not in keys or "obs_rot" not in keys or "dones" not in data:
        raise ValueError("Panel lacks observations, pose, or episode boundaries")
    count = len(data["dones"])
    if any(len(data[key]) != count for key in keys):
        raise ValueError("Panel arrays must have identical temporal lengths")
    batch = data["dones"].shape[1]
    state = torch.zeros(batch, get_rnn_size(cfg), device=device)
    activity, logits, rows = [], [], []
    episode = np.zeros(batch, dtype=int)
    with torch.no_grad():
        for t in range(count):
            obs = {key[4:]: torch.as_tensor(data[key][t], device=device) for key in keys}
            normalized = prepare_and_normalize_obs(actor, obs)
            head = actor.forward_head(normalized)
            core, state = actor.forward_core(head, state)
            n, e = int(cfg.Hippo_n_feature), int(cfg.Hippo_R + cfg.Hippo_L - 1)
            # Keep snapshots independent of the in-place episode-state reset.
            activity.append(core[:, : n * e : e].cpu().numpy().copy())
            logits.append(actor.encoder.DG_projection.last_pre_threshold_logits.detach().cpu().numpy().copy())
            for agent in range(batch):
                p, r = data["obs_pos"][t, agent], data["obs_rot"][t, agent]
                rows.append(
                    dict(
                        frame=t,
                        agent=agent,
                        x=p[0],
                        y=p[1],
                        z=p[2],
                        rot_x=r[0],
                        rot_y=r[1],
                        rot_z=r[2],
                        num_traj=episode[agent],
                    )
                )
            done = torch.as_tensor(data["dones"][t], device=device).bool()
            state[done] = 0
            episode += data["dones"][t].astype(int)
    return pd.DataFrame(rows), np.concatenate(activity), np.concatenate(logits)
