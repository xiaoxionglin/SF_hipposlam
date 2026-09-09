import argparse
import math
import pathlib
import re
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch

from sample_factory.algo.learning.learner import BaseLearner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.utils import log
from sf_working_directories.jannek.dmlab.train_hipposlam import parse_dmlab_args, register_dmlab_components


XBOUND = (100.0, 2000.0)
YBOUND = (100.0, 2000.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-num-frames", type=int, default=50000)
    parser.add_argument("--grain", type=int, default=19)
    parser.add_argument("--checkpoint-rank", type=int, default=1, help="0=newest, 1=second-newest, etc.")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--save-raw-activations", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def run_label(run_dir: pathlib.Path) -> str:
    if run_dir.parent.name.endswith("_"):
        return f"{run_dir.parent.name.rstrip('_')}__{run_dir.name}"
    return run_dir.name


def cli_for_run(run_dir: pathlib.Path, max_num_frames: int, deterministic: bool):
    return [
        "--algo",
        "APPO",
        "--env",
        "openfield_map2_fixed_loc3_noreward",
        "--train_dir",
        str(run_dir.parent),
        "--experiment",
        run_dir.name,
        "--max_num_frames",
        str(max_num_frames),
        "--num_envs",
        "1",
        "--load_checkpoint_kind",
        "latest",
        "--use_jit",
        "False",
        "--with_pos_obs",
        "True",
        "--no_render",
        "--device",
        "cpu",
        "--eval_deterministic",
        str(bool(deterministic)),
    ]


def checkpoint_sort_key(path: str):
    match = re.search(r"_(\d+)_(\d+)\.pth$", pathlib.Path(path).name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return (0, pathlib.Path(path).stat().st_mtime_ns)


def select_checkpoint(cfg, rank: int):
    checkpoint_dir = BaseLearner.checkpoint_dir(cfg, cfg.policy_index)
    checkpoints = BaseLearner.get_checkpoints(checkpoint_dir, "checkpoint_*")
    checkpoints = sorted(checkpoints, key=checkpoint_sort_key)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    idx = max(0, len(checkpoints) - 1 - rank)
    return checkpoints[idx]


def spatial_information(rate_map, occupancy):
    occ = occupancy.astype(np.float64)
    total_occ = np.nansum(occ)
    if total_occ <= 0:
        return np.nan
    p_occ = occ / total_occ
    mean_rate = np.nansum(rate_map * p_occ)
    if mean_rate <= 0 or not np.isfinite(mean_rate):
        return 0.0
    ratio = rate_map / mean_rate
    valid = (p_occ > 0) & np.isfinite(ratio) & (ratio > 0)
    return float(np.nansum(rate_map[valid] * p_occ[valid] * np.log2(ratio[valid])))


def rollout_dg(run_dir: pathlib.Path, max_num_frames: int, deterministic: bool, checkpoint_rank: int):
    register_dmlab_components()
    cfg = parse_dmlab_args(evaluation=True, argv=cli_for_run(run_dir, max_num_frames, deterministic))
    cfg.cli_args = {
        "algo": "APPO",
        "env": cfg.env,
        "train_dir": str(run_dir.parent),
        "experiment": run_dir.name,
        "max_num_frames": max_num_frames,
        "num_envs": 1,
        "load_checkpoint_kind": "latest",
        "use_jit": False,
        "with_pos_obs": True,
        "no_render": True,
        "device": "cpu",
        "eval_deterministic": bool(deterministic),
    }
    cfg = load_from_checkpoint(cfg)
    cfg.max_num_frames = max_num_frames
    cfg.num_envs = 1
    cfg.with_pos_obs = True
    cfg.no_render = True
    cfg.use_jit = False
    cfg.device = "cpu"
    cfg.eval_deterministic = bool(deterministic)

    eval_env_frameskip = cfg.env_frameskip
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=None
    )
    env_info = extract_env_info(env, cfg)
    if hasattr(env.unwrapped, "reset_on_init"):
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()
    device = torch.device("cpu")
    actor_critic.model_to_device(device)

    checkpoint = select_checkpoint(cfg, checkpoint_rank)
    checkpoint_dict = BaseLearner.load_checkpoint([checkpoint], device)
    actor_critic.load_state_dict(checkpoint_dict["model"])
    log.info("Loaded checkpoint %s", checkpoint)

    core_buffers = []

    def core_hook(_module, _inp, out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        core_buffers.append(out.detach().cpu())

    dict(actor_critic.named_modules())["core"].register_forward_hook(core_hook)

    obs, infos = env.reset()
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    true_objectives = [deque([], maxlen=100) for _ in range(env.num_agents)]
    episode_reward = None
    finished_episode = [False for _ in range(env.num_agents)]
    num_frames = 0
    num_traj = 0
    num_episodes = 0
    pose_records = []

    with torch.no_grad():
        while num_frames <= max_num_frames:
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
            policy_outputs = actor_critic(normalized_obs, rnn_states)
            actions = policy_outputs["actions"]
            if cfg.eval_deterministic:
                action_distribution = actor_critic.action_distribution()
                actions = argmax_actions(action_distribution)
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)
            rnn_states = policy_outputs["new_rnn_states"]

            obs, rew, terminated, truncated, infos = env.step(actions)
            pos = obs["DEBUG.POS.TRANS"]
            rot = obs["DEBUG.POS.ROT"]
            for agent_i in range(env.num_agents):
                pose_records.append(
                    {
                        "frame": num_frames,
                        "agent": agent_i,
                        "x": float(pos[agent_i, 0]),
                        "y": float(pos[agent_i, 1]),
                        "z": float(pos[agent_i, 2]),
                        "rot_x": float(rot[agent_i, 0]),
                        "rot_y": float(rot[agent_i, 1]),
                        "rot_z": float(rot[agent_i, 2]),
                        "num_traj": num_traj,
                    }
                )

            dones = make_dones(terminated, truncated).cpu().numpy()
            episode_reward = rew.float().clone() if episode_reward is None else episode_reward + rew.float()
            num_frames += 1
            for agent_i, done_flag in enumerate(dones):
                if done_flag:
                    num_traj += 1
                    finished_episode[agent_i] = True
                    episode_rewards[agent_i].append(episode_reward[agent_i].item())
                    objective = episode_reward[agent_i].item()
                    if isinstance(infos, (list, tuple)):
                        objective = infos[agent_i].get("true_objective", objective)
                    true_objectives[agent_i].append(objective)
                    rnn_states[agent_i] = torch.zeros([get_rnn_size(cfg)], dtype=torch.float32, device=device)
                    episode_reward[agent_i] = 0
                    if cfg.use_record_episode_statistics and isinstance(infos, (list, tuple)):
                        if "episode" in infos[agent_i]:
                            num_episodes += 1
                    else:
                        num_episodes += 1
            if all(finished_episode):
                finished_episode = [False] * env.num_agents

    env.close()

    core = torch.cat(core_buffers, dim=0).numpy()
    n_feature = int(cfg.Hippo_n_feature)
    expanded_length = int(cfg.Hippo_R + cfg.Hippo_L - 1)
    hidden_len = n_feature * expanded_length
    if core.shape[1] < hidden_len:
        raise ValueError(f"Core output {core.shape} is smaller than expected hidden length {hidden_len}")
    dg = core[:, :hidden_len:expanded_length]
    pose = pd.DataFrame(pose_records).iloc[: dg.shape[0]].reset_index(drop=True)
    return cfg, checkpoint, pose, dg


def compute_place_fields(pose: pd.DataFrame, dg: np.ndarray, grain: int):
    bins = (
        np.linspace(*XBOUND, grain + 1),
        np.linspace(*YBOUND, grain + 1),
    )
    occupancy = np.histogramdd((pose["x"], pose["y"]), bins=bins, density=False)[0]
    fields = np.zeros((grain, grain, dg.shape[1]), dtype=np.float64)
    for i in range(dg.shape[1]):
        fields[:, :, i] = np.histogramdd((pose["x"], pose["y"]), bins=bins, weights=dg[:, i], density=False)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_maps = fields / occupancy[:, :, None]
    rate_maps[occupancy <= 0, :] = np.nan
    si = np.array([spatial_information(rate_maps[:, :, i], occupancy) for i in range(dg.shape[1])])
    active_fraction = (dg > 0).mean(axis=0)
    return occupancy, rate_maps, si, active_fraction


def plot_run_grid(rate_maps, occupancy, si, active_fraction, title, out_path):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.size"] = 14

    n_units = rate_maps.shape[-1]
    n_cols = min(8, max(4, int(math.ceil(math.sqrt(n_units)))))
    n_rows = int(math.ceil(n_units / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.5 * n_rows), dpi=180)
    axes = np.asarray(axes).reshape(-1)
    occ_mask = occupancy.T <= 0
    finite = rate_maps[np.isfinite(rate_maps)]
    vmax = np.percentile(finite, 98) if finite.size else 1.0
    vmax = max(vmax, 1e-8)
    for i, ax in enumerate(axes):
        ax.set_axis_off()
        if i >= n_units:
            continue
        data = rate_maps[:, :, i].T.copy()
        data[occ_mask] = np.nan
        im = ax.imshow(data, origin="lower", extent=[*XBOUND, *YBOUND], cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"DG {i}  SI={si[i]:.2f}  act={active_fraction[i]:.3f}", fontsize=11)
    cbar = fig.colorbar(im, ax=axes[:n_units], fraction=0.025, pad=0.01)
    cbar.set_label("mean DG activation")
    fig.suptitle(title, fontsize=18)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary_rows, out_path):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.size"] = 14

    df = pd.DataFrame(summary_rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    x = np.arange(len(df))
    axes[0].bar(x, df["mean_active_fraction"])
    axes[0].set_xticks(x, df["label"], rotation=25, ha="right")
    axes[0].set_ylabel("mean active fraction")
    axes[0].set_title("DG density")
    axes[1].bar(x, df["mean_si"])
    axes[1].set_xticks(x, df["label"], rotation=25, ha="right")
    axes[1].set_ylabel("mean spatial information")
    axes[1].set_title("Place-field selectivity")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for run_dir_str in args.run_dir:
        run_dir = pathlib.Path(run_dir_str)
        label = run_label(run_dir)
        safe_label = label.replace("/", "__")
        run_out = out_dir / safe_label
        run_out.mkdir(parents=True, exist_ok=True)
        cfg, checkpoint, pose, dg = rollout_dg(
            run_dir,
            args.max_num_frames,
            deterministic=args.deterministic,
            checkpoint_rank=args.checkpoint_rank,
        )
        occupancy, rate_maps, si, active_fraction = compute_place_fields(pose, dg, args.grain)
        pose.to_csv(run_out / "pose.csv", index=False)
        np.savez_compressed(
            run_out / "place_fields.npz",
            occupancy=occupancy,
            rate_maps=rate_maps,
            spatial_information=si,
            active_fraction=active_fraction,
            checkpoint=str(checkpoint),
            Hippo_n_feature=int(cfg.Hippo_n_feature),
            Hippo_L=int(cfg.Hippo_L),
            Hippo_R=int(cfg.Hippo_R),
            DG_BN_intercept=float(cfg.DG_BN_intercept),
        )
        if args.save_raw_activations:
            np.savez_compressed(run_out / "dg_activations.npz", dg=dg)
        title = (
            f"{label} | F={cfg.Hippo_n_feature}, L={cfg.Hippo_L}, "
            f"theta={cfg.DG_BN_intercept}, checkpoint={pathlib.Path(checkpoint).name}"
        )
        if not args.no_plots:
            plot_run_grid(rate_maps, occupancy, si, active_fraction, title, run_out / "dg_place_fields.png")
        summary_rows.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "frames": len(pose),
                "Hippo_n_feature": int(cfg.Hippo_n_feature),
                "Hippo_L": int(cfg.Hippo_L),
                "DG_BN_intercept": float(cfg.DG_BN_intercept),
                "mean_active_fraction": float(np.mean(active_fraction)),
                "median_active_fraction": float(np.median(active_fraction)),
                "mean_si": float(np.nanmean(si)),
                "max_si": float(np.nanmax(si)),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    if not args.no_plots:
        plot_summary(summary_rows, out_dir / "summary.png")
    print(f"Wrote DG place-field plots to {out_dir}")


if __name__ == "__main__":
    main()
