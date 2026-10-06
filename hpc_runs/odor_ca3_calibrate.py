"""Calibrate one DG odor gain on a fixed 10k-decision DMLab trajectory.

The actions are independent of policy and odor. Both the frozen visual trunk
and the clean four-channel odor are measured at every decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import deepmind_lab
import numpy as np
import torch

from hpc_runs.intrmotiv_study import load_study
from sf_working_directories.IntrMotiv.dmlab.custom_encoder import HipposlamEncoder
from sf_working_directories.IntrMotiv.dmlab.dmlab_env import make_dmlab_env
from sf_working_directories.IntrMotiv.dmlab.odor import ODOR_CENTERS, ODOR_SPATIAL_SIGMA, clean_odor
from sf_working_directories.IntrMotiv.dmlab.train_hipposlam import parse_dmlab_args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--runfiles-root", type=Path)
    parser.add_argument("--decisions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    if args.decisions != 10000:
        raise ValueError("The production calibration contract uses exactly 10000 decisions")
    if args.runfiles_root is not None:
        deepmind_lab.set_runfiles_path(str(args.runfiles_root.resolve()))
    study = load_study(args.study)
    row = next(row for row in study.expand_runs() if row.factors == {"odor": "off", "goal_set": "all16"} and row.seed == 8)
    scratch = args.output.parent.resolve() / "runtime"
    scratch.mkdir(parents=True, exist_ok=True)
    cfg = parse_dmlab_args(list(row.args) + [
        f"--train_dir={scratch}",
        f"--dmlab_level_cache_path={scratch / 'dmlab_cache'}",
        "--dmlab_use_level_cache=False",
        "--online_spatial_telemetry=False",
        "--exploration_coverage_telemetry=False",
        "--dg_odor_mode=gaussian4",
        "--dg_odor_gain=1.0",
    ])
    torch.set_num_threads(4)
    env = make_dmlab_env(cfg.env, cfg, {"env_id": 0, "worker_index": 0, "vector_index": 0})
    encoder = HipposlamEncoder(cfg, env.observation_space).eval()
    action_rng = np.random.default_rng(args.seed + 1)
    trajectory_hash = hashlib.sha256()
    visual_sum = clean_sum = noisy_sum = 0.0
    visual_square_sum = 0.0
    position_min = np.full(2, np.inf)
    position_max = np.full(2, -np.inf)
    episode = 0
    obs, _ = env.reset(seed=args.seed)
    batch = []

    def flush() -> tuple[float, float]:
        if not batch:
            return 0.0, 0.0
        shapes = {tuple(np.asarray(o["obs"]).shape) for o in batch}
        if len(shapes) != 1:
            raise RuntimeError(f"Calibration observation shapes varied: {shapes}")
        inputs = {
            "obs": torch.from_numpy(np.stack([o["obs"] for o in batch])).float() / 255.0,
            "INSTR": torch.from_numpy(np.stack([o["INSTR"] for o in batch])).long(),
        }
        with torch.no_grad():
            features = encoder.projection_input(inputs)
            norms = torch.linalg.vector_norm(features, dim=-1)
        batch.clear()
        return float(norms.sum().item()), float(norms.square().sum().item())

    try:
        for decision in range(args.decisions):
            position = np.asarray(env.unwrapped.last_debug_position, dtype=np.float64)
            if position.shape[0] < 2 or not np.isfinite(position).all():
                raise RuntimeError("Calibration requires valid DMLab debug position")
            clean_sum += float(np.linalg.norm(clean_odor(position)))
            noisy_sum += float(np.linalg.norm(obs["dg_odor"]))
            position_min = np.minimum(position_min, position[:2])
            position_max = np.maximum(position_max, position[:2])
            # DMLab reuses observation arrays; retain the behavior-time image.
            batch.append({"obs": np.array(obs["obs"], copy=True), "INSTR": np.array(obs["INSTR"], copy=True)})
            trajectory_hash.update(position.tobytes())
            if len(batch) == 64:
                total, squares = flush()
                visual_sum += total
                visual_square_sum += squares
            action = int(action_rng.integers(env.action_space.n))
            trajectory_hash.update(action.to_bytes(2, "little"))
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                episode += 1
                obs, _ = env.reset(seed=args.seed + episode)
            if (decision + 1) % 1000 == 0:
                print(f"calibration decisions: {decision + 1}/{args.decisions}", flush=True)
        total, squares = flush()
        visual_sum += total
        visual_square_sum += squares
    finally:
        env.close()
    mean_visual = visual_sum / args.decisions
    mean_clean = clean_sum / args.decisions
    map_file = (
        args.runfiles_root / "baselab" / "game_scripts" / "levels" / f"{cfg.env}.lua"
        if args.runfiles_root is not None else None
    )
    artifact = {
        "schema": "intrmotiv/odor-gain-calibration/v1",
        "study_sha256": study.fingerprint,
        "decisions": args.decisions,
        "action_seed": args.seed + 1,
        "reset_seed": args.seed,
        "trajectory_sha256": trajectory_hash.hexdigest(),
        "map_lua_sha256": hashlib.sha256(map_file.read_bytes()).hexdigest() if map_file else None,
        "position_min_xy": position_min.tolist(),
        "position_max_xy": position_max.tolist(),
        "odor_centers": ODOR_CENTERS.tolist(),
        "odor_spatial_sigma": ODOR_SPATIAL_SIGMA,
        "odor_noise_std": cfg.dg_odor_noise_std,
        "mean_visual_block_norm": mean_visual,
        "visual_block_norm_std": max(0.0, visual_square_sum / args.decisions - mean_visual**2) ** 0.5,
        "mean_clean_odor_norm": mean_clean,
        "mean_noisy_odor_norm_unscaled": noisy_sum / args.decisions,
        "odor_gain": mean_visual / mean_clean,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2), flush=True)


if __name__ == "__main__":
    main()
