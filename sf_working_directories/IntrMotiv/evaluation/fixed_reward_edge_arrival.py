"""Measure physical reward-site arrival from matched source-DG observations.

This is a frozen-policy Slurm probe. A source observation and action prefix are
replayed exactly for every command, so the requested DG ID is the intervention.
"""

import argparse
import json
import os
from pathlib import Path

import deepmind_lab
import torch

from sf_working_directories.IntrMotiv.evaluation.place_fields import load_policy_env
from sf_working_directories.IntrMotiv.evaluation.target_control_interventions import run_landmark_matched_interventions

SITES = {
    "dg50": dict(target=50, sources=[25], center=(350, 1950), cell=(300, 400, 1900, 2000)),
    "dg51": dict(target=51, sources=[19, 6], center=(250, 1950), cell=(200, 300, 1900, 2000)),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runfiles", type=Path, required=True)
    parser.add_argument("--site", choices=sorted(SITES), required=True)
    parser.add_argument("--sources", type=int, nargs="+", help="Subset of the site's incoming source IDs")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--decision-cap", type=int, default=100000)
    parser.add_argument("--prefix-cap", type=int, default=512)
    parser.add_argument("--physical-horizon", type=int, default=512)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("DMLab edge arrival evaluation requires a Slurm job")
    torch.set_num_threads(1)
    site = SITES[args.site]
    sources = site["sources"] if args.sources is None else args.sources
    if not sources or any(source not in site["sources"] for source in sources):
        parser.error("--sources must select one or more listed incoming source IDs")
    deepmind_lab.set_runfiles_path(str(args.runfiles.resolve(strict=True)))
    cfg, env, info, actor, checkpoint, device = load_policy_env(
        args.run_dir, args.decision_cap, False, 0, args.checkpoint
    )
    cfg.dmlab_use_level_cache = False
    cfg.dmlab_runfiles_path = str(args.runfiles.resolve(strict=True))
    rows, summary = run_landmark_matched_interventions(
        cfg,
        env,
        info,
        actor,
        checkpoint,
        device,
        args.decision_cap,
        max_sources=len(sources),
        targets_per_source=3,
        repeats=args.repeats,
        prefix_cap=args.prefix_cap,
        focus_sources=sources,
        focus_target=site["target"],
        reward_center=site["center"],
        reward_cell=site["cell"],
        physical_horizon=args.physical_horizon,
        discovery_multiplier=20,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.site if args.sources is None else f"{args.site}_source{'_'.join(map(str, sources))}"
    rows.to_csv(args.out_dir / f"{stem}_edge_arrivals.csv", index=False)
    summary.update(site=args.site, source_ids=sources, reward_cell=site["cell"])
    (args.out_dir / f"{stem}_edge_arrivals.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
