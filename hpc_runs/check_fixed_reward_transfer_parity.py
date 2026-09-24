"""Compare source and trained transfer tensors after a qualification run."""

import argparse
import json
from pathlib import Path

import torch


DG = ("encoder.DG_projection.linear.", "encoder.DG_projection.batchnorm1d.")
WORKER = ("decoder.", "action_parameterization.", "core.dg_goal_modulation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("trained", type=Path)
    parser.add_argument("--frozen-worker", action="store_true")
    args = parser.parse_args()
    source = torch.load(args.source, map_location="cpu", weights_only=False)["model"]
    trained = torch.load(args.trained, map_location="cpu", weights_only=False)["model"]
    prefixes = DG + (WORKER if args.frozen_worker else ())
    keys = sorted(key for key in source if key.startswith(prefixes))
    missing = [key for key in keys if key not in trained]
    changed = [key for key in keys if key in trained and not torch.equal(source[key], trained[key])]
    result = {"compared_tensors": len(keys), "missing": missing, "changed": changed,
              "source": str(args.source), "trained": str(args.trained)}
    print(json.dumps(result, indent=2))
    if missing or changed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
