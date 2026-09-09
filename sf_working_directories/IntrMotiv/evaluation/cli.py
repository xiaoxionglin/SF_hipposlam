"""Command line entry point for retrospective IntrMotiv batch diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .report import write_outputs
from .tensorboard import load_batches


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize IntrMotiv TensorBoard batches with a stable schema.")
    parser.add_argument("batch_root", type=Path, nargs="+", help="One or more Sample Factory batch directories.")
    parser.add_argument("--output", type=Path, required=True, help="Workspace directory for CSV/Markdown outputs.")
    parser.add_argument("--target-env-steps", type=int, default=100_000_000)
    parser.add_argument("--workers", type=int, default=1, help="Parallel event-file readers; use 1 for constrained nodes.")
    args = parser.parse_args()
    rows = load_batches(args.batch_root, args.target_env_steps, workers=args.workers)
    if not rows:
        raise SystemExit("No TensorBoard event files found below the supplied batch roots")
    write_outputs(pd.DataFrame(rows), args.output, args.batch_root, args.target_env_steps)
    print(f"Wrote retrospective evaluation for {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()
