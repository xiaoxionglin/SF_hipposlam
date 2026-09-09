"""Generate the production matrix for Graph-Stabilized Recruitment.

This manifest intentionally leaves the corrected-core backbone command bundles
opaque: those bundles are defined in the authoritative NEMO2 checkout and are
not present in this vault repository.  The generated rows contain the exact
new flags and can be joined to the existing C05/C13/C15 bundles there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from hpc_runs.intrmotiv_study import load_study
except ModuleNotFoundError:  # Preserve direct ``python hpc_runs/...py`` use.
    from intrmotiv_study import load_study


SPEC_PATH = Path(__file__).with_name("studies") / "graph_stabilized_recruitment.study.json"
STUDY = load_study(SPEC_PATH)
BATCH_NAME = STUDY.batch_name
BACKBONES = tuple(base.name for base in STUDY.bases)
REDUNDANCY_THRESHOLDS = tuple(level.value for level in STUDY.factors[0].levels)
HALF_LIVES = tuple(level.value for level in STUDY.factors[1].levels)
SEEDS = STUDY.seeds


def rows() -> list[dict[str, object]]:
    result = []
    for run in STUDY.expand_runs():
        # This historical manifest carries supplemental flags only. New complete
        # studies should use training.mode=sample_factory and the shared adapter.
        result.append({
            "name": run.name,
            "batch": run.batch_name,
            "backbone": run.base,
            "seed": run.seed,
            "args": [arg for arg in run.args if not arg.startswith("--seed=")],
            "context_controls": [
                f"original_{run.base}_seed{control_seed}" for control_seed in SEEDS
            ],
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write JSON instead of stdout")
    args = parser.parse_args()
    payload = {
        **STUDY.provenance(),
        "batch": BATCH_NAME,
        "rows": rows(),
        "count": STUDY.expected_runs,
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
