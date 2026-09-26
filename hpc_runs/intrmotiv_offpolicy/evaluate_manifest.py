"""Thin adapter for the established independent-job telemetry submitter."""

import json
import os
import sys
from pathlib import Path

import torch

from .evaluate import main


def run():
    from sf_working_directories.IntrMotiv.evaluation.submit_place_field_sweep import load_manifest

    manifest, index, output = sys.argv[1:]
    row = load_manifest(Path(manifest))[int(index)].values
    parent_dir, parent_checkpoint = row["run_dir"], row["checkpoint"]
    extra = []
    if row["family"] == "ddqn":
        from sf_working_directories.IntrMotiv.evaluation.place_fields import load_checkpoint_dict

        with torch.serialization.safe_globals([type(Path("."))]):
            child = load_checkpoint_dict(Path(row["checkpoint"]), torch.device("cpu"))
        parents = json.loads(Path(child["config"]["parent_manifest"]).read_text())["parents"]
        parent = next(p for p in parents if p["run"] == child["config"]["parent_run"])
        parent_checkpoint = parent["checkpoint"]
        parent_dir = str(Path(parent_checkpoint).parent.parent.parent)
        extra = ["--child-checkpoint", row["checkpoint"]]
    elif row["family"] != "parent":
        raise ValueError("unsupported checkpoint family")
    sys.argv = [
        sys.argv[0],
        "--parent-run-dir",
        parent_dir,
        "--parent-checkpoint",
        parent_checkpoint,
        "--output",
        str(Path(output) / "raw" / row["label_suffix"]),
        "--decision-cap",
        os.environ.get("PLACE_FIELD_MAX_FRAMES", "100000"),
        "--max-sources",
        str(row.get("max_sources", 4)),
        "--repeats",
        str(row.get("repeats", 3)),
    ] + extra
    main()


if __name__ == "__main__":
    run()
