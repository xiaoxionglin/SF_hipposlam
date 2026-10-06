"""Four short qualification jobs derived from the immutable production study."""

from __future__ import annotations

import shlex

from sample_factory.launcher.run_description import Experiment, RunDescription

from .odor_ca3_goal_quality_v2_20261007 import STUDY


QUALIFICATION_CELLS = {
    ("off", "random4"),
    ("off", "hebb8"),
    ("on", "random8"),
    ("on", "hebb4"),
}


def qualification_args(args: tuple[str, ...]) -> list[str]:
    replacements = {
        "train_for_env_steps": "524288",
        "checkpoint_frame_targets": "262144,524288",
        "num_workers": "8",
        "batch_size": "1024",
        "num_batches_per_epoch": "1",
        "with_wandb": "False",
    }
    retained = [arg for arg in args if arg.split("=", 1)[0].removeprefix("--") not in replacements]
    retained.extend(f"--{key}={value}" for key, value in replacements.items())
    retained.append("--online_spatial_telemetry=False")
    return retained


RUN_DESCRIPTION = RunDescription(
    "intrmotiv_odor_ca3_goal_quality_v2_qualification_20261007",
    experiments=[
        Experiment("QUAL_V2_" + run.name, " ".join(shlex.quote(arg) for arg in qualification_args(run.args)), [{}])
        for run in STUDY.expand_runs()
        if run.seed == 99 and (run.factors["odor"], run.factors["goal_set"]) in QUALIFICATION_CELLS
    ],
)
