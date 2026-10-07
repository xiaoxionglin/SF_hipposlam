"""Two short seed-99 qualification runs derived from the production study."""

import shlex

from sample_factory.launcher.run_description import Experiment, RunDescription

from .four_prescribed_dg_controller_20261007 import STUDY


def qualification_args(args: tuple[str, ...]) -> list[str]:
    replacements = {
        "train_for_env_steps": "524288",
        "checkpoint_frame_targets": "262144,524288",
        "num_workers": "8",
        "batch_size": "1024",
        "num_batches_per_epoch": "1",
        "with_wandb": "False",
        "online_spatial_telemetry": "False",
    }
    retained = [arg for arg in args if arg.split("=", 1)[0].removeprefix("--") not in replacements]
    retained.extend(f"--{key}={value}" for key, value in replacements.items())
    return retained


RUN_DESCRIPTION = RunDescription(
    "intrmotiv_four_prescribed_dg_qualification_20261007",
    experiments=[
        Experiment("QUAL_" + run.name, " ".join(shlex.quote(arg) for arg in qualification_args(run.args)), [{}])
        for run in STUDY.expand_runs() if run.seed == 99
    ],
)
