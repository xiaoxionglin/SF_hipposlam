"""One seed-99, 1M-step preflight for every cell in the canonical study."""

from pathlib import Path

from sample_factory.launcher.run_description import Experiment, RunDescription

from hpc_runs.intrmotiv_study import load_study


SPEC = Path(__file__).with_name("studies") / "controllability_edge_exploration.study.json"
STUDY = load_study(SPEC)
BATCH_NAME = f"{STUDY.batch_name}_preflight"


def _preflight_experiment(run):
    args = []
    for arg in run.args:
        if arg == "--train_for_env_steps=75000000":
            arg = "--train_for_env_steps=1000000"
        elif arg == f"--wandb_group={STUDY.batch_name}":
            arg = f"--wandb_group={BATCH_NAME}"
        args.append(arg)
    return Experiment(f"PF_{run.name}", " ".join(args), [{}])


RUN_DESCRIPTION = RunDescription(
    BATCH_NAME,
    experiments=[_preflight_experiment(run) for run in STUDY.expand_runs() if run.seed == 99],
)
