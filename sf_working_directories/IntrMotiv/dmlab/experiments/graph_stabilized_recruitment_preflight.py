"""Five seed-99 2M-frame graph-recruitment preflights."""

from sample_factory.launcher.run_description import Experiment, RunDescription
from sf_working_directories.IntrMotiv.dmlab.experiments.corrected_core_reevaluation import BATCH_NAME as CORRECTED_BATCH
from sf_working_directories.IntrMotiv.dmlab.experiments.corrected_core_reevaluation import CELLS
from sf_working_directories.IntrMotiv.dmlab.experiments.corrected_core_reevaluation import PROJECT as CORRECTED_PROJECT
from sf_working_directories.IntrMotiv.dmlab.experiments.corrected_core_reevaluation import (
    make_experiment as make_corrected_experiment,
)
from sf_working_directories.IntrMotiv.dmlab.experiments.graph_stabilized_recruitment import (
    BATCH_NAME,
    PROJECT,
    make_experiment,
)


def make_flat_fallback_experiment() -> Experiment:
    flat = next(cell for cell in CELLS if cell.number == 1)
    base = make_corrected_experiment(
        flat,
        99,
        train_for_env_steps=2_000_000,
        group_suffix="preflight",
    )
    cli = (
        base.cmd.replace(f"--wandb_project={CORRECTED_PROJECT}", f"--wandb_project={PROJECT}")
        .replace(f"--wandb_group={CORRECTED_BATCH}_preflight", f"--wandb_group={BATCH_NAME}_preflight")
        .replace("--dg_orthogonal_recruitment=False", "--dg_orthogonal_recruitment=True")
        + "--dg_orthogonal_recruitment_mode=graph "
        + "--dg_recruitment_connectivity_threshold=0.25 "
        + "--dg_recruitment_redundancy_max_steps=4 "
        + "--dg_recruitment_passive_half_life_events=5000 "
    )
    return Experiment("PF_GSR_C01_FLAT_D4_H5K_S99", cli, [{}])


PREFLIGHT_CELLS = (
    (5, 4, 5000),
    (5, 8, 10000),
    (13, 4, 10000),
    (15, 8, 5000),
)

RUN_DESCRIPTION = RunDescription(
    f"{BATCH_NAME}_preflight",
    experiments=[
        make_experiment(
            backbone,
            redundancy,
            half_life,
            99,
            train_for_env_steps=2_000_000,
            prefix="PF",
            group_suffix="preflight",
        )
        for backbone, redundancy, half_life in PREFLIGHT_CELLS
    ]
    + [make_flat_fallback_experiment()],
)
