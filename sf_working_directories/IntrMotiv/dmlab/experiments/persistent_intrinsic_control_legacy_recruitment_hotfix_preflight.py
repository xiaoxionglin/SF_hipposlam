"""Force the legacy-recruitment/policy-graph invalidation path once."""

from sample_factory.launcher.run_description import Experiment, RunDescription

from sf_working_directories.IntrMotiv.dmlab.experiments.corrected_core_reevaluation import (
    CELLS,
    make_experiment,
)


BATCH_NAME = "intrmotiv_persistent_intrinsic_control_legacy_recruitment_hotfix_preflight_20260908"
C15 = next(cell for cell in CELLS if cell.number == 15)
BASE = make_experiment(
    C15,
    123,
    train_for_env_steps=1_000_000,
    group_suffix="pic_legacy_recruitment_hotfix_preflight_20260908",
)
COMMAND = " ".join(
    (
        BASE.cmd,
        "--dg_orthogonal_recruitment_mode=legacy",
        "--dg_recruitment_endpoint_gate=open",
        "--dg_recruitment_forced_preflight_row=0",
        "--dg_recruitment_forced_preflight_after_updates=4",
        "--dg_orthogonal_recruitment_max_per_rollout=1",
    )
)

RUN_DESCRIPTION = RunDescription(
    BATCH_NAME,
    experiments=[Experiment("PIC_C15_LEGACY_RECRUITMENT_HOTFIX_S123", COMMAND, [{}])],
)
