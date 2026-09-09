"""Resume all three C15 runs from their last pre-hotfix checkpoints."""

from sample_factory.launcher.run_description import Experiment, RunDescription

from sf_working_directories.IntrMotiv.dmlab.experiments.persistent_intrinsic_control_c15 import (
    STUDY,
    build,
)


RECOVERY_CHECKPOINTS = {
    8: "/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/intrmotiv_persistent_intrinsic_control_c15_r2_20260908/PIC_C15_CONTINUE_S8_/00_PIC_C15_CONTINUE_S8/checkpoint_p0/checkpoint_000010198_167084032.pth",
    99: "/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/intrmotiv_persistent_intrinsic_control_c15_r2_20260908/PIC_C15_CONTINUE_S99_/00_PIC_C15_CONTINUE_S99/checkpoint_p0/checkpoint_000009784_160301056.pth",
    123: "/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/intrmotiv_persistent_intrinsic_control_c15_r2_20260908/PIC_C15_CONTINUE_S123_/00_PIC_C15_CONTINUE_S123/checkpoint_p0/checkpoint_000009076_148701184.pth",
}


def recovery_experiment(run):
    original = build(run)
    old_checkpoint = next(
        arg for arg in run.args if arg.startswith("--load_model_path=")
    )
    command = original.cmd.replace(
        old_checkpoint,
        f"--load_model_path={RECOVERY_CHECKPOINTS[run.seed]}",
    )
    if command == original.cmd:
        raise RuntimeError(f"failed to replace the parent checkpoint for seed {run.seed}")
    return Experiment(run.name, command, [{}])


RUN_DESCRIPTION = RunDescription(
    STUDY.batch_name,
    experiments=[recovery_experiment(run) for run in STUDY.expand_runs()],
)
