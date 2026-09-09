"""Resume the sole node-failed context run from its verified production checkpoint."""
from sample_factory.launcher.run_description import Experiment, RunDescription
from sf_working_directories.IntrMotiv.dmlab.experiments.persistent_intrinsic_control import RUN_DESCRIPTION as FULL
TARGET = "PIC_G_CONTEXT_S99"
CHECKPOINT = "/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/intrmotiv_persistent_intrinsic_control_20260908/PIC_G_CONTEXT_S99_/00_PIC_G_CONTEXT_S99/checkpoint_p0/checkpoint_000005502_90144768.pth"
selected = [e for e in FULL.experiments if e.base_name == TARGET]
assert len(selected) == 1
assert "--load_model_path=" not in selected[0].cmd
RUN_DESCRIPTION = RunDescription(FULL.run_name, experiments=[
    Experiment(TARGET, selected[0].cmd + " --load_model_path=" + CHECKPOINT, [{}])])
