"""Resume only the node-failed W_REF_JOINT seed-123 production run."""

from sample_factory.launcher.run_description import RunDescription
from sf_working_directories.IntrMotiv.dmlab.experiments.persistent_intrinsic_control import (
    RUN_DESCRIPTION as FULL_RUN_DESCRIPTION,
)

TARGET = "PIC_W_REF_JOINT_S123"
EXPERIMENTS = [experiment for experiment in FULL_RUN_DESCRIPTION.experiments if experiment.base_name == TARGET]
if len(EXPERIMENTS) != 1:
    raise RuntimeError(f"expected exactly one {TARGET} experiment, found {len(EXPERIMENTS)}")

RUN_DESCRIPTION = RunDescription(
    FULL_RUN_DESCRIPTION.run_name,
    experiments=EXPERIMENTS,
)
