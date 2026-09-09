from sample_factory.launcher.run_description import RunDescription

from sf_working_directories.IntrMotiv.dmlab.experiments.hrl_persistence_comparison_preflight import (
    long_preflight,
)


# Targeted rerun after validating the fixed/global branch. Keeping it separate
# avoids spending another node allocation on an already-passed preflight.
RUN_DESCRIPTION = RunDescription(
    "intrmotiv_hrl_persistence_comparison_long_preflight_20260821",
    experiments=[long_preflight()],
)
