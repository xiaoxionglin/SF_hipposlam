from sf_working_directories.IntrMotiv.dmlab.experiments.graph_stabilized_recruitment import (
    BACKBONE_NUMBERS,
    BATCH_NAME,
    HALF_LIVES,
    PROJECT,
    REDUNDANCY_THRESHOLDS,
    RUN_DESCRIPTION,
    SEEDS,
)
from sf_working_directories.IntrMotiv.dmlab.experiments.graph_stabilized_recruitment_preflight import (
    RUN_DESCRIPTION as PREFLIGHT_RUN_DESCRIPTION,
)


def test_production_matrix_is_12_conditions_and_36_unique_runs():
    assert RUN_DESCRIPTION.run_name == BATCH_NAME
    assert BACKBONE_NUMBERS == (5, 13, 15)
    assert REDUNDANCY_THRESHOLDS == (4, 8)
    assert HALF_LIVES == (5000, 10000)
    assert SEEDS == (8, 99, 123)
    assert len(RUN_DESCRIPTION.experiments) == 36
    names = [experiment.base_name for experiment in RUN_DESCRIPTION.experiments]
    assert len(names) == len(set(names))


def test_production_commands_use_graph_mode_and_matched_half_lives():
    for experiment in RUN_DESCRIPTION.experiments:
        command = experiment.cmd
        assert "--train_for_env_steps=100000000" in command
        assert "--dg_orthogonal_recruitment=True" in command
        assert "--dg_orthogonal_recruitment_mode=graph" in command
        assert "--dg_recruitment_connectivity_threshold=0.25" in command
        assert f"--wandb_project={PROJECT}" in command
        assert "/work/classic/fr_xl1014-train/" in command
        passive = next(value for value in HALF_LIVES if f"--dg_recruitment_passive_half_life_events={value}" in command)
        assert f"--hrl_fast_weight_half_life_options={passive}" in command


def test_preflight_spans_backbones_thresholds_half_lives_and_flat_fallback():
    assert len(PREFLIGHT_RUN_DESCRIPTION.experiments) == 5
    commands = [experiment.cmd for experiment in PREFLIGHT_RUN_DESCRIPTION.experiments]
    assert all("--train_for_env_steps=2000000" in command for command in commands)
    assert all("--seed=99" in command for command in commands)
    assert any("--hrl_controllable_graph=False" in command for command in commands)
    assert any("GSR_C05_D4_H5K" in experiment.base_name for experiment in PREFLIGHT_RUN_DESCRIPTION.experiments)
    assert any("GSR_C13_D4_H10K" in experiment.base_name for experiment in PREFLIGHT_RUN_DESCRIPTION.experiments)
    assert any("GSR_C15_D8_H5K" in experiment.base_name for experiment in PREFLIGHT_RUN_DESCRIPTION.experiments)
    assert any("--dg_recruitment_redundancy_max_steps=4" in command for command in commands)
    assert any("--dg_recruitment_redundancy_max_steps=8" in command for command in commands)
