from sf_working_directories.IntrMotiv.dmlab.experiments.target_control_her_followup import (
    BATCH_NAME,
    CELLS,
    HER_COEFFICIENT,
    HER_HORIZON,
    PROJECT,
    RUN_DESCRIPTION,
    SEEDS,
)
from sf_working_directories.IntrMotiv.dmlab.experiments.target_control_her_followup_preflight import (
    RUN_DESCRIPTION as PREFLIGHT_RUN_DESCRIPTION,
)


def _command(backbone_number: int, empirical_her: bool, seed: int = 8) -> str:
    marker = f"TCH_C{backbone_number:02d}_HER{'64' if empirical_her else 'OFF'}_S{seed}"
    return next(experiment.cmd for experiment in RUN_DESCRIPTION.experiments if experiment.base_name == marker)


def _without_her_and_wandb_group(command: str) -> str:
    parts = []
    for part in command.split():
        if part.startswith("--hrl_empirical_her="):
            continue
        if part.startswith("--hrl_empirical_her_horizon="):
            continue
        if part.startswith("--hrl_empirical_her_coeff="):
            continue
        if part.startswith("--wandb_group="):
            continue
        parts.append(part)
    return " ".join(parts)


def test_matrix_has_four_cells_three_seeds_and_unique_names():
    assert len(CELLS) == 4
    assert SEEDS == (8, 99, 123)
    assert len(RUN_DESCRIPTION.experiments) == 12
    names = [experiment.base_name for experiment in RUN_DESCRIPTION.experiments]
    assert len(names) == len(set(names))
    assert {cell.backbone_number for cell in CELLS} == {12, 13}


def test_common_configuration_is_corrected_core_and_workspace_safe():
    for experiment in RUN_DESCRIPTION.experiments:
        command = experiment.cmd
        assert "--train_for_env_steps=100000000" in command
        assert "--encoder_conv_architecture=layer2_resnet18" in command
        assert "--encoder_reward_method=encourage" in command
        assert "--encoder_batch_loss=True" in command
        assert "--encoder_multi_activation_loss=True" in command
        assert "--hrl_graph_memory=policy_buffer" in command
        assert "--hrl_manager_mode=visit_direct" in command
        assert "--hrl_target_timing=delayed" in command
        assert "--hrl_timeout_margin_ratio=1.0" in command
        assert "--hrl_timeout_margin_steps=8" in command
        assert "--hrl_bootstrap_horizon=96" in command
        assert f"--wandb_project={PROJECT}" in command
        assert f"--wandb_group={BATCH_NAME}_production" in command
        assert "/work/classic/fr_xl1014-train/" in command


def test_her_is_the_only_condition_difference_within_each_backbone():
    for backbone_number in (12, 13):
        off = _command(backbone_number, False)
        on = _command(backbone_number, True)
        assert _without_her_and_wandb_group(off) == _without_her_and_wandb_group(on)
        assert "--hrl_empirical_her=False" in off
        assert "--hrl_empirical_her=True" in on
        assert f"--hrl_empirical_her_horizon={HER_HORIZON}" in on
        assert f"--hrl_empirical_her_coeff={HER_COEFFICIENT}" in on


def test_c12_c13_backbones_retain_their_manager_difference():
    c12 = _command(12, False)
    c13 = _command(13, False)
    for command in (c12, c13):
        assert "--dg_ca3_temporal_exclusion_coeff=1.0" in command
        assert "--dg_orthogonal_recruitment=True" in command
        assert "--hrl_timeout_margin_ratio=1.0" in command
        assert "--hrl_timeout_margin_steps=8" in command
        assert "--hrl_bootstrap_horizon=96" in command
    assert "--hrl_exploration_mode=False" in c12
    assert "--hrl_manager_exploration_probability=0.0" in c12
    assert "--hrl_exploration_mode=True" in c13
    assert "--hrl_manager_exploration_probability=0.1" in c13


def test_her_on_cells_are_direct_policy_buffer_runs():
    for backbone_number in (12, 13):
        command = _command(backbone_number, True)
        assert "--hrl_controllable_graph=True" in command
        assert "--hrl_graph_memory=policy_buffer" in command
        assert "--hrl_manager_mode=visit_direct" in command


def test_preflight_has_two_her_on_seed99_runs_at_two_million_frames():
    assert PREFLIGHT_RUN_DESCRIPTION.run_name == f"{BATCH_NAME}_preflight"
    assert len(PREFLIGHT_RUN_DESCRIPTION.experiments) == 2
    for experiment in PREFLIGHT_RUN_DESCRIPTION.experiments:
        assert experiment.base_name.startswith("PF_TCH_C")
        assert experiment.base_name.endswith("_S99")
        assert "--train_for_env_steps=2000000" in experiment.cmd
        assert "--hrl_empirical_her=True" in experiment.cmd
        assert f"--wandb_group={BATCH_NAME}_preflight" in experiment.cmd
