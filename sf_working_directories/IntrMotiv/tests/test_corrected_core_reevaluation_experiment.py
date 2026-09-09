from sf_working_directories.IntrMotiv.dmlab.experiments.corrected_core_reevaluation import (
    BATCH_NAME,
    CELLS,
    PROJECT,
    RUN_DESCRIPTION,
    SEEDS,
)
from sf_working_directories.IntrMotiv.dmlab.experiments.corrected_core_reevaluation_preflight import (
    RUN_DESCRIPTION as PREFLIGHT_RUN_DESCRIPTION,
)


def _command(cell_number: int, seed: int = 8) -> str:
    marker = f"C{cell_number:02d}_"
    return next(
        experiment.cmd
        for experiment in RUN_DESCRIPTION.experiments
        if marker in experiment.base_name and experiment.base_name.endswith(f"_S{seed}")
    )


def test_production_matrix_has_16_three_seed_cells_and_unique_names():
    assert RUN_DESCRIPTION.run_name == BATCH_NAME
    assert len(CELLS) == 16
    assert SEEDS == (8, 99, 123)
    assert len(RUN_DESCRIPTION.experiments) == 48
    names = [experiment.base_name for experiment in RUN_DESCRIPTION.experiments]
    assert len(names) == len(set(names))


def test_every_cell_uses_corrected_common_configuration_and_workspace_paths():
    for experiment in RUN_DESCRIPTION.experiments:
        command = experiment.cmd
        assert "--train_for_env_steps=100000000" in command
        assert "--encoder_conv_architecture=layer2_resnet18" in command
        assert "--encoder_reward_method=encourage" in command
        assert "--encoder_batch_loss=True" in command
        assert "--encoder_batch_loss_temperature=0.5" in command
        assert "--encoder_multi_activation_loss=True" in command
        assert "--num_policies=1" in command
        assert "--with_pbt=False" in command
        assert f"--wandb_project={PROJECT}" in command
        assert "/work/classic/fr_xl1014-train/" in command
        assert "--save_best_metric=distance_metric" not in command
        assert "--save_best_metric=z_00_openfield_map2_fixed_loc3_fixedlength_noreward_coverage_auc" in command


def test_direct_timing_iterative_structure_and_her_cells_are_exact():
    assert "--hrl_controllable_graph=False" in _command(1)
    assert "--hrl_target_timing=delayed" in _command(2)
    assert "--hrl_target_timing=immediate" in _command(3)
    assert "--iterative_update=True" in _command(4)
    assert "--dg_global_punishment_coeff=0.01" in _command(5)
    assert "--dg_row_repulsion_coeff=1.0" in _command(5)
    assert "--dg_ca3_temporal_exclusion_coeff=1.0" in _command(6)
    assert "--dg_orthogonal_recruitment=True" in _command(7)
    assert "--hrl_empirical_her=True" in _command(9)
    assert "--hrl_empirical_her_horizon=16" in _command(9)
    assert "--hrl_empirical_her_horizon=64" in _command(10)
    assert "--hrl_target_timing=delayed" in _command(11)


def test_manager_and_topology_comparisons_are_matched():
    assert "--hrl_timeout_margin_ratio=1.0" in _command(12)
    assert "--hrl_exploration_mode=False" in _command(12)
    assert "--hrl_exploration_mode=True" in _command(13)
    assert "--hrl_manager_exploration_probability=0.1" in _command(13)
    assert "--hrl_manager_mode=topology_visit_direct" in _command(14)
    assert "--hrl_manager_mode=frontier_direct" in _command(15)
    assert "--hrl_manager_mode=frontier_waypoint" in _command(16)
    for cell_number in (14, 15, 16):
        command = _command(cell_number)
        assert "--dg_orthogonal_recruitment=True" in command
        assert "--hrl_passive_edge_confidence_threshold=2" in command
        assert "--hrl_action_path_integration=False" in command


def test_preflight_selects_representative_cells_at_2m_frames():
    assert PREFLIGHT_RUN_DESCRIPTION.run_name == f"{BATCH_NAME}_preflight"
    assert len(PREFLIGHT_RUN_DESCRIPTION.experiments) == 4
    names = [experiment.base_name for experiment in PREFLIGHT_RUN_DESCRIPTION.experiments]
    assert all(name.startswith("PF_CCR_") for name in names)
    assert {int(name.split("_CCR_C", 1)[1][:2]) for name in names} == {1, 3, 10, 16}
    for experiment in PREFLIGHT_RUN_DESCRIPTION.experiments:
        assert "--train_for_env_steps=2000000" in experiment.cmd
        assert "--seed=99" in experiment.cmd
