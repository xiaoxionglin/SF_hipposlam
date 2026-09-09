from sf_working_directories.IntrMotiv.dmlab.experiments.frontier_manager_isolation import (
    CONDITIONS as FRONTIER_ISOLATION_CONDITIONS,
)
from sf_working_directories.IntrMotiv.dmlab.experiments.frontier_manager_isolation import (
    PROJECT as FRONTIER_ISOLATION_PROJECT,
)
from sf_working_directories.IntrMotiv.dmlab.experiments.frontier_manager_isolation import (
    RUN_DESCRIPTION as FRONTIER_ISOLATION_RUN_DESCRIPTION,
)
from sf_working_directories.IntrMotiv.dmlab.experiments.frontier_manager_matched_control import (
    PROJECT as MATCHED_CONTROL_PROJECT,
)
from sf_working_directories.IntrMotiv.dmlab.experiments.frontier_manager_matched_control import (
    RUN_DESCRIPTION as MATCHED_CONTROL_RUN_DESCRIPTION,
)
from sf_working_directories.IntrMotiv.dmlab.experiments.frontier_manager_waypoint_extension import (
    RUN_DESCRIPTION as FRONTIER_WAYPOINT_EXTENSION_RUN_DESCRIPTION,
)
from sf_working_directories.IntrMotiv.dmlab.experiments.topological_frontier_motion_preflight import (
    RUN_DESCRIPTION as MOTION_PREFLIGHT_RUN_DESCRIPTION,
)
from sf_working_directories.IntrMotiv.dmlab.experiments.topological_frontier_planning import (
    BATCH_NAME,
    CELLS,
    PROJECT,
    RUN_DESCRIPTION,
)
from sf_working_directories.IntrMotiv.dmlab.experiments.topological_frontier_planning_preflight import (
    RUN_DESCRIPTION as PREFLIGHT_RUN_DESCRIPTION,
)


def test_production_grid_has_16_cells_three_seeds_and_unique_names():
    experiments = RUN_DESCRIPTION.experiments
    assert len(CELLS) == 16
    assert len(experiments) == 48
    assert len({experiment.base_name for experiment in experiments}) == 48
    for cell in CELLS:
        matches = [
            experiment for experiment in experiments if f"C{cell.number:02d}_{cell.tag}_" in experiment.base_name
        ]
        assert len(matches) == 3
        for seed in (8, 99, 123):
            assert sum(f"_S{seed}" in experiment.base_name for experiment in matches) == 1


def test_production_grid_preserves_fixed_architecture_resources_and_workspace_paths():
    for experiment in RUN_DESCRIPTION.experiments:
        command = experiment.cmd
        assert "--train_for_env_steps=100000000" in command
        assert "--num_workers=32" in command
        assert "--num_envs_per_worker=2" in command
        assert "--num_policies=1" in command
        assert "--with_pbt=False" in command
        assert "--encoder_conv_architecture=layer2_resnet18" in command
        assert "--Hippo_n_feature=16" in command
        assert "--Hippo_R=8" in command
        assert "--Hippo_L=64" in command
        assert "--DG_BN_intercept=2.43" in command
        assert "--encoder_reward_method=encourage" in command
        assert "--encoder_batch_loss=True" in command
        assert f"--wandb_project={PROJECT}" in command
        assert f"--wandb_group={BATCH_NAME}" in command
        assert "/work/classic/fr_xl1014-train/" in command


def test_cells_encode_the_requested_ablation_matrix():
    commands = {
        cell.number: next(e.cmd for e in RUN_DESCRIPTION.experiments if f"C{cell.number:02d}_" in e.base_name)
        for cell in CELLS
    }
    assert "--hrl_controllable_graph=False" in commands[1]
    assert "--hrl_manager_mode=visit_direct" in commands[2]
    assert "--hrl_manager_mode=frontier_direct" in commands[3]
    assert "--hrl_manager_mode=frontier_waypoint" in commands[4]
    assert "--hrl_action_path_integration=False" in commands[4]
    assert "--hrl_action_path_integration=True" in commands[5]
    assert "--hrl_motion_policy_input=False" in commands[5]
    assert "--hrl_motion_policy_input=True" in commands[6]
    assert "--dg_path_scatter_coeff=0.01" in commands[8]
    assert "--dg_path_scatter_coeff=0.005" in commands[9]
    assert "--dg_path_scatter_coeff=0.05" in commands[10]
    assert "--dg_path_scatter_min_displacement=4.0" in commands[11]
    assert "--dg_path_scatter_min_displacement=12.0" in commands[12]
    assert "--hrl_frontier_uncertainty_weight=0.5" in commands[13]
    assert "--hrl_frontier_uncertainty_weight=2.0" in commands[14]
    assert "--dg_global_punishment_coeff=0.01" in commands[15]
    assert "--dg_row_repulsion_coeff=1.0" in commands[15]
    assert "--dg_ca3_temporal_exclusion_coeff=1.0" in commands[15]
    assert "--dg_orthogonal_recruitment=False" in commands[15]
    assert "--hrl_landmark_geometry=se2" in commands[16]


def test_preflights_cover_planning_motion_scatter_and_se2_for_multiple_episodes():
    experiments = PREFLIGHT_RUN_DESCRIPTION.experiments
    assert PREFLIGHT_RUN_DESCRIPTION.run_name == f"{BATCH_NAME}_preflight"
    assert len(experiments) == 3
    assert all("--train_for_env_steps=2000000" in experiment.cmd for experiment in experiments)
    assert any("C04_" in experiment.base_name for experiment in experiments)
    assert any("C08_" in experiment.base_name for experiment in experiments)
    assert any("C16_" in experiment.base_name for experiment in experiments)


def test_frontier_manager_isolation_has_matched_direct_and_frontier_conditions():
    assert len(FRONTIER_ISOLATION_RUN_DESCRIPTION.experiments) == 6
    assert [condition.manager for condition in FRONTIER_ISOLATION_CONDITIONS] == [
        "visit_direct",
        "frontier_direct",
    ]
    for experiment in FRONTIER_ISOLATION_RUN_DESCRIPTION.experiments:
        command = experiment.cmd
        assert f"--wandb_project={FRONTIER_ISOLATION_PROJECT}" in command
        assert "--hrl_action_path_integration=False" in command
        assert "--hrl_motion_policy_input=False" in command
        assert "--dg_path_scatter_coeff=0.0" in command
        assert "--hrl_landmark_geometry=none" in command


def test_frontier_waypoint_extension_completes_the_twenty_run_paired_design():
    experiments = FRONTIER_WAYPOINT_EXTENSION_RUN_DESCRIPTION.experiments
    assert len(experiments) == 14
    assert sum("--hrl_manager_mode=visit_direct" in experiment.cmd for experiment in experiments) == 2
    assert sum("--hrl_manager_mode=frontier_direct" in experiment.cmd for experiment in experiments) == 2
    assert sum("--hrl_manager_mode=frontier_waypoint" in experiment.cmd for experiment in experiments) == 10
    assert sum("--hrl_frontier_uncertainty_weight=1.0" in experiment.cmd for experiment in experiments) == 9
    assert sum("--hrl_frontier_uncertainty_weight=0.5" in experiment.cmd for experiment in experiments) == 5
    assert all("--hrl_exploration_mode=False" in experiment.cmd for experiment in experiments)


def test_topology_matched_control_holds_local_exploration_and_validation_constant():
    experiments = MATCHED_CONTROL_RUN_DESCRIPTION.experiments
    assert len(experiments) == 10
    assert sum("--hrl_manager_mode=topology_visit_direct" in experiment.cmd for experiment in experiments) == 5
    assert sum("--hrl_manager_mode=frontier_direct" in experiment.cmd for experiment in experiments) == 5
    assert all(f"--wandb_project={MATCHED_CONTROL_PROJECT}" in experiment.cmd for experiment in experiments)
    assert all("--wandb_tags frontier_manager_causal_control" in experiment.cmd for experiment in experiments)
    assert all("--hrl_passive_edge_confidence_threshold=2" in experiment.cmd for experiment in experiments)
    assert all("--hrl_exploration_horizon=64" in experiment.cmd for experiment in experiments)


def test_motion_preflight_is_action_only_and_has_a_short_run_length():
    experiments = MOTION_PREFLIGHT_RUN_DESCRIPTION.experiments
    assert len(experiments) == 1
    command = experiments[0].cmd
    assert "--env=openfield_map2_fixed_loc3_noreward" in command
    assert "fixedlength_noreward" not in command
    assert "--train_for_env_steps=2000000" in command
    assert "--hrl_action_path_integration=True" in command
    assert "--hrl_motion_policy_input=False" in command
    assert "--dg_path_scatter_coeff=0.0" in command
