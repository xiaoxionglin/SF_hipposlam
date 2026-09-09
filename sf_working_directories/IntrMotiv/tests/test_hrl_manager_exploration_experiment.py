from sf_working_directories.IntrMotiv.dmlab.experiments.hrl_manager_exploration import (
    BATCH_NAME,
    PROJECT,
    RUN_DESCRIPTION,
)
from sf_working_directories.IntrMotiv.dmlab.experiments.hrl_manager_exploration_preflight import (
    RUN_DESCRIPTION as PREFLIGHT_RUN_DESCRIPTION,
)


def test_manager_exploration_grid_has_24_unique_replicated_runs():
    experiments = RUN_DESCRIPTION.experiments
    assert len(experiments) == 24
    assert len({experiment.base_name for experiment in experiments}) == 24
    commands = [experiment.cmd for experiment in experiments]
    for seed in (8, 99, 123):
        for temporal in (0.0, 1.0):
            for enabled, probability in ((False, 0.0), (True, 0.0), (True, 0.1), (True, 0.25)):
                matches = [
                    command
                    for command in commands
                    if f"--seed={seed} " in command
                    and f"--dg_ca3_temporal_exclusion_coeff={temporal}" in command
                    and f"--hrl_exploration_mode={enabled}" in command
                    and f"--hrl_manager_exploration_probability={probability}" in command
                ]
                assert len(matches) == 1


def test_manager_exploration_grid_preserves_architecture_and_uses_workspace():
    for experiment in RUN_DESCRIPTION.experiments:
        command = experiment.cmd
        assert "--train_for_env_steps=100000000" in command
        assert "--hrl_controllable_graph=True" in command
        assert "--hrl_graph_memory=policy_buffer" in command
        assert "--hrl_worker_reward_mode=hit_distance" in command
        assert "--hrl_timeout_margin_ratio=1.0" in command
        assert "--hrl_timeout_margin_steps=8" in command
        assert "--hrl_bootstrap_horizon=96" in command
        assert "--hrl_exploration_horizon=64" in command
        assert "--dg_orthogonal_recruitment=True" in command
        assert "--encoder_conv_architecture=layer2_resnet18" in command
        assert "--num_policies=1" in command
        assert "--with_pbt=False" in command
        assert f"--wandb_project={PROJECT}" in command
        assert "/work/classic/fr_xl1014-train/" in command


def test_manager_exploration_preflight_exercises_forced_and_dense_modes():
    assert PREFLIGHT_RUN_DESCRIPTION.run_name == f"{BATCH_NAME}_preflight"
    assert len(PREFLIGHT_RUN_DESCRIPTION.experiments) == 2
    commands = [experiment.cmd for experiment in PREFLIGHT_RUN_DESCRIPTION.experiments]
    assert all("--train_for_env_steps=500000" in command for command in commands)
    assert any("--hrl_manager_exploration_probability=0.0" in command for command in commands)
    assert any("--hrl_manager_exploration_probability=1.0" in command for command in commands)
