from sf_working_directories.IntrMotiv.dmlab.experiments.online_spatial_telemetry_preflight import (
    BATCH_NAME,
    PROJECT,
    RUN_DESCRIPTION,
)


def test_online_spatial_preflight_is_one_short_wandb_workspace_run():
    assert RUN_DESCRIPTION.run_name == BATCH_NAME
    assert len(RUN_DESCRIPTION.experiments) == 1
    experiment = RUN_DESCRIPTION.experiments[0]
    command = experiment.cmd
    assert experiment.base_name == "ONLINE_SPATIAL_GRAPH_PREFLIGHT_R6_S20260904"
    assert "--train_for_env_steps=131072" in command
    assert "--with_wandb=True" in command
    assert f"--wandb_project={PROJECT}" in command
    assert f"--wandb_group={BATCH_NAME}" in command
    assert "--online_spatial_telemetry=True" in command
    assert "--online_spatial_window_observations=4096" in command
    assert "--online_spatial_scalar_window_observations=1024" in command
    assert "--online_spatial_scalar_interval_frames=32768" in command
    assert "--online_spatial_snapshot_interval_frames=65536" in command
    assert "--online_spatial_snapshot_max_frames=65536" in command
    assert "--online_spatial_snapshot_targets=65536" in command
    assert "--hrl_controllable_graph=True" in command
    assert "--hrl_graph_memory=policy_buffer" in command
    assert "--exploration_coverage_telemetry=False" in command
