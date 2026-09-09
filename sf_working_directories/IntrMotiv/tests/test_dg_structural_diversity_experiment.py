from sf_working_directories.IntrMotiv.dmlab.experiments.dg_structural_diversity import (
    BATCH_NAME,
    PROJECT,
    RUN_DESCRIPTION,
)
from sf_working_directories.IntrMotiv.dmlab.experiments.dg_structural_diversity_preflight import (
    RUN_DESCRIPTION as PREFLIGHT_RUN_DESCRIPTION,
)


def test_production_grid_is_full_replicated_factorial():
    experiments = RUN_DESCRIPTION.experiments
    assert len(experiments) == 48
    names = {experiment.base_name for experiment in experiments}
    assert len(names) == 48
    for experiment in experiments:
        assert "--train_for_env_steps=100000000" in experiment.cmd
        assert "--DG_BN_intercept=2.43" in experiment.cmd
        assert "--encoder_reward_method=encourage" in experiment.cmd
        assert "--encoder_batch_loss=True" in experiment.cmd
        assert "--encoder_conv_architecture=layer2_resnet18" in experiment.cmd
        assert "--iterative_update=False" in experiment.cmd
        assert "--num_policies=1" in experiment.cmd
        assert "--with_pbt=False" in experiment.cmd
        assert f"--wandb_project={PROJECT}" in experiment.cmd
        assert "/work/classic/fr_xl1014-train/" in experiment.cmd


def test_grid_has_every_architecture_background_and_mechanism_cell_for_each_seed():
    commands = [experiment.cmd for experiment in RUN_DESCRIPTION.experiments]
    for seed in (8, 99, 123):
        for hrl in (False, True):
            for global_coeff, row_coeff in ((0.0, 0.0), (0.01, 1.0)):
                for temporal in (0.0, 1.0):
                    for recruitment in (False, True):
                        matches = [
                            command
                            for command in commands
                            if f"--seed={seed} " in command
                            and f"--hrl_controllable_graph={hrl}" in command
                            and f"--dg_global_punishment_coeff={global_coeff}" in command
                            and f"--dg_row_repulsion_coeff={row_coeff}" in command
                            and f"--dg_ca3_temporal_exclusion_coeff={temporal}" in command
                            and f"--dg_orthogonal_recruitment={recruitment}" in command
                        ]
                        assert len(matches) == 1


def test_preflight_calibrates_loss_and_exercises_recruitment():
    assert PREFLIGHT_RUN_DESCRIPTION.run_name == f"{BATCH_NAME}_preflight"
    assert len(PREFLIGHT_RUN_DESCRIPTION.experiments) == 5
    commands = [experiment.cmd for experiment in PREFLIGHT_RUN_DESCRIPTION.experiments]
    assert any("--dg_ca3_temporal_exclusion_coeff=0.03" in command for command in commands)
    assert any("--dg_ca3_temporal_exclusion_coeff=0.3" in command for command in commands)
    assert any("--dg_orthogonal_recruitment=True" in command for command in commands)
    assert all("--train_for_env_steps=2000000" in command for command in commands)
