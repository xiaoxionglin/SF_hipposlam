from sample_factory.launcher.run_description import Experiment, RunDescription
from sf_working_directories.IntrMotiv.dmlab.experiments.dg_anti_collapse_iteration import BATCH_NAME, common_variant

PREFLIGHT_NAME = f"{BATCH_NAME}_preflight"


def experiment(name: str, hrl: bool, row_repulsion: bool) -> Experiment:
    cli = common_variant(
        99,
        "punish",
        2.0 if not row_repulsion else 2.43,
        0.03 if not row_repulsion else 0.0,
        0.01 if row_repulsion else 0.0,
    )
    cli = cli.replace("--train_for_env_steps=80000000", "--train_for_env_steps=1000000")
    cli += f"--wandb_group={PREFLIGHT_NAME} "
    if hrl:
        cli += "--hrl_controllable_graph=True --hrl_graph_memory=policy_buffer --hrl_persistent_fast_weights=False "
        cli += "--hrl_fast_weight_half_life_options=5000 --hrl_worker_reward_mode=hit_distance "
        cli += "--hrl_target_hit_reward=1.0 --hrl_distance_bonus_coeff=0.1 --hrl_timeout_margin_ratio=0.20 "
        cli += "--hrl_timeout_margin_steps=2 --hrl_bootstrap_horizon=64 --hrl_min_target_visits=1 "
        cli += "--hrl_edge_confidence_threshold=0.5 --wandb_tags anti_collapse preflight hrl global5k "
    else:
        cli += "--hrl_controllable_graph=False --wandb_tags anti_collapse preflight flat "
    return Experiment(name, cli, [{}])


RUN_DESCRIPTION = RunDescription(
    PREFLIGHT_NAME,
    experiments=[
        experiment("flat_global_punish", hrl=False, row_repulsion=False),
        experiment("global5k_global_punish", hrl=True, row_repulsion=False),
        experiment("flat_row_repulsion", hrl=False, row_repulsion=True),
        experiment("global5k_row_repulsion", hrl=True, row_repulsion=True),
    ],
)
