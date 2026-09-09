"""Matched flat/global-HRL test of DG global punishment and row repulsion."""

from __future__ import annotations

from dataclasses import dataclass

from sample_factory.launcher.run_description import Experiment, RunDescription

BATCH_NAME = "intrmotiv_encourage_dg_regularizers_20260825"
PROJECT = "SF_IntrMotiv_EncourageDGRegularizers"
WANDB_ROOT = "/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/wandb"
SEEDS = (8, 99, 123)
THRESHOLDS = (2.43, 2.20)


@dataclass(frozen=True)
class LossArm:
    tag: str
    global_coeff: float
    row_coeff: float


LOSS_ARMS = (
    LossArm("CTRL", 0.0, 0.0),
    LossArm("G001", 0.01, 0.0),
    LossArm("G003", 0.03, 0.0),
    LossArm("R100", 0.0, 1.0),
    LossArm("G001_R100", 0.01, 1.0),
)


COMMON_CLI = (
    "--env=openfield_map2_fixed_loc3_fixedlength_noreward --train_for_env_steps=100000000 "
    "--algo=APPO --gamma=0.99 --learning_rate=0.0002 "
    "--exploration_loss_coeff=0.005 --value_loss_coeff=0.3 --ppo_clip_ratio=0.25 "
    "--num_workers=32 --num_envs_per_worker=2 --worker_num_splits=2 "
    "--num_epochs=1 --rollout=64 --recurrence=64 --batch_size=2048 --num_batches_per_epoch=2 "
    "--decorrelate_experience_max_seconds=120 --max_grad_norm=0.0 "
    "--dmlab_renderer=software --dmlab_extended_action_set=False --dmlab_reduced_action_set=True "
    "--dmlab_one_task_per_worker=True --dmlab_use_level_cache=True "
    "--set_workers_cpu_affinity=True --force_envs_single_thread=True "
    "--num_policies=1 --with_pbt=False --max_policy_lag=35 --use_record_episode_statistics=True "
    "--keep_checkpoints=8 --save_every_sec=300 --save_milestones_sec=1800 "
    "--save_best_every_sec=300 --save_best_after=1200 --save_best_metric=distance_metric "
    "--decoder_mlp_layers 128 128 --env_frameskip=8 "
    "--core_name=BypassSS --DG_name=batchnorm_relu --Hippo_n_feature=16 --Hippo_R=8 --Hippo_L=64 "
    "--depth_sensor=True --normalize_input=False --encoder_conv_architecture=layer2_resnet18 "
    "--use_rnn=True --rnn_type=gru --rnn_size=0 --nonlinearity=relu "
    "--with_wandb=True --wandb_user=xiaoxionglin-bernstein-center-freiburg "
    f"--wandb_project={PROJECT} --wandb_dir={WANDB_ROOT} "
    "--benchmark=False --with_number_instruction=True --number_instruction_coef=9 --device=cpu "
    "--rec_distances=True --distance_learning=True --masked_distance_matrix=False "
    "--normalize_advantage=True --advantage_reward_source=internal --extra_encoder_losses=True "
    "--metric=sum --reward_scale=0.1 --double_value=False --reset_critic=False --reset_decoder=False "
    "--encoder_grad_coeff=1 --encoder_batch_loss=True --encoder_multi_activation_loss=False "
    "--encoder_unused_sequence_loss=False --extra_decoder_loss=False "
    "--encoder_population_usage_loss=False --encoder_density_loss_coeff=0.0 "
    "--encoder_collision_loss_coeff=0.0 --encoder_reward_method=encourage "
    "--dg_global_punishment_temperature=0.5 --iterative_update=False "
    "--ca3_predictor_shadow=False --exploration_coverage_telemetry=True "
    "--exploration_coverage_grid_size=100 "
)


def threshold_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def run_name(architecture: str, threshold: float, arm: LossArm, seed: int, prefix: str = "") -> str:
    family = "FREG" if architecture == "flat" else "GREG"
    prefix = f"{prefix}_" if prefix else ""
    return f"{prefix}{family}_F16_L64_T{threshold_tag(threshold)}_{arm.tag}_ENC_S{seed}"


def make_experiment(
    architecture: str,
    threshold: float,
    arm: LossArm,
    seed: int,
    *,
    train_for_env_steps: int = 100_000_000,
    prefix: str = "",
) -> Experiment:
    if architecture not in {"flat", "global_fixed"}:
        raise ValueError(f"Unknown architecture {architecture}")
    name = run_name(architecture, threshold, arm, seed, prefix)
    cli = (
        COMMON_CLI.replace("--train_for_env_steps=100000000", f"--train_for_env_steps={train_for_env_steps}")
        + f"--seed={seed} --DG_BN_intercept={threshold} "
        + f"--dg_global_punishment_coeff={arm.global_coeff} --dg_row_repulsion_coeff={arm.row_coeff} "
    )
    if architecture == "flat":
        cli += f"--hrl_controllable_graph=False --wandb_group={BATCH_NAME}_flat "
        cli += "--wandb_tags encourage batch_loss flat dg_regularizers simultaneous nopbt "
    else:
        cli += "--hrl_controllable_graph=True --hrl_graph_memory=policy_buffer "
        cli += "--hrl_persistent_fast_weights=False --hrl_fast_weight_half_life_options=10000 "
        cli += "--hrl_worker_reward_mode=hit_distance --hrl_target_hit_reward=1.0 "
        cli += "--hrl_distance_bonus_coeff=0.1 --hrl_timeout_margin_ratio=0.20 "
        cli += "--hrl_timeout_margin_steps=2 --hrl_bootstrap_horizon=64 --hrl_min_target_visits=1 "
        cli += "--hrl_edge_confidence_threshold=0.5 "
        cli += f"--wandb_group={BATCH_NAME}_global_fixed "
        cli += "--wandb_tags encourage batch_loss hrl global_graph global10k hit_distance simultaneous nopbt "
    return Experiment(name, cli, [{}])


RUN_DESCRIPTION = RunDescription(
    BATCH_NAME,
    experiments=[
        make_experiment(architecture, threshold, arm, seed)
        for architecture in ("flat", "global_fixed")
        for threshold in THRESHOLDS
        for arm in LOSS_ARMS
        for seed in SEEDS
    ],
)
