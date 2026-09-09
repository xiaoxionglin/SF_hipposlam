"""Factorial test of CA3 temporal exclusion and orthogonal DG recruitment."""

from __future__ import annotations

from dataclasses import dataclass

from sample_factory.launcher.run_description import Experiment, RunDescription


BATCH_NAME = "intrmotiv_dg_structural_diversity_20260826"
PROJECT = "SF_IntrMotiv_DGStructuralDiversity"
WANDB_ROOT = "/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/wandb"
DMLAB_CACHE = "/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/dmlab_cache"
SEEDS = (8, 99, 123)
TEMPORAL_EXCLUSION_COEFF = 1.0


@dataclass(frozen=True)
class Background:
    tag: str
    global_coeff: float
    row_coeff: float


BACKGROUNDS = (
    Background("CTRL", 0.0, 0.0),
    Background("G001_R100", 0.01, 1.0),
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
    f"--dmlab_level_cache_path={DMLAB_CACHE} "
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
    "--exploration_coverage_grid_size=100 --DG_BN_intercept=2.43 "
)


def encoded_float(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def run_name(
    architecture: str,
    background: Background,
    temporal_coeff: float,
    recruitment: bool,
    seed: int,
    prefix: str = "",
) -> str:
    family = "FSD" if architecture == "flat" else "GSD"
    prefix = f"{prefix}_" if prefix else ""
    return (
        f"{prefix}{family}_F16_L64_T2p43_{background.tag}_"
        f"X{encoded_float(temporal_coeff)}_O{int(recruitment)}_S{seed}"
    )


def make_experiment(
    architecture: str,
    background: Background,
    temporal_coeff: float,
    recruitment: bool,
    seed: int,
    *,
    train_for_env_steps: int = 100_000_000,
    prefix: str = "",
    group_suffix: str = "production",
) -> Experiment:
    if architecture not in {"flat", "global_fixed"}:
        raise ValueError(f"Unknown architecture {architecture}")
    name = run_name(architecture, background, temporal_coeff, recruitment, seed, prefix)
    cli = (
        COMMON_CLI.replace("--train_for_env_steps=100000000", f"--train_for_env_steps={train_for_env_steps}")
        + f"--seed={seed} "
        + f"--dg_global_punishment_coeff={background.global_coeff} "
        + f"--dg_row_repulsion_coeff={background.row_coeff} "
        + f"--dg_ca3_temporal_exclusion_coeff={temporal_coeff} "
        + f"--dg_orthogonal_recruitment={recruitment} "
        + "--dg_orthogonal_recruitment_margin=0.5 "
        + "--dg_orthogonal_recruitment_residual_eps=1e-6 "
        + "--dg_orthogonal_recruitment_max_per_rollout=1 "
    )
    architecture_tag = "flat" if architecture == "flat" else "global_fixed"
    cli += f"--wandb_group={BATCH_NAME}_{group_suffix}_{architecture_tag}_{background.tag.lower()} "
    cli += (
        f"--wandb_tags encourage batch_loss {architecture_tag} {background.tag.lower()} "
        f"ca3x_{encoded_float(temporal_coeff)} orthogonal_recruit_{int(recruitment)} simultaneous nopbt "
    )
    if architecture == "flat":
        cli += "--hrl_controllable_graph=False "
    else:
        cli += "--hrl_controllable_graph=True --hrl_graph_memory=policy_buffer "
        cli += "--hrl_persistent_fast_weights=False --hrl_fast_weight_half_life_options=10000 "
        cli += "--hrl_worker_reward_mode=hit_distance --hrl_target_hit_reward=1.0 "
        cli += "--hrl_distance_bonus_coeff=0.1 --hrl_timeout_margin_ratio=0.20 "
        cli += "--hrl_timeout_margin_steps=2 --hrl_bootstrap_horizon=64 --hrl_min_target_visits=1 "
        cli += "--hrl_edge_confidence_threshold=0.5 "
    return Experiment(name, cli, [{}])


RUN_DESCRIPTION = RunDescription(
    BATCH_NAME,
    experiments=[
        make_experiment(architecture, background, temporal_coeff, recruitment, seed)
        for architecture in ("flat", "global_fixed")
        for background in BACKGROUNDS
        for temporal_coeff in (0.0, TEMPORAL_EXCLUSION_COEFF)
        for recruitment in (False, True)
        for seed in SEEDS
    ],
)
