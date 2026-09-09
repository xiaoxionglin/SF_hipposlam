from sample_factory.launcher.run_description import Experiment, RunDescription

BATCH_NAME = "intrmotiv_hrl_persistence_comparison_20260821"
PROJECT = "SF_IntrMotiv_HRLPersistenceComparison"
WANDB_ROOT = "/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/wandb"
SEEDS = [8, 99, 123]
HALF_LIVES = [5000, 10000, 20000]
ITERATIVE_MODES = [("sim", False), ("iter", True)]


COMMON_CLI = (
    "--train_for_env_steps=100000000 "
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
    "--save_best_every_sec=300 --save_best_after=1200 "
    "--decoder_mlp_layers 128 128 --env_frameskip=8 "
    "--core_name=BypassSS --DG_name=batchnorm_relu --Hippo_n_feature=16 --Hippo_L=64 "
    "--DG_BN_intercept=2.43 --depth_sensor=True --normalize_input=False "
    "--encoder_conv_architecture=layer2_resnet18 --use_rnn=True --rnn_type=gru --rnn_size=0 "
    "--nonlinearity=relu --with_wandb=True "
    "--wandb_user=xiaoxionglin-bernstein-center-freiburg "
    f"--wandb_project={PROJECT} "
    f"--wandb_dir={WANDB_ROOT} "
    "--benchmark=False --with_number_instruction=True --number_instruction_coef=9 --device=cpu "
    "--rec_distances=True --distance_learning=True --masked_distance_matrix=False "
    "--normalize_advantage=True --advantage_reward_source=internal --extra_encoder_losses=True "
    "--metric=sum --reward_scale=0.1 --double_value=False --reset_critic=False --reset_decoder=False "
    "--encoder_grad_coeff=1 --encoder_batch_loss=True --encoder_multi_activation_loss=False "
    "--encoder_unused_sequence_loss=False --extra_decoder_loss=False "
    "--encoder_population_usage_loss=False --encoder_density_loss_coeff=0.0 "
    "--encoder_collision_loss_coeff=0.0 --encoder_reward_method=encourage "
    "--hrl_worker_reward_mode=hit_distance --hrl_target_hit_reward=1.0 "
    "--hrl_distance_bonus_coeff=0.1 --hrl_timeout_margin_ratio=0.20 --hrl_timeout_margin_steps=2 "
    "--hrl_bootstrap_horizon=64 --hrl_min_target_visits=1 --hrl_edge_confidence_threshold=0.5 "
    "--ca3_predictor_shadow=False --exploration_coverage_telemetry=True "
    "--exploration_coverage_grid_size=100 "
)


def _schedule_cli(iterative: bool) -> str:
    return (
        f"--iterative_update={iterative} "
        "--iterative_initial_encoder_steps=128 --iterative_decoder_steps=512 "
        "--iterative_encoder_steps=128 --iterative_start_phase=decoder "
    )


def global_fixed(seed: int, half_life: int, mode_tag: str, iterative: bool) -> Experiment:
    name = f"GHRL_F16_L64_T243_HL{half_life}_{mode_tag}_S{seed}"
    cli = (
        COMMON_CLI
        + "--env=openfield_map2_fixed_loc3_fixedlength_noreward "
        + "--save_best_metric=distance_metric --hrl_controllable_graph=True "
        + "--hrl_graph_memory=policy_buffer --hrl_persistent_fast_weights=False "
        + f"--hrl_fast_weight_half_life_options={half_life} "
        + f"--seed={seed} --wandb_group={BATCH_NAME}_global_fixed "
        + "--wandb_tags hrl global_graph fixed_episode policy_buffer hit_distance nopbt "
        + _schedule_cli(iterative)
    )
    return Experiment(name, cli, [{}])


def stream_long(seed: int, half_life: int, mode_tag: str, iterative: bool) -> Experiment:
    name = f"LHRL_F16_L64_T243_HL{half_life}_{mode_tag}_S{seed}"
    cli = (
        COMMON_CLI
        + "--env=openfield_map2_fixed_loc3_longepisode_noreward "
        # Sample Factory incorporates this name into checkpoint filenames, so
        # periodic slash-delimited telemetry remains analysis-only.
        + "--save_best_metric=distance_metric "
        + "--exploration_window_steps=900 --hrl_controllable_graph=True "
        + "--hrl_graph_memory=episode --hrl_persistent_fast_weights=False "
        + f"--hrl_fast_weight_half_life_options={half_life} "
        + f"--seed={seed} --wandb_group={BATCH_NAME}_stream_long "
        + "--wandb_tags hrl stream_graph long_episode hit_distance nopbt "
        + _schedule_cli(iterative)
    )
    return Experiment(name, cli, [{}])


def flat_long(seed: int, mode_tag: str, iterative: bool) -> Experiment:
    name = f"FLATLONG_F16_L64_T243_{mode_tag}_S{seed}"
    cli = (
        COMMON_CLI
        + "--env=openfield_map2_fixed_loc3_longepisode_noreward "
        + "--save_best_metric=distance_metric "
        + "--exploration_window_steps=900 --hrl_controllable_graph=False "
        + f"--seed={seed} --wandb_group={BATCH_NAME}_flat_long "
        + "--wandb_tags flat_baseline long_episode encourage nopbt "
        + _schedule_cli(iterative)
    )
    return Experiment(name, cli, [{}])


RUN_DESCRIPTION = RunDescription(
    BATCH_NAME,
    experiments=[
        *[
            global_fixed(seed, half_life, mode_tag, iterative)
            for seed in SEEDS
            for half_life in HALF_LIVES
            for mode_tag, iterative in ITERATIVE_MODES
        ],
        *[
            stream_long(seed, half_life, mode_tag, iterative)
            for seed in SEEDS
            for half_life in HALF_LIVES
            for mode_tag, iterative in ITERATIVE_MODES
        ],
        *[flat_long(seed, mode_tag, iterative) for seed in SEEDS for mode_tag, iterative in ITERATIVE_MODES],
    ],
)
