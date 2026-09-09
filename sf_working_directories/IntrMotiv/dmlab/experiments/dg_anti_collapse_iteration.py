from sample_factory.launcher.run_description import Experiment, RunDescription


BATCH_NAME = "intrmotiv_dg_anti_collapse_20260824"
PROJECT = "SF_IntrMotiv_DGAntiCollapse"
WANDB_ROOT = "/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/wandb"
SEEDS = [8, 99, 123]
REWARD_METHODS = ["mean", "punish"]
THRESHOLDS = [1.8, 2.0, 2.2]
GLOBAL_COEFFS = [0.0, 0.01, 0.03]


COMMON = (
    "--env=openfield_map2_fixed_loc3_fixedlength_noreward --train_for_env_steps=80000000 "
    "--algo=APPO --gamma=0.99 --learning_rate=0.0002 --exploration_loss_coeff=0.005 --value_loss_coeff=0.3 "
    "--ppo_clip_ratio=0.25 --num_workers=32 --num_envs_per_worker=2 --worker_num_splits=2 "
    "--num_epochs=1 --rollout=64 --recurrence=64 --batch_size=2048 --num_batches_per_epoch=2 "
    "--decorrelate_experience_max_seconds=120 --max_grad_norm=0.0 --dmlab_renderer=software "
    "--dmlab_extended_action_set=False --dmlab_reduced_action_set=True --dmlab_one_task_per_worker=True "
    "--dmlab_use_level_cache=True --set_workers_cpu_affinity=True --force_envs_single_thread=True "
    "--num_policies=1 --with_pbt=False --max_policy_lag=35 --use_record_episode_statistics=True "
    "--keep_checkpoints=8 --save_every_sec=300 --save_milestones_sec=1800 --save_best_every_sec=300 "
    "--save_best_after=1200 --save_best_metric=distance_metric --decoder_mlp_layers 128 128 --env_frameskip=8 "
    "--core_name=BypassSS --DG_name=batchnorm_relu --Hippo_n_feature=16 --Hippo_L=64 --Hippo_R=8 "
    "--depth_sensor=True --normalize_input=False --encoder_conv_architecture=layer2_resnet18 "
    "--use_rnn=True --rnn_type=gru --rnn_size=0 --nonlinearity=relu --with_wandb=True "
    "--wandb_user=xiaoxionglin-bernstein-center-freiburg "
    f"--wandb_project={PROJECT} --wandb_dir={WANDB_ROOT} "
    "--benchmark=False --with_number_instruction=True --number_instruction_coef=9 --device=cpu "
    "--rec_distances=True --distance_learning=True --masked_distance_matrix=False --normalize_advantage=True "
    "--advantage_reward_source=internal --extra_encoder_losses=True --metric=sum --reward_scale=0.1 "
    "--double_value=False --reset_critic=False --reset_decoder=False --encoder_grad_coeff=1 "
    "--encoder_batch_loss=False --encoder_multi_activation_loss=False --encoder_unused_sequence_loss=False "
    "--extra_decoder_loss=False --encoder_population_usage_loss=False --encoder_density_loss_coeff=0.0 "
    "--encoder_collision_loss_coeff=0.0 --ca3_predictor_shadow=False --exploration_coverage_telemetry=True "
    "--exploration_coverage_grid_size=100 --iterative_update=False "
    "--dg_global_punishment_temperature=0.5 "
)


def tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def common_variant(seed: int, method: str, threshold: float, global_coeff: float, row_coeff: float) -> str:
    return (
        COMMON
        + f"--seed={seed} --encoder_reward_method={method} --DG_BN_intercept={threshold} "
        + f"--dg_global_punishment_coeff={global_coeff} --dg_row_repulsion_coeff={row_coeff} "
    )


def flat(seed: int, method: str, threshold: float, global_coeff: float) -> Experiment:
    name = f"FAC_F16_T{tag(threshold)}_G{tag(global_coeff)}_ER{method}_S{seed}"
    cli = common_variant(seed, method, threshold, global_coeff, 0.0)
    cli += f"--hrl_controllable_graph=False --wandb_group={BATCH_NAME}_flat_global_punishment "
    cli += "--wandb_tags anti_collapse flat mean_punish global_punishment nopbt "
    return Experiment(name, cli, [{}])


def global_hrl(seed: int, method: str, threshold: float, global_coeff: float) -> Experiment:
    name = f"GAC_F16_T{tag(threshold)}_G{tag(global_coeff)}_ER{method}_S{seed}"
    cli = common_variant(seed, method, threshold, global_coeff, 0.0)
    cli += "--hrl_controllable_graph=True --hrl_graph_memory=policy_buffer --hrl_persistent_fast_weights=False "
    cli += "--hrl_fast_weight_half_life_options=5000 --hrl_worker_reward_mode=hit_distance "
    cli += "--hrl_target_hit_reward=1.0 --hrl_distance_bonus_coeff=0.1 --hrl_timeout_margin_ratio=0.20 "
    cli += "--hrl_timeout_margin_steps=2 --hrl_bootstrap_horizon=64 --hrl_min_target_visits=1 "
    cli += "--hrl_edge_confidence_threshold=0.5 "
    cli += f"--wandb_group={BATCH_NAME}_global5k_global_punishment "
    cli += "--wandb_tags anti_collapse hrl global_graph global5k simultaneous hit_distance nopbt "
    return Experiment(name, cli, [{}])


def row_repulsion_flat(seed: int, method: str) -> Experiment:
    name = f"FAR_F16_T2p43_R0p01_ER{method}_S{seed}"
    cli = common_variant(seed, method, 2.43, 0.0, 0.01)
    cli += f"--hrl_controllable_graph=False --wandb_group={BATCH_NAME}_row_repulsion "
    cli += "--wandb_tags anti_collapse flat row_repulsion separate_arm nopbt "
    return Experiment(name, cli, [{}])


def row_repulsion_global_hrl(seed: int, method: str) -> Experiment:
    name = f"GAR_F16_T2p43_R0p01_ER{method}_S{seed}"
    cli = common_variant(seed, method, 2.43, 0.0, 0.01)
    cli += "--hrl_controllable_graph=True --hrl_graph_memory=policy_buffer --hrl_persistent_fast_weights=False "
    cli += "--hrl_fast_weight_half_life_options=5000 --hrl_worker_reward_mode=hit_distance "
    cli += "--hrl_target_hit_reward=1.0 --hrl_distance_bonus_coeff=0.1 --hrl_timeout_margin_ratio=0.20 "
    cli += "--hrl_timeout_margin_steps=2 --hrl_bootstrap_horizon=64 --hrl_min_target_visits=1 "
    cli += "--hrl_edge_confidence_threshold=0.5 "
    cli += f"--wandb_group={BATCH_NAME}_row_repulsion "
    cli += "--wandb_tags anti_collapse hrl global_graph global5k row_repulsion separate_arm nopbt "
    return Experiment(name, cli, [{}])


RUN_DESCRIPTION = RunDescription(
    BATCH_NAME,
    experiments=[
        *[
            flat(seed, method, threshold, coeff)
            for seed in SEEDS
            for method in REWARD_METHODS
            for threshold in THRESHOLDS
            for coeff in GLOBAL_COEFFS
        ],
        *[
            global_hrl(seed, method, threshold, coeff)
            for seed in SEEDS
            for method in REWARD_METHODS
            for threshold in THRESHOLDS
            for coeff in GLOBAL_COEFFS
        ],
        *[row_repulsion_flat(seed, method) for seed in SEEDS for method in REWARD_METHODS],
        *[row_repulsion_global_hrl(seed, method) for seed in SEEDS for method in REWARD_METHODS],
    ],
)
