from sample_factory.launcher.run_description import Experiment, RunDescription

BATCH_NAME = "intrmotiv_hrl_iteration2_iterative_sweep3_20260820"
SEEDS = [8, 99, 123]
ENCODER_REWARD_METHODS = ["punish", "encourage", "mean"]
WORKER_REWARD_MODES = ["hit", "hit_distance"]

BASE_CLI = (
    "--env=openfield_map2_fixed_loc3_fixedlength_noreward "
    # Sample Factory applies this stop threshold to each policy. With four
    # PBT policies, 25M per policy gives an approximately 100M population
    # budget while retaining all four inherited policies.
    "--train_for_env_steps=25000000 "
    "--algo=APPO "
    "--gamma=0.99 "
    "--learning_rate=0.0002 "
    "--exploration_loss_coeff=0.005 "
    "--value_loss_coeff=0.3 "
    "--ppo_clip_ratio=0.25 "
    "--num_workers=32 "
    "--num_envs_per_worker=2 "
    "--worker_num_splits=2 "
    "--num_epochs=1 "
    "--rollout=64 "
    "--recurrence=64 "
    "--batch_size=2048 "
    "--num_batches_per_epoch=2 "
    "--decorrelate_experience_max_seconds=120 "
    "--max_grad_norm=0.0 "
    "--dmlab_renderer=software "
    "--dmlab_extended_action_set=False "
    "--dmlab_reduced_action_set=True "
    "--dmlab_one_task_per_worker=True "
    "--dmlab_use_level_cache=True "
    "--set_workers_cpu_affinity=True "
    "--force_envs_single_thread=True "
    "--num_policies=4 "
    "--policy_workers_per_policy=1 "
    "--with_pbt=True "
    "--pbt_mix_policies_in_one_env=False "
    "--pbt_start_mutation=2500000 "
    "--pbt_period_env_steps=1250000 "
    "--pbt_replace_fraction=0.2 "
    "--pbt_replace_reward_gap=0.02 "
    "--pbt_replace_reward_gap_absolute=1.0 "
    "--pbt_mutation_rate=0.15 "
    "--pbt_target_objective=intrmotiv_pbt_objective "
    "--pbt_perturb_min=1.1 "
    "--pbt_perturb_max=1.3 "
    "--max_policy_lag=35 "
    "--use_record_episode_statistics=True "
    "--keep_checkpoints=8 "
    "--save_every_sec=300 "
    "--save_milestones_sec=1800 "
    "--save_best_every_sec=300 "
    "--save_best_after=1250000 "
    "--decoder_mlp_layers 128 128 "
    "--env_frameskip=8 "
    "--core_name=BypassSS "
    "--DG_name=batchnorm_relu "
    "--Hippo_n_feature=16 "
    "--Hippo_L=64 "
    "--DG_BN_intercept=2.0 "
    "--depth_sensor=True "
    "--normalize_input=False "
    "--encoder_conv_architecture=layer2_resnet18 "
    "--use_rnn=True "
    "--rnn_type=gru "
    "--rnn_size=0 "
    "--nonlinearity=relu "
    "--with_wandb=True "
    "--wandb_user=xiaoxionglin-bernstein-center-freiburg "
    "--wandb_project=SF_HRL_Intrinsic_ArchSearch "
    f"--wandb_group={BATCH_NAME} "
    "--wandb_tags hrl_iteration2 pbt4 fixed_length coverage_objective "
    "--benchmark=False "
    "--with_number_instruction=True "
    "--number_instruction_coef=9 "
    "--save_best_metric=z_00_openfield_map2_fixed_loc3_fixedlength_noreward_coverage_auc "
    "--device=cpu "
    "--rec_distances=True "
    "--distance_learning=True "
    "--masked_distance_matrix=False "
    "--normalize_advantage=True "
    "--advantage_reward_source=internal "
    "--extra_encoder_losses=True "
    "--metric=sum "
    "--reward_scale=0.1 "
    "--double_value=False "
    "--reset_critic=False "
    "--reset_decoder=False "
    "--encoder_grad_coeff=1 "
    "--encoder_batch_loss=True "
    "--iterative_update=True "
    "--iterative_initial_encoder_steps=128 "
    "--iterative_decoder_steps=512 "
    "--iterative_encoder_steps=128 "
    "--iterative_start_phase=decoder "
    "--encoder_population_usage_loss=True "
    "--encoder_usage_loss_coeff=0.1 "
    "--encoder_density_loss_coeff=1.0 "
    "--encoder_collision_loss_coeff=0.1 "
    "--encoder_target_density=0.03 "
    "--encoder_multi_activation_loss=True "
    "--hrl_controllable_graph=True "
    "--hrl_bootstrap_horizon=64 "
    "--hrl_min_target_visits=1 "
    "--hrl_target_hit_reward=1.0 "
    "--hrl_distance_bonus_coeff=0.1 "
    "--hrl_timeout_margin_ratio=0.20 "
    "--hrl_timeout_margin_steps=2 "
    "--hrl_pbt_max_silent_fraction=0.5 "
    "--ca3_predictor_shadow=True "
    "--ca3_predictor_hidden_size=128 "
    "--ca3_predictor_horizon=32 "
    "--ca3_predictor_loss_coeff=0.1 "
    "--exploration_coverage_telemetry=True "
    "--exploration_coverage_grid_size=100 "
)


def experiment(seed: int, encoder_method: str, worker_mode: str) -> Experiment:
    mode_tag = "hit" if worker_mode == "hit" else "hitdist"
    name = f"I2_F16_L64_T200_ER{encoder_method}_WR{mode_tag}_S{seed}"
    cli = (
        BASE_CLI
        + f"--seed={seed} "
        + f"--encoder_reward_method={encoder_method} "
        + f"--hrl_worker_reward_mode={worker_mode} "
    )
    return Experiment(name, cli, [{}])


_experiments = [
    experiment(seed, encoder_method, worker_mode)
    for seed in SEEDS
    for encoder_method in ENCODER_REWARD_METHODS
    for worker_mode in WORKER_REWARD_MODES
]

RUN_DESCRIPTION = RunDescription(BATCH_NAME, experiments=_experiments)
