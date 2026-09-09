from sample_factory.launcher.run_description import Experiment, RunDescription

BATCH_NAME = "intrmotiv_hrl_batch1_20260818"
SEEDS = [8, 99, 123, 456]
SEQUENCE_LENGTHS = [32, 64, 128]
DG_THRESHOLDS = [2.0, 2.43]
ENCODER_REWARD_METHODS = ["punish", "encourage", "mean"]

BASE_CLI = (
    "--env=openfield_map2_fixed_loc3_noreward "
    "--train_for_env_steps=90000000 "
    "--train_for_seconds=43200 "
    "--algo=APPO "
    "--gamma=0.99 "
    "--learning_rate=0.0002 "
    "--exploration_loss_coeff=0.005 "
    "--value_loss_coeff=0.3 "
    "--ppo_clip_ratio=0.25 "
    "--num_workers=32 "
    "--num_envs_per_worker=8 "
    "--worker_num_splits=8 "
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
    "--set_workers_cpu_affinity=False "
    "--num_policies=1 "
    "--with_pbt=False "
    "--max_policy_lag=35 "
    "--use_record_episode_statistics=True "
    "--keep_checkpoints=12 "
    "--save_every_sec=300 "
    "--save_milestones_sec=3600 "
    "--save_best_every_sec=300 "
    "--save_best_after=1200 "
    "--decoder_mlp_layers 128 128 "
    "--env_frameskip=8 "
    "--core_name=BypassSS "
    "--DG_name=batchnorm_relu "
    "--Hippo_n_feature=16 "
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
    "--benchmark=False "
    "--with_number_instruction=True "
    "--number_instruction_coef=9 "
    "--save_best_metric=distance_metric "
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
    "--hrl_controllable_graph=True "
    "--hrl_timeout_margin_ratio=0.20 "
    "--hrl_timeout_margin_steps=2 "
)


def experiment(seed: int, seq_len: int, threshold: float, reward_method: str) -> Experiment:
    threshold_tag = int(round(threshold * 100))
    name = f"B1_F16_L{seq_len}_T{threshold_tag:03d}_ER{reward_method}_S{seed}"
    cli = (
        BASE_CLI
        + f"--seed={seed} "
        + f"--Hippo_L={seq_len} "
        + f"--DG_BN_intercept={threshold} "
        + f"--encoder_reward_method={reward_method} "
    )
    return Experiment(name, cli, [{}])


_experiments = [
    experiment(seed, seq_len, threshold, reward_method)
    for seed in SEEDS
    for seq_len in SEQUENCE_LENGTHS
    for threshold in DG_THRESHOLDS
    for reward_method in ENCODER_REWARD_METHODS
]

RUN_DESCRIPTION = RunDescription(BATCH_NAME, experiments=_experiments)
