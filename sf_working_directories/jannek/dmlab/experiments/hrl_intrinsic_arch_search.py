from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription


def seeds(values):
    return ParamGrid([("seed", values)]).generate_params(False)


BASE_CLI = (
    "--env=openfield_map2_fixed_loc3_noreward "
    "--train_for_seconds=68400 "
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
    "--pbt_replace_reward_gap=0.05 "
    "--pbt_replace_reward_gap_absolute=0.2 "
    "--pbt_period_env_steps=2000000 "
    "--pbt_start_mutation=10000000 "
    "--pbt_mix_policies_in_one_env=False "
    "--pbt_target_objective=distance_metric "
    "--pbt_perturb_max=1.3 "
    "--pbt_replace_fraction=0.2 "
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
    "--depth_sensor=True "
    "--normalize_input=False "
    "--fix_encoder_when_load=True "
    "--encoder_load_path=/home/fr/fr_xl1014/training/best_000025288_203030528_reward_94.185.pth "
    "--encoder_conv_architecture=pretrained_resnet "
    "--encoder_conv_mlp_layers=256 "
    "--use_rnn=True "
    "--rnn_type=gru "
    "--rnn_size=0 "
    "--nonlinearity=relu "
    "--with_wandb=True "
    "--wandb_user=xiaoxionglin-bernstein-center-freiburg "
    "--wandb_project=SF_HRL_Intrinsic_ArchSearch "
    "--wandb_group=hrl_intrinsic_arch_search_20260818 "
    "--benchmark=False "
    "--with_number_instruction=True "
    "--number_instruction_coef=9 "
    "--save_best_metric=distance_metric "
    "--device=cpu "
    "--rec_distances=True "
    "--distance_learning=True "
    "--masked_distance_matrix=False "
    "--normalize_advantage=True "
    "--use_internal=False "
    "--use_external=True "
    "--extra_encoder_losses=True "
    "--metric=sum "
    "--reward_scale=0.1 "
    "--double_value=False "
    "--reset_critic=False "
    "--reset_decoder=False "
    "--encoder_grad_coeff=1 "
    "--encoder_reward_method=punish "
)


def cli(n_feature, seq_len, theta):
    return (
        BASE_CLI
        + f"--Hippo_n_feature={n_feature} "
        + f"--Hippo_L={seq_len} "
        + f"--DG_BN_intercept={theta} "
    )


_experiments = [
    # Jannek baseline, replicated because it anchors the search.
    Experiment("HRL_F16_L64_T243", cli(16, 64, 2.43), seeds([8, 99])),
    # Threshold sweep around the baseline: density/sparsity of DG events.
    Experiment("HRL_F16_L64_T220", cli(16, 64, 2.20), seeds([8])),
    Experiment("HRL_F16_L64_T260", cli(16, 64, 2.60), seeds([8])),
    # DG capacity sweep: whether more/fewer candidate sequences help.
    Experiment("HRL_F8_L64_T243", cli(8, 64, 2.43), seeds([8])),
    Experiment("HRL_F32_L64_T243", cli(32, 64, 2.43), seeds([8, 99])),
    # Sequence horizon sweep: whether the reservoir should remember shorter or longer temporal neighborhoods.
    Experiment("HRL_F16_L32_T243", cli(16, 32, 2.43), seeds([8])),
    Experiment("HRL_F16_L128_T243", cli(16, 128, 2.43), seeds([8])),
    # Combined larger reservoir: expensive but plausible if the baseline saturates sequence capacity.
    Experiment("HRL_F32_L128_T243", cli(32, 128, 2.43), seeds([8])),
]


RUN_DESCRIPTION = RunDescription("hipposlam_hrl_intrinsic_arch", experiments=_experiments)
