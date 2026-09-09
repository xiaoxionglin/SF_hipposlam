import argparse
import os
from os.path import join

from sample_factory.utils.utils import str2bool


def hipposlam_override_defaults(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(
        encoder_conv_architecture="convnet_impala",
        obs_subtract_mean=0.0,
        obs_scale=255.0,
        env_frameskip=4,
        nonlinearity="relu",
        rollout=32,
        recurrence=32,
        rnn_type="lstm",
        rnn_size=256,
        num_epochs=1,
        # if observation normalization is used, it is important that we do not normalize INSTRUCTIONS observation
        normalize_input_keys=["obs"],
        decoder_mlp_layers=[128, 128],
    )


def add_hipposlam_env_args(parser: argparse.ArgumentParser) -> None:
    p = parser
    p.add_argument("--decoder_reward_gate", choices=["none", "ca3_absent"], default="none")
    p.add_argument("--dg_ca3_reentry_inhibition", choices=["none", "trace_subtractive", "hard"], default="none")
    p.add_argument("--intrinsic_goal_mode", choices=["none", "ca3_absent_target"], default="none")
    p.add_argument("--intrinsic_goal_horizon", type=int, default=64)
    p.add_argument("--intrinsic_goal_reward_max", type=float, default=6.4)
    p.add_argument("--intrinsic_goal_controller", choices=["shared", "separate"], default="shared")
    p.add_argument("--intrinsic_goal_reference_checkpoint", default=None, type=str)
    p.add_argument("--intrinsic_goal_initialize_live_from_reference", default=False, type=str2bool)
    p.add_argument("--Hippo_n_feature", default=64, type=int, help="number of sequences/features")
    p.add_argument("--Hippo_R", default=8, type=int, help="number of repeats in a sequence")
    p.add_argument("--Hippo_L", default=48, type=int, help="sequence length")

    p.add_argument(
        "--simple_sequence",
        default=False,
        type=bool,
        help="simple sequence, simply shrinking feature dimensions and expanding features to include their history",
    )
    p.add_argument(
        "--core_name",
        default=None,
        type=str,
        help="simple sequence, simply shrinking feature dimensions and expanding features to include their history",
    )
    p.add_argument("--encoder_name", default=None, type=str, help="actually using dmlab encoders")
    p.add_argument("--encoder_load_path", default=None, type=str, help="if loading encoder, the path")

    p.add_argument("--DG_lr", default=None, type=float, help="Dentate Gyrus Pattern separation learning rate")
    p.add_argument("--DG_temperature", default=None, type=float, help="Dentate Gyrus output temperature")
    p.add_argument(
        "--DG_batch_q", default=None, type=bool, help="Dentate Gyrus batch quantile, momentum 0.2, quantile 0.98"
    )
    p.add_argument("--DG_softmax", default=None, type=bool, help="Dentate Gyrus softmax")
    p.add_argument(
        "--DG_name", default=None, type=str, help="model name for the last encoder layer, i.e. Dentate Gyrus"
    )
    p.add_argument(
        "--DG_detect", default=None, type=float, help="batch novelty detection threshold (to activate a sequence)"
    )
    p.add_argument(
        "--DG_novelty", default=None, type=float, help="batch novelty novelty threshold to store a new pattern"
    )
    # p.add_argument("--dense", default=None, type=bool, help="whether encoder gives additional dense output")
    p.add_argument("--head_l1_coef", default=None, type=float, help="L1 penalty to encoder output")
    p.add_argument(
        "--fix_encoder_when_load",
        default=True,
        type=bool,
        help="when loading an encoder, fix its weights at initialization",
    )
    p.add_argument("--depth_sensor", default=False, type=bool, help="having extra depth sensor")
    p.add_argument(
        "--dmlab_reduced_action_set", default=False, type=bool, help="reduced action set to facilitate learning"
    )
    p.add_argument(
        "--with_number_instruction", default=True, type=str2bool, help="instruction input is number, e.g. 1-3"
    )
    p.add_argument("--number_instruction_coef", default=1, type=float, help="instruction strength")
    p.add_argument("--DG_BN_intercept", default=2, type=float, help="instruction strength")
    p.add_argument("--with_pos_obs", default=False, type=str2bool, help="get the true position of agent")
    p.add_argument(
        "--use_jit",
        default=True,
        type=str2bool,
        help="use jit / pytorch script to accelerate decoder. disable it for hooking",
    )

    p.add_argument(
        "--refractory",
        default=0,
        type=int,
        help="when using bypassSS_binary, determine whether to block reentry and how much the refractory. 0: no refractory, -1: entire sequence",
    )

    p.add_argument(
        "--rec_distances",
        default=None,
        type=bool,
        help="Record the distance between the propagation of each individual sequence",
    )
    p.add_argument(
        "--masked_distance_matrix",
        default=False,
        type=str2bool,
        help="Whether to use the masked version of the distance matrix for calculations. Everything is logged",
    )
    p.add_argument(
        "--distance_learning",
        default=True,
        type=str2bool,
        help="Whether to use the distance matrix for learning instead of the advantage",
    )
    p.add_argument(
        "--normalize_advantage",
        default=True,
        type=str2bool,
        help="When using distance learning wether to normalize the metric or nor. just for testing/understanind purposes",
    )
    p.add_argument(
        "--advantage_reward_source",
        default=None,
        choices=["external", "internal"],
        help="Reward stream used for PPO advantages when separate reward streams exist.",
    )
    p.add_argument(
        "--hrl_controllable_graph",
        default=False,
        type=str2bool,
        help="Enable online DG-subgoal HRL with a controllability matrix in rnn_states.",
    )
    p.add_argument(
        "--hrl_graph_memory",
        default="episode",
        choices=["episode", "policy_buffer"],
        help="Store graph fast weights per rollout stream or as learner-owned policy buffers.",
    )
    p.add_argument(
        "--hrl_target_timing",
        default="delayed",
        choices=["delayed", "immediate"],
        help=(
            "Condition actions on the target stored before (delayed, compatibility) or selected during "
            "the current actor forward (immediate)."
        ),
    )
    p.add_argument(
        "--hrl_persistent_fast_weights",
        default=False,
        type=str2bool,
        help="Preserve decayed controllability and novelty weights across terminal resets.",
    )
    p.add_argument(
        "--hrl_fast_weight_half_life_options",
        default=10000.0,
        type=float,
        help="Option-reset half-life for persistent HRL node and edge fast weights.",
    )
    p.add_argument(
        "--hrl_edge_confidence_threshold",
        default=0.5,
        type=float,
        help="Minimum decayed edge strength required for controllability feasibility.",
    )
    p.add_argument(
        "--hrl_timeout_margin_ratio",
        default=0.20,
        type=float,
        help="Relative margin added to learned controllability times when setting an option deadline.",
    )
    p.add_argument(
        "--hrl_timeout_margin_steps",
        default=2,
        type=int,
        help="Fixed step margin added to learned controllability times when setting an option deadline.",
    )
    p.add_argument(
        "--hrl_bootstrap_horizon",
        default=64,
        type=int,
        help="Option horizon used before an intended source-target mean time has been learned.",
    )
    p.add_argument(
        "--hrl_exploration_mode",
        default=False,
        type=str2bool,
        help="Add a dense-reward exploration option to the fixed HRL manager.",
    )
    p.add_argument(
        "--hrl_manager_exploration_probability",
        default=0.0,
        type=float,
        help="Probability of selecting exploration at an option boundary; target timeouts always force it.",
    )
    p.add_argument(
        "--hrl_exploration_horizon",
        default=64,
        type=int,
        help="Decision horizon for one manager exploration option.",
    )
    p.add_argument(
        "--hrl_manager_mode",
        default="visit_direct",
        choices=["visit_direct", "topology_visit_direct", "frontier_direct", "frontier_waypoint", "control_graph"],
        help=(
            "Fixed manager: compatibility visit ranking, topology-matched visit ranking, "
            "graph frontier with direct goals, or explicit next hops."
        ),
    )
    p.add_argument("--hrl_passive_edge_confidence_threshold", default=2.0, type=float)
    p.add_argument("--hrl_passive_min_displacement", default=2.0, type=float)
    p.add_argument("--hrl_passive_max_path_length", default=64.0, type=float)
    p.add_argument("--hrl_frontier_uncertainty_weight", default=1.0, type=float)
    p.add_argument(
        "--hrl_edge_exploration",
        default=False,
        type=str2bool,
        help="Let the common control-graph manager actively probe passive/geometric candidate edges.",
    )
    p.add_argument(
        "--hrl_edge_reliability_threshold",
        default=0.5,
        type=float,
        help="Minimum smoothed hit/attempt ratio for an edge to be usable in routes.",
    )
    p.add_argument(
        "--hrl_edge_connectivity_weight",
        default=0.25,
        type=float,
        help="Connectivity-gain bonus in candidate-edge UCB selection.",
    )
    p.add_argument(
        "--hrl_behavior_mode_condition",
        default=False,
        type=str2bool,
        help="Expose and replay the manager behavior mode alongside the target condition.",
    )
    p.add_argument(
        "--hrl_goal_conditioning",
        default="legacy",
        choices=["legacy", "target_id_additive", "target_trace", "target_id_film"],
        help=(
            "Goal decoder input: legacy/additive concatenated one-hot, target embedding plus selected "
            "CA3 trace, or identity-initialized target-ID FiLM without a selected trace."
        ),
    )
    p.add_argument(
        "--hrl_exploration_policy",
        default="shared",
        choices=["shared", "separate"],
        help="Route free exploration through the goal head or an isolated actor/value head.",
    )
    p.add_argument("--hrl_exploration_decoder_size", default=128, type=int)
    p.add_argument("--hrl_action_path_integration", default=False, type=str2bool)
    p.add_argument("--hrl_motion_policy_input", default=False, type=str2bool)
    p.add_argument("--dg_path_scatter_coeff", default=0.0, type=float)
    p.add_argument("--dg_path_scatter_min_displacement", default=8.0, type=float)
    p.add_argument("--dg_path_scatter_min_straightness", default=0.8, type=float)
    p.add_argument("--dg_path_scatter_temperature", default=0.5, type=float)
    p.add_argument(
        "--hrl_landmark_geometry",
        default="none",
        choices=["none", "se2"],
        help="Optional SE(2) pose-graph control; topological planning remains the default.",
    )
    p.add_argument("--hrl_geometry_neighbors", default=3, type=int)
    p.add_argument("--hrl_geometry_max_distance", default=32.0, type=float)
    p.add_argument("--hrl_geometry_update_steps", default=5, type=int)
    p.add_argument("--hrl_geometry_learning_rate", default=0.05, type=float)
    p.add_argument(
        "--hrl_min_target_visits",
        default=1.0,
        type=float,
        help="Minimum episode-local DG visit count required before a node can be selected as a target.",
    )
    p.add_argument(
        "--hrl_worker_reward_mode",
        default="hit",
        choices=["hit", "hit_distance"],
        help="Always-positive worker reward: target hit only, or hit plus clipped legacy temporal bonus.",
    )
    p.add_argument("--hrl_target_hit_reward", default=1.0, type=float)
    p.add_argument("--hrl_distance_bonus_coeff", default=0.1, type=float)
    p.add_argument(
        "--hrl_control_outcome",
        default="target_hit",
        choices=["target_hit", "first_distinct"],
        help=(
            "Complete intentional options only on the commanded target, or on the first "
            "distinct exclusive landmark with chance-centered wrong-outcome reward."
        ),
    )
    p.add_argument(
        "--hrl_direct_target_selection",
        default="frontier",
        choices=["frontier", "least_tested", "local_successor"],
        help=(
            "Choose direct targets by frontier score, among all observed nodes by lowest pair-attempt "
            "mass, or among directed passive first successors by lowest pair-attempt mass."
        ),
    )
    p.add_argument(
        "--hrl_empirical_her",
        default=False,
        type=str2bool,
        help="Enable the biased PPO-style DG hindsight auxiliary loss for direct policy-buffer HRL.",
    )
    p.add_argument("--hrl_empirical_her_horizon", default=64, type=int)
    p.add_argument("--hrl_empirical_her_coeff", default=0.5, type=float)
    p.add_argument("--ca3_predictor_shadow", default=False, type=str2bool)
    p.add_argument("--ca3_predictor_hidden_size", default=128, type=int)
    p.add_argument("--ca3_predictor_horizon", default=32, type=int)
    p.add_argument("--ca3_predictor_loss_coeff", default=0.1, type=float)
    p.add_argument("--exploration_coverage_telemetry", default=False, type=str2bool)
    p.add_argument("--exploration_coverage_grid_size", default=100.0, type=float)
    p.add_argument(
        "--exploration_window_steps",
        default=0,
        type=int,
        help="Emit non-terminal exploration telemetry windows every N policy decisions (0 disables).",
    )
    p.add_argument(
        "--online_spatial_telemetry",
        default=True,
        type=str2bool,
        help="Collect privileged pose/DG monitoring data without exposing pose to the model.",
    )
    p.add_argument(
        "--online_spatial_window_observations",
        default=100_000,
        type=int,
        help="Latest behavior samples retained per policy and saved in each milestone snapshot.",
    )
    p.add_argument(
        "--online_spatial_scalar_window_observations",
        default=10_000,
        type=int,
        help="Latest retained samples used for W&B scalar calculations (capped by snapshot window).",
    )
    p.add_argument("--online_spatial_scalar_interval_frames", default=1_000_000, type=int)
    p.add_argument("--online_spatial_snapshot_interval_frames", default=25_000_000, type=int)
    p.add_argument("--online_spatial_snapshot_max_frames", default=100_000_000, type=int)
    p.add_argument(
        "--online_spatial_snapshot_targets",
        default="auto",
        type=str,
        help=(
            "Comma-separated milestone frames. 'auto' uses 5M,25M,50M,75M,100M for the default "
            "legacy interval/max settings and otherwise preserves the interval cadence."
        ),
    )
    p.add_argument("--online_spatial_grid_grain", default=19, type=int)
    p.add_argument("--online_spatial_x_min", default=100.0, type=float)
    p.add_argument("--online_spatial_x_max", default=2000.0, type=float)
    p.add_argument("--online_spatial_y_min", default=100.0, type=float)
    p.add_argument("--online_spatial_y_max", default=2000.0, type=float)
    p.add_argument("--online_spatial_stationary_distance", default=1.0, type=float)
    p.add_argument(
        "--online_spatial_max_segment_jump_distance",
        default=250.0,
        type=float,
        help="Split trajectory diagnostics at larger unmarked physical relocations.",
    )
    p.add_argument(
        "--online_spatial_workspace_root",
        default="/work/classic/fr_xl1014-train",
        type=str,
        help="Required containing workspace for all online spatial artifacts.",
    )
    p.add_argument(
        "--online_spatial_output_root",
        default=(
            "/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/"
            "train_dir/analysis/online_spatial"
        ),
        type=str,
        help="Workspace analysis root for batch/run/policy spatial snapshots.",
    )
    p.add_argument("--hrl_pbt_max_silent_fraction", default=0.5, type=float)
    p.add_argument("--encoder_population_usage_loss", default=False, type=str2bool)
    p.add_argument("--encoder_usage_loss_coeff", default=0.1, type=float)
    p.add_argument("--encoder_density_loss_coeff", default=1.0, type=float)
    p.add_argument("--encoder_collision_loss_coeff", default=0.1, type=float)
    p.add_argument("--encoder_target_density", default=0.03, type=float)
    p.add_argument("--dg_global_punishment_coeff", default=0.0, type=float)
    p.add_argument("--dg_global_punishment_temperature", default=0.5, type=float)
    p.add_argument("--dg_row_repulsion_coeff", default=0.0, type=float)
    p.add_argument(
        "--dg_ca3_temporal_exclusion_coeff",
        default=0.0,
        type=float,
        help=(
            "Dominant-onset temporal-margin multiplier. With encourage feedback, 1.0 makes the "
            "combined encoder loss suppress distances below Hippo_R, neutral at Hippo_R, and "
            "reinforce distances above Hippo_R; 0 disables it."
        ),
    )
    p.add_argument(
        "--dg_orthogonal_recruitment",
        default=False,
        type=str2bool,
        help="Recruit never-active DG rows from L-step novelty events using an orthogonal feature residual.",
    )
    p.add_argument(
        "--dg_orthogonal_recruitment_mode",
        default="legacy",
        choices=["legacy", "graph"],
        help="Choose historical least-used recruitment or graph-stabilized victim eligibility.",
    )
    p.add_argument(
        "--dg_recruitment_victim_rule",
        default="incident",
        choices=["incident", "monitor", "directional", "predictive"],
        help=(
            "Graph-mode victim rule: historical incident connectivity, diagnostics only, "
            "fully-tested zero reliable out-degree plus close duplicates, or persistent "
            "predecessor-conditioned outcome inconsistency."
        ),
    )
    p.add_argument("--dg_recruitment_connectivity_threshold", default=0.25, type=float)
    p.add_argument("--dg_recruitment_redundancy_max_steps", default=4, type=int)
    p.add_argument("--dg_recruitment_passive_half_life_events", default=5000.0, type=float)
    p.add_argument("--dg_recruitment_attempt_threshold", default=0.5, type=float)
    p.add_argument("--dg_recruitment_pred_min_context_attempts", default=2.0, type=float)
    p.add_argument("--dg_recruitment_pred_half_life_options", default=5000.0, type=float)
    p.add_argument(
        "--dg_recruitment_endpoint_gate",
        default="silent",
        choices=["silent", "open"],
        help="Require a silent L-step endpoint or permit replacement at active endpoints.",
    )
    p.add_argument(
        "--dg_batchnorm_semantics",
        default="legacy_batch",
        choices=[
            "legacy_batch",
            "running_consistent",
            "running_poststep_atomic",
            "input_centered_atomic",
        ],
        help=(
            "Use legacy train/eval BatchNorm, the original pre-forward running-stat update, "
            "post-step atomic projected moments, or post-step atomic moments with an explicit "
            "running feature-mean subtraction."
        ),
    )
    p.add_argument(
        "--dg_recruitment_reset_goal_adapter",
        default=False,
        type=str2bool,
        help="Atomically reset the replaced landmark's target-specific FiLM row and optimizer state.",
    )
    p.add_argument(
        "--dg_recruitment_forced_preflight_row",
        default=-1,
        type=int,
        help=(
            "Engineering-only: force exactly one replacement of this DG row, then suppress all "
            "further recruitment. Negative disables the hook."
        ),
    )
    p.add_argument(
        "--dg_recruitment_forced_preflight_after_updates",
        default=4,
        type=int,
        help="Learner update after which the engineering-only forced replacement may occur.",
    )
    p.add_argument("--dg_orthogonal_recruitment_margin", default=0.5, type=float)
    p.add_argument("--dg_orthogonal_recruitment_residual_eps", default=1e-6, type=float)
    p.add_argument("--dg_orthogonal_recruitment_max_per_rollout", default=1, type=int)
    p.add_argument(
        "--use_internal",
        default=False,
        type=str2bool,
        help="Deprecated alias for selecting the internal reward stream. Prefer --advantage_reward_source=internal.",
    )
    p.add_argument(
        "--use_external",
        default=True,
        type=str2bool,
        help="Deprecated alias for selecting the external reward stream. Prefer --advantage_reward_source=external.",
    )
    p.add_argument(
        "--extra_encoder_losses",
        default=True,
        type=str2bool,
        help="Use additional encoder losses. Might differentiate in the future",
    )
    p.add_argument(
        "--double_value",
        default=False,
        type=str2bool,
        help="Use two values, one trained on external, the other on internal reward. Which one is used is defined by advantage_reward_source.",
    )
    p.add_argument(
        "--reset_critic",
        default=False,
        type=str2bool,
        help="Reset all critic parameters. Useful after pre-training.",
    )
    p.add_argument(
        "--reset_decoder",
        default=False,
        type=str2bool,
        help="Reset all Decoder parameters (This includes Action-Parametrization/Value-Estimator Layers as well). Useful after pre-training.",
    )
    p.add_argument(
        "--encoder_grad_coeff",
        default=1.0,
        type=float,
        help="A factor by which the gradients of the encoder get multiplied. Only works when the gradients get flipped/",
    )
    p.add_argument(
        "--iterative_update", default=False, type=str2bool,
        help="Alternate encoder and decoder updates using the baseline single optimizer/checkpoint format.",
    )
    p.add_argument(
        "--ppo_dg_gradient",
        default="stop",
        choices=["stop", "joint"],
        help="Stop the actor-critic loss at CA3 or allow the complete PPO decoder loss to update DG.",
    )
    p.add_argument(
        "--dg_context_feedback",
        default="none",
        choices=["none", "gate", "additive"],
        help="Use no DG feedback, a multiplicative visual gate, or a bounded additive CA3 residual.",
    )
    p.add_argument(
        "--dg_context_history",
        default="ca3",
        choices=["ca3", "ca3_action"],
        help="Condition DG feedback on recent CA3 alone or CA3 plus the previous R primitive actions.",
    )
    p.add_argument(
        "--dg_context_gradient",
        default="direct",
        choices=["direct", "bptt"],
        help="Detach CA3 at the feedback adapter or backpropagate through the within-rollout feedback recurrence.",
    )
    p.add_argument(
        "--dg_transition_prediction",
        default="none",
        choices=["none", "passive", "goal"],
        help="Training-only next-landmark prediction from source DG, optionally conditioned on the commanded goal.",
    )
    p.add_argument("--dg_transition_prediction_hidden_size", default=128, type=int)
    p.add_argument("--dg_transition_prediction_coeff", default=0.1, type=float)
    p.add_argument(
        "--iterative_initial_encoder_steps", default=128, type=int,
        help="Initial encoder-only optimizer steps before alternating phases.",
    )
    p.add_argument(
        "--iterative_decoder_steps", default=512, type=int,
        help="Decoder-only optimizer steps in each alternating cycle.",
    )
    p.add_argument(
        "--iterative_encoder_steps", default=128, type=int,
        help="Encoder-only optimizer steps in each alternating cycle.",
    )
    p.add_argument(
        "--iterative_start_phase", default="decoder", choices=["decoder", "encoder"],
        help="Phase immediately after the optional initial encoder warmup.",
    )
    p.add_argument(
        "--metric",
        default=None,
        type=str,
        help="Define which metric should be used for optimization of internal structure. Options currently include 'sum', 'masked_sum', 'minimum'",
    )
    p.add_argument(
        "--encoder_reward_method",
        default=None,
        type=str,
        help="How the encoder reward is shifted. Options include '' ",
    )
    p.add_argument(
        "--encoder_reward_recipient",
        default="arrival",
        choices=["arrival", "source"],
        help="Apply matched temporal encoder credit to the arrival or verified predecessor row.",
    )
    p.add_argument(
        "--encoder_reward_require_local_predecessor",
        default=False,
        type=str2bool,
        help="Drop encoder events without an aligned predecessor onset in the same actor rollout.",
    )
    p.add_argument("--load_model_path", default=None, type=str, help="Path to specific .pth file for the entire model")
    p.add_argument(
        "--encoder_batch_loss",
        default=True,
        type=str2bool,
        help="Recruit DG units unused by the valid minibatch using pre-threshold logits.",
    )
    p.add_argument(
        "--encoder_batch_loss_temperature",
        default=0.5,
        type=float,
        help="Softplus temperature for the pre-threshold unused-unit recruitment loss.",
    )
    p.add_argument(
        "--encoder_multi_activation_loss",
        default=False,
        type=str2bool,
        help="Use multi activation loss",
    )
    p.add_argument(
        "--encoder_unused_sequence_loss",
        default=False,
        type=str2bool,
        help="Use unused sequence loss.",
    )
    p.add_argument(
        "--extra_decoder_loss",
        default=False,
        type=str2bool,
        help="Disabled pending repair: the historical implementation had the wrong optimization sign.",
    )
