import sys
from multiprocessing.context import BaseContext
from typing import Optional

from tensorboardX import SummaryWriter

from sample_factory.algo.runners.runner import AlgoObserver, Runner
from sample_factory.algo.utils.misc import TRAIN_STATS, ExperimentStatus
from sample_factory.algo.utils.model_context import global_learner_factory, global_model_factory
from sample_factory.algo.utils.multiprocessing_utils import get_mp_ctx
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import make_runner
from sample_factory.utils.typing import Config, Env, PolicyID
from sample_factory.utils.utils import experiment_dir
from sf_working_directories.IntrMotiv.dmlab.custom_actor_critic import make_hipposlam_actor_critic

# from sf_working_directories.sf_examples.dmlab.dmlab_model import make_dmlab_encoder
from sf_working_directories.IntrMotiv.dmlab.custom_core import make_hipposlam_core
from sf_working_directories.IntrMotiv.dmlab.custom_decoder import make_hipposlam_decoder
from sf_working_directories.IntrMotiv.dmlab.custom_encoder import make_hipposlam_encoder
from sf_working_directories.IntrMotiv.dmlab.custom_learner import make_hipposlam_learner
from sf_working_directories.IntrMotiv.dmlab.custom_params import add_hipposlam_env_args, hipposlam_override_defaults
from sf_working_directories.IntrMotiv.dmlab.dmlab_env import (
    DMLAB_ENVS,
    dmlab_extra_episodic_stats_processing,
    dmlab_extra_summaries,
    list_all_levels_for_experiment,
    make_dmlab_env,
)
from sf_working_directories.IntrMotiv.dmlab.dmlab_level_cache import DmlabLevelCaches, make_dmlab_caches
from sf_working_directories.IntrMotiv.dmlab.dmlab_params import add_dmlab_env_args, dmlab_override_defaults
from sf_working_directories.IntrMotiv.dmlab.hrl_controllable_graph import (
    hrl_option_state_size,
    hrl_persistent_state_size,
    hrl_state_size,
)
from sf_working_directories.IntrMotiv.dmlab.reward_summaries import write_intrmotiv_summaries
from sf_working_directories.IntrMotiv.dmlab.topological_frontier import (
    ACTION_FEATURE_SIZE,
    GEOMETRY_POLICY_SIZE,
    topological_state_size,
)


class DmlabEnvWithCache:
    def __init__(self, level_caches: Optional[DmlabLevelCaches] = None):
        self.caches = level_caches

    def make_env(self, env_name, cfg, env_config, render_mode) -> Env:
        return make_dmlab_env(env_name, cfg, env_config, render_mode, self.caches)


def register_dmlab_envs(level_caches: Optional[DmlabLevelCaches] = None):
    env_factory = DmlabEnvWithCache(level_caches)
    for env in DMLAB_ENVS:
        register_env(env.name, env_factory.make_env)


def register_dmlab_components(level_caches: Optional[DmlabLevelCaches] = None):
    register_dmlab_envs(level_caches)
    global_model_factory().register_encoder_factory(make_hipposlam_encoder)
    global_model_factory().register_model_core_factory(make_hipposlam_core)
    global_model_factory().register_decoder_factory(make_hipposlam_decoder)

    global_model_factory().register_actor_critic_factory(make_hipposlam_actor_critic)

    global_learner_factory().register_learner_factory(make_hipposlam_learner)


class DmlabExtraSummariesObserver(AlgoObserver):
    def extra_summaries(self, runner: Runner, policy_id: PolicyID, writer: SummaryWriter, env_steps: int) -> None:
        dmlab_extra_summaries(runner, policy_id, writer, env_steps)


def register_msg_handlers(cfg: Config, runner: Runner):
    # Handle IntrMotiv stats first so it can route them outside train/ before
    # Sample Factory's generic handler writes the remaining framework metrics.
    runner.policy_msg_handlers.setdefault(TRAIN_STATS, []).insert(0, write_intrmotiv_summaries)
    if cfg.env == "dmlab_30":
        # extra functions to calculate human-normalized score etc.
        runner.register_episodic_stats_handler(dmlab_extra_episodic_stats_processing)
        runner.register_observer(DmlabExtraSummariesObserver())


def initialize_level_cache(cfg: Config, mp_ctx: BaseContext) -> Optional[DmlabLevelCaches]:
    if not cfg.dmlab_use_level_cache:
        return None

    env_name = cfg.env
    num_policies = cfg.num_policies if hasattr(cfg, "num_policies") else 1
    all_levels = list_all_levels_for_experiment(env_name)
    level_cache_dir = cfg.dmlab_level_cache_path
    caches = make_dmlab_caches(experiment_dir(cfg), all_levels, num_policies, level_cache_dir, mp_ctx)
    return caches


def maybe_overwrite_rnn_size(cfg):
    if getattr(cfg, "extra_decoder_loss", False):
        raise ValueError(
            "extra_decoder_loss is disabled: its historical objective had the wrong sign and must be repaired "
            "and validated before use"
        )
    if getattr(cfg, "with_pbt", False) and getattr(cfg, "pbt_target_objective", "") == "distance_metric":
        raise ValueError(
            "distance_metric is an internal temporal statistic, not a valid PBT objective; "
            "use intrmotiv_pbt_objective or an external exploration metric"
        )

    cfg.extra_policy_output_shapes = (
        (("dg_activity", [int(cfg.Hippo_n_feature)]),) if getattr(cfg, "online_spatial_telemetry", False) else ()
    )
    cfg.wandb_step_metric_namespaces = ("intrmotiv",)
    cfg.head_l1_size = int(cfg.Hippo_n_feature)

    manager_mode = getattr(cfg, "hrl_manager_mode", "visit_direct")
    topological = manager_mode != "visit_direct"
    graph_recruitment = bool(getattr(cfg, "dg_orthogonal_recruitment", False)) and (
        getattr(cfg, "dg_orthogonal_recruitment_mode", "legacy") == "graph"
    )
    if graph_recruitment:
        if getattr(cfg, "DG_name", None) != "batchnorm_relu":
            raise ValueError("Graph-stabilized DG recruitment requires DG_name=batchnorm_relu")
        if getattr(cfg, "core_name", "BypassSS") != "BypassSS":
            raise ValueError("Graph-stabilized DG recruitment currently requires core_name=BypassSS")
        if not 0.0 < float(getattr(cfg, "dg_recruitment_connectivity_threshold", 0.25)) < 1.0:
            raise ValueError("dg_recruitment_connectivity_threshold must be in (0, 1)")
        if int(getattr(cfg, "dg_recruitment_redundancy_max_steps", 4)) <= 0:
            raise ValueError("dg_recruitment_redundancy_max_steps must be positive")
        if float(getattr(cfg, "dg_recruitment_passive_half_life_events", 5000.0)) <= 0:
            raise ValueError("dg_recruitment_passive_half_life_events must be positive")
        victim_rule = getattr(cfg, "dg_recruitment_victim_rule", "incident")
        if victim_rule in ("directional", "predictive") and not (
            getattr(cfg, "hrl_controllable_graph", False)
            and getattr(cfg, "hrl_graph_memory", "episode") == "policy_buffer"
        ):
            raise ValueError(f"{victim_rule} recruitment requires policy-buffer controllability")
        if float(getattr(cfg, "dg_recruitment_attempt_threshold", 0.5)) <= 0:
            raise ValueError("dg_recruitment_attempt_threshold must be positive")
        if float(getattr(cfg, "dg_recruitment_pred_min_context_attempts", 2.0)) <= 0:
            raise ValueError("dg_recruitment_pred_min_context_attempts must be positive")
        if float(getattr(cfg, "dg_recruitment_pred_half_life_options", 5000.0)) <= 0:
            raise ValueError("dg_recruitment_pred_half_life_options must be positive")
    normalization_semantics = getattr(cfg, "dg_batchnorm_semantics", "legacy_batch")
    if normalization_semantics != "legacy_batch" and getattr(cfg, "DG_name", None) != "batchnorm_relu":
        raise ValueError(f"{normalization_semantics} DG normalization requires DG_name=batchnorm_relu")
    context_feedback = getattr(cfg, "dg_context_feedback", "none")
    context_history = getattr(cfg, "dg_context_history", "ca3")
    context_gradient = getattr(cfg, "dg_context_gradient", "direct")
    if context_feedback != "none":
        if getattr(cfg, "DG_name", None) != "batchnorm_relu" or getattr(cfg, "core_name", None) != "BypassSS":
            raise ValueError("Contextual DG feedback requires batchnorm_relu with BypassSS")
        if context_gradient == "bptt" and bool(getattr(cfg, "iterative_update", False)):
            raise ValueError("Contextual DG BPTT requires simultaneous updates")
    if context_history == "ca3_action" and context_feedback == "none":
        raise ValueError("CA3+ACTION context is meaningful only when DG feedback is enabled")
    transition_prediction = getattr(cfg, "dg_transition_prediction", "none")
    if transition_prediction != "none":
        if not (
            getattr(cfg, "hrl_controllable_graph", False)
            and getattr(cfg, "hrl_graph_memory", "episode") == "policy_buffer"
            and getattr(cfg, "hrl_control_outcome", "target_hit") == "first_distinct"
        ):
            raise ValueError("DG transition prediction requires policy-buffer FIRST control")
        if float(getattr(cfg, "dg_transition_prediction_coeff", 0.1)) <= 0:
            raise ValueError("dg_transition_prediction_coeff must be positive")
    if bool(getattr(cfg, "dg_recruitment_reset_goal_adapter", False)):
        if not graph_recruitment:
            raise ValueError("goal-adapter reset requires graph recruitment")
        if getattr(cfg, "hrl_goal_conditioning", "legacy") != "target_id_film":
            raise ValueError("goal-adapter reset requires hrl_goal_conditioning=target_id_film")
        if not (
            getattr(cfg, "hrl_controllable_graph", False)
            and getattr(cfg, "hrl_graph_memory", "episode") == "policy_buffer"
        ):
            raise ValueError("goal-adapter reset requires policy-buffer controllability")
    forced_preflight_row = int(getattr(cfg, "dg_recruitment_forced_preflight_row", -1))
    if forced_preflight_row >= 0:
        if not bool(getattr(cfg, "dg_orthogonal_recruitment", False)):
            raise ValueError("forced recruitment preflight requires orthogonal recruitment")
        if forced_preflight_row >= int(getattr(cfg, "Hippo_n_feature", 0)):
            raise ValueError("forced recruitment preflight row is outside the DG")
        if int(getattr(cfg, "dg_recruitment_forced_preflight_after_updates", 4)) < 1:
            raise ValueError("forced recruitment preflight must wait for at least one learner update")
        if int(getattr(cfg, "dg_orthogonal_recruitment_max_per_rollout", 1)) != 1:
            raise ValueError("forced recruitment preflight requires max one replacement per rollout")
        if getattr(cfg, "dg_recruitment_endpoint_gate", "silent") != "open":
            raise ValueError("forced recruitment preflight requires the open endpoint gate")
        if int(getattr(cfg, "train_for_env_steps", 0)) > 1_000_000:
            raise ValueError("forced recruitment preflight is restricted to runs of at most 1M steps")
    if getattr(cfg, "encoder_reward_recipient", "arrival") == "source":
        if not bool(getattr(cfg, "encoder_reward_require_local_predecessor", False)):
            raise ValueError("source encoder credit requires local predecessor matching")
        if getattr(cfg, "encoder_reward_method", None) != "encourage":
            raise ValueError("source encoder credit is defined only for encoder_reward_method=encourage")
    if topological and (
        not getattr(cfg, "hrl_controllable_graph", False)
        or getattr(cfg, "hrl_graph_memory", "episode") != "policy_buffer"
        or getattr(cfg, "core_name", "BypassSS") != "BypassSS"
    ):
        raise ValueError("Topological frontier management requires BypassSS policy-buffer HRL")
    if getattr(cfg, "hrl_action_path_integration", False) and not topological:
        raise ValueError("Action path integration requires a topological frontier manager")
    if getattr(cfg, "hrl_action_path_integration", False) and not getattr(cfg, "dmlab_reduced_action_set", False):
        raise ValueError("Action path integration requires the five-action DMLab reduced action set")
    if getattr(cfg, "hrl_motion_policy_input", False) and not getattr(cfg, "hrl_action_path_integration", False):
        raise ValueError("Motion policy input requires action path integration")
    if getattr(cfg, "hrl_landmark_geometry", "none") != "none" and not topological:
        raise ValueError("Landmark geometry requires a topological frontier manager")
    if getattr(cfg, "hrl_edge_exploration", False) and manager_mode != "control_graph":
        raise ValueError("Connectivity-aware edge exploration requires hrl_manager_mode=control_graph")
    goal_conditioning = getattr(cfg, "hrl_goal_conditioning", "legacy")
    intrinsic_goal = getattr(cfg, "intrinsic_goal_mode", "none") != "none"
    memory_inhibition = getattr(cfg, "dg_ca3_reentry_inhibition", "none") != "none"
    if intrinsic_goal or memory_inhibition or getattr(cfg, "decoder_reward_gate", "none") != "none":
        if getattr(cfg, "core_name", "BypassSS") != "BypassSS" or getattr(cfg, "hrl_controllable_graph", False):
            raise ValueError("Finite-memory study mechanisms require graph-free BypassSS")
        if graph_recruitment:
            raise ValueError("Finite-memory mechanisms require no landmark recruitment")
        if context_feedback != "none" and not intrinsic_goal:
            raise ValueError("Flat finite-memory mechanisms require no learned feedback")
    if intrinsic_goal:
        if (
            goal_conditioning not in ("target_id_additive", "target_id_film")
            or int(getattr(cfg, "intrinsic_goal_horizon", 64)) < 1
        ):
            raise ValueError("Intrinsic goals require additive/FiLM target identity and a positive horizon")
        if getattr(cfg, "decoder_reward_gate", "none") != "none":
            raise ValueError("Goal cells use their own arrival reward")
    if goal_conditioning in ("target_trace", "target_id_film"):
        if not getattr(cfg, "hrl_controllable_graph", False) and not intrinsic_goal:
            raise ValueError("Custom goal conditioning requires controllable-graph HRL")
        if getattr(cfg, "hrl_target_timing", "delayed") != "immediate":
            raise ValueError("Custom goal conditioning requires immediate behavior targets")
    if getattr(cfg, "hrl_exploration_policy", "shared") == "separate":
        if not topological or not getattr(cfg, "hrl_behavior_mode_condition", False):
            raise ValueError("A separate exploration policy requires a replayed topological manager mode")
    if getattr(cfg, "hrl_behavior_mode_condition", False) and not topological:
        raise ValueError("Manager-mode conditioning requires a topological manager")
    if getattr(cfg, "hrl_empirical_her", False):
        if not (
            getattr(cfg, "hrl_controllable_graph", False)
            and getattr(cfg, "hrl_graph_memory", "episode") == "policy_buffer"
            and manager_mode == "visit_direct"
        ):
            raise ValueError("Empirical PPO-HER requires direct policy-buffer HRL")
        if int(getattr(cfg, "hrl_empirical_her_horizon", 0)) <= 0:
            raise ValueError("hrl_empirical_her_horizon must be positive")
    if (
        getattr(cfg, "hrl_controllable_graph", False)
        and getattr(cfg, "hrl_graph_memory", "episode") == "policy_buffer"
        and getattr(cfg, "core_name", "BypassSS") != "BypassSS"
    ):
        raise ValueError("hrl_graph_memory=policy_buffer is currently implemented for core_name=BypassSS only")
    if getattr(cfg, "cli_args.rnn_size", 0) == 0:
        R = getattr(cfg, "Hippo_R", 8)
        L = getattr(cfg, "Hippo_L", 48)
        hippo_n_feature = getattr(cfg, "Hippo_n_feature", 64)
        rnn_size = hippo_n_feature * (R + L - 1) + 13
        if intrinsic_goal:
            from .ca3_memory import GOAL_STATE_SIZE

            rnn_size += GOAL_STATE_SIZE
            if (
                getattr(cfg, "intrinsic_goal_reference_checkpoint", None)
                or getattr(cfg, "dg_ca3_reentry_inhibition", "none") != "none"
            ):
                rnn_size += int(cfg.Hippo_n_feature)
        if context_feedback != "none" and context_history == "ca3_action":
            action_count = (
                5
                if getattr(cfg, "dmlab_reduced_action_set", False)
                else (15 if getattr(cfg, "dmlab_extended_action_set", False) else 9)
            )
            # Current previous-action observation is retained in the bypass;
            # the separate shift register stores the ordered causal R-history.
            rnn_size += action_count + R * action_count
        if getattr(cfg, "hrl_action_path_integration", False):
            rnn_size += ACTION_FEATURE_SIZE
        if getattr(cfg, "hrl_controllable_graph", False):
            if getattr(cfg, "hrl_graph_memory", "episode") == "policy_buffer":
                rnn_size += hrl_option_state_size(hippo_n_feature)
            else:
                rnn_size += hrl_state_size(hippo_n_feature)
        if topological:
            rnn_size += topological_state_size(hippo_n_feature)
        if graph_recruitment:
            # Last exclusive DG id, decisions since activity, representation generation.
            rnn_size += 3
        if (
            getattr(cfg, "hrl_controllable_graph", False)
            and getattr(cfg, "hrl_graph_memory", "episode") == "policy_buffer"
            and getattr(cfg, "hrl_target_timing", "delayed") == "immediate"
        ):
            # Replay-only behavior descriptor. With manager-mode
            # conditioning this stores target id, four geometry values, and
            # mode id; otherwise it stores only target id.
            rnn_size += 1 + GEOMETRY_POLICY_SIZE + 1 if getattr(cfg, "hrl_behavior_mode_condition", False) else 1
        cfg.cli_args["rnn_size"] = rnn_size
        cfg.rnn_size = rnn_size

    if (
        getattr(cfg, "hrl_controllable_graph", False)
        and getattr(cfg, "hrl_graph_memory", "episode") == "episode"
        and getattr(cfg, "hrl_persistent_fast_weights", False)
    ):
        cfg.rnn_persistent_state_size = hrl_persistent_state_size(getattr(cfg, "Hippo_n_feature", 64))
    elif (
        getattr(cfg, "hrl_controllable_graph", False)
        and getattr(cfg, "hrl_graph_memory", "episode") == "policy_buffer"
        and getattr(cfg, "hrl_target_timing", "delayed") == "immediate"
    ):
        cfg.rnn_persistent_state_size = (
            1 + GEOMETRY_POLICY_SIZE + 1 if getattr(cfg, "hrl_behavior_mode_condition", False) else 1
        )
    else:
        cfg.rnn_persistent_state_size = 0
    if intrinsic_goal:
        cfg.rnn_persistent_state_size = 1


def parse_dmlab_args(argv=None, evaluation=False):
    parser, cfg = parse_sf_args(argv, evaluation=evaluation)
    add_hipposlam_env_args(parser)
    add_dmlab_env_args(parser)
    hipposlam_override_defaults(parser)
    cfg = parse_full_cfg(parser, argv)
    maybe_overwrite_rnn_size(cfg)
    return cfg


def main():
    """Script entry point."""
    cfg = parse_dmlab_args()

    # explicitly create the runner instead of simply calling run_rl()
    # this allows us to register additional message handlers
    cfg, runner = make_runner(cfg)
    register_msg_handlers(cfg, runner)

    level_caches = initialize_level_cache(cfg, get_mp_ctx(cfg.serial_mode))
    register_dmlab_components(level_caches)

    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()

    return status


if __name__ == "__main__":
    sys.exit(main())
