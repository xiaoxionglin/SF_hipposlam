"""Stable metric contract and legacy TensorBoard tag resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


EVALUATION_SCHEMA_VERSION = "intrmotiv/eval/v1"


@dataclass(frozen=True)
class MetricSpec:
    """One analysis metric, including compatible current and legacy tags."""

    key: str
    tags: tuple[str, ...]
    description: str
    source: str
    scope: str
    applicable_to: str = "all"
    suffixes: tuple[str, ...] = ()


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("env_steps", ("train/env_steps",), "Environment frames seen by the policy.", "framework", "run"),
    MetricSpec("throughput", ("train/fps", "train/avg_fps"), "Training throughput.", "framework", "update"),
    MetricSpec("policy_loss", ("train/policy_loss",), "PPO policy loss.", "learner", "update"),
    MetricSpec("value_loss", ("train/value_loss",), "PPO value loss.", "learner", "update"),
    MetricSpec("entropy", ("train/entropy",), "Policy entropy.", "learner", "update"),
    MetricSpec("dg_density", ("intrmotiv/dg/density", "train/dg_density"), "Fraction of active DG entries.", "learner", "minibatch"),
    MetricSpec("dg_multi_activation_fraction", ("intrmotiv/dg/multi_activation_fraction", "train/dg_multi_activation_rate"), "Fraction of transitions with multiple active DGs.", "learner", "minibatch"),
    MetricSpec("dg_silent_unit_fraction", ("intrmotiv/dg/silent_unit_fraction", "train/dg_silent_unit_frac"), "Minibatch-silent DG fraction.", "learner", "minibatch"),
    MetricSpec("dg_unit_duty_cycle_min", ("intrmotiv/dg/unit_duty_cycle_min", "train/dg_unit_duty_cycle_min"), "Least-active DG unit's minibatch duty cycle.", "learner", "minibatch"),
    MetricSpec("dg_unit_duty_cycle_mean", ("intrmotiv/dg/unit_duty_cycle_mean", "train/dg_unit_duty_cycle_mean"), "Mean per-unit DG duty cycle.", "learner", "minibatch"),
    MetricSpec("dg_unit_duty_cycle_max", ("intrmotiv/dg/unit_duty_cycle_max", "train/dg_unit_duty_cycle_max"), "Most-active DG unit's minibatch duty cycle.", "learner", "minibatch"),
    MetricSpec("dg_usage_entropy", ("intrmotiv/dg/usage_entropy", "train/dg_usage_entropy"), "Normalized entropy of DG unit usage; zero means one or no used units.", "learner", "minibatch"),
    MetricSpec("intrinsic_reward_mean", ("intrmotiv/reward/intrinsic_mean", "train/intrinsic_reward_mean"), "Mean internal reward before PPO-specific use.", "learner", "minibatch"),
    MetricSpec("intrinsic_reward_nonzero_fraction", ("intrmotiv/reward/intrinsic_nonzero_fraction", "train/intrinsic_reward_nonzero_frac"), "Fraction of nonzero internal rewards.", "learner", "minibatch"),
    MetricSpec("advantage_reward_mean", ("intrmotiv/reward/advantage_mean", "train/reward_for_advantage_mean"), "Mean reward supplied to PPO/GAE.", "learner", "minibatch"),
    MetricSpec("coverage_auc", ("intrmotiv/exploration/window/coverage_auc",), "Time-average cumulative visited cells.", "environment", "episode_or_window", suffixes=("_coverage_auc",)),
    MetricSpec("coverage_unique_cells", ("intrmotiv/exploration/window/coverage_unique_cells",), "Distinct discretized cells in the measurement interval.", "environment", "episode_or_window", suffixes=("_coverage_unique_cells",)),
    MetricSpec("coverage_entropy", ("intrmotiv/exploration/window/coverage_entropy",), "Occupancy entropy in the measurement interval.", "environment", "episode_or_window", suffixes=("_coverage_entropy",)),
    MetricSpec("active_target_fraction", ("intrmotiv/hrl/active_target_fraction", "train/hrl_active_target_frac"), "Transitions with an active HRL target.", "learner", "minibatch", "hrl"),
    MetricSpec("target_hit_rate", ("intrmotiv/hrl/target_hit_rate", "train/hrl_target_hit_rate"), "Stored target hits per valid policy transition.", "learner", "minibatch", "hrl"),
    MetricSpec("option_timeout_rate", ("intrmotiv/hrl/option_timeout_rate", "train/hrl_option_timeout_rate"), "Stored option timeouts per valid transition.", "learner", "minibatch", "hrl"),
    MetricSpec("option_success_fraction", ("intrmotiv/hrl/option_success_fraction", "train/hrl_option_success_fraction"), "Hits divided by hits plus timeouts.", "learner", "minibatch", "hrl"),
    MetricSpec("tctrl_update_rate", ("intrmotiv/hrl/tctrl_update_rate", "train/hrl_tctrl_update_rate"), "Graph updates per valid transition.", "learner", "minibatch", "hrl"),
    MetricSpec("known_edge_fraction", ("intrmotiv/hrl/known_edge_fraction", "train/hrl_known_edge_fraction"), "Confidence-qualified off-diagonal graph edges.", "learner", "minibatch", "hrl"),
    MetricSpec("forgotten_edge_fraction", ("intrmotiv/hrl/forgotten_edge_fraction", "train/hrl_forgotten_edge_fraction"), "Observed edges below the confidence threshold.", "learner", "minibatch", "hrl"),
    MetricSpec("known_controllability_time_mean", ("intrmotiv/hrl/known_controllability_time_mean", "train/hrl_known_controllability_time_mean"), "Mean T_ctrl over usable edges.", "learner", "minibatch", "hrl"),
    MetricSpec("selected_deadline_mean", ("intrmotiv/hrl/selected_deadline_mean", "train/hrl_selected_deadline_mean"), "Legacy deadline sum divided by all option resets, including targetless resets.", "learner", "minibatch", "hrl"),
    MetricSpec("selected_deadline_positive_mean", ("intrmotiv/hrl/selected_deadline_positive_mean",), "Mean selected deadline over strictly positive deadlines.", "learner", "minibatch", "hrl"),
    MetricSpec("deadline_selection_fraction", ("intrmotiv/hrl/deadline_selection_fraction",), "Fraction of option resets that selected a positive deadline.", "learner", "minibatch", "hrl"),
    MetricSpec("learned_deadline_fraction", ("intrmotiv/hrl/learned_deadline_fraction", "train/hrl_learned_deadline_fraction"), "Option resets using a graph-derived deadline.", "learner", "minibatch", "hrl"),
)

METRIC_BY_KEY = {metric.key: metric for metric in METRICS}


def resolve_tag(available_tags: Iterable[str], metric: MetricSpec) -> str | None:
    """Return an available tag, preferring explicit names to suffixes."""

    available = set(available_tags)
    for tag in metric.tags:
        if tag in available:
            return tag
    for suffix in metric.suffixes:
        matches = sorted(tag for tag in available if tag.endswith(suffix))
        if matches:
            return matches[0]
    return None


def measurement_scope(tag: str | None) -> str:
    if not tag:
        return "missing"
    if tag.startswith("intrmotiv/exploration/window/"):
        return "telemetry_window"
    if tag.startswith("policy_stats/"):
        return "physical_episode"
    return "minibatch_or_framework"
