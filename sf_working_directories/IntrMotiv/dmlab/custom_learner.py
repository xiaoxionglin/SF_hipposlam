from __future__ import annotations

import math
import re
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from sample_factory.algo.learning.learner import BaseLearner, DefaultLearner
from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.misc import (
    EPISODIC,
    LEARNER_ENV_STEPS,
    POLICY_ID_KEY,
    STATS_KEY,
    TRAIN_STATS,
    memory_stats,
)
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.rl_utils import gae_advantages
from sample_factory.algo.utils.tensor_dict import TensorDict, shallow_recursive_copy
from sample_factory.algo.utils.torch_utils import masked_select, synchronize, to_scalar
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.dicts import iterate_recursively
from sample_factory.utils.typing import ActionDistribution, Config, PolicyID
from sample_factory.utils.utils import log
from sf_working_directories.IntrMotiv.dmlab.custom_core import straight_through_binary
from sf_working_directories.IntrMotiv.dmlab.contextual_dg import (
    build_transition_prediction_batch,
    transition_prediction_losses,
)
from sf_working_directories.IntrMotiv.dmlab.dg_recruitment_graph import (
    batch_predictive_events,
    directional_recruitment_eligibility,
    graph_recruitment_eligibility,
)
from sf_working_directories.IntrMotiv.dmlab.empirical_her import (
    build_empirical_her_batch,
    normalize_empirical_her_advantage,
)
from sf_working_directories.IntrMotiv.dmlab.hrl_controllable_graph import (
    HRLStateLayout,
    exploration_mode_mask,
    hrl_option_state_size,
    hrl_state_size,
    option_target_one_hot,
)
from sf_working_directories.IntrMotiv.dmlab.iterative_update import (
    DECODER,
    ENCODER,
    IterativeUpdateSchedule,
    SIMULTANEOUS,
)
from sf_working_directories.IntrMotiv.dmlab.online_spatial_telemetry import TrainingSpatialTelemetry
from sf_working_directories.IntrMotiv.dmlab.topological_frontier import (
    ACTION_FEATURE_SIZE,
    GEOMETRY_POLICY_SIZE,
    MODE_EXPLORE,
    MODE_NAVIGATE,
    MODE_RETURN,
    MODE_VALIDATE,
    N_MANAGER_MODES,
    TopologicalStateLayout,
    dg_path_scatter_loss,
    geometric_candidate_edges,
    topological_state_size,
    update_topological_graph_from_rollout,
)


def reliable_graph_statistics(adjacency: Tensor, confidence: Tensor) -> dict[str, Tensor]:
    """Small directed-graph diagnostics for the 16-node policy graph."""
    adjacency = adjacency.bool().clone()
    adjacency.fill_diagonal_(False)
    n = adjacency.size(0)
    reach = adjacency.clone()
    reach.fill_diagonal_(True)
    for intermediate in range(n):
        reach |= reach[:, intermediate].unsqueeze(1) & reach[intermediate].unsqueeze(0)
    strong = reach & reach.transpose(0, 1)
    largest_scc = strong.sum(dim=1).max().float()
    off_diagonal_reach = reach.sum().float() - float(n)
    reliable_count = adjacency.sum().float()
    reciprocal = (adjacency & adjacency.transpose(0, 1)).sum().float() / reliable_count.clamp_min(1.0)
    weighted = confidence.masked_fill(~adjacency, 0.0)
    incoming = weighted.sum(dim=0)
    top_k = min(3, n)
    top_share = incoming.topk(top_k).values.sum() / incoming.sum().clamp_min(1e-12)
    return {
        "largest_scc": largest_scc,
        "reachable_pair_fraction": off_diagonal_reach / float(max(1, n * (n - 1))),
        "outgoing_node_fraction": adjacency.any(dim=1).float().mean(),
        "outgoing_node_count": adjacency.any(dim=1).sum().float(),
        "reciprocal_fraction": reciprocal,
        "top3_incoming_confidence_share": top_share,
    }


def categorical_action_total_variation(logits: Tensor, alternate_logits: Tensor) -> Tensor:
    """Per-sample total variation between two categorical action policies."""
    if logits.shape != alternate_logits.shape:
        raise ValueError("Action-logit tensors must have identical shapes")
    probabilities = torch.softmax(logits, dim=-1)
    alternate_probabilities = torch.softmax(alternate_logits, dim=-1)
    return 0.5 * (probabilities - alternate_probabilities).abs().sum(dim=-1)


def dg_gradient_interaction_stats(decoder_loss: Tensor, encoder_loss: Tensor, projection) -> dict[str, Tensor]:
    """Measure decoder/encoder DG gradients without accumulating parameter gradients."""
    parameters = [parameter for parameter in projection.parameters() if parameter.requires_grad]
    zero = decoder_loss.detach().new_zeros(())
    if not parameters:
        return {
            "ppo_norm": zero,
            "encoder_norm": zero,
            "ratio": zero,
            "cosine": zero,
            "row_conflict_fraction": zero,
        }
    decoder_grads = torch.autograd.grad(
        decoder_loss, parameters, retain_graph=True, allow_unused=True
    )
    encoder_grads = torch.autograd.grad(
        encoder_loss, parameters, retain_graph=True, allow_unused=True
    )
    ppo_sq = zero.clone()
    encoder_sq = zero.clone()
    dot = zero.clone()
    for parameter, ppo_grad, encoder_grad in zip(parameters, decoder_grads, encoder_grads):
        ppo = torch.zeros_like(parameter) if ppo_grad is None else ppo_grad
        encoder = torch.zeros_like(parameter) if encoder_grad is None else encoder_grad
        ppo_sq = ppo_sq + ppo.square().sum()
        encoder_sq = encoder_sq + encoder.square().sum()
        dot = dot + (ppo * encoder).sum()
    ppo_norm = ppo_sq.sqrt()
    encoder_norm = encoder_sq.sqrt()
    denominator = (ppo_norm * encoder_norm).clamp_min(torch.finfo(decoder_loss.dtype).eps)
    cosine = torch.where((ppo_norm > 0) & (encoder_norm > 0), dot / denominator, zero)

    row_conflict = zero
    linear_weight = getattr(getattr(projection, "linear", None), "weight", None)
    index = next((i for i, parameter in enumerate(parameters) if parameter is linear_weight), -1)
    if index >= 0 and decoder_grads[index] is not None and encoder_grads[index] is not None:
        ppo_rows = decoder_grads[index].reshape(decoder_grads[index].shape[0], -1)
        encoder_rows = encoder_grads[index].reshape(encoder_grads[index].shape[0], -1)
        jointly_active = (ppo_rows.norm(dim=1) > 0) & (encoder_rows.norm(dim=1) > 0)
        conflicts = (ppo_rows * encoder_rows).sum(dim=1) < 0
        row_conflict = (conflicts & jointly_active).sum() / jointly_active.sum().clamp_min(1)
    return {
        "ppo_norm": ppo_norm.detach(),
        "encoder_norm": encoder_norm.detach(),
        "ratio": (ppo_norm / encoder_norm.clamp_min(torch.finfo(decoder_loss.dtype).eps)).detach(),
        "cosine": cosine.detach(),
        "row_conflict_fraction": row_conflict.detach(),
    }


def finite_masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean over selected finite values, returning a finite zero if unsupported."""
    if values.shape != mask.shape:
        raise ValueError("Values and mask must have identical shapes")
    selected = mask.bool() & torch.isfinite(values)
    if not bool(selected.any()):
        return values.masked_fill(~torch.isfinite(values), 0.0).sum() * 0.0
    return values[selected].mean()


def legacy_reward_streams(
    internal_reward: Tensor,
    baseline: float,
    reward_scale: float,
    encoder_reward_method: str,
) -> tuple[Tensor, Tensor]:
    decoder_reward = (baseline - internal_reward[:, 2:]) * reward_scale
    encoder_base = internal_reward[:, 1:-1]
    if encoder_reward_method == "encourage":
        encoder_reward = encoder_base * reward_scale
    elif encoder_reward_method == "punish":
        encoder_reward = (encoder_base - baseline) * reward_scale
    elif encoder_reward_method == "mean":
        encoder_reward = (encoder_base - internal_reward.mean()) * reward_scale
    elif encoder_reward_method == "baseline_adjusted":
        adjusted = internal_reward.clone()
        reward_mask = adjusted != baseline
        adjusted[~reward_mask] -= baseline
        encoder_reward = adjusted[:, 1:-1] * reward_scale
    elif encoder_reward_method == "mean_baseline_adjusted":
        adjusted = internal_reward.clone()
        reward_mask = adjusted != baseline
        baseline_mean = adjusted[reward_mask].mean() if reward_mask.any() else baseline
        adjusted[reward_mask] -= baseline_mean
        adjusted[~reward_mask] -= baseline
        encoder_reward = adjusted[:, 1:-1] * reward_scale
    else:
        raise ValueError(f"Unknown encoder_reward_method: {encoder_reward_method}")
    return decoder_reward, encoder_reward


def dominant_new_activation_masks(
    sequence_core: Tensor,
    progression: Tensor,
    refractory_steps: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Select one behavior-DG event when several units start together.

    ``sequence_core`` has shape ``[B, T, F, K]`` and ``progression`` has
    shape ``[B, T, F]``. A new event starts when a unit is in slot zero now
    and was outside the first ``R`` slots at the preceding decision. The
    largest behavior activation in CA3 slot zero wins; ``argmax`` resolves
    exact ties in favor of the lowest DG index.
    """
    if sequence_core.ndim != 4 or progression.shape != sequence_core.shape[:-1]:
        raise ValueError("Expected sequence_core [B,T,F,K] and matching progression [B,T,F]")

    previous = torch.roll(progression, shifts=1, dims=1)
    candidates = (progression == 0) & (previous >= int(refractory_steps))
    candidates[:, 0] = False

    strengths = sequence_core[..., 0]
    masked_strengths = strengths.masked_fill(~candidates, -torch.inf)
    winner = masked_strengths.argmax(dim=-1)
    dominant = F.one_hot(winner, num_classes=progression.size(-1)).bool()
    dominant &= candidates.any(dim=-1, keepdim=True)
    non_dominant = candidates & ~dominant
    return candidates, dominant, non_dominant


def dg_unused_batch_recruitment_loss(
    pre_threshold_logits: Tensor,
    current_activity: Tensor,
    prior_active: Tensor,
    valids: Tensor,
    intercept: float,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """Encourage units absent from all valid transitions in this minibatch."""
    if temperature <= 0:
        raise ValueError("encoder_batch_loss_temperature must be positive")
    valid = valids.bool().reshape(-1)
    if pre_threshold_logits.shape != current_activity.shape or prior_active.shape != current_activity.shape:
        raise ValueError("DG recruitment tensors must have matching [batch, feature] shapes")
    if valid.numel() != current_activity.size(0):
        raise ValueError("valids must align with the DG minibatch")

    observed = (prior_active.bool() | (current_activity > 0)) & valid.unsqueeze(-1)
    unused = ~observed.any(dim=0)
    if not valid.any() or not unused.any():
        return pre_threshold_logits.sum() * 0.0, unused

    soft_activity = temperature * F.softplus((pre_threshold_logits - float(intercept)) / temperature)
    per_transition = (soft_activity * unused.to(dtype=soft_activity.dtype)).sum(dim=-1)
    per_transition = per_transition / unused.sum().to(dtype=soft_activity.dtype)
    return -per_transition[valid].mean(), unused


@torch.no_grad()
def normalize_dg_projection_rows(linear: torch.nn.Module) -> None:
    """Project trainable DG input rows back onto the unit sphere after SGD."""
    for parameter in linear.parameters():
        if parameter.ndim > 1:
            parameter.div_(parameter.norm(dim=1, keepdim=True).clamp_min(1e-6))


def complete_rollout_generation_mask(
    valids: Tensor,
    generation_matches: Tensor,
) -> tuple[Tensor, Tensor]:
    """Reject stale representation experience at whole-rollout boundaries.

    A replacement can be published while actors are collecting a rollout.
    Such a rollout is semantically mixed even if some individual decisions
    carry the new generation, so every decision in it is rejected together.
    """
    if valids.shape != generation_matches.shape or valids.ndim != 2:
        raise ValueError("Expected aligned [rollout, time] validity and generation masks")
    current_rollout = generation_matches.bool().all(dim=1)
    accepted = valids.bool() & current_rollout.unsqueeze(1)
    stale_rollout = ~current_rollout
    return accepted, stale_rollout


def require_sufficient_generation_batch(
    accepted_valids: Tensor,
    stale_rollouts: Tensor,
    minimum_valid_decisions: int,
) -> bool:
    """Whether a mixed-generation batch has enough fresh data to optimize."""
    if minimum_valid_decisions < 2:
        raise ValueError("A normal learner update requires at least two decisions")
    if not bool(stale_rollouts.any()):
        return True
    return int(accepted_valids.sum().item()) >= int(minimum_valid_decisions)


def forced_recruitment_preflight_selection(
    valids: Tensor,
    forced_row: int,
    n_nodes: int,
    train_step: int,
    after_updates: int,
    completed_replacements: int,
) -> tuple[int, int] | None:
    """Select one valid feature/row for the guarded engineering preflight."""
    if forced_row < 0:
        return None
    if not 0 <= forced_row < n_nodes:
        raise ValueError("Forced recruitment row is outside the DG")
    if completed_replacements > 0 or train_step < after_updates:
        return None
    valid_indices = torch.nonzero(valids.bool().reshape(-1), as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return None
    return int(valid_indices[0].item()), int(forced_row)


def record_poststep_calibration_count(
    loss_summaries: dict,
    reference: Tensor,
    calibrated: bool,
) -> None:
    """Update the loss-summary object that survives into the train loop."""
    if calibrated:
        loss_summaries["additional_stats"]["dg_running_stats_update_count"] = (
            reference.new_tensor(1.0)
        )


def predecessor_distance_for_dominant_events(
    progression: Tensor,
    candidates: Tensor,
    dominant: Tensor,
    baseline: int,
) -> Tensor:
    """Return nearest prior non-simultaneous DG age for each event time."""
    if progression.shape != candidates.shape or progression.shape != dominant.shape:
        raise ValueError("Progression, candidate, and dominant masks must have identical shapes")
    predecessors = progression.masked_fill(candidates, int(baseline) + 100)
    distance = predecessors.min(dim=-1).values.clamp_max(int(baseline))
    default = torch.full_like(distance, int(baseline))
    return torch.where(dominant.any(dim=-1), distance, default).to(dtype=torch.float)


def build_matched_encoder_credit(
    progression: Tensor,
    candidates: Tensor,
    dominant: Tensor,
    valids: Tensor,
    baseline: int,
    reward_scale: float,
    recipient: str,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Align each arrival to a verified within-rollout predecessor onset."""
    if progression.shape != candidates.shape or progression.shape != dominant.shape:
        raise ValueError("progression and activation masks must have identical shapes")
    if valids.shape != progression.shape[:2]:
        raise ValueError("valids must align with the rollout time axes")
    if recipient not in ("arrival", "source"):
        raise ValueError(f"Unknown encoder reward recipient: {recipient}")

    rewards = progression.new_zeros(progression.shape, dtype=torch.float)
    row_mask = torch.zeros_like(dominant, dtype=torch.bool)
    event_mask = dominant.any(dim=-1) & valids.bool()
    # Every row with age zero is simultaneous with the destination decision.
    # This includes continuously active rows that were reinjected into CA3 but
    # are not new-onset candidates. None of them is a predecessor.
    predecessor_age = progression.masked_fill(progression.eq(0), int(baseline) + 100)
    nearest_lag = predecessor_age.min(dim=-1).values
    counts = {name: progression.new_zeros((), dtype=torch.float) for name in (
        "total", "matchable", "credited", "boundary_dropped", "alignment_failure",
        "invalid_interval", "collisions", "reward_mass", "source_lag_sum", "source_lag_max",
    )}

    for stream, arrival_t in torch.nonzero(event_mask, as_tuple=False).tolist():
        counts["total"].add_(1.0)
        lag = int(nearest_lag[stream, arrival_t].item())
        if lag >= int(baseline):
            counts["alignment_failure"].add_(1.0)
            continue
        source_t = int(arrival_t) - lag
        if source_t < 0:
            counts["boundary_dropped"].add_(1.0)
            continue
        counts["matchable"].add_(1.0)
        if not bool(valids[stream, source_t : arrival_t + 1].bool().all()):
            counts["invalid_interval"].add_(1.0)
            continue
        # Several rows can enter CA3 on the same source decision. They have the
        # same nearest lag, but only one carries the behavior-time dominant
        # label. Resolve the row inside that nearest-lag tie rather than using
        # the lowest DG index and spuriously rejecting the event.
        nearest_rows = predecessor_age[stream, arrival_t].eq(lag)
        verified_rows = nearest_rows & dominant[stream, source_t]
        if not bool(verified_rows.any()):
            counts["alignment_failure"].add_(1.0)
            continue
        source_row = int(torch.nonzero(verified_rows, as_tuple=False)[0].item())
        arrival_row = int(torch.nonzero(dominant[stream, arrival_t], as_tuple=False)[0].item())
        credit_t, credit_row = (
            (arrival_t, arrival_row) if recipient == "arrival" else (source_t, source_row)
        )
        if bool(row_mask[stream, credit_t, credit_row]):
            counts["collisions"].add_(1.0)
        reward = float(reward_scale) * float(lag)
        rewards[stream, credit_t, credit_row].add_(reward)
        row_mask[stream, credit_t, credit_row] = True
        counts["credited"].add_(1.0)
        counts["reward_mass"].add_(reward)
        counts["source_lag_sum"].add_(float(lag))
        counts["source_lag_max"].copy_(torch.maximum(counts["source_lag_max"], counts["source_lag_max"].new_tensor(float(lag))))
    return rewards, row_mask, counts


def retirement_endpoint_allowed(endpoint_gate: str, active_units: Tensor) -> bool:
    if endpoint_gate not in ("silent", "open"):
        raise ValueError(f"Unknown DG recruitment endpoint gate: {endpoint_gate}")
    return endpoint_gate == "open" or not bool(active_units.any())


def dg_global_punishment_loss(
    pre_threshold_logits: Tensor,
    intercept: float,
    temperature: float,
    coefficient: float,
    valids: Tensor,
    num_invalids: int,
) -> Tensor:
    """Penalize every DG logit with a smooth pre-threshold surrogate."""
    if coefficient == 0.0:
        return pre_threshold_logits.sum() * 0.0
    if temperature <= 0.0:
        raise ValueError("dg_global_punishment_temperature must be positive")
    per_transition = temperature * F.softplus((pre_threshold_logits - intercept) / temperature).mean(dim=-1)
    return float(coefficient) * masked_select(per_transition, valids, num_invalids).mean()


def dg_row_repulsion_loss(weight: Tensor, coefficient: float) -> Tensor:
    """Spread DG projection rows angularly without using observations."""
    if coefficient == 0.0:
        return weight.sum() * 0.0
    normalized = F.normalize(weight, p=2, dim=1)
    similarities = normalized @ normalized.T
    off_diagonal = ~torch.eye(similarities.size(0), dtype=torch.bool, device=similarities.device)
    return float(coefficient) * similarities.square().masked_select(off_diagonal).mean()


def dg_ca3_temporal_exclusion_loss(
    dg_activity: Tensor,
    rnn_states: Tensor,
    dominant_activation_mask: Tensor,
    n_features: int,
    R: int,
    L: int,
    coefficient: float,
    reward_scale: float,
    valids: Tensor,
    num_invalids: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply an R-step margin to dominant DG onsets and report CA3 conflicts.

    With the ``encourage`` encoder objective, coefficient 1 gives the combined
    event loss ``-reward_scale * (distance - R) * activity``. Consequently,
    dominant onsets below R are suppressed, an onset at R is neutral, and an
    onset above R is reinforced. The CA3 conflict mask remains diagnostic and
    no longer turns this term into a broad penalty on all current DG activity.
    """
    expanded_length = int(R) + int(L) - 1
    ca3 = rnn_states[:, : n_features * expanded_length].view(-1, n_features, expanded_length)
    prior = ca3[:, :, int(R) - 1] > 0
    conflict = prior.sum(dim=-1, keepdim=True) - prior.to(dtype=torch.int64) > 0
    activity = dg_activity[:, :n_features].clamp_min(0.0)
    if dominant_activation_mask.shape != activity.shape:
        raise ValueError("dominant_activation_mask must match the DG activity shape")
    dominant = dominant_activation_mask.bool()
    event_activity = (activity * dominant.to(dtype=activity.dtype)).sum(dim=-1)
    raw_loss = masked_select(event_activity, valids, num_invalids).mean()

    conflict_per_transition = (activity * conflict.to(dtype=activity.dtype)).mean(dim=-1)
    valid_conflict_activity = masked_select(conflict_per_transition, valids, num_invalids)

    valid_conflict = masked_select(conflict.float().mean(dim=-1), valids, num_invalids)
    conflict_fraction = valid_conflict.mean()
    active = activity > 0
    active_count = masked_select(active.float().sum(dim=-1), valids, num_invalids).sum()
    conflicting_active_count = masked_select(
        (active & conflict).float().sum(dim=-1), valids, num_invalids
    ).sum()
    conflicting_activation_fraction = conflicting_active_count / active_count.clamp_min(1.0)
    conflict_count = masked_select(conflict.float().sum(dim=-1), valids, num_invalids).sum()
    conflict_activity = valid_conflict_activity.sum() * n_features / conflict_count.clamp_min(1.0)
    return (
        float(coefficient) * float(reward_scale) * int(R) * raw_loss,
        conflict_fraction.detach(),
        conflicting_activation_fraction.detach(),
        conflict_activity.detach(),
    )


def dg_recruitment_candidate_mask(
    rnn_states: Tensor,
    n_features: int,
    R: int,
    L: int,
    valids: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Find one-shot endpoints exactly L decisions after a lone source DG activation."""
    expanded_length = int(R) + int(L) - 1
    ca3 = rnn_states[..., : n_features * expanded_length].view(*rnn_states.shape[:-1], n_features, expanded_length)
    occupied = ca3 > 0
    unit_seen = occupied.any(dim=-1)
    at_tail = occupied[..., -1]
    previous_tail = torch.zeros_like(at_tail)
    if at_tail.ndim == 2:
        previous_tail[0] = at_tail[0]
        previous_tail[1:] = at_tail[:-1]
    elif at_tail.ndim == 3:
        previous_tail[:, 0] = at_tail[:, 0]
        previous_tail[:, 1:] = at_tail[:, :-1]
    else:
        raise ValueError(f"Expected [time,state] or [batch,time,state], got {tuple(rnn_states.shape)}")
    entered_tail = at_tail & ~previous_tail
    candidate = entered_tail.sum(dim=-1).eq(1) & unit_seen.sum(dim=-1).eq(1) & valids.bool()
    source = entered_tail.to(dtype=torch.int64).argmax(dim=-1)
    return candidate, source, unit_seen


def orthogonal_feature_residual(feature: Tensor, committed_rows: Tensor, eps: float = 1e-6) -> Tensor | None:
    """Return the unit component of feature outside the linear span of committed rows."""
    residual = feature
    if committed_rows.numel() > 0:
        _, singular_values, vh = torch.linalg.svd(committed_rows, full_matrices=False)
        tolerance = max(committed_rows.shape) * torch.finfo(committed_rows.dtype).eps * singular_values.max()
        basis = vh[singular_values > tolerance]
        if basis.numel() > 0:
            residual = feature - (feature @ basis.T) @ basis
    norm = residual.norm()
    if not torch.isfinite(norm) or norm <= float(eps):
        return None
    return residual / norm


def dg_usage_metrics(dg_active: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Summarize thresholded DG use without changing the training objective."""
    if dg_active.ndim != 2:
        raise ValueError(f"Expected [transitions, DG units], got {tuple(dg_active.shape)}")
    duty_cycle = dg_active.float().mean(dim=0)
    usage = duty_cycle / duty_cycle.sum().clamp_min(torch.finfo(duty_cycle.dtype).eps)
    entropy = -(usage * usage.clamp_min(torch.finfo(usage.dtype).eps).log()).sum()
    if duty_cycle.numel() > 1:
        entropy = entropy / math.log(duty_cycle.numel())
    else:
        entropy = entropy * 0.0
    return duty_cycle.min(), duty_cycle.mean(), duty_cycle.max(), entropy


def target_gate_decoder_reward(decoder_reward: Tensor, target_hit: Tensor) -> Tensor:
    return decoder_reward * target_hit[:, 2:]


def selected_deadline_stats(selected_deadline: Tensor, option_reset: Tensor) -> tuple[Tensor, Tensor]:
    """Return the positive-deadline mean and selected-target fraction at resets."""
    selected = selected_deadline > 0
    mean = selected_deadline.sum() / selected.sum().clamp_min(1)
    fraction = selected.sum() / option_reset.sum().clamp_min(1)
    return mean, fraction


def target_reward_magnitude(
    internal_reward: Tensor,
    baseline: float,
    reward_scale: float,
    mode: str,
    hit_reward: float,
    distance_bonus_coeff: float,
) -> Tensor:
    """Current positive hit magnitude, kept separate from outcome semantics."""
    reward = torch.full_like(internal_reward[:, 2:], float(hit_reward))
    if mode == "hit_distance":
        legacy_bonus = ((baseline - internal_reward[:, 2:]) * reward_scale).clamp_min(0.0)
        reward = reward + float(distance_bonus_coeff) * legacy_bonus
    elif mode != "hit":
        raise ValueError(f"Unknown HRL worker reward mode: {mode}")
    return reward


def target_success_worker_reward(
    internal_reward: Tensor,
    target_hit: Tensor,
    baseline: float,
    reward_scale: float,
    mode: str,
    hit_reward: float,
    distance_bonus_coeff: float,
    exploration_mode: Tensor | None = None,
    exploration_reward: Tensor | None = None,
    *,
    control_outcome: str = "target_hit",
    wrong_outcome: Tensor | None = None,
    n_targets: int | None = None,
    command_set_size: Tensor | None = None,
) -> Tensor:
    hit = target_hit[:, 2:].to(dtype=internal_reward.dtype)
    reward = target_reward_magnitude(
        internal_reward, baseline, reward_scale, mode, hit_reward, distance_bonus_coeff
    )
    worker_reward = hit * reward
    if control_outcome == "first_distinct":
        if wrong_outcome is None:
            raise ValueError("first_distinct reward requires wrong_outcome")
        wrong = wrong_outcome.to(dtype=internal_reward.dtype)
        if wrong.shape != worker_reward.shape:
            raise ValueError("Wrong-outcome mask must align with worker reward transitions")
        if command_set_size is None:
            if n_targets is None or int(n_targets) <= 2:
                raise ValueError("first_distinct reward requires n_targets > 2 or command_set_size")
            # One source is excluded, leaving n_targets - 1 possible commands;
            # for a fixed in-set outcome, n_targets - 2 commands are wrong.
            wrong_denominator = float(int(n_targets) - 2)
        else:
            if command_set_size.shape != worker_reward.shape:
                raise ValueError("Command-set size must align with worker reward transitions")
            wrong_denominator = (command_set_size.to(reward.dtype) - 1.0).clamp_min(1.0)
        worker_reward = worker_reward - wrong * reward / wrong_denominator
    elif control_outcome != "target_hit":
        raise ValueError(f"Unknown HRL control outcome: {control_outcome}")
    if exploration_mode is None:
        return worker_reward
    if exploration_reward is None:
        raise ValueError("exploration_reward is required when exploration_mode is provided")
    exploring = exploration_mode[:, 1:-1].to(dtype=torch.bool)
    if exploring.shape != worker_reward.shape or exploration_reward.shape != worker_reward.shape:
        raise ValueError("Exploration mode and reward must align with worker reward transitions")
    return torch.where(exploring, exploration_reward, worker_reward)


def control_outcome_labels(hrl_state: Tensor, dones: Tensor, layout: HRLStateLayout) -> dict[str, Tensor]:
    """Decode transition-aligned intentional outcomes from behavior-time option state."""
    if hrl_state.ndim != 3 or hrl_state.size(1) != dones.size(1) + 2:
        raise ValueError("Outcome labels require [batch, T+2, state] and aligned [batch, T] dones")
    behavior = hrl_state[:, 1:-1]
    event = hrl_state[:, 2:]
    target = behavior[..., layout.target].long() - 1
    source = behavior[..., layout.source].long() - 1
    normal_target = (target >= 0) & (target < layout.n_nodes)
    exploration_target = target == layout.n_nodes
    unsuccessful = event[..., layout.option_expired] > 0
    negative_elapsed = event[..., layout.completion_elapsed] < 0
    wrong = unsuccessful & normal_target & negative_elapsed
    timeout = unsuccessful & normal_target & (~negative_elapsed)
    exploration_timeout = unsuccessful & (
        exploration_target | ((~normal_target) & negative_elapsed)
    )
    correct = event[..., layout.target_hit] > 0
    completed = correct | wrong | timeout
    return {
        "correct": correct,
        "wrong": wrong,
        "timeout": timeout,
        "exploration_timeout": exploration_timeout,
        "censored": dones.bool() & normal_target & (~completed),
        "completed": completed,
        "target": target,
        "source": source,
        "outcome": event[..., layout.active_dg].long() - 1,
        "elapsed": event[..., layout.completion_elapsed].abs(),
    }


def future_target_labels(
    target_onehot: Tensor,
    dg_activity: Tensor,
    dones: Tensor,
    horizon: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Label whether and when each current target appears later in the same trajectory."""
    batch, steps, n_targets = target_onehot.shape
    horizon = min(int(horizon), max(steps - 1, 0))
    target = target_onehot.argmax(dim=-1)
    valid = target_onehot.sum(dim=-1) > 0
    active = dg_activity > 0
    hit = torch.zeros((batch, steps), dtype=torch.bool, device=target_onehot.device)
    hit_time = torch.zeros((batch, steps), dtype=dg_activity.dtype, device=target_onehot.device)
    boundary_free = torch.ones((batch, steps), dtype=torch.bool, device=target_onehot.device)

    for delta in range(1, horizon + 1):
        boundary_free[:, : steps - delta] &= ~dones[:, delta - 1 : steps - 1].bool()
        future_active = active[:, delta:, :]
        current_target = target[:, : steps - delta]
        matched = future_active.gather(2, current_target.unsqueeze(-1)).squeeze(-1)
        available = valid[:, : steps - delta] & boundary_free[:, : steps - delta] & ~hit[:, : steps - delta]
        new_hit = available & matched
        hit[:, : steps - delta] |= new_hit
        hit_time[:, : steps - delta] = torch.where(
            new_hit,
            torch.full_like(hit_time[:, : steps - delta], float(delta)),
            hit_time[:, : steps - delta],
        )

    return hit.to(dtype=dg_activity.dtype), hit_time, valid


class BaseDistanceRecorder(BaseLearner):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        BaseLearner.__init__(self, cfg, env_info, policy_versions_tensor, policy_id, param_server)
        self._online_spatial = None

    def _capture_online_spatial(self, buff: TensorDict) -> None:
        if not bool(getattr(self.cfg, "online_spatial_telemetry", False)):
            return
        if getattr(self, "_online_spatial", None) is None:
            # BaseLearner.init restores env_steps before the first training
            # batch, so this lazy construction implements strict resume cadence.
            self._online_spatial = TrainingSpatialTelemetry(
                self.cfg, self.env_info, self.policy_id, self.env_steps, getattr(self, "actor_critic", None)
            )
        valids = (buff["policy_id"] == self.policy_id) & (
            self.train_step - buff["policy_version"] < self.cfg.max_policy_lag
        )
        self._online_spatial.append_batch(buff, valids)

    def train(self, batch: TensorDict) -> Optional[Dict]:
        # Apply pending PBT/checkpoint changes before filtering behavior policy
        # versions. BaseLearner.train repeats these no-op-safe checks.
        self._maybe_update_cfg()
        self._maybe_load_policy()
        self._capture_online_spatial(batch)
        stats = super().train(batch)
        if stats is None or getattr(self, "_online_spatial", None) is None:
            return stats
        spatial_stats = self._online_spatial.on_env_steps(self.env_steps)
        if spatial_stats:
            stats.setdefault(TRAIN_STATS, {}).update(spatial_stats)
        return stats

    def _advantage_reward_source(self):
        source = getattr(self.cfg, "advantage_reward_source", None)
        if source in (None, "", "legacy"):
            return None
        if source not in ("external", "internal"):
            raise ValueError(f"Unknown advantage_reward_source: {source}")
        return source

    def _use_external_reward_for_advantage(self) -> bool:
        source = self._advantage_reward_source()
        if source is not None:
            return source == "external"
        return bool(getattr(self.cfg, "use_external", True))

    def _use_internal_reward_for_advantage(self) -> bool:
        source = self._advantage_reward_source()
        if source is not None:
            return source == "internal"
        return bool(getattr(self.cfg, "use_internal", False))

    def _hrl_layout(self):
        return HRLStateLayout(getattr(self.cfg, "Hippo_n_feature", 64))

    def _hrl_state_offset(self) -> int:
        R = getattr(self.cfg, "Hippo_R", 8)
        L = getattr(self.cfg, "Hippo_L", 48)
        hippo_n_feature = getattr(self.cfg, "Hippo_n_feature", 64)
        action_features = ACTION_FEATURE_SIZE if getattr(self.cfg, "hrl_action_path_integration", False) else 0
        context_actions = 0
        if (
            getattr(self.cfg, "dg_context_feedback", "none") != "none"
            and getattr(self.cfg, "dg_context_history", "ca3") == "ca3_action"
        ):
            context_actions = 5 if getattr(self.cfg, "dmlab_reduced_action_set", False) else (
                15 if getattr(self.cfg, "dmlab_extended_action_set", False) else 9
            )
        return hippo_n_feature * (R + L - 1) + 13 + action_features + context_actions

    def _uses_topological_manager(self) -> bool:
        return self._uses_policy_graph() and getattr(self.cfg, "hrl_manager_mode", "visit_direct") != "visit_direct"

    def _hrl_state_from_rnn(self, rnn_states: Tensor) -> Tensor:
        layout = self._hrl_layout()
        offset = self._hrl_state_offset()
        state_size = (
            hrl_option_state_size(layout.n_nodes)
            if getattr(self.cfg, "hrl_graph_memory", "episode") == "policy_buffer"
            else hrl_state_size(layout.n_nodes)
        )
        expected = offset + state_size
        if rnn_states.size(-1) < expected:
            raise RuntimeError(f"HRL rnn_state expects at least {expected} features, got {rnn_states.size(-1)}")
        return rnn_states[..., offset:expected]

    def _topological_state_from_rnn(self, rnn_states: Tensor) -> Tensor:
        if not self._uses_topological_manager():
            raise RuntimeError("Topological RNN state requested while the topological manager is disabled")
        offset = self._hrl_state_offset() + hrl_option_state_size(self._hrl_layout().n_nodes)
        size = topological_state_size(self._hrl_layout().n_nodes)
        expected = offset + size
        if rnn_states.size(-1) < expected:
            raise RuntimeError(f"Topological rnn_state expects at least {expected} features, got {rnn_states.size(-1)}")
        return rnn_states[..., offset:expected]

    def _uses_policy_graph(self) -> bool:
        return bool(getattr(self.cfg, "hrl_controllable_graph", False)) and (
            getattr(self.cfg, "hrl_graph_memory", "episode") == "policy_buffer"
        )

    def _uses_graph_recruitment(self) -> bool:
        return bool(getattr(self.cfg, "dg_orthogonal_recruitment", False)) and (
            getattr(self.cfg, "dg_orthogonal_recruitment_mode", "legacy") == "graph"
        )

    def _invalidate_recruited_node(self, row: int) -> None:
        """Invalidate state whose landmark identity changed after recruitment."""
        graph_recruitment = self._uses_graph_recruitment()
        if graph_recruitment:
            self._passive_recruitment_graph().invalidate_node(row)
        if not self._uses_policy_graph():
            return
        self._policy_graph().invalidate_node(row)
        if graph_recruitment:
            invalidated = self._predictive_recruitment_evidence().invalidate_node(row)
            self._last_recruitment_stats["predictive_invalidation_mass"] += float(invalidated.item())

    def _passive_recruitment_graph(self):
        graph = getattr(getattr(self.actor_critic, "core", None), "passive_recruitment_graph", None)
        if graph is None:
            raise RuntimeError("Graph recruitment requires a passive recruitment graph on the model core")
        return graph

    def _predictive_recruitment_evidence(self):
        evidence = getattr(getattr(self.actor_critic, "core", None), "predictive_recruitment_evidence", None)
        if evidence is None:
            raise RuntimeError("Persistent PRED requires evidence buffers on the model core")
        return evidence

    def _uses_immediate_target_timing(self) -> bool:
        return self._uses_policy_graph() and getattr(self.cfg, "hrl_target_timing", "delayed") == "immediate"

    def _behavior_targets_from_states(self, rnn_states: Tensor) -> Tensor:
        """Decode the target that actually conditioned an actor decision."""
        if getattr(self.cfg, "intrinsic_goal_mode", "none") != "none":
            n = int(self.cfg.Hippo_n_feature)
            ids = rnn_states[..., -1].round().long()
            return F.one_hot((ids - 1).clamp(0, n - 1), n).to(rnn_states.dtype) * ids.gt(0).unsqueeze(-1)
        n_nodes = self._hrl_layout().n_nodes
        if self._uses_immediate_target_timing():
            descriptor_size = int(getattr(self.actor_critic.core, "behavior_goal_state_size", 1))
            target_id = rnn_states[..., -descriptor_size].round().long()
            valid = (target_id > 0) & (target_id <= n_nodes)
            safe = (target_id - 1).clamp(min=0, max=n_nodes - 1)
            return F.one_hot(safe, num_classes=n_nodes).to(dtype=rnn_states.dtype) * valid.unsqueeze(-1)
        return option_target_one_hot(self._hrl_state_from_rnn(rnn_states), n_nodes)

    def _behavior_condition_from_states(self, rnn_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        target = self._behavior_targets_from_states(rnn_states)
        shape = rnn_states.shape[:-1]
        geometry = rnn_states.new_zeros(shape + (GEOMETRY_POLICY_SIZE,))
        mode = rnn_states.new_zeros(shape + (N_MANAGER_MODES,))
        descriptor_size = int(getattr(self.actor_critic.core, "behavior_goal_state_size", 1))
        if descriptor_size > 1:
            descriptor = rnn_states[..., -descriptor_size:]
            geometry = descriptor[..., 1 : 1 + GEOMETRY_POLICY_SIZE]
            mode_id = descriptor[..., -1].round().long().clamp(0, N_MANAGER_MODES - 1)
            mode = F.one_hot(mode_id, N_MANAGER_MODES).to(dtype=rnn_states.dtype)
        return target, geometry, mode

    def _policy_graph(self):
        graph = getattr(getattr(self.actor_critic, "core", None), "policy_graph", None)
        if graph is None:
            raise RuntimeError("Policy graph mode requires a controllable graph buffer on the model core")
        return graph

    def _current_generation_mask(self, rnn_states: Tensor) -> Tensor:
        """Whether each behavior sample belongs to the current landmark generation."""
        if not self._uses_policy_graph():
            return torch.ones(rnn_states.shape[:-1], dtype=torch.bool, device=rnn_states.device)
        option_state = self._hrl_state_from_rnn(rnn_states)
        stored = option_state[..., self._hrl_layout().persistent_start]
        current = self._policy_graph().representation_generation.to(
            device=stored.device, dtype=stored.dtype
        )
        return stored.eq(current)

    def _override_core_outputs_for_replay(self, core_outputs: Tensor, mb: AttrDict) -> Tensor:
        """Teacher-force the behavior target before PPO evaluates the decoder."""
        if getattr(self.cfg, "intrinsic_goal_mode", "none") != "none":
            target = self._behavior_targets_from_states(mb.rnn_states)
            result = self._with_worker_target(core_outputs, target)
            start = self.actor_critic.core.target_condition_start
            self._last_behavior_replay_mismatch = (result[:, start:start + target.size(-1)] - target).abs().max().detach()
            return result
        if not self._uses_policy_graph():
            return core_outputs
        target, geometry, mode = self._behavior_condition_from_states(mb.rnn_states)
        if "hrl_behavior_targets" in mb:
            target = mb.hrl_behavior_targets
        target_start = self.actor_critic.core.target_condition_start
        replay_outputs = core_outputs.clone()
        replay_outputs[:, target_start : target_start + target.size(-1)] = target.to(dtype=core_outputs.dtype)
        if bool(getattr(self.cfg, "hrl_behavior_mode_condition", False)):
            if getattr(self.cfg, "hrl_landmark_geometry", "none") == "se2":
                geometry_start = self.actor_critic.core.geometry_condition_start
                replay_outputs[:, geometry_start : geometry_start + GEOMETRY_POLICY_SIZE] = geometry.to(
                    dtype=core_outputs.dtype
                )
            mode_start = self.actor_critic.core.mode_condition_start
            replay_outputs[:, mode_start : mode_start + N_MANAGER_MODES] = mode.to(dtype=core_outputs.dtype)
        mismatches = [
            (replay_outputs[:, target_start : target_start + target.size(-1)] - target).abs().max()
        ]
        if bool(getattr(self.cfg, "hrl_behavior_mode_condition", False)):
            if getattr(self.cfg, "hrl_landmark_geometry", "none") == "se2":
                mismatches.append(
                    (
                        replay_outputs[:, geometry_start : geometry_start + GEOMETRY_POLICY_SIZE]
                        - geometry
                    ).abs().max()
                )
            mismatches.append(
                (replay_outputs[:, mode_start : mode_start + N_MANAGER_MODES] - mode).abs().max()
            )
        self._last_behavior_replay_mismatch = torch.stack(mismatches).max().detach()
        return replay_outputs

    def _with_worker_target(self, core_outputs: Tensor, target: Tensor) -> Tensor:
        target_start = self.actor_critic.core.target_condition_start
        conditioned = core_outputs.clone()
        conditioned[:, target_start : target_start + target.size(-1)] = target.to(dtype=core_outputs.dtype)
        return conditioned

    def _empirical_her_loss(
        self,
        core_outputs: Tensor,
        head_outputs: Tensor,
        mb: AttrDict,
        recurrence: int,
        clip_ratio_low: float,
        clip_ratio_high: float,
        clip_value: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Biased PPO-style hindsight objective, isolated from ordinary PPO."""
        zero = core_outputs.sum() * 0.0
        if not bool(getattr(self.cfg, "hrl_empirical_her", False)):
            return zero, zero.detach(), zero.detach(), zero.detach(), zero.detach(), zero.detach()
        if not self._uses_policy_graph() or getattr(self.cfg, "hrl_manager_mode", "visit_direct") != "visit_direct":
            raise ValueError("hrl_empirical_her requires direct policy-buffer HRL")

        if head_outputs.size(0) % recurrence:
            raise RuntimeError("Empirical HER requires complete recurrent trajectories")
        trajectories = head_outputs.size(0) // recurrence
        with torch.no_grad():
            dg = head_outputs[:, : self._hrl_layout().n_nodes]
            active = dg.argmax(dim=-1)
            exclusive = dg.gt(0).sum(dim=-1).eq(1)
            active_ids = (active + 1).where(exclusive, torch.zeros_like(active)).view(trajectories, recurrence)
            her = build_empirical_her_batch(
                active_ids,
                mb.dones.view(trajectories, recurrence),
                mb.valids.view(trajectories, recurrence),
                torch.full_like(active_ids, float(self.cfg.hrl_target_hit_reward), dtype=head_outputs.dtype),
                self._hrl_layout().n_nodes,
                int(self.cfg.hrl_empirical_her_horizon),
                float(self.cfg.gamma),
            )
            her_targets = her.targets.reshape(head_outputs.size(0), -1)
            segment_lengths = her.valid.float().sum(dim=1)
            if self.cfg.hrl_worker_reward_mode == "hit_distance":
                baseline = float(self.cfg.Hippo_L + self.cfg.Hippo_R - 1)
                terminal_values = float(self.cfg.hrl_target_hit_reward) + float(self.cfg.hrl_distance_bonus_coeff) * (
                    (baseline - (segment_lengths - 1.0)).clamp_min(0.0) * float(self.cfg.reward_scale)
                )
                terminal_scale = terminal_values / float(self.cfg.hrl_target_hit_reward)
            else:
                terminal_scale = torch.ones_like(segment_lengths)
            her_returns = (her.returns * terminal_scale.unsqueeze(-1)).reshape(-1)
            her_terminal_reward = her.terminal_reward * terminal_scale.unsqueeze(-1)
            valid = her.valid.reshape(-1)
            accepted = her.accepted_segments.clamp_min(1.0)
            self._last_empirical_her_stats = dict(
                accepted_segments=float(her.accepted_segments.item()),
                skipped_no_endpoint=float(her.skipped_no_endpoint.item()),
                skipped_same_source=float(her.skipped_same_source.item()),
                segment_length=float(her.valid.float().sum().div(accepted).item()),
                positive_fraction=float(her.valid.float().mean().item()),
                terminal_reward=float(her_terminal_reward.sum().div(accepted).item()),
            )
        valid_count = valid.sum()
        if valid_count.item() == 0:
            return zero, zero.detach(), zero.detach(), zero.detach(), zero.detach(), zero.detach()

        conditioned = self._with_worker_target(core_outputs, her_targets)
        result = self.actor_critic.forward_tail(conditioned, values_only=False, sample_actions=False)
        distribution = get_action_distribution(self.actor_critic.action_space, result["action_logits"])
        log_prob = distribution.log_prob(mb.actions)
        ratio = torch.exp(log_prob - mb.log_prob_actions).clamp(0.05, 20.0)
        values = result["values"].squeeze()
        advantage = her_returns - values.detach()
        advantage = normalize_empirical_her_advantage(advantage, valid, self.cfg.normalize_advantage)

        num_invalids = int((~valid).sum().item())
        policy_loss = self._policy_loss(ratio, advantage, clip_ratio_low, clip_ratio_high, valid, num_invalids)
        value_loss = self._value_loss(values, values.detach(), her_returns, clip_value, valid, num_invalids)
        weighted = float(self.cfg.hrl_empirical_her_coeff) * (policy_loss + value_loss)
        clip_fraction = ((ratio < clip_ratio_low) | (ratio > clip_ratio_high))[valid].float().mean()
        return weighted, policy_loss.detach(), value_loss.detach(), ratio[valid].mean().detach(), clip_fraction.detach(), valid.float().mean().detach()

    @torch.no_grad()
    def _update_policy_graph_from_rollout(self, rnn_states: Tensor, valid_steps: Tensor) -> dict[str, Tensor] | None:
        if not self._uses_policy_graph():
            return None
        option_states = self._hrl_state_from_rnn(rnn_states)
        stats = self._policy_graph().update_from_option_rollout(
            option_states,
            valid_steps,
            float(self.cfg.hrl_fast_weight_half_life_options),
            float(self.cfg.hrl_edge_confidence_threshold),
            float(getattr(self.cfg, "hrl_edge_reliability_threshold", 0.5)),
        )
        if self._uses_graph_recruitment():
            self._passive_recruitment_graph().decay_birth_support(
                stats["completion_count"],
                float(self.cfg.hrl_fast_weight_half_life_options),
            )
        if self._uses_topological_manager():
            topological_stats = update_topological_graph_from_rollout(
                self._policy_graph(),
                option_states,
                self._topological_state_from_rnn(rnn_states),
                valid_steps,
                geometry=getattr(self.cfg, "hrl_landmark_geometry", "none"),
                pose_steps=int(getattr(self.cfg, "hrl_geometry_update_steps", 5)),
                pose_learning_rate=float(getattr(self.cfg, "hrl_geometry_learning_rate", 0.05)),
            )
            stats.update(topological_stats)
        return stats

    @torch.no_grad()
    def _update_passive_recruitment_graph_from_rollout(
        self, rnn_states: Tensor, valid_steps: Tensor
    ) -> dict[str, Tensor] | None:
        if not self._uses_graph_recruitment():
            return None
        core = self.actor_critic.core
        history = core.recruitment_history_from_rnn(rnn_states)
        return self._passive_recruitment_graph().update_from_rollout(
            history,
            valid_steps,
            int(self.cfg.Hippo_L),
            float(self.cfg.dg_recruitment_passive_half_life_events),
            decay_birth=not self._uses_policy_graph(),
        )

    def _hrl_graph_views(self, hrl_state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        layout = self._hrl_layout()
        if self._uses_policy_graph():
            graph = self._policy_graph()
            count = hrl_state.shape[0]
            return (
                graph.node_visits.unsqueeze(0).expand(count, -1),
                graph.tctrl.unsqueeze(0).expand(count, -1, -1),
                graph.edge_confidence.unsqueeze(0).expand(count, -1, -1),
            )
        visits = hrl_state[:, layout.visits_start : layout.visits_end]
        tctrl = hrl_state[:, layout.tctrl_start : layout.tctrl_end].view(-1, layout.n_nodes, layout.n_nodes)
        strength = hrl_state[:, layout.edge_strength_start : layout.edge_strength_end].view(
            -1, layout.n_nodes, layout.n_nodes
        )
        return visits, tctrl, strength

    def _maybe_reset_critic(self):
        if self.cfg.reset_critic:
            try:
                self.actor_critic.critic_linear.reset_parameters()
                log.debug(f"Reset Critic Parameters.")
            except AttributeError:
                log.warning(f"Failed resetting the Critic parameters in a double Critic experiment. Something's wrong!")

    def _maybe_reset_decoder(self):
        if self.cfg.reset_decoder:
            try:
                self.actor_critic.decoder.reset_parameters()
                self.actor_critic.action_parameterization.reset_parameters()
                self.actor_critic.critic_linear.reset_parameters()
                log.debug(f"Reset Decoder Parameters.")
            except AttributeError:
                log.warning(
                    f"Failed resetting the Decoder parameters in a double Critic experiment. Something's wrong!"
                )

    def _replace_checkpoint_policy_id(self, checkpoint_path, policy_id):
        return checkpoint_path
        # return re.sub(r'checkpoint_p\d+', f'checkpoint_p{policy_id}', checkpoint_path)

    def _replace_checkpoint_seed(self, checkpoint_path):
        # return checkpoint_path
        return re.sub(r"see_\d+", f"see_{self.cfg.seed}", checkpoint_path)

    def load_from_checkpoint(self, policy_id: PolicyID, load_progress: bool = True) -> None:
        """
        Docstring for load_from_checkpoint

        :param policy_id: Description
        :type policy_id: PolicyID
        :param load_progress: Description
        :type load_progress: bool
        """
        name_prefix = dict(latest="checkpoint", best="best")[self.cfg.load_checkpoint_kind]
        checkpoints = self.get_checkpoints(self.checkpoint_dir(self.cfg, policy_id), pattern=f"{name_prefix}_*")
        if (
            self.cfg.load_model_path and load_progress
        ):  # Hacky way to prevent this injection from happening every time pbt replaces a policy
            log.debug(f"Injecting custom load_model_path")
            checkpoints.append(self._replace_checkpoint_policy_id(self.cfg.load_model_path, policy_id))
        checkpoint_dict = self.load_checkpoint(checkpoints, self.device)
        if checkpoint_dict is None:
            log.debug("Did not load from checkpoint, starting from scratch!")
        else:
            log.debug("Loading model from checkpoint")
            # if we're replacing our policy with another policy (under PBT), let's not reload the env_steps
            self._load_state(checkpoint_dict, load_progress=load_progress)
            if load_progress:  # see above
                self._maybe_reset_critic()
                self._maybe_reset_decoder()

    def _calculate_sequence_core(self, rnn_state: Tensor, minibatch_size: int | tuple):
        """
        From rnn states this returns just the sequence core with all units that are part of a sequence.
        Additionally it returns L+R-1 as a value.

        :param rnn_state: Batched rnn states.
        :type rnn_state: Tensor
        :param minibatch_size: The number of minibatches (forward pass records this automatically during calculation of the head output.)
        :type minibatch_size: int | tuple
        """
        #     log.debug(f'minibatch_size: {minibatch_size}')
        R = getattr(self.cfg, "Hippo_R", 8)
        L = getattr(self.cfg, "Hippo_L", 48)
        hippo_n_feature = getattr(self.cfg, "Hippo_n_feature", 64)
        # Total length of the shift register.
        expanded_length = R + L - 1
        # Core (shift register) output dimension.
        core_output_size = hippo_n_feature * expanded_length
        return rnn_state[:, :core_output_size].view(minibatch_size, hippo_n_feature, expanded_length), expanded_length

    def _calculate_progression(self, sequence_core):
        """
        For one time step this gives a tensor where each entry is a number showing how far the corresponding indexed sequence activation progressed.
        Must be batched? Might be easily adaptable though.
        0: just activated
        L+R-1: Fading out

        :param sequence_core: The sequence core. Can be batched I think.
        """
        return torch.argmax(
            torch.cat(
                (
                    (sequence_core != 0).to(dtype=torch.int),
                    torch.ones(sequence_core.shape[:-1] + (1,), dtype=torch.int),
                ),
                dim=-1,
            ),
            dim=-1,
        ).squeeze(0)

    def _record_distance_matrix(
        self, core_outputs, minibatch_size: int, masked_matrix: bool = True, return_progression: bool = False
    ):
        """
        Calculates the distance matrix (see Janneks report) and returns a full matrix as well as a masked version for
        the distances between active sequences only.

        :param core_outputs: The core outputs as recorded from a forward pass. Must be batched? Might be easily adaptable though.
        :param minibatch_size: The number of minibatches (forward pass records this automatically during calculation of the head output.)
        :type minibatch_size: int
        :param masked_matrix: Wether the masked_matrix should be returned. If *False*: returns a null-tensor instead
        :type masked_matrix: bool
        :param return_progression: Wether the progression tensor is returned. If *False*: Only the two matrices are returned
        :type return_progression: bool
        """
        locale_verbose = False
        if getattr(self.cfg, "rec_distances", None) or getattr(self.cfg, "distance_learning", None):
            sequence_core, _ = self._calculate_sequence_core(core_outputs, minibatch_size)

            progression = self._calculate_progression(sequence_core)
            distance_matrix = torch.abs(progression.unsqueeze(-1) - progression.unsqueeze(-2)).to(dtype=torch.float)

            # sum = torch.sum(torch.sum(distance_matrix.to(dtype=torch.float),dim=-1),dim=-1)
            # value = sum/(distance_matrix.shape[1]**2)
            # meaned_value = value.mean().detach()
            if locale_verbose:
                log.info(f"RNN States shape: {core_outputs.shape}")
                log.info(f"SeququenceCore shape: {sequence_core.shape}")
                log.info(f"Progression: {progression}")
                log.info(f"Progression shape: {progression.shape}")
                log.info(f"Distance Matrix: {distance_matrix}")
                log.info(f"Distance Matrix shape: {distance_matrix.shape}")
                # log.info(f'Summed values: {value}')
                # log.info(f'Summed values shape: {value.shape}')

            if masked_matrix:
                masked_progression = torch.where(progression == sequence_core.shape[-1], False, True)
                distance_matrix_mask = torch.logical_and(
                    masked_progression.unsqueeze(-1), masked_progression.unsqueeze(-2)
                )
                masked_distance_matrix = distance_matrix * distance_matrix_mask
            else:
                masked_distance_matrix = None
        else:
            distance_matrix = None
        if return_progression:
            return distance_matrix, masked_distance_matrix, progression
        else:
            return distance_matrix, masked_distance_matrix

    def _manipulate_gradients(self):
        pass

    def _train(
        self, gpu_buffer: TensorDict, batch_size: int, experience_size: int, num_invalids: int
    ) -> Optional[AttrDict]:
        timing = self.timing
        with torch.no_grad():
            early_stopping_tolerance = 1e-6
            early_stop = False
            prev_epoch_actor_loss = 1e9
            epoch_actor_losses = [0] * self.cfg.num_batches_per_epoch

            # recent mean KL-divergences per minibatch, this used by LR schedulers
            recent_kls = []

            if self.cfg.with_vtrace:
                assert (
                    self.cfg.recurrence == self.cfg.rollout and self.cfg.recurrence > 1
                ), "V-trace requires to recurrence and rollout to be equal"

            num_sgd_steps = 0
            stats_and_summaries: Optional[AttrDict] = None

            # When it is time to record train summaries, we randomly sample epoch/batch for which the summaries are
            # collected to get equal representation from different stages of training.
            # Half the time, we record summaries from the very large step of training. There we will have the highest
            # KL-divergence and ratio of PPO-clipped samples, which makes this data even more useful for analysis.
            # Something to consider: maybe we should have these last-batch metrics in a separate summaries category?
            with_summaries = self._should_save_summaries()
            if np.random.rand() < 0.5:
                summaries_epoch = np.random.randint(0, self.cfg.num_epochs)
                summaries_batch = np.random.randint(0, self.cfg.num_batches_per_epoch)
            else:
                summaries_epoch = self.cfg.num_epochs - 1
                summaries_batch = self.cfg.num_batches_per_epoch - 1

            assert self.actor_critic.training

        for epoch in range(self.cfg.num_epochs):
            with timing.add_time("epoch_init"):
                if early_stop:
                    break

                force_summaries = False
                minibatches = self._get_minibatches(batch_size, experience_size)

            for batch_num in range(len(minibatches)):
                with torch.no_grad(), timing.add_time("minibatch_init"):
                    indices = minibatches[batch_num]

                    # current minibatch consisting of short trajectory segments with length == recurrence
                    mb = self._get_minibatch(gpu_buffer, indices)

                    # enable syntactic sugar that allows us to access dict's keys as object attributes
                    mb = AttrDict(mb)

                with timing.add_time("calculate_losses"):
                    (
                        action_distribution,
                        policy_loss,
                        exploration_loss,
                        kl_old,
                        kl_loss,
                        value_loss,
                        loss_summaries,
                    ) = self._calculate_losses(mb, num_invalids)

                with timing.add_time("losses_postprocess"):
                    # noinspection PyTypeChecker
                    actor_loss: Tensor = policy_loss + exploration_loss + kl_loss
                    critic_loss = value_loss
                    loss: Tensor = actor_loss + critic_loss

                    epoch_actor_losses[batch_num] = float(actor_loss)

                    high_loss = 30.0
                    if torch.abs(loss) > high_loss:
                        log.warning(
                            "High loss value: l:%.4f pl:%.4f vl:%.4f exp_l:%.4f kl_l:%.4f (recommended to adjust the --reward_scale parameter)",
                            to_scalar(loss),
                            to_scalar(policy_loss),
                            to_scalar(value_loss),
                            to_scalar(exploration_loss),
                            to_scalar(kl_loss),
                        )

                        # perhaps something weird is happening, we definitely want summaries from this step
                        force_summaries = True

                with torch.no_grad(), timing.add_time("kl_divergence"):
                    # if kl_old is not None it is already calculated above
                    if kl_old is None:
                        # calculate KL-divergence with the behaviour policy action distribution
                        old_action_distribution = get_action_distribution(
                            self.actor_critic.action_space,
                            mb.action_logits,
                        )
                        kl_old = action_distribution.kl_divergence(old_action_distribution)
                        kl_old = masked_select(kl_old, mb.valids, num_invalids)

                    kl_old_mean = float(kl_old.mean().item())
                    recent_kls.append(kl_old_mean)
                    if kl_old.numel() > 0 and kl_old.max().item() > 100:
                        log.warning(f"KL-divergence is very high: {kl_old.max().item():.4f}")

                # update the weights
                with timing.add_time("update"):
                    # following advice from https://youtu.be/9mS1fIYj1So set grad to None instead of optimizer.zero_grad()
                    for p in self.actor_critic.parameters():
                        p.grad = None

                    loss.backward()

                    self._manipulate_gradients()

                    if self.cfg.max_grad_norm > 0.0:
                        with timing.add_time("clip"):
                            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)

                    curr_policy_version = self.train_step  # policy version before the weight update

                    actual_lr = self.curr_lr
                    if num_invalids > 0:
                        # if we have masked (invalid) data we should reduce the learning rate accordingly
                        # this prevents a situation where most of the data in the minibatch is invalid
                        # and we end up doing SGD with super noisy gradients
                        actual_lr = self.curr_lr * (experience_size - num_invalids) / experience_size
                    self._apply_lr(actual_lr)

                    with self.param_server.policy_lock:
                        self.optimizer.step()

                    num_sgd_steps += 1

                with torch.no_grad(), timing.add_time("after_optimizer"):
                    self._after_optimizer_step()


                    if self.lr_scheduler.invoke_after_each_minibatch():
                        self.curr_lr = self.lr_scheduler.update(self.curr_lr, recent_kls)

                    # collect and report summaries
                    should_record_summaries = with_summaries
                    should_record_summaries &= epoch == summaries_epoch and batch_num == summaries_batch
                    should_record_summaries |= force_summaries
                    if should_record_summaries:
                        # hacky way to collect all of the intermediate variables for summaries
                        summary_vars = {**locals(), **loss_summaries}
                        stats_and_summaries = self._record_summaries(AttrDict(summary_vars))
                        del summary_vars
                        force_summaries = False

                    # make sure everything (such as policy weights) is committed to shared device memory
                    synchronize(self.cfg, self.device)
                    # this will force policy update on the inference worker (policy worker)
                    self.policy_versions_tensor[self.policy_id] = self.train_step

            # end of an epoch
            if self.lr_scheduler.invoke_after_each_epoch():
                self.curr_lr = self.lr_scheduler.update(self.curr_lr, recent_kls)

            new_epoch_actor_loss = float(np.mean(epoch_actor_losses))
            loss_delta_abs = abs(prev_epoch_actor_loss - new_epoch_actor_loss)
            if loss_delta_abs < early_stopping_tolerance:
                early_stop = True
                log.debug(
                    "Early stopping after %d epochs (%d sgd steps), loss delta %.7f",
                    epoch + 1,
                    num_sgd_steps,
                    loss_delta_abs,
                )
                break

            prev_epoch_actor_loss = new_epoch_actor_loss

        return stats_and_summaries

    def _record_summaries(self, train_loop_vars):
        var = train_loop_vars  # TODO: Think of a better way, why is this necessary? Just redirecting pointer?
        stats = super()._record_summaries(train_loop_vars)
        if var.additional_stats["Distance Matrix"] != None:
            summed = torch.sum(torch.sum(var.additional_stats["Distance Matrix"].to(dtype=torch.float), dim=-1), dim=-1)
            value = summed / (var.additional_stats["Distance Matrix"].shape[1] ** 2)
            meaned_value, stded_value = torch.std_mean(value)
            stats.distance_metric = meaned_value.detach()
            stats.distance_metric_max = value.max().detach()
            stats.distance_metric_min = value.min().detach()
            stats.distance_metric_std = stded_value.detach()

            summed_masked = torch.sum(
                torch.sum(var.additional_stats["Distance Matrix Masked"].to(dtype=torch.float), dim=-1), dim=-1
            )
            value_masked = summed_masked / (var.additional_stats["Distance Matrix Masked"].shape[1] ** 2)
            meaned_value_masked, stded_value_masked = torch.std_mean(value_masked)
            stats.distance_metric_masked = meaned_value_masked.detach()
            stats.distance_metric_masked_max = value_masked.max().detach()
            stats.distance_metric_masked_min = value_masked.min().detach()
            stats.distance_metric_masked_std = stded_value_masked.detach()

            activated_sequences = var.additional_stats["Head Output"].count_nonzero(dim=-1).to(dtype=torch.float)
            meaned_activated_sequences, stded_activated_sequences = torch.std_mean(activated_sequences)
            stats.activated_sequences = meaned_activated_sequences.detach()
            stats.activated_sequences_max = activated_sequences.max().detach()
            stats.activated_sequences_min = activated_sequences.min().detach()
            stats.activated_sequences_std = stded_activated_sequences.detach()
        return stats

    """def train(self, batch: TensorDict) -> Optional[Dict]:
        with self.timing.add_time("misc"):
            self._maybe_update_cfg()
            self._maybe_load_policy()

        with self.timing.add_time("prepare_batch"):
            buff, experience_size, num_invalids = self._prepare_batch(batch)

        if num_invalids >= experience_size:
            if self.cfg.with_pbt:
                log.warning("No valid samples in the batch, with PBT this must mean we just replaced weights")
            else:
                log.error(f"Learner {self.policy_id=} received an entire batch of invalid data, skipping...")
            return None
        else:
            with self.timing.add_time("train"):
                train_stats = self._train(buff, self.cfg.batch_size, experience_size, num_invalids)

            # multiply the number of samples by frameskip so that FPS metrics reflect the number
            # of environment steps actually simulated
            if self.cfg.summaries_use_frameskip:
                self.env_steps += experience_size * self.env_info.frameskip
            else:
                self.env_steps += experience_size

            stats = {LEARNER_ENV_STEPS: self.env_steps, POLICY_ID_KEY: self.policy_id}
            if train_stats is not None:
                if train_stats is not None: #?
                    stats[TRAIN_STATS] = train_stats
                    stats[EPISODIC]["distance_metric] = train_stats["distance_metric"]
                stats[STATS_KEY] = memory_stats("learner", self.device)

            return stats"""  # Assigning stats[EPISODIC] here does not seem to work and you will not be able to stop the program anymore by keyboard interupt


class DistanceLearnerSimple(BaseDistanceRecorder):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        BaseLearner.__init__(self, cfg, env_info, policy_versions_tensor, policy_id, param_server)

    def _calculate_losses(
        self, mb: AttrDict, num_invalids: int
    ) -> Tuple[ActionDistribution, Tensor, Tensor | float, Optional[Tensor], Tensor | float, Tensor, Dict]:
        additional_stats = AttrDict()
        with torch.no_grad(), self.timing.add_time("losses_init"):
            recurrence: int = self.cfg.recurrence

            # PPO clipping
            clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
            # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
            clip_ratio_low = 1.0 / clip_ratio_high
            clip_value = self.cfg.ppo_clip_value

            valids = mb.valids

        outputs = self._forward_pass(
            mb=mb,
            recurrence=recurrence,
            valids=valids,
            return_outputs=[True, True, True],
        )

        additional_stats["Head Output"] = outputs.head_outputs[:, : getattr(self.cfg, "Hippo_n_feature", 64)]
        with self.timing.add_time("post_forward"):
            action_distribution = self.actor_critic.action_distribution()
            log_prob_actions = action_distribution.log_prob(mb.actions)
            ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

            # super large/small values can cause numerical problems and are probably noise anyway
            ratio = torch.clamp(ratio, 0.05, 20.0)

            values = outputs.result["values"].squeeze()

        # these computations are not the part of the computation graph
        with torch.no_grad(), self.timing.add_time("advantages_returns"):
            with self.timing.add_time("Distance Matrix"):
                distance_matrix, masked_distance_matrix = self._record_distance_matrix(
                    outputs.core_outputs, minibatch_size=outputs.minibatch_size, masked_matrix=True
                )
                additional_stats["Distance Matrix"] = distance_matrix
                additional_stats["Distance Matrix Masked"] = masked_distance_matrix

            if self.cfg.masked_distance_matrix:
                adv = -torch.sum(torch.sum(masked_distance_matrix.to(dtype=torch.float), dim=-1), dim=-1)
            else:
                adv = -torch.sum(torch.sum(distance_matrix.to(dtype=torch.float), dim=-1), dim=-1)

            adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
            if self.cfg.normalize_advantage:
                adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)  # normalize advantage
            # log.info(f'Advantage Shape: {adv.shape}')

        with self.timing.add_time("losses"):
            # noinspection PyTypeChecker
            policy_loss = self._policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids)
            l1_loss = self._l1_loss(outputs.head_outputs)

            policy_loss += l1_loss

            exploration_loss = self.exploration_loss_func(action_distribution, valids, num_invalids)
            kl_old, kl_loss = self.kl_loss_func(
                self.actor_critic.action_space, mb.action_logits, action_distribution, valids, num_invalids
            )
            old_values = mb["values"]
            # value_loss = self._value_loss(values, old_values, targets, clip_value, valids, num_invalids)
            value_loss = torch.zeros(1)

        loss_summaries = dict(
            ratio=ratio,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            values=outputs.result["values"],
            adv=adv,
            adv_std=adv_std,
            adv_mean=adv_mean,
            additional_stats=additional_stats,
        )
        del outputs

        return action_distribution, policy_loss, exploration_loss, kl_old, kl_loss, value_loss, loss_summaries


class DistanceLearnerEncoderDecoderSeparate(BaseDistanceRecorder):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        BaseLearner.__init__(self, cfg, env_info, policy_versions_tensor, policy_id, param_server)

    @staticmethod
    def _make_grad_flip_hook(name):
        def _grad_flip_hook(module, grad_output: Tensor) -> Tensor:
            # log.debug(f"Flipping gradients on module {name}")
            return tuple(-g if g is not None else None for g in grad_output)

        return _grad_flip_hook

    def _register_backward_hooks(self):
        self.actor_critic.encoder.DG_projection.register_full_backward_pre_hook(
            DistanceLearnerEncoderDecoderSeparate._make_grad_flip_hook("encoder")
        )
        log.info("Succesfully registered backward hooks.")

    def _calculate_losses(
        self, mb: AttrDict, num_invalids: int
    ) -> Tuple[ActionDistribution, Tensor, Tensor | float, Optional[Tensor], Tensor | float, Tensor, Dict]:
        additional_stats = AttrDict()

        # for param_group in self.optimizer.param_groups:
        # log.info(f'Parameter Group: {self.optimizer.param_groups[-1]}')

        with torch.no_grad(), self.timing.add_time("losses_init"):
            recurrence: int = self.cfg.recurrence

            # PPO clipping
            clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
            # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
            clip_ratio_low = 1.0 / clip_ratio_high
            clip_value = self.cfg.ppo_clip_value

            valids = mb.valids

        outputs = self._forward_pass(
            mb=mb,
            recurrence=recurrence,
            valids=valids,
            # grad_context=grad_context,
            return_outputs=[True, True, True],
        )

        additional_stats["Head Output"] = outputs.head_outputs[:, : getattr(self.cfg, "Hippo_n_feature", 64)]
        with self.timing.add_time("post_forward"):
            action_distribution = self.actor_critic.action_distribution()
            log_prob_actions = action_distribution.log_prob(mb.actions)
            ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

            # super large/small values can cause numerical problems and are probably noise anyway
            ratio = torch.clamp(ratio, 0.05, 20.0)

            values = outputs.result["values"].squeeze()

        # these computations are not the part of the computation graph
        with torch.no_grad(), self.timing.add_time("advantages_returns"):
            with self.timing.add_time("Distance Matrix"):
                distance_matrix, masked_distance_matrix = self._record_distance_matrix(
                    outputs.core_outputs, minibatch_size=outputs.minibatch_size, masked_matrix=True
                )
                additional_stats["Distance Matrix"] = distance_matrix
                additional_stats["Distance Matrix Masked"] = masked_distance_matrix

            if self.cfg.masked_distance_matrix:
                adv = -torch.sum(torch.sum(masked_distance_matrix.to(dtype=torch.float), dim=-1), dim=-1)
            else:
                adv = -torch.sum(torch.sum(distance_matrix.to(dtype=torch.float), dim=-1), dim=-1)

            adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
            if self.cfg.normalize_advantage:
                adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)  # normalize advantage
            # log.info(f'Advantage Shape: {adv.shape}')

        with self.timing.add_time("losses"):
            # noinspection PyTypeChecker
            policy_loss = self._policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids)
            l1_loss = self._l1_loss(outputs.head_outputs)

            policy_loss += l1_loss

            exploration_loss = self.exploration_loss_func(action_distribution, valids, num_invalids)
            kl_old, kl_loss = self.kl_loss_func(
                self.actor_critic.action_space, mb.action_logits, action_distribution, valids, num_invalids
            )
            old_values = mb["values"]
            # value_loss = self._value_loss(values, old_values, targets, clip_value, valids, num_invalids)
            value_loss = torch.zeros(1)

        loss_summaries = dict(
            ratio=ratio,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            values=outputs.result["values"],
            adv=adv,
            adv_std=adv_std,
            adv_mean=adv_mean,
            additional_stats=additional_stats,
        )
        del outputs

        return action_distribution, policy_loss, exploration_loss, kl_old, kl_loss, value_loss, loss_summaries


class DistanceLearnerCombined(BaseDistanceRecorder):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        BaseLearner.__init__(self, cfg, env_info, policy_versions_tensor, policy_id, param_server)

    def _calculate_losses(
        self, mb: AttrDict, num_invalids: int
    ) -> Tuple[ActionDistribution, Tensor, Tensor | float, Optional[Tensor], Tensor | float, Tensor, Dict]:
        additional_stats = AttrDict()
        with torch.no_grad(), self.timing.add_time("losses_init"):
            recurrence: int = self.cfg.recurrence

            # PPO clipping
            clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
            # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
            clip_ratio_low = 1.0 / clip_ratio_high
            clip_value = self.cfg.ppo_clip_value

            valids = mb.valids

        outputs = self._forward_pass(mb=mb, recurrence=recurrence, valids=valids, return_outputs=[True, True, True])

        additional_stats["Head Output"] = outputs.head_outputs[:, : getattr(self.cfg, "Hippo_n_feature", 64)]

        with self.timing.add_time("post_forward"):
            action_distribution = self.actor_critic.action_distribution()
            log_prob_actions = action_distribution.log_prob(mb.actions)
            ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

            # super large/small values can cause numerical problems and are probably noise anyway
            ratio = torch.clamp(ratio, 0.05, 20.0)

            values = outputs.result["values"].squeeze()

        # these computations are not the part of the computation graph
        with torch.no_grad(), self.timing.add_time("advantages_returns"):
            with self.timing.add_time("Distance Matrix"):
                distance_matrix, masked_distance_matrix = self._record_distance_matrix(
                    outputs.core_outputs, minibatch_size=outputs.minibatch_size, masked_matrix=True
                )
                additional_stats["Distance Matrix"] = distance_matrix
                additional_stats["Distance Matrix Masked"] = masked_distance_matrix
            if self.cfg.with_vtrace:
                # V-trace parameters
                rho_hat = torch.Tensor([self.cfg.vtrace_rho])
                c_hat = torch.Tensor([self.cfg.vtrace_c])

                ratios_cpu = ratio.cpu()
                values_cpu = values.cpu()
                rewards_cpu = mb.rewards_cpu
                dones_cpu = mb.dones_cpu

                vtrace_rho = torch.min(rho_hat, ratios_cpu)
                vtrace_c = torch.min(c_hat, ratios_cpu)

                vs = torch.zeros((outputs.num_trajectories * recurrence))
                adv = torch.zeros((outputs.num_trajectories * recurrence))

                next_values = values_cpu[recurrence - 1 :: recurrence] - rewards_cpu[recurrence - 1 :: recurrence]
                next_values /= self.cfg.gamma
                next_vs = next_values

                for i in reversed(range(self.cfg.recurrence)):
                    rewards = rewards_cpu[i::recurrence]
                    dones = dones_cpu[i::recurrence]
                    not_done = 1.0 - dones
                    not_done_gamma = not_done * self.cfg.gamma

                    curr_values = values_cpu[i::recurrence]
                    curr_vtrace_rho = vtrace_rho[i::recurrence]
                    curr_vtrace_c = vtrace_c[i::recurrence]

                    delta_s = curr_vtrace_rho * (rewards + not_done_gamma * next_values - curr_values)
                    adv[i::recurrence] = curr_vtrace_rho * (rewards + not_done_gamma * next_vs - curr_values)
                    next_vs = curr_values + delta_s + not_done_gamma * curr_vtrace_c * (next_vs - next_values)
                    vs[i::recurrence] = next_vs

                    next_values = curr_values

                targets = vs.to(self.device)
                adv = adv.to(self.device)
            else:
                # using regular GAE
                adv = mb.advantages
                targets = mb.returns

            if self.cfg.masked_distance_matrix:
                advA = -torch.sum(torch.sum(masked_distance_matrix.to(dtype=torch.float), dim=-1), dim=-1)
            else:
                advA = -torch.sum(torch.sum(distance_matrix.to(dtype=torch.float), dim=-1), dim=-1)

            adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
            advA_std, advA_mean = torch.std_mean(masked_select(advA, valids, num_invalids))
            if self.cfg.normalize_advantage:
                adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)  # normalize advantage
                advA = (advA - advA_mean) / torch.clamp_min(advA_std, 1e-7)  # normalize advantage

            adv += advA

            # log.info(f'Advantage Shape: {adv.shape}')

        with self.timing.add_time("losses"):
            # noinspection PyTypeChecker
            policy_loss = self._policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids)
            l1_loss = self._l1_loss(outputs.head_outputs)

            policy_loss += l1_loss

            exploration_loss = self.exploration_loss_func(action_distribution, valids, num_invalids)
            kl_old, kl_loss = self.kl_loss_func(
                self.actor_critic.action_space, mb.action_logits, action_distribution, valids, num_invalids
            )
            old_values = mb["values"]
            value_loss = self._value_loss(values, old_values, targets, clip_value, valids, num_invalids)

        loss_summaries = dict(
            ratio=ratio,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            values=outputs.result["values"],
            adv=adv,
            adv_std=adv_std,
            adv_mean=adv_mean,
            additional_stats=additional_stats,
        )
        del outputs

        return action_distribution, policy_loss, exploration_loss, kl_old, kl_loss, value_loss, loss_summaries


class DistanceRecorder(BaseDistanceRecorder):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        BaseLearner.__init__(self, cfg, env_info, policy_versions_tensor, policy_id, param_server)

    def _calculate_losses(
        self, mb: AttrDict, num_invalids: int
    ) -> Tuple[ActionDistribution, Tensor, Tensor | float, Optional[Tensor], Tensor | float, Tensor, Dict]:
        additional_stats = AttrDict()
        with torch.no_grad(), self.timing.add_time("losses_init"):
            recurrence: int = self.cfg.recurrence

            # PPO clipping
            clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
            # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
            clip_ratio_low = 1.0 / clip_ratio_high
            clip_value = self.cfg.ppo_clip_value

            valids = mb.valids

        outputs = self._forward_pass(mb=mb, recurrence=recurrence, valids=valids, return_outputs=[True, True, True])

        additional_stats["Head Output"] = outputs.head_outputs[:, : getattr(self.cfg, "Hippo_n_feature", 64)]

        with self.timing.add_time("post_forward"):
            action_distribution = self.actor_critic.action_distribution()
            log_prob_actions = action_distribution.log_prob(mb.actions)
            ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

            # super large/small values can cause numerical problems and are probably noise anyway
            ratio = torch.clamp(ratio, 0.05, 20.0)

            values = outputs.result["values"].squeeze()

        # these computations are not the part of the computation graph
        with torch.no_grad(), self.timing.add_time("advantages_returns"):
            with self.timing.add_time("Distance Matrix"):
                distance_matrix, masked_distance_matrix = self._record_distance_matrix(
                    outputs.core_outputs, minibatch_size=outputs.minibatch_size, masked_matrix=True
                )
                additional_stats["Distance Matrix"] = distance_matrix
                additional_stats["Distance Matrix Masked"] = masked_distance_matrix
            if self.cfg.with_vtrace:
                # V-trace parameters
                rho_hat = torch.Tensor([self.cfg.vtrace_rho])
                c_hat = torch.Tensor([self.cfg.vtrace_c])

                ratios_cpu = ratio.cpu()
                values_cpu = values.cpu()
                rewards_cpu = mb.rewards_cpu
                dones_cpu = mb.dones_cpu

                vtrace_rho = torch.min(rho_hat, ratios_cpu)
                vtrace_c = torch.min(c_hat, ratios_cpu)

                vs = torch.zeros((outputs.num_trajectories * recurrence))
                adv = torch.zeros((outputs.num_trajectories * recurrence))

                next_values = values_cpu[recurrence - 1 :: recurrence] - rewards_cpu[recurrence - 1 :: recurrence]
                next_values /= self.cfg.gamma
                next_vs = next_values

                for i in reversed(range(self.cfg.recurrence)):
                    rewards = rewards_cpu[i::recurrence]
                    dones = dones_cpu[i::recurrence]
                    not_done = 1.0 - dones
                    not_done_gamma = not_done * self.cfg.gamma

                    curr_values = values_cpu[i::recurrence]
                    curr_vtrace_rho = vtrace_rho[i::recurrence]
                    curr_vtrace_c = vtrace_c[i::recurrence]

                    delta_s = curr_vtrace_rho * (rewards + not_done_gamma * next_values - curr_values)
                    adv[i::recurrence] = curr_vtrace_rho * (rewards + not_done_gamma * next_vs - curr_values)
                    next_vs = curr_values + delta_s + not_done_gamma * curr_vtrace_c * (next_vs - next_values)
                    vs[i::recurrence] = next_vs

                    next_values = curr_values

                targets = vs.to(self.device)
                adv = adv.to(self.device)
            else:
                # using regular GAE
                adv = mb.advantages
                targets = mb.returns

            adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
            if self.cfg.normalize_advantage:
                adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)  # normalize advantage
            # log.info(f'Advantage Shape: {adv.shape}')

        with self.timing.add_time("losses"):
            # noinspection PyTypeChecker
            policy_loss = self._policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids)
            l1_loss = self._l1_loss(outputs.head_outputs, valids, num_invalids)

            # policy_loss += l1_loss

            exploration_loss = self.exploration_loss_func(action_distribution, valids, num_invalids)
            kl_old, kl_loss = self.kl_loss_func(
                self.actor_critic.action_space, mb.action_logits, action_distribution, valids, num_invalids
            )
            old_values = mb["values"]
            value_loss = self._value_loss(values, old_values, targets, clip_value, valids, num_invalids)

        loss_summaries = dict(
            ratio=ratio,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            values=outputs.result["values"],
            adv=adv,
            adv_std=adv_std,
            adv_mean=adv_mean,
            additional_stats=additional_stats,
        )
        del outputs

        return action_distribution, policy_loss, exploration_loss, kl_old, kl_loss, value_loss, loss_summaries


class DistanceLearnerMaster(BaseDistanceRecorder):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        BaseLearner.__init__(self, cfg, env_info, policy_versions_tensor, policy_id, param_server)

    @staticmethod
    def make_grad_flip_hook(
        name,
    ):  # name useful for debugging only. This wrapper preserves the variable for the actual hook
        def grad_flip_hook(module, grad_output: Tensor) -> Tensor:  # full_backward_pre_hook needs these inputs.
            # log.debug(f"Flipping gradients on module {name}")
            return tuple(-g if g is not None else None for g in grad_output)

        return grad_flip_hook

    def _register_backward_hooks(self):
        if self.cfg.encoder_decoder_share_losses:
            pass
        else:
            self.actor_critic.encoder.DG_projection.register_full_backward_pre_hook(
                DistanceLearnerMaster.make_grad_flip_hook("encoder.DG_projection")
            )
            log.info("Succesfully registered backward hooks.")

    def _calculate_losses(
        self, mb: AttrDict, num_invalids: int
    ) -> Tuple[ActionDistribution, Tensor, Tensor | float, Optional[Tensor], Tensor | float, Tensor, Dict]:
        additional_stats = AttrDict()
        with torch.no_grad(), self.timing.add_time("losses_init"):
            recurrence: int = self.cfg.recurrence

            # PPO clipping
            clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
            # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
            clip_ratio_low = 1.0 / clip_ratio_high
            clip_value = self.cfg.ppo_clip_value

            valids = mb.valids

        outputs = self._forward_pass(
            mb=mb,
            recurrence=recurrence,
            valids=valids,
            # grad_context=[True,True,True],
            return_outputs=[True, True, True],
        )

        additional_stats["Head Output"] = outputs.head_outputs[:, : getattr(self.cfg, "Hippo_n_feature", 64)]

        with self.timing.add_time("post_forward"):
            action_distribution = self.actor_critic.action_distribution()
            log_prob_actions = action_distribution.log_prob(mb.actions)
            ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

            # super large/small values can cause numerical problems and are probably noise anyway
            ratio = torch.clamp(ratio, 0.05, 20.0)

            values = outputs.result["values"].squeeze()

        # these computations are not the part of the computation graph
        with torch.no_grad(), self.timing.add_time("advantages_returns"):
            with self.timing.add_time("Distance Matrix"):
                distance_matrix, masked_distance_matrix = self._record_distance_matrix(
                    outputs.core_outputs, minibatch_size=outputs.minibatch_size, masked_matrix=True
                )
                additional_stats["Distance Matrix"] = distance_matrix
                additional_stats["Distance Matrix Masked"] = masked_distance_matrix

            if self._use_external_reward_for_advantage():
                if self.cfg.with_vtrace:
                    # V-trace parameters
                    rho_hat = torch.Tensor([self.cfg.vtrace_rho])
                    c_hat = torch.Tensor([self.cfg.vtrace_c])

                    ratios_cpu = ratio.cpu()
                    values_cpu = values.cpu()
                    rewards_cpu = mb.rewards_cpu
                    dones_cpu = mb.dones_cpu

                    vtrace_rho = torch.min(rho_hat, ratios_cpu)
                    vtrace_c = torch.min(c_hat, ratios_cpu)

                    vs = torch.zeros((outputs.num_trajectories * recurrence))
                    adv = torch.zeros((outputs.num_trajectories * recurrence))

                    next_values = values_cpu[recurrence - 1 :: recurrence] - rewards_cpu[recurrence - 1 :: recurrence]
                    next_values /= self.cfg.gamma
                    next_vs = next_values

                    for i in reversed(range(self.cfg.recurrence)):
                        rewards = rewards_cpu[i::recurrence]
                        dones = dones_cpu[i::recurrence]
                        not_done = 1.0 - dones
                        not_done_gamma = not_done * self.cfg.gamma

                        curr_values = values_cpu[i::recurrence]
                        curr_vtrace_rho = vtrace_rho[i::recurrence]
                        curr_vtrace_c = vtrace_c[i::recurrence]

                        delta_s = curr_vtrace_rho * (rewards + not_done_gamma * next_values - curr_values)
                        adv[i::recurrence] = curr_vtrace_rho * (rewards + not_done_gamma * next_vs - curr_values)
                        next_vs = curr_values + delta_s + not_done_gamma * curr_vtrace_c * (next_vs - next_values)
                        vs[i::recurrence] = next_vs

                        next_values = curr_values

                    targets = vs.to(self.device)
                    adv = adv.to(self.device)
                else:
                    # using regular GAE
                    adv = mb.advantages
                    targets = mb.returns
            else:
                # Could this cause problems down the line?
                adv = torch.zeros(outputs.minibatch_size)
                # targets = torch.zeros(1)
            if self._use_internal_reward_for_advantage():
                if self.cfg.metric == "minimum":
                    metric = -torch.sum(torch.min(masked_distance_matrix.to(dtype=torch.float), dim=-1).values, dim=-1)
                elif self.cfg.metric == "masked_sum":
                    metric = -torch.sum(torch.sum(masked_distance_matrix.to(dtype=torch.float), dim=-1), dim=-1)
                elif self.cfg.metric == "sum":
                    metric = -torch.sum(torch.sum(distance_matrix.to(dtype=torch.float), dim=-1), dim=-1)
                else:
                    raise NotImplementedError()
                adv += metric

            adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
            if self.cfg.normalize_advantage:
                adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)  # normalize advantage
            # log.info(f'Advantage Shape: {adv.shape}')

        with self.timing.add_time("losses"):
            # noinspection PyTypeChecker
            policy_loss = self._policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids)
            l1_loss = self._l1_loss(outputs.head_outputs)

            policy_loss += l1_loss

            exploration_loss = self.exploration_loss_func(action_distribution, valids, num_invalids)
            kl_old, kl_loss = self.kl_loss_func(
                self.actor_critic.action_space, mb.action_logits, action_distribution, valids, num_invalids
            )
            old_values = mb["values"]
            if self._use_external_reward_for_advantage():
                value_loss = self._value_loss(values, old_values, targets, clip_value, valids, num_invalids)
            else:
                value_loss = torch.zeros(1)

        loss_summaries = dict(
            ratio=ratio,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            values=outputs.result["values"],
            adv=adv,
            adv_std=adv_std,
            adv_mean=adv_mean,
            additional_stats=additional_stats,
        )
        del outputs

        return action_distribution, policy_loss, exploration_loss, kl_old, kl_loss, value_loss, loss_summaries


class DoubleDistanceLearnerReward(BaseDistanceRecorder):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        BaseLearner.__init__(self, cfg, env_info, policy_versions_tensor, policy_id, param_server)

    def flip_module_grads(self, module: torch.nn.Module):
        """
        Multiply the .grad of every Parameter belonging to *module*
        by -1, in‑place.
        """
        # log.warn(f'Flipping gradients on module {module}')
        for p in module.parameters():
            if p.grad is not None:
                # log.debug(p.grad)
                p.grad.detach_()  # detach from graph – we only need the tensor
                p.grad.mul_(-self.cfg.encoder_grad_coeff)  # in‑place negation
                # log.debug(p.grad)

    def _manipulate_gradients(self):
        # flip the gradients of the Encoder
        if self._use_internal_reward_for_advantage():
            self.flip_module_grads(self.actor_critic.encoder.DG_projection)

    def _extra_encoder_loss(self, head_outputs, rnn_states, progression, minibatch_size):
        straight_through = straight_through_binary(head_outputs)
        # log.debug(f'Straight_Through: {straight_through}')
        sequence_core, _ = self._calculate_sequence_core(rnn_states, minibatch_size)
        # log.info(f'Shapes: {head_outputs.shape}, {sequence_core.shape}')
        # Punishment for multi-activation
        mask_new_activations = progression == 0
        penalty_mask = mask_new_activations & (mask_new_activations.sum(dim=1) > 1).unsqueeze(1)
        penalty_mask = F.pad(
            penalty_mask, pad=(0, head_outputs.shape[-1] - self.cfg.Hippo_n_feature), mode="constant", value=0
        )
        loss_penalty = -(straight_through * penalty_mask).sum() / (penalty_mask.sum() + 1e-6)
        # Reward for new activations of not used sequences
        mask_active_now = sequence_core != 0
        reward_mask = mask_new_activations & (mask_active_now.sum(dim=2) == self.cfg.Hippo_R)
        reward_mask = F.pad(
            reward_mask, pad=(0, head_outputs.shape[-1] - self.cfg.Hippo_n_feature), mode="constant", value=0
        )
        loss_reward = (straight_through * reward_mask).sum() / (reward_mask.sum() + 1e-6)
        # Reward for not used sequences in this mini batch
        batch_mask = mask_active_now.sum(dim=2) > 0
        batch_mask = torch.logical_not(torch.any(batch_mask, dim=0))
        batch_mask = F.pad(
            batch_mask, pad=(0, head_outputs.shape[-1] - self.cfg.Hippo_n_feature), mode="constant", value=0
        ).unsqueeze(0)
        batch_penalty = (straight_through * batch_mask).sum() / (batch_mask.sum() + 1e-6)
        log.info(f"ADDITIONAL LOSSES: {loss_penalty.item()}; {loss_reward.item()}; {batch_penalty.item()}")
        return loss_penalty, loss_reward, batch_penalty

    def _calculate_losses(
        self, mb: AttrDict, num_invalids: int
    ) -> Tuple[ActionDistribution, Tensor, Tensor | float, Optional[Tensor], Tensor | float, Tensor, Dict]:
        additional_stats = AttrDict()
        with torch.no_grad(), self.timing.add_time("losses_init"):
            recurrence: int = self.cfg.recurrence

            # PPO clipping
            clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
            # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
            clip_ratio_low = 1.0 / clip_ratio_high
            clip_value = self.cfg.ppo_clip_value

            valids = mb.valids

        outputs = self._forward_pass(
            mb=mb,
            recurrence=recurrence,
            valids=valids,
            # grad_context=[True,True,True],
            return_outputs=[True, True, True],
        )

        additional_stats["Head Output"] = outputs.head_outputs[:, : getattr(self.cfg, "Hippo_n_feature", 64)]

        with self.timing.add_time("post_forward"):
            action_distribution = self.actor_critic.action_distribution()
            log_prob_actions = action_distribution.log_prob(mb.actions)
            ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

            # super large/small values can cause numerical problems and are probably noise anyway
            ratio = torch.clamp(ratio, 0.05, 20.0)

            values_external = outputs.result["values_external"].squeeze()
            values_internal = outputs.result["values_internal"].squeeze()

        # these computations are not the part of the computation graph
        with torch.no_grad(), self.timing.add_time("advantages_returns"):
            with self.timing.add_time("Distance Matrix"):
                distance_matrix, masked_distance_matrix, progression = self._record_distance_matrix(
                    outputs.core_outputs,
                    minibatch_size=outputs.minibatch_size,
                    masked_matrix=True,
                    return_progression=True,
                )
                additional_stats["Distance Matrix"] = distance_matrix
                additional_stats["Distance Matrix Masked"] = masked_distance_matrix

            # using regular GAE
            adv = mb.advantages
            targets_external = mb.returns_external
            targets_internal = mb.returns_internal

            adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
            if self.cfg.normalize_advantage:
                adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)  # normalize advantage
            # log.info(f'Advantage Shape: {adv.shape}')

        with self.timing.add_time("losses"):
            # noinspection PyTypeChecker
            policy_loss = self._policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids)
            l1_loss = self._l1_loss(outputs.head_outputs)

            encoder_penalty_loss, encoder_reward_loss, encoder_batch_loss = self._extra_encoder_loss(
                outputs.head_outputs, mb["rnn_states"].clone(), progression, outputs.minibatch_size
            )

            additional_stats["intrinsic_rewards"] = mb["rewards"]
            additional_stats["encoder_penalty_loss"] = encoder_penalty_loss
            additional_stats["encoder_reward_loss"] = encoder_reward_loss
            additional_stats["batch_penalty_loss"] = encoder_batch_loss

            encoder_losses = l1_loss + encoder_reward_loss + encoder_penalty_loss + encoder_batch_loss

            policy_loss += encoder_losses

            exploration_loss = self.exploration_loss_func(action_distribution, valids, num_invalids)
            kl_old, kl_loss = self.kl_loss_func(
                self.actor_critic.action_space, mb.action_logits, action_distribution, valids, num_invalids
            )
            old_values_external = mb["values_external"]
            old_values_internal = mb["values_internal"]
            value_loss_external = self._value_loss(
                values_external, old_values_external, targets_external, clip_value, valids, num_invalids
            )
            value_loss_internal = self._value_loss(
                values_internal, old_values_internal, targets_internal, clip_value, valids, num_invalids
            )
            additional_stats["value_loss_internal"] = value_loss_internal
            additional_stats["value_loss_external"] = value_loss_external
            value_loss = value_loss_external + value_loss_internal

        loss_summaries = dict(
            ratio=ratio,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            values=torch.zeros_like(values_external),
            values_external=outputs.result["values_external"],
            values_internal=outputs.result["values_internal"],
            adv=adv,
            adv_std=adv_std,
            adv_mean=adv_mean,
            additional_stats=additional_stats,
        )
        del outputs

        return action_distribution, policy_loss, exploration_loss, kl_old, kl_loss, value_loss, loss_summaries

    def _prepare_batch(self, batch: TensorDict) -> Tuple[TensorDict, int, int]:
        with torch.no_grad():
            # create a shallow copy so we can modify the dictionary
            # we still reference the same buffers though
            buff = shallow_recursive_copy(batch)

            # ignore experience from other agents (i.e. on episode boundary) and from inactive agents
            valids: Tensor = buff["policy_id"] == self.policy_id
            # ignore experience that was older than the threshold even before training started
            curr_policy_version: int = self.train_step
            buff["valids"][:, :-1] = valids & (curr_policy_version - buff["policy_version"] < self.cfg.max_policy_lag)
            # for last T+1 step, we want to use the validity of the previous step
            buff["valids"][:, -1] = buff["valids"][:, -2]
            # log.info(f'RNN_states Shape: {buff["rnn_states"].shape}')
            # log.info(f'Internal Reward1: {buff["rewards"][:,:10]}')
            # log.info(f'Reward Shape1: {buff["rewards"].shape}')
            # Calculate Internal Reward

            if True:  # self.cfg.replace_reward:#False: #
                rnn_state_shape = buff["rnn_states"].shape
                dataset_size = rnn_state_shape[0] * rnn_state_shape[1]
                distance_matrix, _, progression = self._record_distance_matrix(
                    buff["rnn_states"].clone().reshape((dataset_size,) + tuple(rnn_state_shape[2:])),
                    dataset_size,
                    masked_matrix=False,
                    return_progression=True,
                )
                dataset_idx, row_idx = torch.where(progression == 0)
                # Only use first (row, col) pair for each row_idx
                lookup = {}
                for r, c in zip(dataset_idx.tolist(), row_idx.tolist()):
                    if r not in lookup:
                        lookup[r] = c
                # Prepare result tensor, filled with fallback value
                baseline = progression.shape[-1]
                internal_reward = torch.full((dataset_size, 1), baseline, dtype=torch.int32)

                # Fill in values where match was found
                # If all sequences got activated have a fallback_value
                fallback_value = torch.tensor(2 * baseline)
                for r, c in lookup.items():
                    vec = distance_matrix[r, c]
                    vec_mask = vec != 0
                    internal_reward[r] = vec[vec_mask].min() if vec_mask.any() else fallback_value
                    # log.info(f'Adjusting internal reward at position {r,c} to be the minimum of {distance_matrix[r, c]}')
                buff["rewards_external"] = buff["rewards"].clone()
                buff["rewards_internal"] = (
                    -internal_reward.view(*rnn_state_shape[:2])[:, 1:] + baseline
                ) * self.cfg.reward_scale
                # log.info(f'Internal Reward2: {buff["rewards"][:,:10]}')
                # log.info(f'Reward Shape2: {buff["rewards"].shape}')
                del (
                    lookup,
                    rnn_state_shape,
                    distance_matrix,
                    progression,
                    dataset_idx,
                    row_idx,
                    baseline,
                    internal_reward,
                    fallback_value,
                )

            # ensure we're in train mode so that normalization statistics are updated
            if not self.actor_critic.training:
                self.actor_critic.train()

            buff["normalized_obs"] = self._prepare_and_normalize_obs(buff["obs"])
            del buff["obs"]  # don't need non-normalized obs anymore

            # calculate estimated value for the next step (T+1)
            normalized_last_obs = buff["normalized_obs"][:, -1]
            last_values = self.actor_critic(normalized_last_obs, buff["rnn_states"][:, -1], values_only=True)
            next_values_external = last_values["values_external"]
            next_values_internal = last_values["values_internal"]
            buff["values_external"][:, -1] = next_values_external
            buff["values_internal"][:, -1] = next_values_internal

            if self.cfg.normalize_returns:
                # Since our value targets are normalized, the values will also have normalized statistics.
                # We need to denormalize them before using them for GAE caculation and value bootstrapping.
                # rl_games PPO uses a similar approach, see:
                # https://github.com/Denys88/rl_games/blob/7b5f9500ee65ae0832a7d8613b019c333ecd932c/rl_games/algos_torch/models.py#L51
                denormalized_values_external = buff[
                    "values_external"
                ].clone()  # need to clone since normalizer is in-place
                denormalized_values_internal = buff[
                    "values_internal"
                ].clone()  # need to clone since normalizer is in-place
                self.actor_critic.returns_normalizer(denormalized_values_external, denormalize=True)
                self.actor_critic.returns_normalizer(denormalized_values_internal, denormalize=True)
            else:
                # values are not normalized in this case, so we can use them as is
                denormalized_values_external = buff["values_external"]
                denormalized_values_internal = buff["values_internal"]

            if self.cfg.value_bootstrap:
                # Value bootstrapping is a technique that reduces the surprise for the critic in case
                # we're ending the episode by timeout. Intuitively, in this case the cumulative return for the last step
                # should not be zero, but rather what the critic expects. This improves learning in many envs
                # because otherwise the critic cannot predict the abrupt change in rewards in a timed-out episode.
                # What we really want here is v(t+1) which we don't have because we don't have obs(t+1) (since
                # the episode ended). Using v(t) is an approximation that requires that rew(t) can be generally ignored.

                # Multiply by both time_out and done flags to make sure we count only timeouts in terminal states.
                # There was a bug in older versions of isaacgym where timeouts were reported for non-terminal states.
                buff["rewards_external"].add_(
                    self.cfg.gamma * denormalized_values_external[:, :-1] * buff["time_outs"] * buff["dones"]
                )
                buff["rewards_internal"].add_(
                    self.cfg.gamma * denormalized_values_internal[:, :-1] * buff["time_outs"] * buff["dones"]
                )

            if not self.cfg.with_vtrace:
                # calculate advantage estimate (in case of V-trace it is done separately for each minibatch)
                advantages_external = gae_advantages(
                    buff["rewards_external"],
                    buff["dones"],
                    denormalized_values_external,
                    buff["valids"],
                    self.cfg.gamma,
                    self.cfg.gae_lambda,
                )
                # here returns are not normalized yet, so we should use denormalized values
                buff["returns_external"] = (
                    advantages_external + buff["valids"][:, :-1] * denormalized_values_external[:, :-1]
                )
                advantages_internal = gae_advantages(
                    buff["rewards_internal"],
                    buff["dones"],
                    denormalized_values_internal,
                    buff["valids"],
                    self.cfg.gamma,
                    self.cfg.gae_lambda,
                )
                # here returns are not normalized yet, so we should use denormalized values
                buff["returns_internal"] = (
                    advantages_internal + buff["valids"][:, :-1] * denormalized_values_internal[:, :-1]
                )

                if self._use_external_reward_for_advantage():
                    buff["advantages"] = advantages_external
                elif self._use_internal_reward_for_advantage():
                    buff["advantages"] = advantages_internal
                else:
                    log.error(f"Both use_internal and use_external are set to FALSE")
                    raise NotImplementedError
            # remove next step obs, rnn_states, and values from the batch, we don't need them anymore
            for key in ["normalized_obs", "rnn_states", "values_external", "values_internal", "valids"]:
                buff[key] = buff[key][:, :-1]

            dataset_size = buff["actions"].shape[0] * buff["actions"].shape[1]
            for d, k, v in iterate_recursively(buff):
                # collapse first two dimensions (batch and time) into a single dimension
                d[k] = v.reshape((dataset_size,) + tuple(v.shape[2:]))

            buff["dones_cpu"] = buff["dones"].to("cpu", copy=True, dtype=torch.float, non_blocking=True)
            buff["rewards_cpu"] = buff["rewards"].to("cpu", copy=True, dtype=torch.float, non_blocking=True)

            # return normalization parameters are only used on the learner, no need to lock the mutex
            if self.cfg.normalize_returns:
                self.actor_critic.returns_normalizer(buff["returns_external"])  # in-place
                self.actor_critic.returns_normalizer(buff["returns_internal"])  # in-place

            num_invalids = dataset_size - buff["valids"].sum().item()
            if num_invalids > 0:
                invalid_fraction = num_invalids / dataset_size
                if invalid_fraction > 0.5:
                    log.warning(f"{self.policy_id=} batch has {invalid_fraction:.2%} of invalid samples")

                # invalid action values can cause problems when we calculate logprobs
                # here we set them to 0 just to be safe
                invalid_indices = (buff["valids"] == 0).nonzero().squeeze()
                buff["actions"][invalid_indices] = 0
                # likewise, some invalid values of log_prob_actions can cause NaNs or infs
                buff["log_prob_actions"][invalid_indices] = -1  # -1 seems like a safe value

            return buff, dataset_size, num_invalids

    def _record_summaries(self, train_loop_vars):
        var = train_loop_vars  # TODO: Think of a better way, why is this necessary? Just redirecting pointer?
        stats = super()._record_summaries(train_loop_vars)

        stats.intrinsic_rewards = var.additional_stats["intrinsic_rewards"].mean().detach().float()
        stats.encoder_penalty_loss = var.additional_stats["encoder_penalty_loss"].detach().float()
        stats.encoder_reward_loss = var.additional_stats["encoder_reward_loss"].detach().float()
        stats.batch_penalty_loss = var.additional_stats["batch_penalty_loss"].detach().float()

        stats.value_external = var.values_external.mean()
        stats.value_internal = var.values_internal.mean()

        stats.value_loss_external = var.additional_stats["value_loss_external"].detach()
        stats.value_loss_internal = var.additional_stats["value_loss_internal"].detach()

        return stats


class DistanceLearnerReward(BaseDistanceRecorder):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        BaseLearner.__init__(self, cfg, env_info, policy_versions_tensor, policy_id, param_server)
        self._last_recruitment_stats = dict(
            candidate_count=0.0,
            silent_endpoint_count=0.0,
            recruited_count=0.0,
            residual_norm=0.0,
            connected_fraction=0.0,
            isolated_fraction=0.0,
            redundant_pair_count=0.0,
            eligible_vertex_count=0.0,
            birth_protected_count=0.0,
            repeat_assignment_count=0.0,
            isolated_assignment_count=0.0,
            redundant_assignment_count=0.0,
            passive_graph_density=0.0,
            passive_update_count=0.0,
            passive_stale_count=0.0,
            passive_over_gap_count=0.0,
            attempt_coverage_fraction=0.0,
            fully_tested_count=0.0,
            zero_outdegree_count=0.0,
            untested_zero_outdegree_count=0.0,
            bad_source_count=0.0,
            reliable_out_degree_mean=0.0,
            reliable_outgoing_confidence_mean=0.0,
            predictive_event_count=0.0,
            predictive_context_group_count=0.0,
            predictive_eligible_count=0.0,
            predictive_reliability_gap=0.0,
            bad_source_assignment_count=0.0,
            predictive_assignment_count=0.0,
            active_endpoint_count=0.0,
            activity_blocked_count=0.0,
            eligible_victim_endpoint_count=0.0,
            residual_pass_count=0.0,
            residual_reject_count=0.0,
            endpoint_active_unit_count=0.0,
            victim_active_count=0.0,
            predictive_decayed_attempt_mass=0.0,
            predictive_supported_context_count=0.0,
            predictive_invalidation_mass=0.0,
            predictive_context_coverage_fraction=0.0,
            eligible_bad_source_endpoint_count=0.0,
            eligible_redundant_endpoint_count=0.0,
            eligible_predictive_endpoint_count=0.0,
            victim_active_bad_source_count=0.0,
            victim_active_redundant_count=0.0,
            victim_active_predictive_count=0.0,
            victim_active_bad_source_fraction=0.0,
            victim_active_redundant_fraction=0.0,
            victim_active_predictive_fraction=0.0,
            goal_adapter_reset_count=0.0,
            forced_preflight_assignment_count=0.0,
        )
        self._last_graph_rollout_stats: dict[str, float] = {}
        self._last_passive_recruitment_stats: dict[str, float] = {}
        self._current_predictive_eligibility = None
        self._last_predictive_update_stats: dict[str, float] = {}
        self._last_encoder_credit_stats: dict[str, float] = {}
        self._last_stale_generation_stats = dict(
            rejected_count=0.0,
            rejected_fraction=0.0,
            dropped_rollouts=0.0,
            dropped_decisions=0.0,
            update_deferred=0.0,
        )
        self._generation_dropped_rollouts_total = 0.0
        self._generation_dropped_decisions_total = 0.0
        self._generation_deferred_updates_total = 0.0
        self._goal_adapter_reset_count = 0.0
        self._last_empirical_her_stats = dict(
            accepted_segments=0.0,
            skipped_no_endpoint=0.0,
            skipped_same_source=0.0,
            segment_length=0.0,
            positive_fraction=0.0,
            terminal_reward=0.0,
        )

    def _queue_dg_recruitment_candidates(self, buff: TensorDict) -> None:
        if not bool(getattr(self.cfg, "dg_orthogonal_recruitment", False)):
            return
        projection = self.actor_critic.encoder.DG_projection
        required = ("linear", "batchnorm1d", "recruitment_committed", "recruitment_activation_counts")
        if any(not hasattr(projection, name) for name in required):
            raise RuntimeError("Orthogonal DG recruitment requires trainable batchnorm_relu DG projection")

        states = buff["rnn_states"][:, :-1]
        valids = buff["valids"][:, :-1].bool()
        candidate, source, _ = dg_recruitment_candidate_mask(
            states,
            int(self.cfg.Hippo_n_feature),
            int(self.cfg.Hippo_R),
            int(self.cfg.Hippo_L),
            valids,
        )
        expanded_length = int(self.cfg.Hippo_R) + int(self.cfg.Hippo_L) - 1
        ca3 = states[..., : int(self.cfg.Hippo_n_feature) * expanded_length].view(
            *states.shape[:-1], int(self.cfg.Hippo_n_feature), expanded_length
        )
        observed = (ca3[..., 0] > 0) & valids.unsqueeze(-1)
        projection.recruitment_activation_counts.add_(observed.float().sum(dim=(0, 1)))
        buff["dg_recruit_candidate"] = candidate
        buff["dg_recruit_source"] = source

        self._current_predictive_eligibility = None
        if self._uses_graph_recruitment() and self._uses_policy_graph():
            all_states = buff["rnn_states"]
            all_ca3 = all_states[..., : int(self.cfg.Hippo_n_feature) * expanded_length].view(
                *all_states.shape[:-1], int(self.cfg.Hippo_n_feature), expanded_length
            )
            option_states = self._hrl_state_from_rnn(all_states)
            pred_source, pred_target, pred_context, pred_success = batch_predictive_events(
                option_states,
                all_ca3,
                valids,
                int(self.cfg.Hippo_n_feature),
                int(self.cfg.Hippo_R),
            )
            evidence = self._predictive_recruitment_evidence()
            update_stats = evidence.update(
                pred_source,
                pred_target,
                pred_context,
                pred_success,
                float(getattr(self.cfg, "dg_recruitment_pred_half_life_options", 5000.0)),
            )
            self._last_predictive_update_stats = {
                key: float(value.detach().cpu().item()) for key, value in update_stats.items()
            }
            self._current_predictive_eligibility = evidence.eligibility(
                self._passive_recruitment_graph().birth_support,
                float(getattr(self.cfg, "dg_recruitment_pred_min_context_attempts", 2.0)),
                float(getattr(self.cfg, "hrl_edge_reliability_threshold", 0.5)),
                float(self.cfg.dg_recruitment_connectivity_threshold),
            )

    @staticmethod
    def _reset_optimizer_row(optimizer: torch.optim.Optimizer, parameter: Tensor, row: int) -> None:
        state = optimizer.state.get(parameter, {})
        for value in state.values():
            if torch.is_tensor(value) and value.shape == parameter.shape:
                value[row].zero_()

    @torch.no_grad()
    def _reset_goal_adapter_row(self, row: int) -> None:
        target_modulation = getattr(self.actor_critic.decoder, "target_modulation", None)
        if target_modulation is None:
            raise RuntimeError("goal-adapter reset requires target-ID FiLM conditioning")
        target_modulation[int(row)].zero_()
        self._reset_optimizer_row(self.optimizer, target_modulation, int(row))
        self._goal_adapter_reset_count += 1.0

    def _graph_recruitment_eligibility(self):
        passive_graph = self._passive_recruitment_graph()
        if self._uses_policy_graph():
            policy_graph = self._policy_graph()
            confidence, elapsed = policy_graph.edge_confidence, policy_graph.tctrl
        else:
            confidence, elapsed = passive_graph.confidence, passive_graph.elapsed
        return graph_recruitment_eligibility(
            confidence,
            elapsed,
            passive_graph.birth_support,
            float(self.cfg.dg_recruitment_connectivity_threshold),
            int(self.cfg.dg_recruitment_redundancy_max_steps),
        )

    def _directional_recruitment_eligibility(self):
        policy_graph = self._policy_graph()
        return directional_recruitment_eligibility(
            policy_graph.edge_confidence,
            policy_graph.control_attempts,
            policy_graph.tctrl,
            self._passive_recruitment_graph().birth_support,
            float(self.cfg.hrl_edge_confidence_threshold),
            float(getattr(self.cfg, "hrl_edge_reliability_threshold", 0.5)),
            float(getattr(self.cfg, "dg_recruitment_attempt_threshold", 0.5)),
            float(self.cfg.dg_recruitment_connectivity_threshold),
            int(self.cfg.dg_recruitment_redundancy_max_steps),
        )

    def _selected_recruitment_eligibility(self, predictive_eligibility):
        rule = getattr(self.cfg, "dg_recruitment_victim_rule", "incident")
        if rule == "monitor":
            return None
        if rule == "directional":
            return self._directional_recruitment_eligibility()
        if rule == "predictive":
            return predictive_eligibility
        return self._graph_recruitment_eligibility()

    def _record_graph_recruitment_telemetry(self, predictive_eligibility=None) -> None:
        legacy = self._graph_recruitment_eligibility()
        n_nodes = legacy.adjacency.size(0)
        connected = ~legacy.isolated
        denominator = max(1, n_nodes * (n_nodes - 1))
        birth_support = self._passive_recruitment_graph().birth_support
        threshold = float(self.cfg.dg_recruitment_connectivity_threshold)
        directional = self._directional_recruitment_eligibility() if self._uses_policy_graph() else None
        if directional is not None:
            connected = directional.adjacency.any(dim=0) | directional.adjacency.any(dim=1)
        selected = self._selected_recruitment_eligibility(predictive_eligibility)
        self._last_recruitment_stats.update(
            connected_fraction=float(connected.float().mean().item()),
            isolated_fraction=float(legacy.isolated.float().mean().item()),
            redundant_pair_count=float(
                directional.redundant_pair_count if directional is not None else legacy.redundant_pair_count
            ),
            eligible_vertex_count=float(selected.eligible.sum().item()) if selected is not None else 0.0,
            birth_protected_count=float((birth_support > threshold).sum().item()),
            passive_graph_density=float(
                (self._passive_recruitment_graph().confidence > threshold).fill_diagonal_(False).sum().item()
                / denominator
            ),
        )
        if directional is not None:
            covered = self._policy_graph().control_attempts >= float(
                getattr(self.cfg, "dg_recruitment_attempt_threshold", 0.5)
            )
            covered = covered.clone()
            covered.fill_diagonal_(False)
            self._last_recruitment_stats.update(
                attempt_coverage_fraction=float(covered.sum().item() / denominator),
                fully_tested_count=float(directional.fully_tested.sum().item()),
                zero_outdegree_count=float(directional.zero_outdegree.sum().item()),
                untested_zero_outdegree_count=float(
                    (directional.zero_outdegree & ~directional.fully_tested).sum().item()
                ),
                bad_source_count=float(directional.bad_source.sum().item()),
                reliable_out_degree_mean=float(directional.out_degree.float().mean().item()),
                reliable_outgoing_confidence_mean=float(directional.outgoing_confidence.mean().item()),
            )
        if predictive_eligibility is not None:
            pred_update = self._last_predictive_update_stats
            pred_evidence = self._predictive_recruitment_evidence()
            pred_supported = pred_evidence.attempts >= float(
                getattr(self.cfg, "dg_recruitment_pred_min_context_attempts", 2.0)
            )
            possible_groups = max(1, pred_evidence.n_nodes * (pred_evidence.n_nodes - 1) ** 2)
            self._last_recruitment_stats.update(
                predictive_event_count=float(pred_update.get("accepted_count", 0.0)),
                predictive_context_group_count=float(predictive_eligibility.context_group_count),
                predictive_eligible_count=float(predictive_eligibility.eligible.sum().item()),
                predictive_reliability_gap=float(predictive_eligibility.reliability_gap.max().item()),
                predictive_decayed_attempt_mass=float(pred_update.get("decayed_attempt_mass", 0.0)),
                predictive_supported_context_count=float(pred_supported.sum().item()),
                predictive_context_coverage_fraction=float(pred_supported.sum().item() / possible_groups),
            )

    @torch.no_grad()
    def _apply_dg_recruitment(self, buff: TensorDict) -> None:
        predictive_eligibility = self._current_predictive_eligibility
        self._current_predictive_eligibility = None
        self._last_recruitment_stats = dict(
            candidate_count=0.0,
            silent_endpoint_count=0.0,
            recruited_count=0.0,
            residual_norm=0.0,
            connected_fraction=0.0,
            isolated_fraction=0.0,
            redundant_pair_count=0.0,
            eligible_vertex_count=0.0,
            birth_protected_count=0.0,
            repeat_assignment_count=0.0,
            isolated_assignment_count=0.0,
            redundant_assignment_count=0.0,
            passive_graph_density=0.0,
            passive_update_count=0.0,
            passive_stale_count=0.0,
            passive_over_gap_count=0.0,
            attempt_coverage_fraction=0.0,
            fully_tested_count=0.0,
            zero_outdegree_count=0.0,
            untested_zero_outdegree_count=0.0,
            bad_source_count=0.0,
            reliable_out_degree_mean=0.0,
            reliable_outgoing_confidence_mean=0.0,
            predictive_event_count=0.0,
            predictive_context_group_count=0.0,
            predictive_eligible_count=0.0,
            predictive_reliability_gap=0.0,
            bad_source_assignment_count=0.0,
            predictive_assignment_count=0.0,
            active_endpoint_count=0.0,
            activity_blocked_count=0.0,
            eligible_victim_endpoint_count=0.0,
            residual_pass_count=0.0,
            residual_reject_count=0.0,
            endpoint_active_unit_count=0.0,
            victim_active_count=0.0,
            predictive_decayed_attempt_mass=0.0,
            predictive_supported_context_count=0.0,
            predictive_invalidation_mass=0.0,
            predictive_context_coverage_fraction=0.0,
            eligible_bad_source_endpoint_count=0.0,
            eligible_redundant_endpoint_count=0.0,
            eligible_predictive_endpoint_count=0.0,
            victim_active_bad_source_count=0.0,
            victim_active_redundant_count=0.0,
            victim_active_predictive_count=0.0,
            victim_active_bad_source_fraction=0.0,
            victim_active_redundant_fraction=0.0,
            victim_active_predictive_fraction=0.0,
            goal_adapter_reset_count=0.0,
            forced_preflight_assignment_count=0.0,
        )
        if not bool(getattr(self.cfg, "dg_orthogonal_recruitment", False)):
            return

        if self._uses_graph_recruitment():
            passive_stats = self._last_passive_recruitment_stats
            self._last_recruitment_stats["passive_update_count"] = passive_stats.get("accepted_count", 0.0)
            self._last_recruitment_stats["passive_stale_count"] = passive_stats.get("stale_count", 0.0)
            self._last_recruitment_stats["passive_over_gap_count"] = passive_stats.get("over_gap_count", 0.0)
            self._record_graph_recruitment_telemetry(predictive_eligibility)

        encoder = self.actor_critic.encoder
        projection = encoder.DG_projection
        forced_row = int(getattr(self.cfg, "dg_recruitment_forced_preflight_row", -1))
        forced_preflight = forced_row >= 0
        forced_selection = None
        if forced_preflight:
            forced_selection = forced_recruitment_preflight_selection(
                buff["valids"],
                forced_row,
                int(self.cfg.Hippo_n_feature),
                self.train_step,
                int(getattr(self.cfg, "dg_recruitment_forced_preflight_after_updates", 4)),
                int(projection.recruitment_count.item()),
            )
            if forced_selection is None:
                return

        candidate = buff.get("dg_recruit_candidate", None)
        if candidate is None:
            return
        if forced_preflight:
            candidate = buff["valids"].bool()
        else:
            candidate = candidate.bool() & buff["valids"].bool()
        candidate_indices = torch.nonzero(candidate, as_tuple=False).flatten()
        if forced_preflight:
            candidate_indices = candidate_indices.new_tensor([forced_selection[0]])
        self._last_recruitment_stats["candidate_count"] = float(candidate_indices.numel())
        if candidate_indices.numel() == 0:
            return

        was_training = encoder.training
        encoder.eval()
        try:
            features = encoder.projection_input(buff["normalized_obs"][candidate_indices]).detach()
        finally:
            encoder.train(was_training)

        weight = projection.linear.weight
        batchnorm = projection.batchnorm1d
        projection_features = (
            projection.centered_projection_input(features)
            if hasattr(projection, "centered_projection_input")
            else features
        )
        raw = F.linear(projection_features, weight)
        normalized = (raw - batchnorm.running_mean) / torch.sqrt(batchnorm.running_var + batchnorm.eps)
        activity = torch.relu(normalized - float(projection.intercept))
        max_recruits = max(0, int(self.cfg.dg_orthogonal_recruitment_max_per_rollout))
        endpoint_gate = getattr(self.cfg, "dg_recruitment_endpoint_gate", "silent")
        residual_norms = []

        with self.param_server.policy_lock:
            for feature, active in zip(projection_features, activity):
                active_units = active > 0
                projection.recruitment_activation_counts.add_(active_units.float())
                active_count = float(active_units.sum().item())
                self._last_recruitment_stats["endpoint_active_unit_count"] += active_count
                if active_units.any():
                    self._last_recruitment_stats["active_endpoint_count"] += 1.0
                else:
                    self._last_recruitment_stats["silent_endpoint_count"] += 1.0

                assignment_reason = None
                if forced_preflight:
                    row = forced_selection[1]
                    assignment_reason = "forced_preflight"
                elif self._uses_graph_recruitment():
                    eligibility = self._selected_recruitment_eligibility(predictive_eligibility)
                    if eligibility is None or eligibility.victim is None:
                        continue
                    row = eligibility.victim
                    assignment_reason = eligibility.reason
                else:
                    available = torch.nonzero(~projection.recruitment_committed, as_tuple=False).flatten()
                    if available.numel() == 0:
                        continue
                    available_counts = projection.recruitment_activation_counts[available]
                    row = int(available[torch.argmin(available_counts)].item())
                self._last_recruitment_stats["eligible_victim_endpoint_count"] += 1.0
                eligible_reason_key = {
                    "bad_source": "eligible_bad_source_endpoint_count",
                    "redundant": "eligible_redundant_endpoint_count",
                    "predictive": "eligible_predictive_endpoint_count",
                }.get(assignment_reason)
                if eligible_reason_key is not None:
                    self._last_recruitment_stats[eligible_reason_key] += 1.0
                if bool(active_units[row]):
                    self._last_recruitment_stats["victim_active_count"] += 1.0
                    active_reason_key = {
                        "bad_source": "victim_active_bad_source_count",
                        "redundant": "victim_active_redundant_count",
                        "predictive": "victim_active_predictive_count",
                    }.get(assignment_reason)
                    if active_reason_key is not None:
                        self._last_recruitment_stats[active_reason_key] += 1.0
                existing_rows = torch.cat((weight[:row], weight[row + 1 :]), dim=0)
                residual = orthogonal_feature_residual(
                    feature,
                    existing_rows,
                    float(self.cfg.dg_orthogonal_recruitment_residual_eps),
                )
                if residual is None:
                    projection.recruitment_tiny_residual_count.add_(1)
                    self._last_recruitment_stats["residual_reject_count"] += 1.0
                    continue
                self._last_recruitment_stats["residual_pass_count"] += 1.0
                if not retirement_endpoint_allowed(endpoint_gate, active_units):
                    self._last_recruitment_stats["activity_blocked_count"] += 1.0
                    continue
                if self._last_recruitment_stats["recruited_count"] >= max_recruits:
                    continue

                weight[row].copy_(residual)
                self._reset_optimizer_row(self.optimizer, weight, row)
                reference_var = torch.cat((batchnorm.running_var[:row], batchnorm.running_var[row + 1 :]))
                variance = reference_var.median() if reference_var.numel() else weight.new_tensor(1.0)
                variance = variance.clamp_min(batchnorm.eps)
                new_raw = torch.dot(weight[row], feature)
                target_z = float(projection.intercept) + float(self.cfg.dg_orthogonal_recruitment_margin)
                batchnorm.running_var[row].copy_(variance)
                batchnorm.running_mean[row].copy_(new_raw - target_z * torch.sqrt(variance + batchnorm.eps))
                mark_replacement = getattr(projection, "mark_replacement_committed", None)
                if mark_replacement is not None:
                    mark_replacement()
                was_repeat = bool(projection.recruitment_row_counts[row].item() > 0)
                projection.recruitment_committed[row] = True
                projection.recruitment_activation_counts[row] += 1.0
                projection.recruitment_row_counts[row] += 1
                projection.recruitment_count.add_(1)
                if was_repeat:
                    projection.recruitment_repeat_count.add_(1)
                    self._last_recruitment_stats["repeat_assignment_count"] += 1.0
                self._invalidate_recruited_node(row)
                if bool(getattr(self.cfg, "dg_recruitment_reset_goal_adapter", False)):
                    self._reset_goal_adapter_row(row)
                    self._last_recruitment_stats["goal_adapter_reset_count"] += 1.0
                if assignment_reason == "isolated":
                    self._last_recruitment_stats["isolated_assignment_count"] += 1.0
                elif assignment_reason == "bad_source":
                    self._last_recruitment_stats["bad_source_assignment_count"] += 1.0
                elif assignment_reason == "redundant":
                    self._last_recruitment_stats["redundant_assignment_count"] += 1.0
                elif assignment_reason == "predictive":
                    self._last_recruitment_stats["predictive_assignment_count"] += 1.0
                elif assignment_reason == "forced_preflight":
                    self._last_recruitment_stats["forced_preflight_assignment_count"] += 1.0
                residual_norms.append(float(residual.norm().item()))
                self._last_recruitment_stats["recruited_count"] += 1.0

            if self._uses_graph_recruitment():
                self._record_graph_recruitment_telemetry(predictive_eligibility)

        if residual_norms:
            self._last_recruitment_stats["residual_norm"] = float(np.mean(residual_norms))

    def _iterative_phase(self) -> str:
        schedule = IterativeUpdateSchedule(
            enabled=bool(getattr(self.cfg, "iterative_update", False)),
            initial_encoder_steps=int(getattr(self.cfg, "iterative_initial_encoder_steps", 128)),
            decoder_steps=int(getattr(self.cfg, "iterative_decoder_steps", 512)),
            encoder_steps=int(getattr(self.cfg, "iterative_encoder_steps", 128)),
            start_phase=getattr(self.cfg, "iterative_start_phase", DECODER),
        )
        return schedule.phase(self.train_step)

    def _register_forward_hooks(self):
        return super()._register_forward_hooks()

    def _extra_encoder_loss(
        self,
        head_outputs,
        rnn_states,
        dominant_activation_mask,
        non_dominant_activation_mask,
        minibatch_size,
        valids,
        num_invalids,
    ):
        straight_through = head_outputs  # straight_through_binary(head_outputs)
        # log.debug(f'Straight_Through: {straight_through}')
        sequence_core, _ = self._calculate_sequence_core(rnn_states, minibatch_size)
        mask_active_now = sequence_core != 0
        if self.cfg.encoder_multi_activation_loss:
            penalty_mask = non_dominant_activation_mask.bool()
            penalty_mask = F.pad(
                penalty_mask, pad=(0, head_outputs.shape[-1] - self.cfg.Hippo_n_feature), mode="constant", value=0
            )
            loss_penalty = (straight_through * penalty_mask).sum(dim=1)
            loss_penalty = masked_select(loss_penalty, valids, num_invalids).mean(dim=0)
        else:
            loss_penalty = 0
        # Optional reward for a dominant event whose own CA3 chain was empty
        # immediately before this action.
        if self.cfg.encoder_unused_sequence_loss:
            reward_mask = dominant_activation_mask.bool() & (mask_active_now.sum(dim=2) == 0)
            reward_mask = F.pad(
                reward_mask, pad=(0, head_outputs.shape[-1] - self.cfg.Hippo_n_feature), mode="constant", value=0
            )
            loss_reward = (straight_through * reward_mask).sum(dim=1)  # .sum() / (reward_mask.sum() + 1e-6)
            loss_reward = -masked_select(loss_reward, valids, num_invalids).mean(dim=0)
        else:
            loss_reward = 0
        # Reward for not used sequences in this mini batch
        if self.cfg.encoder_batch_loss:
            projection = self.actor_critic.encoder.DG_projection
            pre_threshold = getattr(projection, "last_pre_threshold_logits", None)
            if pre_threshold is None:
                raise RuntimeError("encoder_batch_loss requires DG pre-threshold logits")
            batch_penalty, unused_mask = dg_unused_batch_recruitment_loss(
                pre_threshold[:, : self.cfg.Hippo_n_feature],
                head_outputs[:, : self.cfg.Hippo_n_feature],
                mask_active_now.sum(dim=2) > 0,
                valids,
                float(projection.intercept),
                float(self.cfg.encoder_batch_loss_temperature),
            )
        # log.info(f'ADDITIONAL LOSSES: {loss_penalty.item()}; {loss_reward.item()}; {batch_penalty.item()}')
        else:
            batch_penalty = 0
            unused_mask = torch.zeros(self.cfg.Hippo_n_feature, device=head_outputs.device, dtype=torch.bool)
        return loss_penalty, loss_reward, batch_penalty, unused_mask.float().sum()

    def _population_usage_loss(self, head_outputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if not self.cfg.encoder_population_usage_loss:
            zero = head_outputs.sum() * 0.0
            return zero, zero, zero, zero

        dg = head_outputs[:, : self.cfg.Hippo_n_feature].clamp_min(0.0)
        eps = torch.finfo(dg.dtype).eps
        usage = dg.mean(dim=0) + eps
        usage_prob = usage / usage.sum()
        usage_loss = (usage_prob * torch.log(usage_prob * self.cfg.Hippo_n_feature)).sum()
        density_loss = (dg.mean() - float(self.cfg.encoder_target_density)).square()
        collision_loss = torch.relu(dg.sum(dim=-1) - 1.0).square().mean()
        total = (
            float(self.cfg.encoder_usage_loss_coeff) * usage_loss
            + float(self.cfg.encoder_density_loss_coeff) * density_loss
            + float(self.cfg.encoder_collision_loss_coeff) * collision_loss
        )
        return total, usage_loss, density_loss, collision_loss

    def _anti_collapse_losses(
        self,
        head_outputs: Tensor,
        rnn_states: Tensor,
        dominant_activation_mask: Tensor,
        valids: Tensor,
        num_invalids: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        projection = self.actor_critic.encoder.DG_projection
        global_coeff = float(getattr(self.cfg, "dg_global_punishment_coeff", 0.0))
        row_coeff = float(getattr(self.cfg, "dg_row_repulsion_coeff", 0.0))
        temporal_coeff = float(getattr(self.cfg, "dg_ca3_temporal_exclusion_coeff", 0.0))
        scatter_coeff = float(getattr(self.cfg, "dg_path_scatter_coeff", 0.0))
        zero = head_outputs.sum() * 0.0

        pre_threshold = getattr(projection, "last_pre_threshold_logits", None)
        if pre_threshold is not None:
            pre_threshold_mean = pre_threshold.mean()
            pre_threshold_above = (pre_threshold > float(projection.intercept)).float().mean()
        else:
            pre_threshold_mean = zero.detach()
            pre_threshold_above = zero.detach()

        if global_coeff:
            if pre_threshold is None:
                raise RuntimeError("Global DG punishment requires batchnorm_relu pre-threshold logits")
            global_loss = dg_global_punishment_loss(
                pre_threshold,
                float(projection.intercept),
                float(self.cfg.dg_global_punishment_temperature),
                global_coeff,
                valids,
                num_invalids,
            )
        else:
            global_loss = zero

        if row_coeff and not hasattr(projection, "linear"):
            raise RuntimeError("DG row repulsion requires a projection with linear weights")
        row_loss = dg_row_repulsion_loss(projection.linear.weight, row_coeff) if row_coeff else zero
        if temporal_coeff and self.cfg.encoder_reward_method != "encourage":
            raise RuntimeError(
                "DG CA3 temporal exclusion is calibrated as a margin on encoder_reward_method=encourage"
            )
        (
            temporal_loss,
            conflict_fraction,
            conflicting_activation_fraction,
            conflict_activity,
        ) = dg_ca3_temporal_exclusion_loss(
            head_outputs,
            rnn_states,
            dominant_activation_mask,
            int(self.cfg.Hippo_n_feature),
            int(self.cfg.Hippo_R),
            int(self.cfg.Hippo_L),
            temporal_coeff,
            float(self.cfg.reward_scale),
            valids,
            num_invalids,
        )
        if scatter_coeff:
            if not self._uses_topological_manager() or not getattr(self.cfg, "hrl_action_path_integration", False):
                raise RuntimeError("DG path scatter loss requires topological HRL with action path integration")
            if pre_threshold is None:
                raise RuntimeError("DG path scatter loss requires batchnorm_relu pre-threshold logits")
            scatter_loss, scatter_conflict_fraction = dg_path_scatter_loss(
                pre_threshold,
                rnn_states,
                head_outputs[:, -ACTION_FEATURE_SIZE:],
                self._hrl_state_offset() + hrl_option_state_size(self._hrl_layout().n_nodes),
                self._hrl_layout().n_nodes,
                float(projection.intercept),
                scatter_coeff,
                float(getattr(self.cfg, "dg_path_scatter_min_displacement", 8.0)),
                float(getattr(self.cfg, "dg_path_scatter_min_straightness", 0.8)),
                float(getattr(self.cfg, "dg_path_scatter_temperature", 0.5)),
                valids,
            )
        else:
            scatter_loss = zero
            scatter_conflict_fraction = zero.detach()
        diagnostics = torch.stack(
            (
                pre_threshold_mean,
                pre_threshold_above,
                conflict_fraction,
                conflicting_activation_fraction,
                conflict_activity,
                scatter_conflict_fraction,
            )
        )
        return global_loss, row_loss, temporal_loss, scatter_loss, diagnostics

    def _ca3_predictor_loss(self, core_outputs: Tensor, head_outputs: Tensor, mb: AttrDict, recurrence: int):
        core = self.actor_critic.core
        if not self.cfg.ca3_predictor_shadow or getattr(core, "ca3_predictor", None) is None:
            zero = core_outputs.sum() * 0.0
            return zero, zero.detach(), zero.detach(), zero.detach()

        n_sequences = core_outputs.size(0) // recurrence
        ca3_size = core.core_output_size
        n_targets = self.cfg.Hippo_n_feature
        sequence_outputs = core_outputs.view(n_sequences, recurrence, -1)
        sequence_heads = head_outputs.view(n_sequences, recurrence, -1)
        targets = sequence_outputs[..., -n_targets:].detach()
        ca3 = sequence_outputs[..., :ca3_size].detach()
        dg = sequence_heads[..., :n_targets].detach()
        dones = mb.dones.view(n_sequences, recurrence)

        hit, hit_time, valid = future_target_labels(
            targets,
            dg,
            dones,
            int(self.cfg.ca3_predictor_horizon),
        )
        prediction = core.predict_target(ca3.reshape(-1, ca3_size), targets.reshape(-1, n_targets))
        hit_logit = prediction[:, 0].view_as(hit)
        time_prediction = torch.sigmoid(prediction[:, 1]).view_as(hit_time)

        valid_count = valid.sum().clamp_min(1)
        positives = (hit * valid).sum()
        negatives = ((1.0 - hit) * valid).sum()
        pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)
        hit_loss_raw = F.binary_cross_entropy_with_logits(
            hit_logit,
            hit,
            pos_weight=pos_weight,
            reduction="none",
        )
        hit_loss = (hit_loss_raw * valid).sum() / valid_count

        positive = (hit > 0) & valid
        normalized_time = hit_time / max(float(self.cfg.ca3_predictor_horizon), 1.0)
        if positive.any():
            time_loss = F.smooth_l1_loss(time_prediction[positive], normalized_time[positive])
            time_mae = (time_prediction[positive] - normalized_time[positive]).abs().mean()
        else:
            time_loss = time_prediction.sum() * 0.0
            time_mae = time_loss.detach()

        loss = hit_loss + time_loss
        with torch.no_grad():
            hit_accuracy = ((hit_logit >= 0) == (hit > 0)).logical_and(valid).sum() / valid_count
            positive_fraction = positives / valid_count
        return loss, hit_accuracy, time_mae, positive_fraction

    def _encoder_loss(
        self, head_outputs: Tensor, rewards: Tensor, dominant_activation_mask, valids, num_invalids
    ) -> Tensor:
        reward_by_row = rewards.unsqueeze(1) if rewards.ndim == 1 else rewards
        encoder_loss = (
            reward_by_row
            * head_outputs[:, : getattr(self.cfg, "Hippo_n_feature", 64)]
            * dominant_activation_mask
        ).sum(
            dim=1
        )  # / (rewards.sum() + 1e-6)
        return -masked_select(encoder_loss, valids, num_invalids).mean(dim=0)

    def _extra_decoder_loss(
        self, ratio, dominant_activation_mask, clip_ratio_low, clip_ratio_high, valids, num_invalids
    ):
        if self.cfg.extra_decoder_loss:
            # FIXME: the historical positive ratio term was added to a minimized
            # loss and therefore suppressed event actions. Repair and validate a
            # properly signed objective before this option is re-enabled.
            raise RuntimeError("extra_decoder_loss is disabled pending a corrected, validated objective")
        else:
            return 0

    def _calculate_losses(
        self, mb: AttrDict, num_invalids: int, iterative_phase: str
    ) -> Tuple[ActionDistribution, Tensor, Tensor | float, Optional[Tensor], Tensor | float, Tensor, Dict]:
        additional_stats = AttrDict()
        with torch.no_grad(), self.timing.add_time("losses_init"):
            recurrence: int = self.cfg.recurrence

            # PPO clipping
            clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
            # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
            clip_ratio_low = 1.0 / clip_ratio_high
            clip_value = self.cfg.ppo_clip_value

            valids = mb.valids
            if int(valids.sum().item()) < 2:
                raise RuntimeError(
                    "Learner update invariant violated: fewer than two valid decisions reached PPO"
                )

        projection = self.actor_critic.encoder.DG_projection
        update_running_stats = iterative_phase in (SIMULTANEOUS, ENCODER)
        stats_context = getattr(projection, "running_stats_update", None)
        if stats_context is None:
            outputs = self._forward_pass(
                mb=mb, recurrence=recurrence, valids=valids, return_outputs=[True, True, True]
            )
        else:
            with stats_context(update_running_stats):
                outputs = self._forward_pass(
                    mb=mb, recurrence=recurrence, valids=valids, return_outputs=[True, True, True]
                )
        additional_stats["dg_forward_count"] = outputs.head_outputs.new_tensor(1.0)
        additional_stats["dg_running_stats_update_count"] = outputs.head_outputs.new_tensor(
            float(bool(getattr(projection, "last_running_stats_updated", False)))
        )

        # CA3 slot zero is exactly the current post-feedback DG activation.
        # Reconstructing it here keeps every existing DG loss on the true
        # landmark representation without another visual or DG forward.
        n_dg = int(self.cfg.Hippo_n_feature)
        expanded = int(self.cfg.Hippo_R) + int(self.cfg.Hippo_L) - 1
        learner_dg = outputs.core_outputs[:, : n_dg * expanded].view(-1, n_dg, expanded)[:, :, 0]
        outputs.head_outputs = torch.cat((learner_dg, outputs.head_outputs[:, n_dg:]), dim=-1)
        feedback_stats = getattr(self.actor_critic.core, "last_context_feedback_stats", {})
        feedback_zero = learner_dg.detach().sum() * 0.0
        for name in (
            "created_fraction",
            "suppressed_fraction",
            "unchanged_fraction",
            "modulation_abs_mean",
            "modulation_saturation_fraction",
        ):
            additional_stats[f"dg_context_{name}"] = feedback_stats.get(name, feedback_zero).detach()

        raw_logits = getattr(projection, "last_raw_logits", None)
        normalized_logits = getattr(projection, "last_pre_threshold_logits", None)
        valid_logits = valids.bool()
        if torch.is_tensor(raw_logits) and torch.is_tensor(normalized_logits) and valid_logits.any():
            raw_valid = raw_logits[valid_logits]
            normalized_valid = normalized_logits[valid_logits]
            raw_row_mean = raw_valid.mean(dim=0)
            raw_row_var = raw_valid.var(dim=0, unbiased=False)
            normalized_row_mean = normalized_valid.mean(dim=0)
            normalized_row_var = normalized_valid.var(dim=0, unbiased=False)
            additional_stats["dg_raw_logit_mean_abs"] = raw_row_mean.abs().mean().detach()
            additional_stats["dg_raw_logit_variance_mean"] = raw_row_var.mean().detach()
            additional_stats["dg_normalized_logit_mean_abs"] = normalized_row_mean.abs().mean().detach()
            additional_stats["dg_normalized_logit_variance_error"] = (
                normalized_row_var - 1.0
            ).abs().mean().detach()
        else:
            zero = outputs.head_outputs.detach().sum() * 0.0
            additional_stats["dg_raw_logit_mean_abs"] = zero
            additional_stats["dg_raw_logit_variance_mean"] = zero
            additional_stats["dg_normalized_logit_mean_abs"] = zero
            additional_stats["dg_normalized_logit_variance_error"] = zero

        additional_stats["Head Output"] = outputs.head_outputs[:, : getattr(self.cfg, "Hippo_n_feature", 64)]

        with self.timing.add_time("post_forward"):
            action_distribution = self.actor_critic.action_distribution()
            log_prob_actions = action_distribution.log_prob(mb.actions)
            ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

            # super large/small values can cause numerical problems and are probably noise anyway
            ratio = torch.clamp(ratio, 0.05, 20.0)

            values = outputs.result["values"].squeeze()

        # these computations are not the part of the computation graph
        with torch.no_grad(), self.timing.add_time("advantages_returns"):
            with self.timing.add_time("Distance Matrix"):
                distance_matrix, masked_distance_matrix, progression = self._record_distance_matrix(
                    outputs.core_outputs.detach(),
                    minibatch_size=outputs.minibatch_size,
                    masked_matrix=True,
                    return_progression=True,
                )
                additional_stats["Distance Matrix"] = distance_matrix
                additional_stats["Distance Matrix Masked"] = masked_distance_matrix

            # using regular GAE
            adv = mb.advantages
            targets = mb.returns

            adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
            if self.cfg.normalize_advantage:
                adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)  # normalize advantage
            # log.info(f'Advantage Shape: {adv.shape}')

        with self.timing.add_time("decoder_losses"):
            # noinspection PyTypeChecker
            old_values = mb["values"]
            value_loss = self._value_loss(values, old_values, targets, clip_value, valids, num_invalids)
            policy_loss = self._policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids)

            exploration_loss = self.exploration_loss_func(action_distribution, valids, num_invalids)
            additional_stats["behavior_replay_mismatch"] = getattr(
                self, "_last_behavior_replay_mismatch", outputs.core_outputs.detach().sum() * 0.0
            )
            if bool(getattr(self.cfg, "hrl_behavior_mode_condition", False)):
                _, _, behavior_mode = self._behavior_condition_from_states(mb.rnn_states)
                free_mask = behavior_mode[:, MODE_EXPLORE].bool()
                for name, branch_valid in (
                    ("goal", valids & ~free_mask),
                    ("free", valids & free_mask),
                ):
                    branch_invalid = int((~branch_valid).sum().item())
                    if branch_valid.any():
                        additional_stats[f"{name}_policy_loss"] = self._policy_loss(
                            ratio, adv, clip_ratio_low, clip_ratio_high, branch_valid, branch_invalid
                        ).detach()
                        additional_stats[f"{name}_value_loss"] = self._value_loss(
                            values, old_values, targets, clip_value, branch_valid, branch_invalid
                        ).detach()
                        additional_stats[f"{name}_entropy_loss"] = self.exploration_loss_func(
                            action_distribution, branch_valid, branch_invalid
                        ).detach()
                    else:
                        zero = values.detach().sum() * 0.0
                        additional_stats[f"{name}_policy_loss"] = zero
                        additional_stats[f"{name}_value_loss"] = zero
                        additional_stats[f"{name}_entropy_loss"] = zero
            else:
                zero = values.detach().sum() * 0.0
                for name in ("goal", "free"):
                    additional_stats[f"{name}_policy_loss"] = zero
                    additional_stats[f"{name}_value_loss"] = zero
                    additional_stats[f"{name}_entropy_loss"] = zero

            extra_decoder_loss = self._extra_decoder_loss(
                ratio,
                mb["encoder_dominant_activation_mask"],
                clip_ratio_low,
                clip_ratio_high,
                valids,
                num_invalids,
            )
            predictor_loss, predictor_hit_accuracy, predictor_time_mae, predictor_positive_fraction = (
                self._ca3_predictor_loss(outputs.core_outputs, outputs.head_outputs, mb, recurrence)
            )
            extra_decoder_loss = extra_decoder_loss + float(self.cfg.ca3_predictor_loss_coeff) * predictor_loss
            (
                empirical_her_loss,
                empirical_her_policy_loss,
                empirical_her_value_loss,
                empirical_her_ratio,
                empirical_her_clip_fraction,
                empirical_her_valid_fraction,
            ) = self._empirical_her_loss(
                outputs.core_outputs,
                outputs.head_outputs,
                mb,
                recurrence,
                clip_ratio_low,
                clip_ratio_high,
                clip_value,
            )
            extra_decoder_loss = extra_decoder_loss + empirical_her_loss
            additional_stats["ca3_predictor_loss"] = predictor_loss
            additional_stats["ca3_predictor_hit_accuracy"] = predictor_hit_accuracy
            additional_stats["ca3_predictor_time_mae"] = predictor_time_mae
            additional_stats["ca3_predictor_positive_fraction"] = predictor_positive_fraction
            additional_stats["empirical_her_loss"] = empirical_her_loss
            additional_stats["empirical_her_policy_loss"] = empirical_her_policy_loss
            additional_stats["empirical_her_value_loss"] = empirical_her_value_loss
            additional_stats["empirical_her_ratio"] = empirical_her_ratio
            additional_stats["empirical_her_clip_fraction"] = empirical_her_clip_fraction
            additional_stats["empirical_her_valid_fraction"] = empirical_her_valid_fraction
            with torch.no_grad():
                if self._uses_policy_graph():
                    behavior_target = self._behavior_targets_from_states(mb.rnn_states)
                    alternate_target = behavior_target.roll(1, dims=0)
                    alternate = self.actor_critic.forward_tail(
                        self._with_worker_target(outputs.core_outputs.detach(), alternate_target),
                        values_only=False,
                        sample_actions=False,
                    )
                    goal_valid = behavior_target.sum(dim=-1).gt(0)
                    action_delta = (
                        outputs.result["action_logits"].detach() - alternate["action_logits"]
                    ).abs().mean(dim=-1)
                    action_probability_tv = categorical_action_total_variation(
                        outputs.result["action_logits"].detach(), alternate["action_logits"]
                    )
                    value_delta = (
                        outputs.result["values"].detach() - alternate["values"]
                    ).abs()
                    additional_stats["goal_condition_target_valid_fraction"] = goal_valid.float().mean()
                    additional_stats["goal_condition_action_sensitivity"] = (
                        action_delta[goal_valid].mean() if goal_valid.any() else action_delta.sum() * 0.0
                    )
                    additional_stats["goal_condition_action_probability_tv"] = (
                        action_probability_tv[goal_valid].mean()
                        if goal_valid.any()
                        else action_probability_tv.sum() * 0.0
                    )
                    additional_stats["goal_condition_value_span"] = (
                        value_delta[goal_valid].mean() if goal_valid.any() else value_delta.sum() * 0.0
                    )
                else:
                    goal_zero = outputs.core_outputs.detach().sum() * 0.0
                    additional_stats["goal_condition_target_valid_fraction"] = goal_zero
                    additional_stats["goal_condition_action_sensitivity"] = goal_zero
                    additional_stats["goal_condition_action_probability_tv"] = goal_zero
                    additional_stats["goal_condition_value_span"] = goal_zero

            kl_old, kl_loss = self.kl_loss_func(
                self.actor_critic.action_space, mb.action_logits, action_distribution, valids, num_invalids
            )

        with self.timing.add_time("encoder_losses"):
            learner_active = outputs.head_outputs[:, : int(self.cfg.Hippo_n_feature)] > 0
            scheduled_credit_mask = mb["encoder_credit_activation_mask"].bool()
            applied_credit_mask = scheduled_credit_mask & learner_active
            reward_by_row = mb["rewards_encoder"]
            if reward_by_row.ndim == 1:
                reward_by_row = reward_by_row.unsqueeze(1).expand_as(applied_credit_mask)
            scheduled_reward_mass = (reward_by_row * scheduled_credit_mask).sum()
            applied_reward_mass = (reward_by_row * applied_credit_mask).sum()
            scheduled_credit_count = scheduled_credit_mask.sum()
            applied_credit_count = applied_credit_mask.sum()
            additional_stats["encoder_credit_scheduled_count"] = scheduled_credit_count.detach().float()
            additional_stats["encoder_credit_applied_count"] = applied_credit_count.detach().float()
            additional_stats["encoder_credit_scheduled_mass"] = scheduled_reward_mass.detach().float()
            additional_stats["encoder_credit_applied_mass"] = applied_reward_mass.detach().float()
            additional_stats["encoder_credit_replay_match"] = (
                applied_credit_count.float() / scheduled_credit_count.clamp_min(1).float()
            ).detach()
            # noinspection PyTypeChecker
            encoder_loss = self._encoder_loss(
                outputs.head_outputs,
                mb["rewards_encoder"],
                applied_credit_mask,
                valids,
                num_invalids,
            )
            encoder_credit_loss = encoder_loss
            l1_loss = self._l1_loss(outputs.head_outputs, valids, num_invalids)

            if self.cfg.extra_encoder_losses:
                (
                    encoder_penalty_loss,
                    encoder_reward_loss,
                    encoder_batch_loss,
                    encoder_batch_unused_count,
                ) = self._extra_encoder_loss(
                    outputs.head_outputs,
                    mb["rnn_states"].detach(),
                    mb["encoder_dominant_activation_mask"],
                    mb["encoder_non_dominant_activation_mask"],
                    outputs.minibatch_size,
                    valids,
                    num_invalids,
                )
                encoder_loss += encoder_reward_loss + encoder_penalty_loss + encoder_batch_loss
            else:
                encoder_loss += l1_loss
                encoder_batch_unused_count = outputs.head_outputs.sum() * 0.0
            population_loss, usage_loss, density_loss, collision_loss = self._population_usage_loss(
                outputs.head_outputs
            )
            global_punishment_loss, row_repulsion_loss, temporal_exclusion_loss, path_scatter_loss, dg_regularizer_stats = (
                self._anti_collapse_losses(
                    outputs.head_outputs,
                    mb["rnn_states"],
                    mb["encoder_dominant_activation_mask"],
                    valids,
                    num_invalids,
                )
            )
            encoder_loss += (
                population_loss
                + global_punishment_loss
                + row_repulsion_loss
                + temporal_exclusion_loss
                + path_scatter_loss
            )
            transition_mode = getattr(self.cfg, "dg_transition_prediction", "none")
            transition_loss = encoder_loss * 0.0
            transition_stats = {
                "main_loss": transition_loss.detach(),
                "control_loss": transition_loss.detach(),
                "validation_main_ce": transition_loss.detach(),
                "validation_control_ce": transition_loss.detach(),
                "validation_state_gain": transition_loss.detach(),
                "validation_accuracy": transition_loss.detach(),
                "validation_count": transition_loss.detach(),
            }
            scheduled_prediction = transition_loss.detach()
            applied_prediction = transition_loss.detach()
            boundary_prediction = transition_loss.detach()
            predictor = getattr(self.actor_critic, "dg_transition_predictor", None)
            if transition_mode != "none":
                if predictor is None:
                    raise RuntimeError("DG transition prediction is enabled without a predictor")
                prediction_batch = build_transition_prediction_batch(
                    learner_dg,
                    mb["hrl_control_completed"],
                    mb["hrl_control_target_timeout"],
                    mb["hrl_control_source"],
                    mb["hrl_control_command_target"],
                    mb["hrl_control_outcome_id"],
                    mb["hrl_control_elapsed"],
                    valids,
                    recurrence,
                    n_dg,
                )
                transition_loss, transition_stats = transition_prediction_losses(
                    predictor, prediction_batch
                )
                scheduled_prediction = prediction_batch.scheduled_count.detach().float()
                applied_prediction = prediction_batch.applied_count.detach().float()
                boundary_prediction = prediction_batch.boundary_drop_count.detach().float()
                encoder_loss = encoder_loss + float(
                    getattr(self.cfg, "dg_transition_prediction_coeff", 0.1)
                ) * transition_loss
            encoder_loss *= self.cfg.encoder_grad_coeff
            additional_stats["encoder_loss"] = encoder_loss
            additional_stats["dg_transition_prediction_loss"] = transition_loss.detach()
            additional_stats["dg_transition_prediction_main_loss"] = transition_stats["main_loss"]
            additional_stats["dg_transition_prediction_control_loss"] = transition_stats["control_loss"]
            additional_stats["dg_transition_prediction_validation_main_ce"] = transition_stats[
                "validation_main_ce"
            ]
            additional_stats["dg_transition_prediction_validation_control_ce"] = transition_stats[
                "validation_control_ce"
            ]
            additional_stats["dg_transition_prediction_validation_state_gain"] = transition_stats[
                "validation_state_gain"
            ]
            additional_stats["dg_transition_prediction_validation_accuracy"] = transition_stats[
                "validation_accuracy"
            ]
            additional_stats["dg_transition_prediction_validation_count"] = transition_stats[
                "validation_count"
            ]
            additional_stats["dg_transition_prediction_scheduled_count"] = scheduled_prediction
            additional_stats["dg_transition_prediction_applied_count"] = applied_prediction
            additional_stats["dg_transition_prediction_boundary_drop_count"] = boundary_prediction
            additional_stats["dg_transition_prediction_replay_match"] = (
                applied_prediction / (scheduled_prediction - boundary_prediction).clamp_min(1.0)
            )
            gradient_zero = encoder_loss.detach().new_zeros(())
            additional_stats["dg_ppo_gradient_norm"] = gradient_zero
            additional_stats["dg_encoder_gradient_norm"] = gradient_zero
            additional_stats["dg_gradient_norm_ratio"] = gradient_zero
            additional_stats["dg_gradient_cosine"] = gradient_zero
            additional_stats["dg_gradient_row_conflict_fraction"] = gradient_zero
            zero_credit = encoder_loss.detach() * 0.0
            if getattr(self.cfg, "encoder_reward_recipient", "arrival") == "source":
                additional_stats["encoder_source_credit_loss"] = encoder_credit_loss.detach()
                additional_stats["encoder_arrival_credit_loss"] = zero_credit
            else:
                additional_stats["encoder_arrival_credit_loss"] = encoder_credit_loss.detach()
                additional_stats["encoder_source_credit_loss"] = zero_credit
            additional_stats["intrinsic_rewards"] = mb["rewards"]
            additional_stats["encoder_penalty_loss"] = encoder_penalty_loss
            additional_stats["encoder_reward_loss"] = encoder_reward_loss
            additional_stats["batch_reward_loss"] = encoder_batch_loss
            additional_stats["encoder_batch_unused_count"] = encoder_batch_unused_count
            additional_stats["encoder_population_loss"] = population_loss
            additional_stats["encoder_usage_loss"] = usage_loss
            additional_stats["encoder_density_loss"] = density_loss
            additional_stats["encoder_collision_loss"] = collision_loss
            additional_stats["encoder_global_punishment_loss"] = global_punishment_loss
            additional_stats["encoder_row_repulsion_loss"] = row_repulsion_loss
            additional_stats["encoder_ca3_temporal_exclusion_loss"] = temporal_exclusion_loss
            additional_stats["encoder_path_scatter_loss"] = path_scatter_loss
            additional_stats["dg_pre_threshold_mean"] = dg_regularizer_stats[0]
            additional_stats["dg_pre_threshold_above_fraction"] = dg_regularizer_stats[1]
            additional_stats["dg_ca3_conflict_fraction"] = dg_regularizer_stats[2]
            additional_stats["dg_ca3_conflicting_activation_fraction"] = dg_regularizer_stats[3]
            additional_stats["dg_ca3_conflict_activity"] = dg_regularizer_stats[4]
            additional_stats["dg_path_scatter_conflict_fraction"] = dg_regularizer_stats[5]

        loss_summaries = dict(
            ratio=ratio,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            values=outputs.result["values"],
            adv=adv,
            adv_std=adv_std,
            adv_mean=adv_mean,
            additional_stats=additional_stats,
        )
        del outputs

        return (
            action_distribution,
            policy_loss,
            exploration_loss,
            kl_old,
            kl_loss,
            value_loss,
            extra_decoder_loss,
            encoder_loss,
            loss_summaries,
        )

    def _train(
        self, gpu_buffer: TensorDict, batch_size: int, experience_size: int, num_invalids: int
    ) -> Optional[AttrDict]:
        timing = self.timing
        with torch.no_grad():
            early_stopping_tolerance = 1e-6
            early_stop = False
            prev_epoch_actor_loss = 1e9
            epoch_actor_losses = [0] * self.cfg.num_batches_per_epoch

            # recent mean KL-divergences per minibatch, this used by LR schedulers
            recent_kls = []

            if self.cfg.with_vtrace:
                assert (
                    self.cfg.recurrence == self.cfg.rollout and self.cfg.recurrence > 1
                ), "V-trace requires to recurrence and rollout to be equal"

            num_sgd_steps = 0
            stats_and_summaries: Optional[AttrDict] = None
            recruitment_applied = False

            # When it is time to record train summaries, we randomly sample epoch/batch for which the summaries are
            # collected to get equal representation from different stages of training.
            # Half the time, we record summaries from the very large step of training. There we will have the highest
            # KL-divergence and ratio of PPO-clipped samples, which makes this data even more useful for analysis.
            # Something to consider: maybe we should have these last-batch metrics in a separate summaries category?
            with_summaries = self._should_save_summaries()
            if np.random.rand() < 0.5:
                summaries_epoch = np.random.randint(0, self.cfg.num_epochs)
                summaries_batch = np.random.randint(0, self.cfg.num_batches_per_epoch)
            else:
                summaries_epoch = self.cfg.num_epochs - 1
                summaries_batch = self.cfg.num_batches_per_epoch - 1

            assert self.actor_critic.training

        for epoch in range(self.cfg.num_epochs):
            with timing.add_time("epoch_init"):
                if early_stop:
                    break

                force_summaries = False
                minibatches = self._get_minibatches(batch_size, experience_size)

            for batch_num in range(len(minibatches)):
                with torch.no_grad(), timing.add_time("minibatch_init"):
                    indices = minibatches[batch_num]

                    # current minibatch consisting of short trajectory segments with length == recurrence
                    mb = self._get_minibatch(gpu_buffer, indices)

                    # enable syntactic sugar that allows us to access dict's keys as object attributes
                    mb = AttrDict(mb)

                with timing.add_time("calculate_losses"):
                    iterative_phase = self._iterative_phase()
                    (
                        action_distribution,
                        policy_loss,
                        exploration_loss,
                        kl_old,
                        kl_loss,
                        value_loss,
                        extra_decoder_loss,
                        encoder_loss,
                        loss_summaries,
                    ) = self._calculate_losses(mb, num_invalids, iterative_phase)

                with timing.add_time("losses_postprocess"):
                    # noinspection PyTypeChecker
                    actor_loss: Tensor = policy_loss + exploration_loss + kl_loss
                    critic_loss = value_loss
                    decoder_loss: Tensor = actor_loss + critic_loss + extra_decoder_loss

                    if iterative_phase == DECODER:
                        loss = decoder_loss
                    elif iterative_phase == ENCODER:
                        loss = encoder_loss
                    else:
                        loss = decoder_loss + encoder_loss

                    epoch_actor_losses[batch_num] = float(actor_loss)

                    high_loss = 30.0
                    if torch.abs(decoder_loss) > high_loss:
                        log.warning(
                            "High loss value: decl:%.4f encl:%.4f pl:%.4f vl:%.4f exp_l:%.4f kl_l:%.4f (recommended to adjust the --reward_scale parameter)",
                            to_scalar(decoder_loss),
                            to_scalar(encoder_loss),
                            to_scalar(policy_loss),
                            to_scalar(value_loss),
                            to_scalar(exploration_loss),
                            to_scalar(kl_loss),
                        )

                        # perhaps something weird is happening, we definitely want summaries from this step
                        force_summaries = True

                with torch.no_grad(), timing.add_time("kl_divergence"):
                    # if kl_old is not None it is already calculated above
                    if kl_old is None:
                        # calculate KL-divergence with the behaviour policy action distribution
                        old_action_distribution = get_action_distribution(
                            self.actor_critic.action_space,
                            mb.action_logits,
                        )
                        kl_old = action_distribution.kl_divergence(old_action_distribution)
                        kl_old = masked_select(kl_old, mb.valids, num_invalids)

                    kl_old_mean = float(kl_old.mean().item())
                    recent_kls.append(kl_old_mean)
                    if kl_old.numel() > 0 and kl_old.max().item() > 100:
                        log.warning(f"KL-divergence is very high: {kl_old.max().item():.4f}")

                # update the weights
                with timing.add_time("update"):
                    # following advice from https://youtu.be/9mS1fIYj1So set grad to None instead of optimizer.zero_grad()
                    for p in self.actor_critic.parameters():
                        p.grad = None

                    if with_summaries or force_summaries:
                        gradient_stats = dg_gradient_interaction_stats(
                            decoder_loss,
                            encoder_loss,
                            self.actor_critic.encoder.DG_projection,
                        )
                        additional = loss_summaries["additional_stats"]
                        additional["dg_ppo_gradient_norm"] = gradient_stats["ppo_norm"]
                        additional["dg_encoder_gradient_norm"] = gradient_stats["encoder_norm"]
                        additional["dg_gradient_norm_ratio"] = gradient_stats["ratio"]
                        additional["dg_gradient_cosine"] = gradient_stats["cosine"]
                        additional["dg_gradient_row_conflict_fraction"] = gradient_stats[
                            "row_conflict_fraction"
                        ]

                    # The forward graph defines ownership: CA3 is stopped before
                    # the controller and the DG input is detached from the bypass.
                    # Consequently the selected phase loss can be differentiated
                    # once without deleting or rewriting gradients afterwards.
                    loss.backward()

                    # self._manipulate_gradients()

                    if self.cfg.max_grad_norm > 0.0:
                        with timing.add_time("clip"):
                            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)

                    curr_policy_version = self.train_step  # policy version before the weight update

                    actual_lr = self.curr_lr
                    if num_invalids > 0:
                        # if we have masked (invalid) data we should reduce the learning rate accordingly
                        # this prevents a situation where most of the data in the minibatch is invalid
                        # and we end up doing SGD with super noisy gradients
                        actual_lr = self.curr_lr * (experience_size - num_invalids) / experience_size
                    self._apply_lr(actual_lr)

                    with self.param_server.policy_lock:
                        self.optimizer.step()
                        if iterative_phase in (SIMULTANEOUS, ENCODER):
                            projection = self.actor_critic.encoder.DG_projection
                            normalize_dg_projection_rows(projection.linear)
                            poststep_update = getattr(projection, "post_step_update_running_stats", None)
                            if poststep_update is not None:
                                calibrated = bool(poststep_update(mb.valids))
                                record_poststep_calibration_count(loss_summaries, loss, calibrated)
                            weight_generation = getattr(projection, "weight_generation", None)
                            statistics_generation = getattr(projection, "statistics_generation", None)
                            if (
                                torch.is_tensor(weight_generation)
                                and torch.is_tensor(statistics_generation)
                                and not torch.equal(weight_generation, statistics_generation)
                            ):
                                raise RuntimeError(
                                    "DG weights and normalization statistics were published at different generations"
                                )

                    num_sgd_steps += 1

                with torch.no_grad(), timing.add_time("after_optimizer"):
                    self._after_optimizer_step()

                    final_minibatch = epoch == self.cfg.num_epochs - 1 and batch_num == len(minibatches) - 1
                    if final_minibatch and not recruitment_applied:
                        self._apply_dg_recruitment(gpu_buffer)
                        recruitment_applied = True

                    if self.lr_scheduler.invoke_after_each_minibatch():
                        self.curr_lr = self.lr_scheduler.update(self.curr_lr, recent_kls)

                    # collect and report summaries
                    should_record_summaries = with_summaries
                    should_record_summaries &= epoch == summaries_epoch and batch_num == summaries_batch
                    should_record_summaries |= force_summaries
                    if should_record_summaries:
                        # hacky way to collect all of the intermediate variables for summaries
                        summary_vars = {**locals(), **loss_summaries}
                        stats_and_summaries = self._record_summaries(AttrDict(summary_vars))
                        del summary_vars
                        force_summaries = False

                    # make sure everything (such as policy weights) is committed to shared device memory
                    synchronize(self.cfg, self.device)
                    # this will force policy update on the inference worker (policy worker)
                    self.policy_versions_tensor[self.policy_id] = self.train_step

            # end of an epoch
            if self.lr_scheduler.invoke_after_each_epoch():
                self.curr_lr = self.lr_scheduler.update(self.curr_lr, recent_kls)

            new_epoch_actor_loss = float(np.mean(epoch_actor_losses))
            loss_delta_abs = abs(prev_epoch_actor_loss - new_epoch_actor_loss)
            if loss_delta_abs < early_stopping_tolerance:
                early_stop = True
                log.debug(
                    "Early stopping after %d epochs (%d sgd steps), loss delta %.7f",
                    epoch + 1,
                    num_sgd_steps,
                    loss_delta_abs,
                )
                break

            prev_epoch_actor_loss = new_epoch_actor_loss

        if not recruitment_applied:
            self._apply_dg_recruitment(gpu_buffer)
            synchronize(self.cfg, self.device)
            self.policy_versions_tensor[self.policy_id] = self.train_step

        return stats_and_summaries

    def _calculate_temporal_internal_reward(self, buff, additional_step):
        # rnn_state_shape = buff["rnn_states"].shape
        # log.debug(f'rnn_state_shape: {rnn_state_shape}')
        # dataset_size = rnn_state_shape[0]*rnn_state_shape[1]
        rnn_states_clone: Tensor = buff["rnn_states"].clone()
        # log.debug(f'additional_step["new_rnn_states"] Shape: {additional_step["new_rnn_states"].unsqueeze(1).shape}; rnn_state_shape: {rnn_state_shape}')
        rnn_states_clone = torch.cat((rnn_states_clone, additional_step["new_rnn_states"].unsqueeze(1)), dim=1)

        rnn_state_shape = rnn_states_clone.shape
        rnn_states_sequence = rnn_states_clone
        # log.debug(f'rnn_state_shape: {rnn_state_shape}')
        dataset_size = rnn_state_shape[0] * rnn_state_shape[1]
        rnn_states_clone = rnn_states_clone.reshape((dataset_size,) + tuple(rnn_state_shape[2:]))
        sequence_core, _ = self._calculate_sequence_core(rnn_states_clone, dataset_size)
        # log.debug(f'sequence_core: {sequence_core}')

        progression = self._calculate_progression(sequence_core)
        progression = progression.view(*rnn_state_shape[:2], self.cfg.Hippo_n_feature)
        sequence_core = sequence_core.view(
            *rnn_state_shape[:2], self.cfg.Hippo_n_feature, self.cfg.Hippo_R + self.cfg.Hippo_L - 1
        )
        # log.debug(f'progression: {progression.long()}')
        # log.debug(f'progression == 0: {(progression == 0).long()}')
        # log.debug(f'rolled progression: {(torch.roll(progression, shifts = 1, dims = 1) >= self.cfg.Hippo_R).long()}')

        mask_new_activations, dominant_activations, non_dominant_activations = dominant_new_activation_masks(
            sequence_core, progression, self.cfg.Hippo_R
        )
        # progression = progression.view(dataset_size, self.cfg.Hippo_n_feature)
        # mask_new_activations = mask_new_activations.view(dataset_size, self.cfg.Hippo_n_feature)

        # sequence_core = sequence_core.view(*rnn_state_shape[:2] + (self.cfg.Hippo_n_feature, expanded_length))
        # progression = progression.view(*rnn_state_shape[:2], self.cfg.Hippo_n_feature)
        # Prepare result tensor, filled with baseline value
        baseline = self.cfg.Hippo_L + self.cfg.Hippo_R - 1

        internal_reward = predecessor_distance_for_dominant_events(
            progression, mask_new_activations, dominant_activations, baseline
        )

        return (
            internal_reward,
            rnn_states_sequence,
            progression,
            mask_new_activations,
            dominant_activations,
            non_dominant_activations,
        )

    def _calculate_internal_reward(self, buff, additional_step):
        buff["rewards_external"] = buff["rewards"].clone()
        internal_reward, rnn_states, progression, candidate_activations, dominant_activations, non_dominant_activations = (
            self._calculate_temporal_internal_reward(buff, additional_step)
        )
        baseline = self.cfg.Hippo_L + self.cfg.Hippo_R - 1
        decoder_reward, encoder_reward = legacy_reward_streams(
            internal_reward,
            baseline,
            self.cfg.reward_scale,
            self.cfg.encoder_reward_method,
        )
        from .ca3_memory import absent_event_gate
        novel = absent_event_gate(progression, dominant_activations, baseline)[:, 2:]
        onset = dominant_activations[:, 2:].any(-1)
        buff["memory_onset"] = onset.float()
        buff["memory_novel_onset"] = novel.float()
        buff["memory_familiar_onset"] = (onset & ~novel).float()
        if getattr(self.cfg, "decoder_reward_gate", "none") == "ca3_absent":
            decoder_reward = decoder_reward * novel
        if getattr(self.cfg, "intrinsic_goal_mode", "none") != "none":
            # Same action/outcome alignment as legacy decoder rewards. The
            # stored pulse describes arrival under the preceding command.
            decoder_reward = rnn_states[:, 2:, -4].clone()
            buff["memory_goal_ambiguous"] = rnn_states[:, 2:, -3].clone()
            command = rnn_states[:, 1:-1, -1].long()
            buff["memory_goal_active"] = command.gt(0).float()
            buff["memory_goal_hit"] = decoder_reward.gt(0).float()
        else:
            for name in ("goal_ambiguous", "goal_active", "goal_hit"):
                buff["memory_" + name] = torch.zeros_like(decoder_reward)
        buff["memory_gate_violation"] = (
            decoder_reward.ne(0) & ~novel
        ).float() if getattr(self.cfg, "decoder_reward_gate", "none") == "ca3_absent" else torch.zeros_like(decoder_reward)
        if getattr(self.cfg, "hrl_controllable_graph", False):
            behavior_states = torch.cat((buff["rnn_states"], additional_step["new_rnn_states"].unsqueeze(1)), dim=1)
            hrl_state = self._hrl_state_from_rnn(behavior_states if self._uses_policy_graph() else rnn_states)
            layout = self._hrl_layout()
            hit = hrl_state[..., layout.target_hit]
            exploring = exploration_mode_mask(hrl_state, layout.n_nodes)
            outcome_labels = control_outcome_labels(hrl_state, buff["dones"], layout)
            wrong_outcome = outcome_labels["wrong"]
            command_set_size = None
            if getattr(self.cfg, "hrl_direct_target_selection", "frontier") == "local_successor":
                topo_state = self._topological_state_from_rnn(
                    behavior_states if self._uses_policy_graph() else rnn_states
                )
                topo_layout = TopologicalStateLayout(layout.n_nodes)
                command_set_size = topo_state[:, 1:-1, topo_layout.local_candidate_count]
            magnitude = target_reward_magnitude(
                internal_reward,
                baseline,
                self.cfg.reward_scale,
                self.cfg.hrl_worker_reward_mode,
                self.cfg.hrl_target_hit_reward,
                self.cfg.hrl_distance_bonus_coeff,
            )
            decoder_reward = target_success_worker_reward(
                internal_reward,
                hit,
                baseline,
                self.cfg.reward_scale,
                self.cfg.hrl_worker_reward_mode,
                self.cfg.hrl_target_hit_reward,
                self.cfg.hrl_distance_bonus_coeff,
                exploring,
                decoder_reward,
                control_outcome=getattr(self.cfg, "hrl_control_outcome", "target_hit"),
                wrong_outcome=wrong_outcome,
                n_targets=layout.n_nodes,
                command_set_size=command_set_size,
            )
            buff["hrl_control_correct_outcome"] = outcome_labels["correct"]
            buff["hrl_control_wrong_outcome"] = wrong_outcome
            buff["hrl_control_target_timeout"] = outcome_labels["timeout"]
            buff["hrl_control_exploration_timeout"] = outcome_labels["exploration_timeout"]
            buff["hrl_control_censored"] = outcome_labels["censored"]
            buff["hrl_control_completed"] = outcome_labels["completed"]
            buff["hrl_control_reward_magnitude"] = magnitude
            buff["hrl_control_elapsed"] = outcome_labels["elapsed"]
            buff["hrl_control_command_target"] = outcome_labels["target"]
            buff["hrl_control_source"] = outcome_labels["source"]
            buff["hrl_control_outcome_id"] = outcome_labels["outcome"]
            if command_set_size is not None:
                buff["hrl_control_command_set_size"] = command_set_size
        buff["rewards"] = decoder_reward
        dominant_rollout = dominant_activations[:, 1:-1]
        credit_mask = dominant_rollout
        if bool(getattr(self.cfg, "encoder_reward_require_local_predecessor", False)):
            encoder_reward, credit_mask, credit_stats = build_matched_encoder_credit(
                progression[:, 1:-1],
                candidate_activations[:, 1:-1],
                dominant_rollout,
                buff["valids"][:, :-1],
                baseline,
                self.cfg.reward_scale,
                getattr(self.cfg, "encoder_reward_recipient", "arrival"),
            )
            credited = max(1.0, float(credit_stats["credited"].item()))
            self._last_encoder_credit_stats = {
                key: float(value.detach().cpu().item()) for key, value in credit_stats.items()
            }
            self._last_encoder_credit_stats["source_lag_mean"] = (
                self._last_encoder_credit_stats["source_lag_sum"] / credited
            )
        else:
            self._last_encoder_credit_stats = {}
        buff["rewards_encoder"] = encoder_reward
        buff["encoder_credit_activation_mask"] = credit_mask
        buff["encoder_dominant_activation_mask"] = dominant_rollout
        buff["encoder_non_dominant_activation_mask"] = non_dominant_activations[:, 1:-1]

    def _prepare_batch(self, batch: TensorDict) -> Tuple[TensorDict, int, int]:
        with torch.no_grad():
            # create a shallow copy so we can modify the dictionary
            # we still reference the same buffers though
            buff = shallow_recursive_copy(batch)

            # ignore experience from other agents (i.e. on episode boundary) and from inactive agents
            valids: Tensor = buff["policy_id"] == self.policy_id
            # ignore experience that was older than the threshold even before training started
            curr_policy_version: int = self.train_step
            buff["valids"][:, :-1] = valids & (curr_policy_version - buff["policy_version"] < self.cfg.max_policy_lag)
            eligible_before_generation = buff["valids"][:, :-1].clone()
            generation_matches = self._current_generation_mask(buff["rnn_states"][:, :-1])
            generation_valids, stale_rollouts = complete_rollout_generation_mask(
                eligible_before_generation,
                generation_matches,
            )
            stale_generation = eligible_before_generation & ~generation_valids
            buff["valids"][:, :-1] = generation_valids
            eligible_count = eligible_before_generation.sum().clamp_min(1)
            self._last_stale_generation_stats = dict(
                rejected_count=float(stale_generation.sum().item()),
                rejected_fraction=float((stale_generation.sum() / eligible_count).item()),
                dropped_rollouts=float(stale_rollouts.sum().item()),
                dropped_decisions=float(stale_generation.sum().item()),
                update_deferred=0.0,
            )
            self._generation_dropped_rollouts_total += float(stale_rollouts.sum().item())
            self._generation_dropped_decisions_total += float(stale_generation.sum().item())
            dataset_size = int(buff["actions"].shape[0] * buff["actions"].shape[1])
            minimum_valid_decisions = min(int(self.cfg.batch_size), dataset_size)
            if not require_sufficient_generation_batch(
                generation_valids,
                stale_rollouts,
                minimum_valid_decisions,
            ):
                # A representation change is a rollout barrier, not an
                # element-wise PPO mask. Skip before graph updates, value
                # bootstrap, reward construction, GAE, or DG statistics.
                buff["valids"].zero_()
                self._last_stale_generation_stats["update_deferred"] = 1.0
                self._generation_deferred_updates_total += 1.0
                log.warning(
                    "Representation-generation barrier deferred learner update: "
                    "%d/%d fresh decisions, %d stale rollouts",
                    int(generation_valids.sum().item()),
                    minimum_valid_decisions,
                    int(stale_rollouts.sum().item()),
                )
                return buff, dataset_size, dataset_size
            # for last T+1 step, we want to use the validity of the previous step
            buff["valids"][:, -1] = buff["valids"][:, -2]
            graph_stats = self._update_policy_graph_from_rollout(buff["rnn_states"], buff["valids"][:, :-1])
            if graph_stats is not None:
                self._last_graph_rollout_stats = {
                    key: float(value.detach().cpu().item()) for key, value in graph_stats.items()
                }
            passive_stats = self._update_passive_recruitment_graph_from_rollout(
                buff["rnn_states"], buff["valids"][:, :-1]
            )
            if passive_stats is not None:
                self._last_passive_recruitment_stats = {
                    key: float(value.detach().cpu().item()) for key, value in passive_stats.items()
                }
            self._queue_dg_recruitment_candidates(buff)
            # log.info(f'RNN_states Shape: {buff["rnn_states"].shape}')
            # log.info(f'Internal Reward1: {buff["rewards"][:,:10]}')
            # log.info(f'Reward Shape1: {buff["rewards"].shape}')
            # Calculate Internal Reward

            # ensure we're in train mode so that normalization statistics are updated
            if not self.actor_critic.training:
                self.actor_critic.train()

            buff["normalized_obs"] = self._prepare_and_normalize_obs(buff["obs"])
            del buff["obs"]  # don't need non-normalized obs anymore

            # calculate estimated value for the next step (T+1)
            normalized_last_obs = buff["normalized_obs"][:, -1]
            additional_step = self.actor_critic(normalized_last_obs, buff["rnn_states"][:, -1], values_only=True)
            next_values = additional_step["values"]
            buff["values"][:, -1] = next_values

            self._calculate_internal_reward(buff, additional_step)

            if self.cfg.normalize_returns:
                # Since our value targets are normalized, the values will also have normalized statistics.
                # We need to denormalize them before using them for GAE caculation and value bootstrapping.
                # rl_games PPO uses a similar approach, see:
                # https://github.com/Denys88/rl_games/blob/7b5f9500ee65ae0832a7d8613b019c333ecd932c/rl_games/algos_torch/models.py#L51
                denormalized_values = buff["values"].clone()  # need to clone since normalizer is in-place
                self.actor_critic.returns_normalizer(denormalized_values, denormalize=True)
            else:
                # values are not normalized in this case, so we can use them as is
                denormalized_values = buff["values"]

            if self.cfg.value_bootstrap:
                # Value bootstrapping is a technique that reduces the surprise for the critic in case
                # we're ending the episode by timeout. Intuitively, in this case the cumulative return for the last step
                # should not be zero, but rather what the critic expects. This improves learning in many envs
                # because otherwise the critic cannot predict the abrupt change in rewards in a timed-out episode.
                # What we really want here is v(t+1) which we don't have because we don't have obs(t+1) (since
                # the episode ended). Using v(t) is an approximation that requires that rew(t) can be generally ignored.

                # Multiply by both time_out and done flags to make sure we count only timeouts in terminal states.
                # There was a bug in older versions of isaacgym where timeouts were reported for non-terminal states.
                buff["rewards"].add_(self.cfg.gamma * denormalized_values[:, :-1] * buff["time_outs"] * buff["dones"])

            if not self.cfg.with_vtrace:
                # calculate advantage estimate (in case of V-trace it is done separately for each minibatch)
                buff["advantages"] = gae_advantages(
                    buff["rewards"],
                    buff["dones"],
                    denormalized_values,
                    buff["valids"],
                    self.cfg.gamma,
                    self.cfg.gae_lambda,
                )
                # here returns are not normalized yet, so we should use denormalized values
                buff["returns"] = buff["advantages"] + buff["valids"][:, :-1] * denormalized_values[:, :-1]

            # remove next step obs, rnn_states, and values from the batch, we don't need them anymore
            for key in ["normalized_obs", "rnn_states", "values", "valids"]:
                buff[key] = buff[key][:, :-1]
            dataset_size = buff["actions"].shape[0] * buff["actions"].shape[1]
            for d, k, v in iterate_recursively(buff):
                # collapse first two dimensions (batch and time) into a single dimension
                d[k] = v.reshape((dataset_size,) + tuple(v.shape[2:]))

            buff["dones_cpu"] = buff["dones"].to("cpu", copy=True, dtype=torch.float, non_blocking=True)
            buff["rewards_cpu"] = buff["rewards"].to("cpu", copy=True, dtype=torch.float, non_blocking=True)

            # return normalization parameters are only used on the learner, no need to lock the mutex
            if self.cfg.normalize_returns:
                self.actor_critic.returns_normalizer(buff["returns"])  # in-place

            num_invalids = dataset_size - buff["valids"].sum().item()
            if num_invalids > 0:
                invalid_fraction = num_invalids / dataset_size
                if invalid_fraction > 0.5:
                    log.warning(f"{self.policy_id=} batch has {invalid_fraction:.2%} of invalid samples")

                # invalid action values can cause problems when we calculate logprobs
                # here we set them to 0 just to be safe
                invalid_indices = (buff["valids"] == 0).nonzero().squeeze()
                buff["actions"][invalid_indices] = 0
                # likewise, some invalid values of log_prob_actions can cause NaNs or infs
                buff["log_prob_actions"][invalid_indices] = -1  # -1 seems like a safe value

            return buff, dataset_size, num_invalids

    def _record_summaries(self, train_loop_vars):
        var = train_loop_vars  # TODO: Think of a better way, why is this necessary? Just redirecting pointer?
        stats = super()._record_summaries(train_loop_vars)
        stats.encoder_loss = var.additional_stats["encoder_loss"].detach().float()
        stats.decoder_loss = var.decoder_loss.detach().float()
        for name in ("onset", "novel_onset", "familiar_onset", "gate_violation", "goal_ambiguous", "goal_active", "goal_hit"):
            key = "memory_" + name
            if key in var.mb:
                stats[key] = var.mb[key][var.mb.valids.bool()].float().mean().detach()
        if getattr(self.cfg, "intrinsic_goal_mode", "none") != "none":
            stats.memory_replay_mismatch = getattr(self, "_last_behavior_replay_mismatch", 0.0)
        stats.loss = var.loss.detach().float()
        stats.iterative_phase = {SIMULTANEOUS: 0, DECODER: 1, ENCODER: 2}[var.iterative_phase]
        rewards_for_advantage = var.mb.rewards.float()
        intrinsic_rewards = var.additional_stats["intrinsic_rewards"].float()
        stats.reward_for_advantage_mean = rewards_for_advantage.mean().detach().float()
        stats.reward_for_advantage_sum = rewards_for_advantage.sum().detach().float()
        stats.reward_for_advantage_abs_mean = rewards_for_advantage.abs().mean().detach().float()
        stats.reward_for_advantage_nonzero_frac = (rewards_for_advantage != 0).float().mean().detach().float()
        stats.reward_for_advantage_min = rewards_for_advantage.min().detach().float()
        stats.reward_for_advantage_max = rewards_for_advantage.max().detach().float()
        stats.intrinsic_reward_mean = intrinsic_rewards.mean().detach().float()
        stats.intrinsic_reward_sum = intrinsic_rewards.sum().detach().float()
        stats.intrinsic_reward_nonzero_frac = (intrinsic_rewards != 0).float().mean().detach().float()
        if "rewards_external" in var.mb:
            env_rewards = var.mb.rewards_external.float()
            stats.env_reward_mean = env_rewards.mean().detach().float()
            stats.env_reward_sum = env_rewards.sum().detach().float()
            stats.env_reward_nonzero_frac = (env_rewards != 0).float().mean().detach().float()
        dg_active = var.additional_stats["Head Output"] > 0
        stats.dg_density = dg_active.float().mean().detach().float()
        stats.dg_multi_activation_rate = dg_active.sum(dim=1).gt(1).float().mean().detach().float()
        stats.dg_silent_unit_frac = dg_active.any(dim=0).logical_not().float().mean().detach().float()
        valid = var.mb.valids.bool()
        valid_count = valid.sum().clamp_min(1)
        learner_active_transition = dg_active.any(dim=-1) & valid
        dominant = var.mb["encoder_dominant_activation_mask"].bool()
        credited_rows = var.mb["encoder_credit_activation_mask"].bool()
        non_dominant = var.mb["encoder_non_dominant_activation_mask"].bool()
        dominant_transition = dominant.any(dim=-1) & valid
        multi_onset_transition = non_dominant.any(dim=-1) & dominant_transition
        dominant_count = dominant_transition.sum().clamp_min(1)
        feedback = var.mb.rewards_encoder.float()
        feedback_by_transition = feedback.sum(dim=-1) if feedback.ndim > 1 else feedback
        stats.dg_learner_active_transition_fraction = (
            learner_active_transition.sum() / valid_count
        ).detach().float()
        stats.dg_behavior_dominant_event_fraction = (dominant_transition.sum() / valid_count).detach().float()
        stats.dg_behavior_multi_onset_event_fraction = (
            multi_onset_transition.sum() / dominant_count
        ).detach().float()
        stats.dg_behavior_non_dominant_onsets_per_event = (
            (non_dominant & valid.unsqueeze(-1)).sum() / dominant_count
        ).detach().float()
        stats.encoder_dominant_event_count = dominant_transition.sum().detach().float()
        if dominant_transition.any():
            event_feedback = feedback_by_transition[dominant_transition]
            stats.encoder_feedback_on_dominant_event_mean = event_feedback.mean().detach().float()
            stats.encoder_feedback_abs_on_dominant_event_mean = event_feedback.abs().mean().detach().float()
        else:
            stats.encoder_feedback_on_dominant_event_mean = (feedback.sum() * 0.0).detach().float()
            stats.encoder_feedback_abs_on_dominant_event_mean = (feedback.sum() * 0.0).detach().float()
        stats.encoder_credited_row_count = (credited_rows & valid.unsqueeze(-1)).sum().detach().float()
        stats.encoder_credit_scheduled_count = var.additional_stats[
            "encoder_credit_scheduled_count"
        ].detach().float()
        stats.encoder_credit_applied_count = var.additional_stats[
            "encoder_credit_applied_count"
        ].detach().float()
        stats.encoder_credit_scheduled_mass = var.additional_stats[
            "encoder_credit_scheduled_mass"
        ].detach().float()
        stats.encoder_credit_applied_mass = var.additional_stats[
            "encoder_credit_applied_mass"
        ].detach().float()
        stats.encoder_credit_replay_match = var.additional_stats[
            "encoder_credit_replay_match"
        ].detach().float()
        stats.dg_forward_count = var.additional_stats["dg_forward_count"].detach().float()
        stats.dg_running_stats_update_count = var.additional_stats[
            "dg_running_stats_update_count"
        ].detach().float()
        stats.dg_raw_logit_mean_abs = var.additional_stats["dg_raw_logit_mean_abs"].detach().float()
        stats.dg_raw_logit_variance_mean = var.additional_stats[
            "dg_raw_logit_variance_mean"
        ].detach().float()
        stats.dg_normalized_logit_mean_abs = var.additional_stats[
            "dg_normalized_logit_mean_abs"
        ].detach().float()
        stats.dg_normalized_logit_variance_error = var.additional_stats[
            "dg_normalized_logit_variance_error"
        ].detach().float()
        stats.dg_ppo_gradient_norm = var.additional_stats["dg_ppo_gradient_norm"].detach().float()
        stats.dg_encoder_gradient_norm = var.additional_stats["dg_encoder_gradient_norm"].detach().float()
        stats.dg_gradient_norm_ratio = var.additional_stats["dg_gradient_norm_ratio"].detach().float()
        stats.dg_gradient_cosine = var.additional_stats["dg_gradient_cosine"].detach().float()
        stats.dg_gradient_row_conflict_fraction = var.additional_stats[
            "dg_gradient_row_conflict_fraction"
        ].detach().float()
        for name in (
            "created_fraction",
            "suppressed_fraction",
            "unchanged_fraction",
            "modulation_abs_mean",
            "modulation_saturation_fraction",
        ):
            stats[f"dg_context_{name}"] = var.additional_stats[
                f"dg_context_{name}"
            ].detach().float()
        feedback_module = getattr(self.actor_critic.core, "context_feedback", None)
        if feedback_module is None:
            stats.dg_context_adapter_gradient_norm = 0.0
        else:
            squared = sum(
                parameter.grad.detach().square().sum()
                for parameter in feedback_module.adapter.parameters()
                if parameter.grad is not None
            )
            stats.dg_context_adapter_gradient_norm = (
                squared.sqrt().float() if torch.is_tensor(squared) else 0.0
            )
        stats.stale_generation_rejected_count = float(
            self._last_stale_generation_stats["rejected_count"]
        )
        stats.stale_generation_rejected_fraction = float(
            self._last_stale_generation_stats["rejected_fraction"]
        )
        stats.stale_generation_dropped_rollouts_total = float(
            self._generation_dropped_rollouts_total
        )
        stats.stale_generation_dropped_decisions_total = float(
            self._generation_dropped_decisions_total
        )
        stats.stale_generation_deferred_updates_total = float(
            self._generation_deferred_updates_total
        )
        projection = self.actor_critic.encoder.DG_projection
        weight_generation = getattr(projection, "weight_generation", None)
        statistics_generation = getattr(projection, "statistics_generation", None)
        if torch.is_tensor(weight_generation) and torch.is_tensor(statistics_generation):
            stats.dg_weight_generation = float(weight_generation.item())
            stats.dg_statistics_generation = float(statistics_generation.item())
            stats.dg_publication_generation_mismatch = float(
                not torch.equal(weight_generation, statistics_generation)
            )
        stats.dg_valid_minibatch_unused_unit_count = var.additional_stats[
            "encoder_batch_unused_count"
        ].detach().float()
        stats.dg_valid_minibatch_unused_unit_fraction = (
            var.additional_stats["encoder_batch_unused_count"] / float(self.cfg.Hippo_n_feature)
        ).detach().float()
        duty_min, duty_mean, duty_max, usage_entropy = dg_usage_metrics(dg_active)
        stats.dg_unit_duty_cycle_min = duty_min.detach().float()
        stats.dg_unit_duty_cycle_mean = duty_mean.detach().float()
        stats.dg_unit_duty_cycle_max = duty_max.detach().float()
        stats.dg_usage_entropy = usage_entropy.detach().float()
        if getattr(self.cfg, "hrl_controllable_graph", False):
            hrl = self._hrl_state_from_rnn(var.mb.rnn_states)
            valid_hrl = var.mb.valids.bool()
            hrl_dg_active = dg_active
            if valid_hrl.any():
                hrl = hrl[valid_hrl]
                hrl_rewards = intrinsic_rewards[valid_hrl]
                hrl_dg_active = hrl_dg_active[valid_hrl]
            else:
                hrl_rewards = intrinsic_rewards
            layout = self._hrl_layout()
            control_correct = var.mb["hrl_control_correct_outcome"].bool() & valid_hrl
            control_wrong = var.mb["hrl_control_wrong_outcome"].bool() & valid_hrl
            control_timeout = var.mb["hrl_control_target_timeout"].bool() & valid_hrl
            control_exploration_timeout = var.mb["hrl_control_exploration_timeout"].bool() & valid_hrl
            control_censored = var.mb["hrl_control_censored"].bool() & valid_hrl
            control_completed = var.mb["hrl_control_completed"].bool() & valid_hrl
            control_magnitude = var.mb["hrl_control_reward_magnitude"].float()
            control_source = var.mb["hrl_control_source"].long()
            control_command = var.mb["hrl_control_command_target"].long()
            control_outcome = var.mb["hrl_control_outcome_id"].long()
            outcome_event = control_correct | control_wrong
            alternate = (control_command + 1).remainder(layout.n_nodes)
            alternate = torch.where(alternate == control_source, (alternate + 1).remainder(layout.n_nodes), alternate)
            shuffled_success = outcome_event & (control_outcome == alternate)
            stats.hrl_control_correct_count = control_correct.sum().detach().float()
            stats.hrl_control_wrong_count = control_wrong.sum().detach().float()
            stats.hrl_control_timeout_count = control_timeout.sum().detach().float()
            stats.hrl_control_censored_count = control_censored.sum().detach().float()
            stats.hrl_control_completed_count = control_completed.sum().detach().float()
            stats.hrl_first_outcome_commanded_numerator = control_correct.sum().detach().float()
            stats.hrl_first_outcome_commanded_event_count = outcome_event.sum().detach().float()
            stats.hrl_first_outcome_shuffled_numerator = shuffled_success.sum().detach().float()
            stats.hrl_first_outcome_shuffled_event_count = outcome_event.sum().detach().float()
            control_elapsed = var.mb["hrl_control_elapsed"].float()
            stats.hrl_control_correct_elapsed_mean = finite_masked_mean(
                control_elapsed, control_correct
            ).detach().float()
            stats.hrl_control_wrong_elapsed_mean = finite_masked_mean(
                control_elapsed, control_wrong
            ).detach().float()
            stats.hrl_control_reward_magnitude_correct_mean = finite_masked_mean(
                control_magnitude, control_correct
            ).detach().float()
            stats.hrl_control_reward_magnitude_wrong_mean = finite_masked_mean(
                control_magnitude, control_wrong
            ).detach().float()
            target = hrl[:, layout.target].long() - 1
            normal_target = (target >= 0) & (target < layout.n_nodes)
            exploring = target == layout.n_nodes
            stats.hrl_active_target_frac = normal_target.float().mean().detach().float()
            stats.hrl_active_option_frac = (normal_target | exploring).float().mean().detach().float()
            stats.hrl_exploration_mode_fraction = exploring.float().mean().detach().float()
            stats.hrl_source_frac = (hrl[:, layout.source] > 0).float().mean().detach().float()
            stats.hrl_target_hit_rate = hrl[:, layout.target_hit].float().mean().detach().float()
            stats.hrl_tctrl_update_rate = hrl[:, layout.tctrl_updated].float().mean().detach().float()
            stats.hrl_option_reset_rate = hrl[:, layout.option_reset].float().mean().detach().float()
            expired = hrl[:, layout.option_expired].float()
            hits = hrl[:, layout.target_hit].float()
            resets = hrl[:, layout.option_reset].float()
            learned_deadline = hrl[:, layout.deadline_learned].float()
            selected_deadline = hrl[:, layout.selected_deadline].float()
            completed_elapsed = hrl[:, layout.completion_elapsed].float()
            if valid_hrl.any():
                exploration_completed = control_exploration_timeout[valid_hrl]
                target_expired = control_timeout[valid_hrl]
                wrong_completed = control_wrong[valid_hrl]
            else:
                exploration_completed = control_exploration_timeout
                target_expired = control_timeout
                wrong_completed = control_wrong
            selected_exploration = (resets > 0) & exploring
            forced_exploration = selected_exploration & target_expired
            selected_target = (resets > 0) & normal_target
            stats.hrl_option_timeout_rate = target_expired.float().mean().detach().float()
            stats.hrl_option_success_fraction = (
                hits.sum()
                / (hits.sum() + target_expired.float().sum() + wrong_completed.float().sum()).clamp_min(1.0)
            ).detach().float()
            stats.hrl_learned_deadline_fraction = (
                learned_deadline.sum() / selected_target.float().sum().clamp_min(1.0)
            ).detach().float()
            stats.hrl_selected_deadline_mean = (
                selected_deadline.sum() / resets.sum().clamp_min(1.0)
            ).detach().float()
            positive_deadline_mean, deadline_selection_fraction = selected_deadline_stats(
                selected_deadline, resets
            )
            stats.hrl_selected_deadline_positive_mean = positive_deadline_mean.detach().float()
            stats.hrl_deadline_selection_fraction = deadline_selection_fraction.detach().float()
            stats.hrl_elapsed_on_hit_mean = (
                (completed_elapsed * hits).sum() / hits.sum().clamp_min(1.0)
            ).detach().float()
            stats.hrl_elapsed_on_timeout_mean = (
                (completed_elapsed * target_expired).sum() / target_expired.float().sum().clamp_min(1.0)
            ).detach().float()
            stats.hrl_exploration_selection_fraction = (
                selected_exploration.float().sum() / resets.sum().clamp_min(1.0)
            ).detach().float()
            stats.hrl_forced_exploration_fraction = (
                forced_exploration.float().sum() / selected_exploration.float().sum().clamp_min(1.0)
            ).detach().float()
            stats.hrl_exploration_completion_rate = exploration_completed.float().mean().detach().float()
            stats.hrl_exploration_elapsed_mean = (
                (-completed_elapsed * exploration_completed).sum()
                / exploration_completed.float().sum().clamp_min(1.0)
            ).detach().float()
            stats.hrl_target_selected_deadline_mean = (
                (selected_deadline * selected_target).sum() / selected_target.float().sum().clamp_min(1.0)
            ).detach().float()
            stats.hrl_exploration_selected_deadline_mean = (
                (selected_deadline * selected_exploration).sum()
                / selected_exploration.float().sum().clamp_min(1.0)
            ).detach().float()
            stats.hrl_exploration_reward_mean = (
                (hrl_rewards * exploring).sum() / exploring.float().sum().clamp_min(1.0)
            ).detach().float()
            stats.hrl_exploration_reward_nonzero_fraction = (
                ((hrl_rewards != 0) & exploring).float().sum() / exploring.float().sum().clamp_min(1.0)
            ).detach().float()

            visits, tctrl, edge_strength = self._hrl_graph_views(hrl)
            visits = visits.float()
            stats.hrl_node_coverage_fraction = (visits > 0).float().mean().detach().float()
            stats.hrl_node_visit_weight_mean = visits.mean().detach().float()
            valid_selected = (resets > 0) & normal_target
            selected_visits = visits.gather(
                1, target.clamp(min=0, max=layout.n_nodes - 1).unsqueeze(1)
            ).squeeze(1)
            stats.hrl_selected_target_visit_mean = (
                (selected_visits * valid_selected).sum() / valid_selected.sum().clamp_min(1)
            ).detach().float()

            tctrl = tctrl.float()
            off_diagonal = ~torch.eye(layout.n_nodes, dtype=torch.bool, device=tctrl.device)
            edge_strength = edge_strength.float()
            confidence_threshold = (
                self.cfg.hrl_edge_confidence_threshold
                if self._uses_policy_graph() or getattr(self.cfg, "hrl_persistent_fast_weights", False)
                else 0.0
            )
            known = (tctrl > 0) & (edge_strength >= confidence_threshold) & off_diagonal.unsqueeze(0)
            if self._uses_policy_graph():
                graph = self._policy_graph()
                edge_reliability = (graph.edge_confidence + 1.0) / (graph.control_attempts + 2.0)
                reliable_2d = edge_reliability >= float(
                    getattr(self.cfg, "hrl_edge_reliability_threshold", 0.5)
                )
                known &= reliable_2d.unsqueeze(0)
                attempted_edges = graph.control_attempts > 0
                observed = graph.node_visits >= float(getattr(self.cfg, "hrl_min_target_visits", 1.0))
                if getattr(self.cfg, "hrl_direct_target_selection", "frontier") == "local_successor":
                    eligible_pairs = (graph.passive_confidence > 0) & off_diagonal
                    local_counts = eligible_pairs.sum(dim=1)
                    stats.hrl_control_local_candidate_pair_count = eligible_pairs.sum().detach().float()
                    stats.hrl_control_local_candidate_source_fraction = (
                        (local_counts > 0).float().mean().detach().float()
                    )
                    stats.hrl_control_local_candidate_count_mean = (
                        local_counts[local_counts > 0].float().mean().detach()
                        if (local_counts > 0).any()
                        else local_counts.float().sum().detach()
                    )
                else:
                    eligible_pairs = observed.unsqueeze(1) & observed.unsqueeze(0) & off_diagonal
                stats.hrl_control_observed_pair_coverage = (
                    (attempted_edges & eligible_pairs).sum() / eligible_pairs.sum().clamp_min(1)
                ).detach().float()
                row_entropies = []
                for source_id in torch.where(observed)[0].tolist():
                    eligible_targets = eligible_pairs[source_id]
                    row_attempts = graph.control_attempts[source_id, eligible_targets]
                    if row_attempts.sum() < 30 or row_attempts.numel() <= 1:
                        continue
                    probabilities = row_attempts / row_attempts.sum()
                    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
                    row_entropies.append(entropy / math.log(row_attempts.numel()))
                stats.hrl_control_command_entropy = (
                    torch.stack(row_entropies).mean().detach().float()
                    if row_entropies
                    else graph.control_attempts.sum().detach().float() * 0.0
                )
                if "hrl_control_command_set_size" in var.mb:
                    command_sizes = var.mb["hrl_control_command_set_size"].float()
                    command_ids = var.mb["hrl_control_command_target"].long()
                    selected = (command_ids >= 0) & (command_ids < layout.n_nodes)
                    stats.hrl_control_behavior_candidate_count_mean = finite_masked_mean(
                        command_sizes, selected & valid_hrl
                    ).detach().float()
                stats.hrl_edge_reliability_mean = (
                    edge_reliability.masked_fill(~attempted_edges, 0).sum()
                    / attempted_edges.sum().clamp_min(1)
                ).detach().float()
                reliable_stats = reliable_graph_statistics(known[0], graph.edge_confidence)
                stats.hrl_reliable_largest_scc = reliable_stats["largest_scc"].detach().float()
                stats.hrl_reliable_reachable_pair_fraction = reliable_stats[
                    "reachable_pair_fraction"
                ].detach().float()
                stats.hrl_reliable_outgoing_node_fraction = reliable_stats[
                    "outgoing_node_fraction"
                ].detach().float()
                stats.hrl_reliable_outgoing_node_count = reliable_stats[
                    "outgoing_node_count"
                ].detach().float()
                stats.hrl_reliable_reciprocal_fraction = reliable_stats[
                    "reciprocal_fraction"
                ].detach().float()
                stats.hrl_reliable_top3_incoming_confidence_share = reliable_stats[
                    "top3_incoming_confidence_share"
                ].detach().float()
                stats.hrl_edge_promotions = float(self._last_graph_rollout_stats.get("promotion_count", 0.0))
                stats.hrl_edge_demotions = float(self._last_graph_rollout_stats.get("demotion_count", 0.0))
                completion_count = max(
                    1.0, float(self._last_graph_rollout_stats.get("completion_count", 0.0))
                )
                stats.hrl_edge_promotion_rate = stats.hrl_edge_promotions / completion_count
                stats.hrl_edge_demotion_rate = stats.hrl_edge_demotions / completion_count
                empirical = graph.edge_confidence / graph.control_attempts.clamp_min(1.0)
                calibration_edges = graph.control_attempts > 0
                stats.hrl_edge_reliability_brier = (
                    (edge_reliability - empirical).square().masked_fill(~calibration_edges, 0).sum()
                    / calibration_edges.sum().clamp_min(1)
                ).detach().float()
            seen = (tctrl > 0) & off_diagonal.unsqueeze(0)
            stats.hrl_known_edge_fraction = (
                known.float().sum() / float(tctrl.size(0) * layout.n_nodes * (layout.n_nodes - 1))
            ).detach().float()
            stats.hrl_forgotten_edge_fraction = (
                (seen & ~known).float().sum() / float(tctrl.size(0) * layout.n_nodes * (layout.n_nodes - 1))
            ).detach().float()
            stats.hrl_known_controllability_time_mean = (
                tctrl.masked_fill(~known, 0).sum() / known.sum().clamp_min(1)
            ).detach().float()
            stats.hrl_edge_confidence_mean = (
                edge_strength.masked_fill(~off_diagonal.unsqueeze(0), 0).sum()
                / float(tctrl.size(0) * layout.n_nodes * (layout.n_nodes - 1))
            ).detach().float()
            target_index = target.clamp(min=0, max=layout.n_nodes - 1)
            actual_activation = normal_target & hrl_dg_active.gather(1, target_index.unsqueeze(1)).squeeze(1)
            shuffled_target = target_index.roll(1)
            shuffled_valid = normal_target & normal_target.roll(1)
            shuffled_activation = shuffled_valid & hrl_dg_active.gather(1, shuffled_target.unsqueeze(1)).squeeze(1)
            actual_rate = actual_activation.float().sum() / normal_target.sum().clamp_min(1)
            shuffled_rate = shuffled_activation.float().sum() / shuffled_valid.sum().clamp_min(1)
            stats.hrl_target_hit_numerator = actual_activation.float().sum().detach()
            stats.hrl_target_hit_event_count = normal_target.float().sum().detach()
            stats.hrl_shuffled_hit_numerator = shuffled_activation.float().sum().detach()
            stats.hrl_shuffled_hit_event_count = shuffled_valid.float().sum().detach()
            stats.hrl_target_hit_lift = (actual_rate / shuffled_rate.clamp_min(1e-6)).detach().float()

            if self._uses_topological_manager():
                topo = self._topological_state_from_rnn(var.mb.rnn_states)
                if valid_hrl.any():
                    topo = topo[valid_hrl]
                topo_layout = TopologicalStateLayout(layout.n_nodes)
                graph = self._policy_graph()
                passive_known = graph.passive_confidence >= float(self.cfg.hrl_passive_edge_confidence_threshold)
                passive_known.fill_diagonal_(False)
                reliability = (graph.edge_confidence + 1.0) / (graph.control_attempts + 2.0)
                reliable = (
                    (graph.tctrl > 0)
                    & (graph.edge_confidence >= float(self.cfg.hrl_edge_confidence_threshold))
                    & (reliability >= float(getattr(self.cfg, "hrl_edge_reliability_threshold", 0.5)))
                )
                reliable.fill_diagonal_(False)
                candidate = passive_known & ~reliable
                denominator = float(layout.n_nodes * (layout.n_nodes - 1))
                stats.hrl_passive_updates = float(self._last_graph_rollout_stats.get("passive_update_count", 0.0))
                stats.hrl_passive_known_edge_fraction = passive_known.float().sum().div(denominator).detach().float()
                stats.hrl_passive_candidate_edge_fraction = candidate.float().sum().div(denominator).detach().float()
                stats.hrl_edge_candidate_rate = stats.hrl_passive_candidate_edge_fraction
                observed = graph.passive_confidence > 0
                stats.hrl_passive_time_mean = (
                    graph.passive_time.masked_fill(~observed, 0).sum() / observed.sum().clamp_min(1)
                ).detach().float()
                stats.hrl_passive_path_length_mean = (
                    graph.passive_path_length.masked_fill(~observed, 0).sum() / observed.sum().clamp_min(1)
                ).detach().float()
                stats.hrl_passive_reject_nonexclusive_rate = topo[:, topo_layout.passive_reject_nonexclusive].mean().detach().float()
                stats.hrl_passive_reject_time_rate = topo[:, topo_layout.passive_reject_time].mean().detach().float()
                stats.hrl_passive_reject_path_rate = topo[:, topo_layout.passive_reject_path].mean().detach().float()
                stats.hrl_passive_reject_motion_rate = topo[:, topo_layout.passive_reject_motion].mean().detach().float()

                frontier_selected = topo[:, topo_layout.frontier_selected]
                frontier_reached = topo[:, topo_layout.final_reached]
                stats.hrl_frontier_score_mean = (
                    (topo[:, topo_layout.frontier_score] * frontier_selected).sum()
                    / frontier_selected.sum().clamp_min(1)
                ).detach().float()
                stats.hrl_frontier_selection_rate = frontier_selected.mean().detach().float()
                stats.hrl_frontier_attempts = float(self._last_graph_rollout_stats.get("frontier_attempt_count", 0.0))
                stats.hrl_frontier_discoveries = float(self._last_graph_rollout_stats.get("frontier_discovery_count", 0.0))
                stats.hrl_frontier_yield = (
                    graph.frontier_discoveries.sum() / graph.frontier_attempts.sum().clamp_min(1e-6)
                ).detach().float()
                stats.hrl_frontier_reached_fraction = (
                    frontier_reached.sum() / frontier_selected.sum().clamp_min(1)
                ).detach().float()

                route = topo[:, topo_layout.route_available]
                stats.hrl_planning_route_available_rate = route.mean().detach().float()
                stats.hrl_planning_hop_count_mean = finite_masked_mean(
                    topo[:, topo_layout.plan_hops], route > 0
                ).detach().float()
                # A target hit can belong to direct targeting, validation, a
                # return, or a routed waypoint. Keep the generic rate separate
                # from the routed-waypoint diagnostic.
                stats.hrl_planning_target_hit_rate = hits.mean().detach().float()
                waypoint_navigation = (topo[:, topo_layout.mode] == MODE_NAVIGATE) & (
                    topo[:, topo_layout.plan_hops] > 1
                )
                stats.hrl_planning_waypoint_navigation_fraction = waypoint_navigation.float().mean().detach().float()
                stats.hrl_planning_waypoint_step_hit_rate = (
                    (hits * waypoint_navigation).sum() / waypoint_navigation.sum().clamp_min(1)
                ).detach().float()
                stats.hrl_planning_replan_rate = resets.mean().detach().float()
                stats.hrl_planning_final_frontier_reach_rate = frontier_reached.mean().detach().float()

                stats.hrl_validation_queued_edges = candidate.float().sum().detach().float()
                stats.hrl_validation_return_success_rate = topo[:, topo_layout.return_success].mean().detach().float()
                stats.hrl_validation_success_rate = topo[:, topo_layout.validation_success].mean().detach().float()
                stats.hrl_validation_timeout_rate = topo[:, topo_layout.validation_timeout].mean().detach().float()
                _, _, replay_mode = self._behavior_condition_from_states(var.mb.rnn_states)
                if valid_hrl.any():
                    replay_mode = replay_mode[valid_hrl]
                mode = replay_mode.argmax(dim=-1)
                goal_mode = (mode == MODE_NAVIGATE) | (mode == MODE_RETURN)
                stats.hrl_mode_free_fraction = (mode == MODE_EXPLORE).float().mean().detach().float()
                stats.hrl_mode_navigate_fraction = (mode == MODE_NAVIGATE).float().mean().detach().float()
                stats.hrl_mode_goal_fraction = goal_mode.float().mean().detach().float()
                stats.hrl_mode_probe_fraction = (mode == MODE_VALIDATE).float().mean().detach().float()
                stats.hrl_edge_probe_rate = stats.hrl_mode_probe_fraction
                stats.hrl_edge_probe_successes = float(
                    self._last_graph_rollout_stats.get("edge_probe_success_count", 0.0)
                )
                stats.hrl_edge_probe_timeouts = float(
                    self._last_graph_rollout_stats.get("edge_probe_timeout_count", 0.0)
                )
                probe_completions = max(
                    1.0, stats.hrl_edge_probe_successes + stats.hrl_edge_probe_timeouts
                )
                stats.hrl_edge_probe_success_rate = stats.hrl_edge_probe_successes / probe_completions
                stats.hrl_edge_probe_timeout_rate = stats.hrl_edge_probe_timeouts / probe_completions

                segment_path = topo[:, topo_layout.segment_path]
                displacement = torch.sqrt(
                    topo[:, topo_layout.segment_x].square() + topo[:, topo_layout.segment_y].square()
                )
                straightness = displacement / segment_path.clamp_min(1e-6)
                stats.path_length_mean = segment_path.mean().detach().float()
                stats.path_displacement_mean = displacement.mean().detach().float()
                stats.path_straightness_mean = straightness.mean().detach().float()
                mode_masks = {
                    "goal": goal_mode,
                    "free": mode == MODE_EXPLORE,
                    "probe": mode == MODE_VALIDATE,
                }
                for mode_name, mode_mask in mode_masks.items():
                    count = mode_mask.sum().clamp_min(1)
                    mode_hits = hits.bool() & mode_mask
                    stats[f"hrl_{mode_name}_target_hit_rate"] = (
                        mode_hits.sum() / count
                    ).detach().float()
                    stats[f"hrl_{mode_name}_time_to_hit"] = (
                        (completed_elapsed * mode_hits).sum() / mode_hits.sum().clamp_min(1)
                    ).detach().float()
                    stats[f"hrl_{mode_name}_path_length"] = (
                        (segment_path * mode_mask).sum() / count
                    ).detach().float()
                    mode_straightness = (straightness * mode_mask).sum() / count
                    stats[f"hrl_{mode_name}_straightness"] = mode_straightness.detach().float()
                    stats[f"hrl_{mode_name}_loop_fraction"] = (
                        ((straightness < 0.25) & mode_mask).sum() / count
                    ).detach().float()
                stats.geometry_se2_stress = graph.pose_stress.detach().float()
                stats.geometry_valid_landmark_fraction = graph.pose_valid.float().mean().detach().float()
                geometric_candidates = geometric_candidate_edges(
                    graph,
                    float(self.cfg.hrl_edge_confidence_threshold),
                    int(getattr(self.cfg, "hrl_geometry_neighbors", 3)),
                    float(getattr(self.cfg, "hrl_geometry_max_distance", 32.0)),
                )
                stats.geometry_proposed_edge_fraction = geometric_candidates.float().sum().div(denominator).detach().float()
        if self.cfg.encoder_multi_activation_loss:
            stats.encoder_penalty_loss = var.additional_stats["encoder_penalty_loss"].detach().float()
        else:
            stats.encoder_penalty_loss = float(var.additional_stats["encoder_penalty_loss"])
        if self.cfg.encoder_unused_sequence_loss:
            stats.encoder_reward_loss = var.additional_stats["encoder_reward_loss"].detach().float()
        else:
            stats.encoder_reward_loss = float(var.additional_stats["encoder_reward_loss"])
        if self.cfg.encoder_batch_loss:
            stats.batch_reward_loss = var.additional_stats["batch_reward_loss"].detach().float()
        else:
            stats.batch_reward_loss = float(var.additional_stats["batch_reward_loss"])
        if self.cfg.extra_decoder_loss:
            stats.extra_decoder_loss = var.extra_decoder_loss.detach().float()
        else:
            stats.extra_decoder_loss = float(var.extra_decoder_loss)
        encoder_feedback = var.mb.rewards_encoder.float()
        if encoder_feedback.ndim > 1:
            encoder_feedback = encoder_feedback.sum(dim=-1)
        stats.encoder_punishment = encoder_feedback.mean()
        stats.encoder_arrival_credit_loss = var.additional_stats["encoder_arrival_credit_loss"].detach().float()
        stats.encoder_source_credit_loss = var.additional_stats["encoder_source_credit_loss"].detach().float()
        credit_stats = self._last_encoder_credit_stats
        for key in (
            "total", "matchable", "credited", "boundary_dropped", "alignment_failure",
            "invalid_interval", "collisions", "reward_mass", "source_lag_mean", "source_lag_max",
        ):
            stats[f"encoder_credit_{key}"] = float(credit_stats.get(key, 0.0))
        stats.intrinsic_reward_negative_frac = (intrinsic_rewards < 0).float().mean().detach().float()
        stats.ca3_predictor_loss = var.additional_stats["ca3_predictor_loss"].detach().float()
        stats.ca3_predictor_hit_accuracy = var.additional_stats["ca3_predictor_hit_accuracy"].detach().float()
        stats.ca3_predictor_time_mae = var.additional_stats["ca3_predictor_time_mae"].detach().float()
        stats.ca3_predictor_positive_fraction = var.additional_stats[
            "ca3_predictor_positive_fraction"
        ].detach().float()
        for name in (
            "loss",
            "main_loss",
            "control_loss",
            "validation_main_ce",
            "validation_control_ce",
            "validation_state_gain",
            "validation_accuracy",
            "validation_count",
            "scheduled_count",
            "applied_count",
            "boundary_drop_count",
            "replay_match",
        ):
            stats[f"dg_transition_prediction_{name}"] = var.additional_stats[
                f"dg_transition_prediction_{name}"
            ].detach().float()
        stats.hrl_goal_condition_target_valid_fraction = var.additional_stats[
            "goal_condition_target_valid_fraction"
        ].detach().float()
        stats.hrl_goal_condition_action_sensitivity = var.additional_stats[
            "goal_condition_action_sensitivity"
        ].detach().float()
        stats.hrl_goal_condition_action_probability_tv = var.additional_stats[
            "goal_condition_action_probability_tv"
        ].detach().float()
        stats.hrl_goal_condition_value_span = var.additional_stats["goal_condition_value_span"].detach().float()
        stats.hrl_behavior_replay_mismatch = var.additional_stats[
            "behavior_replay_mismatch"
        ].detach().float()
        for branch in ("goal", "free"):
            stats[f"hrl_{branch}_policy_loss"] = var.additional_stats[
                f"{branch}_policy_loss"
            ].detach().float()
            stats[f"hrl_{branch}_value_loss"] = var.additional_stats[
                f"{branch}_value_loss"
            ].detach().float()
            stats[f"hrl_{branch}_entropy_loss"] = var.additional_stats[
                f"{branch}_entropy_loss"
            ].detach().float()
        stats.empirical_her_loss = var.additional_stats["empirical_her_loss"].detach().float()
        stats.empirical_her_policy_loss = var.additional_stats["empirical_her_policy_loss"].detach().float()
        stats.empirical_her_value_loss = var.additional_stats["empirical_her_value_loss"].detach().float()
        stats.empirical_her_behavior_ratio = var.additional_stats["empirical_her_ratio"].detach().float()
        stats.empirical_her_clip_fraction = var.additional_stats["empirical_her_clip_fraction"].detach().float()
        stats.empirical_her_valid_fraction = var.additional_stats["empirical_her_valid_fraction"].detach().float()
        her_stats = self._last_empirical_her_stats
        stats.empirical_her_accepted_segments = her_stats["accepted_segments"]
        stats.empirical_her_skipped_no_endpoint = her_stats["skipped_no_endpoint"]
        stats.empirical_her_skipped_same_source = her_stats["skipped_same_source"]
        stats.empirical_her_segment_length = her_stats["segment_length"]
        stats.empirical_her_positive_fraction = her_stats["positive_fraction"]
        stats.empirical_her_terminal_reward = her_stats["terminal_reward"]
        stats.encoder_population_loss = var.additional_stats["encoder_population_loss"].detach().float()
        stats.encoder_usage_loss = var.additional_stats["encoder_usage_loss"].detach().float()
        stats.encoder_density_loss = var.additional_stats["encoder_density_loss"].detach().float()
        stats.encoder_collision_loss = var.additional_stats["encoder_collision_loss"].detach().float()
        stats.encoder_global_punishment_loss = var.additional_stats["encoder_global_punishment_loss"].detach().float()
        stats.encoder_row_repulsion_loss = var.additional_stats["encoder_row_repulsion_loss"].detach().float()
        stats.encoder_ca3_temporal_exclusion_loss = var.additional_stats[
            "encoder_ca3_temporal_exclusion_loss"
        ].detach().float()
        stats.encoder_path_scatter_loss = var.additional_stats["encoder_path_scatter_loss"].detach().float()
        stats.dg_pre_threshold_mean = var.additional_stats["dg_pre_threshold_mean"].detach().float()
        stats.dg_pre_threshold_above_fraction = var.additional_stats["dg_pre_threshold_above_fraction"].detach().float()
        stats.dg_ca3_conflict_fraction = var.additional_stats["dg_ca3_conflict_fraction"].detach().float()
        stats.dg_ca3_conflicting_activation_fraction = var.additional_stats[
            "dg_ca3_conflicting_activation_fraction"
        ].detach().float()
        stats.dg_ca3_conflict_activity = var.additional_stats["dg_ca3_conflict_activity"].detach().float()
        stats.dg_path_scatter_conflict_fraction = var.additional_stats[
            "dg_path_scatter_conflict_fraction"
        ].detach().float()
        projection = self.actor_critic.encoder.DG_projection
        if hasattr(projection, "recruitment_committed"):
            stats.dg_recruitment_committed_fraction = projection.recruitment_committed.float().mean().detach()
            stats.dg_recruitment_total = projection.recruitment_count.float().detach()
            stats.dg_recruitment_repeat_total = projection.recruitment_repeat_count.float().detach()
            stats.dg_recruitment_tiny_residual_total = projection.recruitment_tiny_residual_count.float().detach()
        stats.dg_recruitment_candidate_count = float(self._last_recruitment_stats["candidate_count"])
        stats.dg_recruitment_silent_endpoint_count = float(self._last_recruitment_stats["silent_endpoint_count"])
        stats.dg_recruitment_rollout_count = float(self._last_recruitment_stats["recruited_count"])
        stats.dg_recruitment_residual_norm = float(self._last_recruitment_stats["residual_norm"])
        stats.dg_recruitment_connected_fraction = float(self._last_recruitment_stats["connected_fraction"])
        stats.dg_recruitment_isolated_fraction = float(self._last_recruitment_stats["isolated_fraction"])
        stats.dg_recruitment_redundant_pair_count = float(self._last_recruitment_stats["redundant_pair_count"])
        stats.dg_recruitment_eligible_vertex_count = float(self._last_recruitment_stats["eligible_vertex_count"])
        stats.dg_recruitment_birth_protected_count = float(self._last_recruitment_stats["birth_protected_count"])
        stats.dg_recruitment_repeat_assignment_count = float(
            self._last_recruitment_stats["repeat_assignment_count"]
        )
        stats.dg_recruitment_isolated_assignment_count = float(
            self._last_recruitment_stats["isolated_assignment_count"]
        )
        stats.dg_recruitment_redundant_assignment_count = float(
            self._last_recruitment_stats["redundant_assignment_count"]
        )
        stats.dg_recruitment_passive_graph_density = float(
            self._last_recruitment_stats["passive_graph_density"]
        )
        stats.dg_recruitment_passive_update_count = float(
            self._last_recruitment_stats["passive_update_count"]
        )
        stats.dg_recruitment_passive_stale_count = float(
            self._last_recruitment_stats["passive_stale_count"]
        )
        stats.dg_recruitment_passive_over_gap_count = float(
            self._last_recruitment_stats["passive_over_gap_count"]
        )
        stats.dg_recruitment_attempt_coverage_fraction = float(
            self._last_recruitment_stats["attempt_coverage_fraction"]
        )
        stats.dg_recruitment_fully_tested_count = float(
            self._last_recruitment_stats["fully_tested_count"]
        )
        stats.dg_recruitment_zero_outdegree_count = float(
            self._last_recruitment_stats["zero_outdegree_count"]
        )
        stats.dg_recruitment_untested_zero_outdegree_count = float(
            self._last_recruitment_stats["untested_zero_outdegree_count"]
        )
        stats.dg_recruitment_bad_source_count = float(
            self._last_recruitment_stats["bad_source_count"]
        )
        stats.dg_recruitment_reliable_out_degree_mean = float(
            self._last_recruitment_stats["reliable_out_degree_mean"]
        )
        stats.dg_recruitment_reliable_outgoing_confidence_mean = float(
            self._last_recruitment_stats["reliable_outgoing_confidence_mean"]
        )
        stats.dg_recruitment_predictive_event_count = float(
            self._last_recruitment_stats["predictive_event_count"]
        )
        stats.dg_recruitment_predictive_context_group_count = float(
            self._last_recruitment_stats["predictive_context_group_count"]
        )
        stats.dg_recruitment_predictive_eligible_count = float(
            self._last_recruitment_stats["predictive_eligible_count"]
        )
        stats.dg_recruitment_predictive_reliability_gap = float(
            self._last_recruitment_stats["predictive_reliability_gap"]
        )
        stats.dg_recruitment_bad_source_assignment_count = float(
            self._last_recruitment_stats["bad_source_assignment_count"]
        )
        stats.dg_recruitment_predictive_assignment_count = float(
            self._last_recruitment_stats["predictive_assignment_count"]
        )
        for key in (
            "active_endpoint_count", "activity_blocked_count", "eligible_victim_endpoint_count",
            "residual_pass_count", "residual_reject_count", "endpoint_active_unit_count",
            "victim_active_count", "predictive_decayed_attempt_mass",
            "predictive_supported_context_count", "predictive_invalidation_mass",
            "predictive_context_coverage_fraction", "eligible_bad_source_endpoint_count",
            "eligible_redundant_endpoint_count", "eligible_predictive_endpoint_count",
            "victim_active_bad_source_count", "victim_active_redundant_count",
            "victim_active_predictive_count",
            "forced_preflight_assignment_count",
        ):
            stats[f"dg_recruitment_{key}"] = float(self._last_recruitment_stats[key])
        eligible_endpoints = max(1.0, self._last_recruitment_stats["eligible_victim_endpoint_count"])
        stats.dg_recruitment_victim_active_fraction = float(
            self._last_recruitment_stats["victim_active_count"] / eligible_endpoints
        )
        for reason in ("bad_source", "redundant", "predictive"):
            reason_eligible = max(
                1.0, self._last_recruitment_stats[f"eligible_{reason}_endpoint_count"]
            )
            stats[f"dg_recruitment_victim_active_{reason}_fraction"] = float(
                self._last_recruitment_stats[f"victim_active_{reason}_count"] / reason_eligible
            )
        endpoint_count = max(1.0, self._last_recruitment_stats["candidate_count"])
        stats.dg_recruitment_endpoint_active_unit_mean = float(
            self._last_recruitment_stats["endpoint_active_unit_count"] / endpoint_count
        )
        residual_pass = max(1.0, self._last_recruitment_stats["residual_pass_count"])
        stats.dg_recruitment_replacement_conversion = float(
            self._last_recruitment_stats["recruited_count"] / residual_pass
        )
        stats.dg_recruitment_goal_adapter_reset_count = float(
            self._last_recruitment_stats["goal_adapter_reset_count"]
        )
        stats.dg_recruitment_goal_adapter_reset_total = float(self._goal_adapter_reset_count)
        decoder = self.actor_critic.decoder
        stats.hrl_goal_decoder_parameter_count = float(
            sum(parameter.numel() for parameter in decoder.parameters())
        )
        target_modulation = getattr(decoder, "target_modulation", None)
        if torch.is_tensor(target_modulation):
            row_norm = target_modulation.detach().norm(dim=-1)
            stats.hrl_goal_condition_film_modulation_norm_mean = row_norm.mean().float()
            stats.hrl_goal_condition_film_modulation_norm_max = row_norm.max().float()
        else:
            stats.hrl_goal_condition_film_modulation_norm_mean = 0.0
            stats.hrl_goal_condition_film_modulation_norm_max = 0.0

        return stats


class OnlineSpatialDefaultLearner(DefaultLearner):
    """Default PPO learner with the same privileged behavior telemetry path."""

    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        super().__init__(cfg, env_info, policy_versions_tensor, policy_id, param_server)
        self._online_spatial = None

    def train(self, batch: TensorDict) -> Optional[Dict]:
        self._maybe_update_cfg()
        self._maybe_load_policy()
        if self._online_spatial is None:
            self._online_spatial = TrainingSpatialTelemetry(
                self.cfg, self.env_info, self.policy_id, self.env_steps, self.actor_critic
            )
        valids = (batch["policy_id"] == self.policy_id) & (
            self.train_step - batch["policy_version"] < self.cfg.max_policy_lag
        )
        self._online_spatial.append_batch(batch, valids)
        stats = super().train(batch)
        if stats is None:
            return stats
        spatial_stats = self._online_spatial.on_env_steps(self.env_steps)
        if spatial_stats:
            stats.setdefault(TRAIN_STATS, {}).update(spatial_stats)
        return stats


def make_hipposlam_learner(
    cfg: Config, env_info: EnvInfo, policy_versions_tensor: Tensor, policy_id: PolicyID, param_server: ParameterServer
) -> BaseLearner:
    if cfg.distance_learning:
        if cfg.double_value:
            return DoubleDistanceLearnerReward(cfg, env_info, policy_versions_tensor, policy_id, param_server)
        else:
            return DistanceLearnerReward(cfg, env_info, policy_versions_tensor, policy_id, param_server)
        # if not cfg.encoder_decoder_share_losses:
        #     if cfg.combined_learning:
        #         log.warn("Using encoder_decoder_share_losses & combined learning at the same time! Choosing DistanceLearnerEncoderDecoderSeparate.")
        #     return DistanceLearnerEncoderDecoderSeparate(cfg, env_info, policy_versions_tensor, policy_id, param_server)
        # elif cfg.combined_learning:
        #     return DistanceLearnerCombined(cfg, env_info, policy_versions_tensor, policy_id, param_server)
        # else:
        #     return DistanceLearnerSimple(cfg, env_info, policy_versions_tensor, policy_id, param_server)
    else:
        if cfg.rec_distances:
            return DistanceRecorder(cfg, env_info, policy_versions_tensor, policy_id, param_server)
        else:
            learner_cls = OnlineSpatialDefaultLearner if getattr(cfg, "online_spatial_telemetry", False) else DefaultLearner
            return learner_cls(cfg, env_info, policy_versions_tensor, policy_id, param_server)
