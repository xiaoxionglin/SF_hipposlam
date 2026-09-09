from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class HRLStateLayout:
    """Packed HRL state with a contiguous terminal-persistent graph suffix."""

    n_nodes: int

    @property
    def target(self) -> int:
        return 0

    @property
    def source(self) -> int:
        return 1

    @property
    def age(self) -> int:
        return 2

    @property
    def countdown(self) -> int:
        return 3

    @property
    def diag_start(self) -> int:
        return 4

    @property
    def active_dg(self) -> int:
        return self.diag_start

    @property
    def target_hit(self) -> int:
        return self.diag_start + 1

    @property
    def tctrl_updated(self) -> int:
        return self.diag_start + 2

    @property
    def option_reset(self) -> int:
        return self.diag_start + 3

    @property
    def multi_activation(self) -> int:
        return self.diag_start + 4

    @property
    def option_expired(self) -> int:
        return self.diag_start + 5

    @property
    def deadline_learned(self) -> int:
        return self.diag_start + 6

    @property
    def selected_deadline(self) -> int:
        return self.diag_start + 7

    @property
    def completion_elapsed(self) -> int:
        return self.diag_start + 8

    @property
    def persistent_start(self) -> int:
        return self.diag_start + 9

    @property
    def visits_start(self) -> int:
        return self.persistent_start

    @property
    def visits_end(self) -> int:
        return self.visits_start + self.n_nodes

    @property
    def tctrl_start(self) -> int:
        return self.visits_end

    @property
    def tctrl_end(self) -> int:
        return self.tctrl_start + self.n_nodes * self.n_nodes

    @property
    def edge_strength_start(self) -> int:
        return self.tctrl_end

    @property
    def edge_strength_end(self) -> int:
        return self.edge_strength_start + self.n_nodes * self.n_nodes

    # Compatibility aliases for existing diagnostics and old graph snapshots.
    @property
    def tctrl_count_start(self) -> int:
        return self.edge_strength_start

    @property
    def tctrl_count_end(self) -> int:
        return self.edge_strength_end

    @property
    def persistent_size(self) -> int:
        return self.edge_strength_end - self.persistent_start

    @property
    def size(self) -> int:
        return self.edge_strength_end


def hrl_state_size(n_nodes: int) -> int:
    return HRLStateLayout(n_nodes).size


def hrl_persistent_state_size(n_nodes: int) -> int:
    return HRLStateLayout(n_nodes).persistent_size


def hrl_option_state_size(n_nodes: int) -> int:
    """Transient option state used when graph fast weights live in the policy."""
    return HRLStateLayout(n_nodes).persistent_start + 1


def initial_hrl_state(batch_size: int, n_nodes: int, device=None, dtype=None) -> Tensor:
    return torch.zeros(batch_size, hrl_state_size(n_nodes), device=device, dtype=dtype)


def split_hrl_state(rnn_states: Tensor, base_state_size: int, n_nodes: int) -> tuple[Tensor, Tensor]:
    layout = HRLStateLayout(n_nodes)
    expected = base_state_size + layout.size
    if rnn_states.size(-1) != expected:
        raise RuntimeError(f"Expected rnn state size {expected}, got {rnn_states.size(-1)}")
    return rnn_states[..., :base_state_size], rnn_states[..., base_state_size:expected]


def split_hrl_option_state(rnn_states: Tensor, base_state_size: int, n_nodes: int) -> tuple[Tensor, Tensor]:
    """Split the CA3 state from the compact per-stream option state."""
    option_size = hrl_option_state_size(n_nodes)
    expected = base_state_size + option_size
    if rnn_states.size(-1) != expected:
        raise RuntimeError(f"Expected rnn state size {expected}, got {rnn_states.size(-1)}")
    return rnn_states[..., :base_state_size], rnn_states[..., base_state_size:expected]


def _decode_id(stored: Tensor) -> Tensor:
    return stored.long() - 1


def _encode_id(idx: Tensor, dtype: torch.dtype) -> Tensor:
    return (idx + 1).to(dtype=dtype)


def exploration_mode_mask(hrl_state: Tensor, n_nodes: int) -> Tensor:
    """Return where the stored manager action is the reserved exploration option."""
    layout = HRLStateLayout(n_nodes)
    return _decode_id(hrl_state[..., layout.target]) == n_nodes


def current_dg_from_activity(dg_activity: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    active = dg_activity > 0
    n_active = active.sum(dim=-1)
    has_active = n_active > 0
    dg_id = dg_activity.argmax(dim=-1)
    dg_id = torch.where(has_active, dg_id, torch.full_like(dg_id, -1))
    return dg_id, has_active, n_active


def source_from_trace(sequence_core: Tensor) -> Tensor:
    """Return the DG with the most recent occupied CA3 slot, or -1 if empty."""
    occupied = sequence_core != 0
    n_slots = sequence_core.size(-1)
    slot = torch.arange(n_slots, device=sequence_core.device)
    age = torch.where(occupied, slot, n_slots).amin(dim=-1)
    source = age.argmin(dim=-1)
    return torch.where(age.amin(dim=-1) < n_slots, source, torch.full_like(source, -1))


def target_one_hot(hrl_state: Tensor, n_nodes: int) -> Tensor:
    layout = HRLStateLayout(n_nodes)
    target = _decode_id(hrl_state[:, layout.target])
    valid_target = (target >= 0) & (target < n_nodes)
    safe_target = target.clamp(min=0, max=n_nodes - 1)
    one_hot = F.one_hot(safe_target, num_classes=n_nodes).to(dtype=hrl_state.dtype, device=hrl_state.device)
    return one_hot * valid_target.unsqueeze(-1).to(dtype=hrl_state.dtype)


def option_target_one_hot(option_state: Tensor, n_nodes: int) -> Tensor:
    """Decode the behavior target already stored in a compact option state."""
    layout = HRLStateLayout(n_nodes)
    target = _decode_id(option_state[:, layout.target])
    valid_target = (target >= 0) & (target < n_nodes)
    safe_target = target.clamp(min=0, max=n_nodes - 1)
    one_hot = F.one_hot(safe_target, num_classes=n_nodes).to(dtype=option_state.dtype, device=option_state.device)
    return one_hot * valid_target.unsqueeze(-1).to(dtype=option_state.dtype)


def _graph_views(hrl_state: Tensor, layout: HRLStateLayout) -> tuple[Tensor, Tensor, Tensor]:
    visits = hrl_state[:, layout.visits_start : layout.visits_end]
    tctrl = hrl_state[:, layout.tctrl_start : layout.tctrl_end].view(-1, layout.n_nodes, layout.n_nodes)
    strength = hrl_state[:, layout.edge_strength_start : layout.edge_strength_end].view(
        -1, layout.n_nodes, layout.n_nodes
    )
    return visits, tctrl, strength


def controllability_distances(
    hrl_state: Tensor,
    layout: HRLStateLayout,
    edge_confidence_threshold: float = 0.0,
) -> Tensor:
    """Return shortest controllable travel times through currently credible edges."""
    _, tctrl, strength = _graph_views(hrl_state, layout)
    known = (tctrl > 0) & (strength >= edge_confidence_threshold)
    infinity = torch.full_like(tctrl, torch.inf)
    distances = torch.where(known, tctrl, infinity)
    diagonal = torch.eye(layout.n_nodes, device=hrl_state.device, dtype=torch.bool).unsqueeze(0)
    distances = torch.where(diagonal, torch.zeros_like(distances), distances)
    for intermediate in range(layout.n_nodes):
        via_intermediate = distances[:, :, intermediate].unsqueeze(-1) + distances[:, intermediate, :].unsqueeze(1)
        distances = torch.minimum(distances, via_intermediate)
    return distances


def select_target_for_layout(
    hrl_state: Tensor,
    source: Tensor,
    layout: HRLStateLayout,
    excluded_target: Tensor | None = None,
    min_visits: float = 1.0,
    edge_confidence_threshold: float = 0.0,
) -> Tensor:
    visits, _, _ = _graph_views(hrl_state, layout)
    node_ids = torch.arange(layout.n_nodes, device=hrl_state.device)
    eligible = visits >= min_visits
    eligible = eligible & (source.unsqueeze(-1) != node_ids)
    if excluded_target is not None:
        eligible = eligible & (excluded_target.unsqueeze(-1) != node_ids)

    distances = controllability_distances(hrl_state, layout, edge_confidence_threshold)
    batch_idx = torch.arange(hrl_state.size(0), device=hrl_state.device)
    safe_source = source.clamp(min=0, max=layout.n_nodes - 1)
    reachable = torch.isfinite(distances[batch_idx, safe_source])
    reachable = reachable & (source >= 0).unsqueeze(-1)
    reachable_candidates = eligible & reachable
    use_reachable = reachable_candidates.any(dim=-1, keepdim=True)
    candidates = torch.where(use_reachable, reachable_candidates, eligible)

    score = visits.masked_fill(~candidates, torch.inf)
    has_candidate = candidates.any(dim=-1)
    selected = score.argmin(dim=-1)
    return torch.where(has_candidate, selected, torch.full_like(selected, -1))


def option_deadline(
    hrl_state: Tensor,
    source: Tensor,
    target: Tensor,
    layout: HRLStateLayout,
    fallback_horizon: int,
    margin_ratio: float,
    margin_steps: int,
    edge_confidence_threshold: float = 0.0,
) -> tuple[Tensor, Tensor]:
    distances = controllability_distances(hrl_state, layout, edge_confidence_threshold)
    batch_idx = torch.arange(hrl_state.size(0), device=hrl_state.device)
    safe_source = source.clamp(min=0, max=layout.n_nodes - 1)
    safe_target = target.clamp(min=0, max=layout.n_nodes - 1)
    cost = distances[batch_idx, safe_source, safe_target]
    known = torch.isfinite(cost) & (source >= 0) & (target >= 0)
    learned = torch.ceil(cost * (1.0 + margin_ratio)) + float(margin_steps)
    deadline = torch.where(known, learned.clamp_min(1.0), torch.full_like(cost, float(fallback_horizon)))
    return deadline, known


def update_hrl_state(
    prev_hrl_state: Tensor,
    dg_activity: Tensor,
    prev_sequence_core: Tensor,
    fallback_horizon: int,
    timeout_margin_ratio: float = 0.20,
    timeout_margin_steps: int = 2,
    min_target_visits: float = 1.0,
    persistent_fast_weights: bool = False,
    fast_weight_half_life_options: float = 10000.0,
    edge_confidence_threshold: float = 0.5,
    exploration_mode: bool = False,
    manager_exploration_probability: float = 0.0,
    exploration_horizon: int = 64,
) -> tuple[Tensor, Tensor]:
    layout = HRLStateLayout(dg_activity.size(-1))
    n_nodes = layout.n_nodes
    state = prev_hrl_state.clone()
    state[:, layout.diag_start : layout.persistent_start] = 0

    current_dg, has_active, n_active = current_dg_from_activity(dg_activity.detach())
    state[:, layout.active_dg] = _encode_id(current_dg, state.dtype)
    state[:, layout.multi_activation] = (n_active > 1).to(dtype=state.dtype)

    if not 0.0 <= manager_exploration_probability <= 1.0:
        raise ValueError("manager_exploration_probability must be in [0, 1]")
    if exploration_mode and exploration_horizon <= 0:
        raise ValueError("exploration_horizon must be positive")

    target = _decode_id(state[:, layout.target])
    source = _decode_id(state[:, layout.source])
    age = state[:, layout.age]
    countdown = state[:, layout.countdown]
    normal_target = (target >= 0) & (target < n_nodes)
    exploring = exploration_mode & (target == n_nodes)
    has_option = normal_target | exploring
    hit = has_active & normal_target & (current_dg == target)
    elapsed = age + 1.0
    expired = (~hit) & has_option & (countdown <= 1.0)
    target_expired = expired & normal_target
    exploration_expired = expired & exploring
    reset = hit | expired | (~has_option)

    visits, tctrl, edge_strength = _graph_views(state, layout)
    if persistent_fast_weights and reset.any():
        if fast_weight_half_life_options <= 0:
            raise ValueError("fast_weight_half_life_options must be positive")
        decay = math.pow(0.5, 1.0 / fast_weight_half_life_options)
        visits[reset] *= decay
        edge_strength[reset] *= decay

    if has_active.any():
        visits[has_active] = visits[has_active].scatter_add(
            1,
            current_dg[has_active].unsqueeze(-1),
            torch.ones((has_active.sum(), 1), device=state.device, dtype=state.dtype),
        )

    updated = torch.zeros_like(hit)
    intended_success = hit & (source >= 0) & (current_dg != source)
    if intended_success.any():
        batch_idx = torch.arange(state.size(0), device=state.device)[intended_success]
        src_idx = source[intended_success]
        dst_idx = current_dg[intended_success]
        old_strength = edge_strength[batch_idx, src_idx, dst_idx]
        old_time = tctrl[batch_idx, src_idx, dst_idx]
        elapsed_success = elapsed[intended_success]
        if persistent_fast_weights:
            new_strength = old_strength + 1.0
            tctrl[batch_idx, src_idx, dst_idx] = (old_strength * old_time + elapsed_success) / new_strength
            edge_strength[batch_idx, src_idx, dst_idx] = new_strength
            updated[intended_success] = True
        else:
            new_strength = old_strength + 1.0
            is_better = (old_time <= 0) | (elapsed_success < old_time)
            tctrl[batch_idx, src_idx, dst_idx] = torch.where(is_better, elapsed_success, old_time)
            edge_strength[batch_idx, src_idx, dst_idx] = new_strength
            updated[intended_success] = is_better

    if reset.any():
        fallback_source = source_from_trace(prev_sequence_core.detach())
        new_source = torch.where(has_active, current_dg, fallback_source)
        old_target = target.clone()
        excluded = torch.where(target_expired, old_target, torch.full_like(old_target, -1))
        selected = select_target_for_layout(
            state[reset],
            new_source[reset],
            layout,
            excluded_target=excluded[reset],
            min_visits=min_target_visits,
            edge_confidence_threshold=edge_confidence_threshold if persistent_fast_weights else 0.0,
        )
        if exploration_mode:
            random_exploration = torch.rand(selected.shape, device=selected.device) < float(
                manager_exploration_probability
            )
            choose_exploration = target_expired[reset] | random_exploration
            selected = torch.where(choose_exploration, torch.full_like(selected, n_nodes), selected)
        selected_valid = (selected >= 0) & (selected < n_nodes)
        selected_exploration = exploration_mode & (selected == n_nodes)
        deadline = torch.zeros_like(selected, dtype=state.dtype)
        deadline_learned = torch.zeros_like(selected_valid)
        if selected_valid.any():
            valid_deadline, valid_learned = option_deadline(
                state[reset][selected_valid],
                new_source[reset][selected_valid],
                selected[selected_valid],
                layout,
                fallback_horizon,
                timeout_margin_ratio,
                timeout_margin_steps,
                edge_confidence_threshold if persistent_fast_weights else 0.0,
            )
            deadline[selected_valid] = valid_deadline
            deadline_learned[selected_valid] = valid_learned
        deadline[selected_exploration] = float(exploration_horizon)
        target[reset] = selected
        source[reset] = new_source[reset]
        age[reset] = 0.0
        countdown[reset] = deadline
        state[reset, layout.option_reset] = 1.0
        state[reset, layout.deadline_learned] = deadline_learned.to(dtype=state.dtype)
        state[reset, layout.selected_deadline] = deadline
        completed = (hit | expired)[reset]
        signed_elapsed = torch.where(exploration_expired[reset], -elapsed[reset], elapsed[reset])
        state[reset, layout.completion_elapsed] = torch.where(
            completed, signed_elapsed, torch.zeros_like(elapsed[reset])
        )

    keep = ~reset
    age[keep] = age[keep] + 1.0
    countdown[keep] = torch.clamp(countdown[keep] - 1.0, min=0.0)

    state[:, layout.target] = _encode_id(target, state.dtype)
    state[:, layout.source] = _encode_id(source, state.dtype)
    state[:, layout.age] = age
    state[:, layout.countdown] = countdown
    state[:, layout.target_hit] = hit.to(dtype=state.dtype)
    state[:, layout.tctrl_updated] = updated.to(dtype=state.dtype)
    state[:, layout.option_expired] = expired.to(dtype=state.dtype)
    return state, target_one_hot(state, n_nodes)


class PolicyControllableGraph(nn.Module):
    """Policy-scoped Hebbian controllability memory stored as model buffers."""

    def __init__(self, n_nodes: int):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.register_buffer("node_visits", torch.zeros(self.n_nodes))
        self.register_buffer("tctrl", torch.zeros(self.n_nodes, self.n_nodes))
        self.register_buffer("edge_confidence", torch.zeros(self.n_nodes, self.n_nodes))
        self.register_buffer("control_attempts", torch.zeros(self.n_nodes, self.n_nodes))
        self.register_buffer("passive_confidence", torch.zeros(self.n_nodes, self.n_nodes))
        self.register_buffer("passive_time", torch.zeros(self.n_nodes, self.n_nodes))
        self.register_buffer("passive_path_length", torch.zeros(self.n_nodes, self.n_nodes))
        self.register_buffer("passive_dx", torch.zeros(self.n_nodes, self.n_nodes))
        self.register_buffer("passive_dy", torch.zeros(self.n_nodes, self.n_nodes))
        self.register_buffer("passive_dtheta_sin", torch.zeros(self.n_nodes, self.n_nodes))
        self.register_buffer("passive_dtheta_cos", torch.zeros(self.n_nodes, self.n_nodes))
        self.register_buffer("frontier_attempts", torch.zeros(self.n_nodes))
        self.register_buffer("frontier_discoveries", torch.zeros(self.n_nodes))
        self.register_buffer("landmark_pose", torch.zeros(self.n_nodes, 3))
        self.register_buffer("pose_valid", torch.zeros(self.n_nodes, dtype=torch.bool))
        self.register_buffer("pose_stress", torch.zeros(()))
        self.register_buffer("representation_generation", torch.zeros((), dtype=torch.int64))
        # Cumulative outcomes evaluated against the graph as it existed before
        # each learner batch. These are diagnostic-only and never condition the
        # policy or graph updates.
        for name in (
            "prospective_attempts",
            "prospective_successes",
            "prospective_probability_sum",
            "prospective_brier_sum",
            "prospective_timing_count",
            "prospective_timing_sum",
            "prospective_predicted_timing_sum",
            "prospective_timing_absolute_error_sum",
        ):
            self.register_buffer(name, torch.zeros(self.n_nodes, self.n_nodes))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for name in (
            "control_attempts",
            "passive_confidence",
            "passive_time",
            "passive_path_length",
            "passive_dx",
            "passive_dy",
            "passive_dtheta_sin",
            "passive_dtheta_cos",
            "frontier_attempts",
            "frontier_discoveries",
            "landmark_pose",
            "pose_valid",
            "pose_stress",
            "representation_generation",
            "prospective_attempts",
            "prospective_successes",
            "prospective_probability_sum",
            "prospective_brier_sum",
            "prospective_timing_count",
            "prospective_timing_sum",
            "prospective_predicted_timing_sum",
            "prospective_timing_absolute_error_sum",
        ):
            state_dict.setdefault(prefix + name, getattr(self, name).clone())
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.no_grad()
    def invalidate_node(self, node: int) -> None:
        """Forget graph evidence tied to a reassigned DG representation."""
        node = int(node)
        if node < 0 or node >= self.n_nodes:
            raise IndexError(f"DG node {node} is outside [0, {self.n_nodes})")
        self.node_visits[node] = 0
        self.tctrl[node, :] = 0
        self.tctrl[:, node] = 0
        self.edge_confidence[node, :] = 0
        self.edge_confidence[:, node] = 0
        for matrix in (
            self.control_attempts,
            self.passive_confidence,
            self.passive_time,
            self.passive_path_length,
            self.passive_dx,
            self.passive_dy,
            self.passive_dtheta_sin,
            self.passive_dtheta_cos,
            self.prospective_attempts,
            self.prospective_successes,
            self.prospective_probability_sum,
            self.prospective_brier_sum,
            self.prospective_timing_count,
            self.prospective_timing_sum,
            self.prospective_predicted_timing_sum,
            self.prospective_timing_absolute_error_sum,
        ):
            matrix[node, :] = 0
            matrix[:, node] = 0
        self.frontier_attempts[node] = 0
        self.frontier_discoveries[node] = 0
        self.landmark_pose[node] = 0
        self.pose_valid[node] = False
        self.representation_generation.add_(1)

    def expanded_state(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        """Materialize a read-only snapshot in the legacy packed graph layout."""
        layout = HRLStateLayout(self.n_nodes)
        state = torch.zeros(batch_size, layout.size, dtype=dtype, device=device)
        state[:, layout.visits_start : layout.visits_end] = self.node_visits.to(device=device, dtype=dtype)
        state[:, layout.tctrl_start : layout.tctrl_end] = self.tctrl.to(device=device, dtype=dtype).flatten()
        state[:, layout.edge_strength_start : layout.edge_strength_end] = self.edge_confidence.to(
            device=device, dtype=dtype
        ).flatten()
        return state

    @torch.no_grad()
    def update_from_option_rollout(
        self,
        option_states: Tensor,
        valid_steps: Tensor,
        half_life_options: float,
        confidence_threshold: float = 0.5,
        reliability_threshold: float = 0.5,
    ) -> dict[str, Tensor]:
        """Apply a rollout's accepted events exactly once in flattened order."""
        if half_life_options <= 0:
            raise ValueError("fast_weight_half_life_options must be positive")
        before_reliability = (self.edge_confidence + 1.0) / (self.control_attempts + 2.0)
        before_known = (
            (self.tctrl > 0)
            & (self.edge_confidence >= float(confidence_threshold))
            & (before_reliability >= float(reliability_threshold))
        )
        before_known.fill_diagonal_(False)
        layout = HRLStateLayout(self.n_nodes)
        prev = option_states[:, :-1].reshape(-1, option_states.shape[-1])
        nxt = option_states[:, 1:].reshape(-1, option_states.shape[-1])
        valid = valid_steps.bool().reshape(-1)
        if valid.numel() != prev.shape[0]:
            raise ValueError("valid_steps must align with option rollout transitions")

        generation = self.representation_generation.to(device=option_states.device, dtype=option_states.dtype)
        generation_matches = (prev[:, layout.persistent_start] == generation) & (
            nxt[:, layout.persistent_start] == generation
        )
        valid = valid & generation_matches

        hit = (nxt[:, layout.target_hit] > 0) & valid
        unsuccessful = (nxt[:, layout.option_expired] > 0) & valid
        source = prev[:, layout.source].long() - 1
        target = prev[:, layout.target].long() - 1
        normal_target = (target >= 0) & (target < self.n_nodes)
        exploration_target = target == self.n_nodes
        negative_elapsed = nxt[:, layout.completion_elapsed] < 0
        wrong_outcome = unsuccessful & normal_target & negative_elapsed
        exploration_timeout = unsuccessful & (exploration_target | (~normal_target & negative_elapsed))
        target_timeout = unsuccessful & normal_target & (~negative_elapsed)
        timeout = target_timeout | exploration_timeout
        completed = hit | wrong_outcome | timeout
        future_events = torch.flip(torch.cumsum(torch.flip(completed.to(torch.int64), (0,)), 0), (0,))
        future_events = future_events - completed.to(torch.int64)
        gamma_future = torch.pow(
            torch.as_tensor(0.5, device=self.node_visits.device),
            future_events.to(dtype=self.node_visits.dtype) / float(half_life_options),
        )
        gamma_total = torch.pow(
            torch.as_tensor(0.5, device=self.node_visits.device),
            completed.sum().to(dtype=self.node_visits.dtype) / float(half_life_options),
        )

        self.node_visits.mul_(gamma_total)
        self.edge_confidence.mul_(gamma_total)
        self.control_attempts.mul_(gamma_total)
        self.passive_confidence.mul_(gamma_total)
        self.frontier_attempts.mul_(gamma_total)
        self.frontier_discoveries.mul_(gamma_total)
        active = nxt[:, layout.active_dg].long() - 1
        active_mask = valid & (active >= 0) & (active < self.n_nodes)
        if active_mask.any():
            self.node_visits.scatter_add_(0, active[active_mask], gamma_future[active_mask])

        attempted = (hit | wrong_outcome | target_timeout) & (source >= 0) & (source < self.n_nodes)
        attempted = attempted & (target >= 0) & (target < self.n_nodes) & (source != target)
        prospective = attempted.clone()
        if prospective.any():
            prospective = (
                prospective & before_known[source.clamp(0, self.n_nodes - 1), target.clamp(0, self.n_nodes - 1)]
            )
        if prospective.any():
            prospective_indices = source[prospective] * self.n_nodes + target[prospective]
            outcomes = hit[prospective].to(dtype=self.node_visits.dtype)
            probabilities = before_reliability[source[prospective], target[prospective]].to(
                dtype=self.node_visits.dtype
            )
            ones = torch.ones_like(outcomes)
            self.prospective_attempts.flatten().scatter_add_(0, prospective_indices, ones)
            self.prospective_successes.flatten().scatter_add_(0, prospective_indices, outcomes)
            self.prospective_probability_sum.flatten().scatter_add_(0, prospective_indices, probabilities)
            self.prospective_brier_sum.flatten().scatter_add_(
                0, prospective_indices, (probabilities - outcomes).square()
            )
            prospective_success = prospective & hit
            if prospective_success.any():
                timing_indices = source[prospective_success] * self.n_nodes + target[prospective_success]
                actual_timing = nxt[:, layout.completion_elapsed][prospective_success].to(dtype=self.node_visits.dtype)
                predicted_timing = self.tctrl[source[prospective_success], target[prospective_success]].clone()
                timing_ones = torch.ones_like(actual_timing)
                self.prospective_timing_count.flatten().scatter_add_(0, timing_indices, timing_ones)
                self.prospective_timing_sum.flatten().scatter_add_(0, timing_indices, actual_timing)
                self.prospective_predicted_timing_sum.flatten().scatter_add_(0, timing_indices, predicted_timing)
                self.prospective_timing_absolute_error_sum.flatten().scatter_add_(
                    0, timing_indices, (actual_timing - predicted_timing).abs()
                )
        if attempted.any():
            attempt_index = source[attempted] * self.n_nodes + target[attempted]
            attempt_weights = gamma_future[attempted]
            self.control_attempts.flatten().scatter_add_(0, attempt_index, attempt_weights)
        success = hit & (source >= 0) & (source < self.n_nodes) & (target >= 0) & (target < self.n_nodes)
        success = success & (source != target)
        if success.any():
            edge_index = source[success] * self.n_nodes + target[success]
            weights = gamma_future[success]
            elapsed = nxt[:, layout.completion_elapsed][success].to(dtype=self.node_visits.dtype)
            old_confidence = self.edge_confidence.flatten().clone()
            old_time = self.tctrl.flatten().clone()
            weighted_hits = torch.zeros_like(old_confidence)
            weighted_time = torch.zeros_like(old_time)
            weighted_hits.scatter_add_(0, edge_index, weights)
            weighted_time.scatter_add_(0, edge_index, weights * elapsed)
            new_confidence = old_confidence + weighted_hits
            updated = weighted_hits > 0
            new_time = torch.where(
                updated,
                (old_confidence * old_time + weighted_time) / new_confidence.clamp_min(1e-12),
                old_time,
            )
            self.edge_confidence.copy_(new_confidence.view_as(self.edge_confidence))
            self.tctrl.copy_(new_time.view_as(self.tctrl))
        after_reliability = (self.edge_confidence + 1.0) / (self.control_attempts + 2.0)
        after_known = (
            (self.tctrl > 0)
            & (self.edge_confidence >= float(confidence_threshold))
            & (after_reliability >= float(reliability_threshold))
        )
        after_known.fill_diagonal_(False)
        return {
            "completion_count": completed.sum().to(dtype=self.node_visits.dtype),
            "success_count": success.sum().to(dtype=self.node_visits.dtype),
            "wrong_outcome_count": wrong_outcome.sum().to(dtype=self.node_visits.dtype),
            "timeout_count": timeout.sum().to(dtype=self.node_visits.dtype),
            "target_timeout_count": target_timeout.sum().to(dtype=self.node_visits.dtype),
            "exploration_timeout_count": exploration_timeout.sum().to(dtype=self.node_visits.dtype),
            "promotion_count": ((~before_known) & after_known).sum().to(dtype=self.node_visits.dtype),
            "demotion_count": (before_known & (~after_known)).sum().to(dtype=self.node_visits.dtype),
        }


def update_option_state_from_policy_graph(
    prev_option_state: Tensor,
    dg_activity: Tensor,
    prev_sequence_core: Tensor,
    graph: PolicyControllableGraph,
    fallback_horizon: int,
    timeout_margin_ratio: float = 0.20,
    timeout_margin_steps: int = 2,
    min_target_visits: float = 1.0,
    edge_confidence_threshold: float = 0.5,
    exploration_mode: bool = False,
    manager_exploration_probability: float = 0.0,
    exploration_horizon: int = 64,
    target_timing: str = "delayed",
    fast_weight_half_life_options: float = 10000.0,
) -> tuple[Tensor, Tensor]:
    """Advance option state without mutating global graph buffers.

    The output target is the behavior target already in ``prev_option_state``;
    selection based on the graph snapshot is committed only for the next step.
    """
    n_nodes = dg_activity.size(-1)
    layout = HRLStateLayout(n_nodes)
    if prev_option_state.size(-1) != hrl_option_state_size(n_nodes):
        raise RuntimeError("Policy graph mode expects compact option state")
    behavior_target = option_target_one_hot(prev_option_state, n_nodes)
    option_state = prev_option_state[:, : layout.persistent_start].clone()
    generation = graph.representation_generation.to(device=prev_option_state.device, dtype=prev_option_state.dtype)
    has_option = (option_state[:, layout.target] > 0) | (option_state[:, layout.source] > 0)
    stale_option = has_option & (prev_option_state[:, layout.persistent_start] != generation)
    if stale_option.any():
        option_state[stale_option, layout.target] = 0
        option_state[stale_option, layout.source] = 0
        option_state[stale_option, layout.age] = 0
        option_state[stale_option, layout.countdown] = 0
    state = graph.expanded_state(prev_option_state.size(0), prev_option_state.dtype, prev_option_state.device)
    state[:, : layout.persistent_start] = option_state
    next_state, _ = update_hrl_state(
        state,
        dg_activity,
        prev_sequence_core,
        fallback_horizon,
        timeout_margin_ratio,
        timeout_margin_steps,
        min_target_visits,
        persistent_fast_weights=True,
        fast_weight_half_life_options=fast_weight_half_life_options,
        edge_confidence_threshold=edge_confidence_threshold,
        exploration_mode=exploration_mode,
        manager_exploration_probability=manager_exploration_probability,
        exploration_horizon=exploration_horizon,
    )
    compact_state = prev_option_state.new_zeros(prev_option_state.shape)
    compact_state[:, : layout.persistent_start] = next_state[:, : layout.persistent_start]
    compact_state[:, layout.persistent_start] = generation
    if target_timing == "delayed":
        return compact_state, behavior_target
    if target_timing == "immediate":
        return compact_state, option_target_one_hot(compact_state, n_nodes)
    raise ValueError(f"Unknown HRL target timing: {target_timing}")
