from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from sf_working_directories.IntrMotiv.dmlab.hrl_controllable_graph import (
    HRLStateLayout,
    current_dg_from_activity,
    option_target_one_hot,
)


MODE_NONE = 0
MODE_NAVIGATE = 1
MODE_EXPLORE = 2
MODE_RETURN = 3
MODE_VALIDATE = 4
MODE_EDGE_PROBE = MODE_VALIDATE
N_MANAGER_MODES = 5

# [net forward, net strafe, net yaw, traversed path length].  The first three
# values describe the net SE(2) transform of one policy action; the last keeps
# straightness meaningful when DMLab repeats a turning command.
ACTION_FEATURE_SIZE = 4
MOTION_POLICY_SIZE = 6
GEOMETRY_POLICY_SIZE = 4


@dataclass(frozen=True)
class TopologicalStateLayout:
    """Per-stream topological manager and path-integration state."""

    n_nodes: int

    mode: int = 0
    final_goal: int = 1
    pending_source: int = 2
    pending_destination: int = 3
    last_landmark: int = 4
    last_landmark_age: int = 5
    segment_x: int = 6
    segment_y: int = 7
    segment_heading: int = 8
    segment_path: int = 9
    segment_turn: int = 10
    episode_x: int = 11
    episode_y: int = 12
    episode_heading: int = 13
    validation_defer_countdown: int = 14

    @property
    def local_candidate_count(self) -> int:
        """Behavior-time local command count for the local-successor manager.

        This aliases ``pending_source`` without changing recurrent-state or
        checkpoint shapes. The two meanings are mutually exclusive:
        local-successor selection is used only with edge validation disabled.
        """
        return self.pending_source

    @property
    def anchors_start(self) -> int:
        return 15

    @property
    def anchors_end(self) -> int:
        return self.anchors_start + 3 * self.n_nodes

    @property
    def diag_start(self) -> int:
        return self.anchors_end

    @property
    def passive_event(self) -> int:
        return self.diag_start

    @property
    def passive_source(self) -> int:
        return self.diag_start + 1

    @property
    def passive_destination(self) -> int:
        return self.diag_start + 2

    @property
    def passive_elapsed(self) -> int:
        return self.diag_start + 3

    @property
    def passive_path_length(self) -> int:
        return self.diag_start + 4

    @property
    def passive_dx(self) -> int:
        return self.diag_start + 5

    @property
    def passive_dy(self) -> int:
        return self.diag_start + 6

    @property
    def passive_dtheta(self) -> int:
        return self.diag_start + 7

    @property
    def passive_reject_nonexclusive(self) -> int:
        return self.diag_start + 8

    @property
    def passive_reject_time(self) -> int:
        return self.diag_start + 9

    @property
    def passive_reject_path(self) -> int:
        return self.diag_start + 10

    @property
    def passive_reject_motion(self) -> int:
        return self.diag_start + 11

    @property
    def frontier_selected(self) -> int:
        return self.diag_start + 12

    @property
    def frontier_score(self) -> int:
        return self.diag_start + 13

    @property
    def final_reached(self) -> int:
        return self.diag_start + 14

    @property
    def discovery(self) -> int:
        return self.diag_start + 15

    @property
    def return_success(self) -> int:
        return self.diag_start + 16

    @property
    def validation_success(self) -> int:
        return self.diag_start + 17

    @property
    def validation_timeout(self) -> int:
        return self.diag_start + 18

    @property
    def route_available(self) -> int:
        return self.diag_start + 19

    @property
    def plan_hops(self) -> int:
        return self.diag_start + 20

    @property
    def geometry_start(self) -> int:
        return self.diag_start + 21

    @property
    def geometry_end(self) -> int:
        return self.geometry_start + GEOMETRY_POLICY_SIZE

    @property
    def size(self) -> int:
        return self.geometry_end


def topological_state_size(n_nodes: int) -> int:
    return TopologicalStateLayout(n_nodes).size


def encode_node(node: Tensor | int, dtype: torch.dtype, device=None) -> Tensor:
    value = torch.as_tensor(node, device=device)
    return (value + 1).to(dtype=dtype)


def decode_node(value: Tensor) -> Tensor:
    return value.long() - 1


def reduced_action_features(action: Tensor, action_repeat: int = 1) -> Tensor:
    """Return the repeated-command SE(2) transform for the reduced action set.

    DMLab applies the selected command ``action_repeat`` times.  A turning
    command therefore follows an arc, not one unit of translation followed by
    a large turn.  The translation is expressed in the frame at the midpoint
    of the resulting turn, which is the convention used by ``integrate_motion``.
    """
    action = action.long().view(-1)
    table = torch.tensor(
        (
            (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, -math.radians(20.0)),
            (1.0, 0.0, math.radians(20.0)),
            (0.0, 0.0, 0.0),
        ),
        dtype=torch.float32,
        device=action.device,
    )
    if action_repeat <= 0:
        raise ValueError("action_repeat must be positive")
    features = table[action.clamp(min=0, max=5)]
    forward, strafe, yaw_step = features.unbind(dim=-1)
    repeats = float(action_repeat)
    half_yaw = 0.5 * yaw_step
    # Sum the repeated midpoint translations in a frame centered on the total
    # turn. The limit is exactly ``action_repeat`` for zero yaw.
    scale = torch.where(
        half_yaw.abs() < 1e-6,
        torch.full_like(half_yaw, repeats),
        torch.sin(repeats * half_yaw) / torch.sin(half_yaw),
    )
    total_yaw = yaw_step * repeats
    path_length = torch.sqrt(forward.square() + strafe.square()) * repeats
    return torch.stack((forward * scale, strafe * scale, total_yaw, path_length), dim=-1)


def integrate_motion(
    x: Tensor,
    y: Tensor,
    heading: Tensor,
    path_length: Tensor,
    turn: Tensor,
    action_features: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    forward, strafe, yaw, action_path_length = action_features.unbind(dim=-1)
    midpoint = heading + 0.5 * yaw
    dx = forward * torch.cos(midpoint) - strafe * torch.sin(midpoint)
    dy = forward * torch.sin(midpoint) + strafe * torch.cos(midpoint)
    heading = torch.atan2(torch.sin(heading + yaw), torch.cos(heading + yaw))
    path_length = path_length + action_path_length
    turn = turn + yaw.abs()
    return x + dx, y + dy, heading, path_length, turn


def motion_policy_features(state: Tensor, layout: TopologicalStateLayout, horizon: int) -> Tensor:
    dx = state[:, layout.segment_x]
    dy = state[:, layout.segment_y]
    path = state[:, layout.segment_path]
    heading = state[:, layout.segment_heading]
    displacement = torch.sqrt(dx.square() + dy.square())
    straightness = displacement / path.clamp_min(1e-6)
    scale = float(max(1, horizon))
    return torch.stack(
        (
            (dx / scale).clamp(-1.0, 1.0),
            (dy / scale).clamp(-1.0, 1.0),
            (path / scale).clamp(0.0, 1.0),
            torch.sin(heading),
            torch.cos(heading),
            straightness.clamp(0.0, 1.0),
        ),
        dim=-1,
    )


def reliable_edges(graph, confidence_threshold: float, reliability_threshold: float = 0.5) -> Tensor:
    """Directed edges supported by recent hits and an acceptable hit/attempt ratio."""
    known = (graph.tctrl > 0) & (graph.edge_confidence >= confidence_threshold)
    if hasattr(graph, "control_attempts"):
        reliability = (graph.edge_confidence + 1.0) / (graph.control_attempts + 2.0)
        known = known & (reliability >= float(reliability_threshold))
    known.fill_diagonal_(False)
    return known


def select_least_tested_target(graph, source: int, min_visits: float = 1.0) -> int | None:
    """Choose an observed alternative with the least decayed intentional evidence."""
    source = int(source)
    if source < 0 or source >= graph.n_nodes:
        return None
    ids = torch.arange(graph.n_nodes, device=graph.node_visits.device)
    eligible = (graph.node_visits >= float(min_visits)) & (ids != source)
    candidates = torch.where(eligible)[0]
    if not candidates.numel():
        return None
    attempts = graph.control_attempts[source, candidates]
    minimum = attempts.min()
    # torch.where is sorted, giving the required lowest-index deterministic tie break.
    tied = candidates[attempts == minimum]
    return int(tied[0].item())


def select_least_tested_successor(graph, source: int) -> tuple[int | None, int]:
    """Choose among directed passive first successors, returning target and set size."""
    source = int(source)
    if source < 0 or source >= graph.n_nodes:
        return None, 0
    ids = torch.arange(graph.n_nodes, device=graph.passive_confidence.device)
    eligible = (graph.passive_confidence[source] > 0) & (ids != source)
    candidates = torch.where(eligible)[0]
    count = int(candidates.numel())
    if count == 0:
        return None, 0
    attempts = graph.control_attempts[source, candidates]
    tied = candidates[attempts == attempts.min()]
    return int(tied[0].item()), count


def validated_paths(
    graph,
    confidence_threshold: float,
    reliability_threshold: float = 0.5,
) -> tuple[Tensor, Tensor, Tensor]:
    """All-pairs controlled cost, first hop, and hop count for one policy graph."""
    n = graph.n_nodes
    known = reliable_edges(graph, confidence_threshold, reliability_threshold)
    dist = torch.where(known, graph.tctrl, torch.full_like(graph.tctrl, torch.inf)).clone()
    idx = torch.arange(n, device=dist.device)
    dist[idx, idx] = 0.0
    next_hop = torch.where(known, idx.view(1, n).expand(n, n), torch.full((n, n), -1, device=dist.device))
    next_hop[idx, idx] = idx
    hops = torch.where(known, torch.ones_like(graph.tctrl), torch.full_like(graph.tctrl, torch.inf))
    hops[idx, idx] = 0.0
    for k in range(n):
        candidate = dist[:, k].unsqueeze(1) + dist[k, :].unsqueeze(0)
        better = candidate < dist
        dist = torch.where(better, candidate, dist)
        next_hop = torch.where(better, next_hop[:, k].unsqueeze(1), next_hop)
        candidate_hops = hops[:, k].unsqueeze(1) + hops[k, :].unsqueeze(0)
        hops = torch.where(better, candidate_hops, hops)
    return dist, next_hop, hops


def frontier_scores(graph, uncertainty_weight: float) -> Tensor:
    visits = graph.node_visits
    observed = visits > 0
    n = visits.numel()
    order = torch.argsort(visits, stable=True)
    rank = torch.empty_like(visits)
    rank[order] = torch.arange(n, dtype=visits.dtype, device=visits.device)
    novelty = 1.0 - rank / float(max(1, n - 1))
    attempts = graph.frontier_attempts
    discoveries = graph.frontier_discoveries
    uncertainty = torch.sqrt(torch.log(2.0 + attempts.sum()) / (1.0 + attempts)).clamp(max=1.0)
    discovery_yield = (discoveries + 1.0) / (attempts + 2.0)
    score = novelty + float(uncertainty_weight) * uncertainty + 0.5 * discovery_yield
    return score.masked_fill(~observed, -torch.inf)


def visit_scores(graph) -> Tensor:
    """Novelty-only manager score used by the topology-matched control."""
    visits = graph.node_visits
    observed = visits > 0
    n = visits.numel()
    order = torch.argsort(visits, stable=True)
    rank = torch.empty_like(visits)
    rank[order] = torch.arange(n, dtype=visits.dtype, device=visits.device)
    return (1.0 - rank / float(max(1, n - 1))).masked_fill(~observed, -torch.inf)


def _candidate_edges(
    graph,
    passive_threshold: float,
    controllability_threshold: float,
    reliability_threshold: float,
    geometry: str,
    geometry_k: int,
    geometry_max_distance: float,
):
    candidates = (graph.passive_confidence >= passive_threshold) & ~reliable_edges(
        graph, controllability_threshold, reliability_threshold
    )
    candidates.fill_diagonal_(False)
    if geometry == "se2":
        not_reliable = ~reliable_edges(graph, controllability_threshold, reliability_threshold)
        candidates |= geometric_candidate_edges(
            graph,
            controllability_threshold,
            geometry_k,
            geometry_max_distance,
            unvalidated=not_reliable,
        )
    return candidates


def geometric_candidate_edges(
    graph,
    controllability_threshold: float,
    geometry_k: int,
    geometry_max_distance: float,
    unvalidated: Tensor | None = None,
) -> Tensor:
    """Return only SE(2)-proposed, still-unvalidated directed edges."""
    candidates = torch.zeros_like(graph.edge_confidence, dtype=torch.bool)
    if graph.pose_valid.sum() > 1:
        pose = graph.landmark_pose
        distance = torch.cdist(pose[:, :2], pose[:, :2])
        valid_pair = graph.pose_valid.unsqueeze(1) & graph.pose_valid.unsqueeze(0)
        geometric = valid_pair & (distance > 0) & (distance <= geometry_max_distance)
        nearest = torch.zeros_like(geometric)
        for source in range(graph.n_nodes):
            ids = torch.where(geometric[source])[0]
            if ids.numel():
                selected = ids[torch.argsort(distance[source, ids])[:geometry_k]]
                nearest[source, selected] = True
        if unvalidated is None:
            unvalidated = graph.edge_confidence < controllability_threshold
        candidates = nearest & unvalidated
        candidates.fill_diagonal_(False)
    return candidates


def _select_validation_edge(candidates: Tensor, reachable: Tensor, scores: Tensor) -> tuple[int, int] | None:
    """Select a reachable unvalidated edge without depending on DG index order."""
    edge_ids = torch.nonzero(candidates & reachable.unsqueeze(1), as_tuple=False)
    if edge_ids.numel() == 0:
        return None
    source_score = scores[edge_ids[:, 0]]
    destination_score = scores[edge_ids[:, 1]]
    # Prefer an exploratory source and an exploratory destination. ``nonzero``
    # is row-major, so argmax provides a deterministic source/destination tie
    # break without introducing actor-side randomness.
    edge_score = source_score + destination_score
    selected = edge_ids[torch.argmax(edge_score)]
    return int(selected[0].item()), int(selected[1].item())


def _transitive_reachability(adjacency: Tensor) -> Tensor:
    reach = adjacency.bool().clone()
    reach.fill_diagonal_(True)
    for intermediate in range(reach.size(0)):
        reach |= reach[:, intermediate].unsqueeze(1) & reach[intermediate].unsqueeze(0)
    return reach


def connectivity_gain(adjacency: Tensor, source: int, destination: int) -> float:
    """Increase in ordered reachable pairs after adding one directed edge."""
    before = _transitive_reachability(adjacency)
    after_graph = adjacency.clone()
    after_graph[source, destination] = True
    after = _transitive_reachability(after_graph)
    diagonal = torch.eye(adjacency.size(0), dtype=torch.bool, device=adjacency.device)
    return float(((after & ~before) & ~diagonal).sum().item())


def select_connectivity_probe(
    graph,
    candidates: Tensor,
    reachable_sources: Tensor,
    confidence_threshold: float,
    reliability_threshold: float,
    connectivity_weight: float = 0.25,
) -> tuple[tuple[int, int] | None, float]:
    """Return the highest-UCB actionable edge and its task score."""
    edge_ids = torch.nonzero(candidates & reachable_sources.unsqueeze(1), as_tuple=False)
    if edge_ids.numel() == 0:
        return None, -math.inf
    reliable = reliable_edges(graph, confidence_threshold, reliability_threshold)
    attempts = graph.control_attempts
    successes = graph.edge_confidence
    total_attempts = attempts.sum()
    gains = [connectivity_gain(reliable, int(pair[0]), int(pair[1])) for pair in edge_ids]
    max_gain = max(gains) if gains else 0.0
    best_pair = None
    best_score = -math.inf
    for pair, gain in zip(edge_ids.tolist(), gains):
        source, destination = int(pair[0]), int(pair[1])
        attempt = float(attempts[source, destination].item())
        success = float(successes[source, destination].item())
        probability = (success + 1.0) / (attempt + 2.0)
        uncertainty = min(1.0, math.sqrt(math.log(2.0 + float(total_attempts.item())) / (1.0 + attempt)))
        normalized_gain = gain / max_gain if max_gain > 0 else 0.0
        score = probability + uncertainty + float(connectivity_weight) * normalized_gain
        # Edge-first and row-major deterministic tie breaking.
        if score > best_score:
            best_pair = (source, destination)
            best_score = score
    return best_pair, best_score


def _deadline_from_value(value: float, fallback: int, ratio: float, steps: int) -> float:
    if not math.isfinite(value) or value <= 0:
        return float(fallback)
    return float(max(1, math.ceil(value * (1.0 + ratio)) + steps))


def _set_option(
    option: Tensor,
    topo: Tensor,
    row: int,
    target: int,
    source: int,
    mode: int,
    deadline: float,
    option_layout: HRLStateLayout,
    topo_layout: TopologicalStateLayout,
    geometry_condition: Tensor | None = None,
):
    option[row, option_layout.target] = float(target + 1)
    option[row, option_layout.source] = float(source + 1)
    option[row, option_layout.age] = 0.0
    option[row, option_layout.countdown] = deadline
    option[row, option_layout.option_reset] = 1.0
    option[row, option_layout.selected_deadline] = deadline
    topo[row, topo_layout.mode] = float(mode)
    topo[row, topo_layout.geometry_start : topo_layout.geometry_end] = 0.0
    if geometry_condition is not None:
        topo[row, topo_layout.geometry_start : topo_layout.geometry_end] = geometry_condition


def _geometry_condition(graph, source: int, target: int, dtype: torch.dtype, device) -> Tensor:
    out = torch.zeros(GEOMETRY_POLICY_SIZE, dtype=dtype, device=device)
    if source < 0 or target < 0 or not (graph.pose_valid[source] and graph.pose_valid[target]):
        return out
    source_pose = graph.landmark_pose[source]
    target_pose = graph.landmark_pose[target]
    delta_world = target_pose[:2] - source_pose[:2]
    c, s = torch.cos(source_pose[2]), torch.sin(source_pose[2])
    dx = c * delta_world[0] + s * delta_world[1]
    dy = -s * delta_world[0] + c * delta_world[1]
    dtheta = target_pose[2] - source_pose[2]
    return torch.stack((dx / 32.0, dy / 32.0, torch.sin(dtheta), torch.cos(dtheta))).to(dtype=dtype)


def advance_topological_manager(
    prev_option_state: Tensor,
    prev_topological_state: Tensor,
    dg_activity: Tensor,
    action_features: Tensor,
    graph,
    *,
    fallback_horizon: int,
    margin_ratio: float,
    margin_steps: int,
    confidence_threshold: float,
    passive_threshold: float,
    passive_min_displacement: float,
    passive_max_length: float,
    use_motion_filter: bool,
    frontier_uncertainty_weight: float,
    frontier_selection: bool = True,
    waypoint_planning: bool,
    exploration_horizon: int,
    geometry: str = "none",
    geometry_k: int = 3,
    geometry_max_distance: float = 32.0,
    edge_exploration: bool = True,
    reliability_threshold: float = 0.5,
    connectivity_weight: float = 0.25,
    target_timing: str = "delayed",
    include_behavior_mode: bool = False,
    common_manager: bool = False,
    control_outcome: str = "target_hit",
    direct_target_selection: str = "frontier",
    min_target_visits: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Advance topological state while preserving the sampled behavior condition."""
    if control_outcome not in ("target_hit", "first_distinct"):
        raise ValueError(f"Unknown control_outcome={control_outcome}")
    if direct_target_selection not in ("frontier", "least_tested", "local_successor"):
        raise ValueError(f"Unknown direct_target_selection={direct_target_selection}")
    n_nodes = dg_activity.size(-1)
    option_layout = HRLStateLayout(n_nodes)
    topo_layout = TopologicalStateLayout(n_nodes)
    option = prev_option_state.clone()
    topo = prev_topological_state.clone()
    behavior_target = option_target_one_hot(prev_option_state, n_nodes)
    behavior_geometry = prev_topological_state[:, topo_layout.geometry_start : topo_layout.geometry_end].clone()
    generation = graph.representation_generation.to(device=option.device, dtype=option.dtype)
    has_option = (option[:, option_layout.target] > 0) | (option[:, option_layout.source] > 0)
    stale = has_option & (option[:, option_layout.persistent_start] != generation)
    if stale.any():
        option[stale, : option_layout.persistent_start] = 0.0
        topo[stale] = 0.0
    option[:, option_layout.diag_start : option_layout.persistent_start] = 0.0
    topo[:, topo_layout.diag_start : topo_layout.size] = 0.0
    deferred = topo[:, topo_layout.validation_defer_countdown] > 0
    topo[deferred, topo_layout.validation_defer_countdown] -= 1.0
    mode_now = topo[:, topo_layout.mode].long()
    inactive_validation = (mode_now != MODE_NAVIGATE) & (mode_now != MODE_RETURN) & (mode_now != MODE_VALIDATE)
    clear_deferred = (topo[:, topo_layout.validation_defer_countdown] <= 0) & inactive_validation
    topo[clear_deferred, topo_layout.pending_source : topo_layout.pending_destination + 1] = 0.0

    # Apply the action that produced the current observation.
    for prefix in ("segment", "episode"):
        fields = [getattr(topo_layout, f"{prefix}_{name}") for name in ("x", "y", "heading")]
        if prefix == "segment":
            values = integrate_motion(
                topo[:, fields[0]], topo[:, fields[1]], topo[:, fields[2]],
                topo[:, topo_layout.segment_path], topo[:, topo_layout.segment_turn], action_features,
            )
            topo[:, fields[0]], topo[:, fields[1]], topo[:, fields[2]] = values[:3]
            topo[:, topo_layout.segment_path], topo[:, topo_layout.segment_turn] = values[3:]
        else:
            zeros = torch.zeros(topo.size(0), dtype=topo.dtype, device=topo.device)
            values = integrate_motion(topo[:, fields[0]], topo[:, fields[1]], topo[:, fields[2]], zeros, zeros, action_features)
            topo[:, fields[0]], topo[:, fields[1]], topo[:, fields[2]] = values[:3]

    current, has_active, n_active = current_dg_from_activity(dg_activity.detach())
    exclusive = has_active & (n_active == 1)
    option[:, option_layout.active_dg] = encode_node(current, option.dtype)
    option[:, option_layout.multi_activation] = (n_active > 1).to(option.dtype)

    # Episode-local DG anchors are set once; far reactivations remain measured from the original anchor.
    for row in range(topo.size(0)):
        if not exclusive[row]:
            continue
        node = int(current[row].item())
        anchor = topo_layout.anchors_start + 3 * node
        if topo[row, anchor + 2] <= 0:
            topo[row, anchor] = topo[row, topo_layout.episode_x]
            topo[row, anchor + 1] = topo[row, topo_layout.episode_y]
            topo[row, anchor + 2] = 1.0

    # Emit stable, local passive transitions.
    last = decode_node(topo[:, topo_layout.last_landmark])
    for row in range(topo.size(0)):
        if not exclusive[row]:
            if n_active[row] > 1 and last[row] >= 0:
                topo[row, topo_layout.passive_reject_nonexclusive] = 1.0
            topo[row, topo_layout.last_landmark_age] += 1.0
            continue
        node = int(current[row].item())
        old = int(last[row].item())
        if old < 0:
            topo[row, topo_layout.last_landmark] = float(node + 1)
            topo[row, topo_layout.last_landmark_age] = 0.0
            topo[row, topo_layout.segment_x : topo_layout.segment_turn + 1] = 0.0
            continue
        if old == node:
            topo[row, topo_layout.last_landmark_age] += 1.0
            continue
        dx = float(topo[row, topo_layout.segment_x].item())
        dy = float(topo[row, topo_layout.segment_y].item())
        path = float(topo[row, topo_layout.segment_path].item())
        displacement = math.hypot(dx, dy)
        elapsed = float(topo[row, topo_layout.last_landmark_age].item() + 1.0)
        elapsed_ok = 0.0 < elapsed <= passive_max_length
        if use_motion_filter:
            path_ok = path <= passive_max_length
            motion_ok = displacement >= passive_min_displacement
        else:
            path = elapsed
            path_ok = True
            motion_ok = True
        accepted = elapsed_ok and path_ok and motion_ok
        if accepted:
            topo[row, topo_layout.passive_event] = 1.0
            topo[row, topo_layout.passive_source] = float(old + 1)
            topo[row, topo_layout.passive_destination] = float(node + 1)
            topo[row, topo_layout.passive_elapsed] = elapsed
            topo[row, topo_layout.passive_path_length] = path
            topo[row, topo_layout.passive_dx] = dx
            topo[row, topo_layout.passive_dy] = dy
            topo[row, topo_layout.passive_dtheta] = topo[row, topo_layout.segment_heading]
        else:
            topo[row, topo_layout.passive_reject_time] = float(not elapsed_ok)
            topo[row, topo_layout.passive_reject_path] = float(not path_ok)
            topo[row, topo_layout.passive_reject_motion] = float(not motion_ok)
        topo[row, topo_layout.last_landmark] = float(node + 1)
        topo[row, topo_layout.last_landmark_age] = 0.0
        topo[row, topo_layout.segment_x : topo_layout.segment_turn + 1] = 0.0

    dist, next_hop, hop_count = validated_paths(graph, confidence_threshold, reliability_threshold)
    # The common controllability manager's node objective is exactly visit-rank
    # novelty.  Legacy frontier managers retain their discovery-UCB score.
    scores = (
        visit_scores(graph)
        if common_manager or not frontier_selection
        else frontier_scores(graph, frontier_uncertainty_weight)
    )
    candidates = (
        _candidate_edges(
            graph,
            passive_threshold,
            confidence_threshold,
            reliability_threshold,
            geometry,
            geometry_k,
            geometry_max_distance,
        )
        if edge_exploration
        else torch.zeros_like(graph.edge_confidence, dtype=torch.bool)
    )

    def start_exploration(row: int, source: int, frontier_score: float):
        _set_option(option, topo, row, n_nodes, source, MODE_EXPLORE, float(exploration_horizon), option_layout, topo_layout)
        topo[row, topo_layout.final_goal] = float(source + 1)
        topo[row, topo_layout.frontier_selected] = 1.0
        topo[row, topo_layout.frontier_score] = frontier_score
        if direct_target_selection == "local_successor":
            topo[row, topo_layout.local_candidate_count] = 0.0

    def start_target(row: int, source: int, target: int, mode: int, final: int, deadline_value: float):
        geometry_condition = _geometry_condition(graph, source, target, option.dtype, option.device) if geometry == "se2" else None
        deadline = _deadline_from_value(deadline_value, fallback_horizon, margin_ratio, margin_steps)
        _set_option(option, topo, row, target, source, mode, deadline, option_layout, topo_layout, geometry_condition)
        topo[row, topo_layout.final_goal] = float(final + 1)

    def choose_task(row: int, source: int):
        if source < 0:
            option[row, option_layout.target] = 0.0
            topo[row, topo_layout.mode] = float(MODE_NONE)
            return
        if direct_target_selection in ("least_tested", "local_successor"):
            if direct_target_selection == "local_successor":
                destination, candidate_count = select_least_tested_successor(graph, source)
            else:
                destination = select_least_tested_target(graph, source, min_target_visits)
                candidate_count = max(0, int((graph.node_visits >= float(min_target_visits)).sum().item()) - 1)
            if destination is None:
                start_exploration(row, source, float(scores[source].item()) if torch.isfinite(scores[source]) else 0.0)
                return
            reliable = reliable_edges(graph, confidence_threshold, reliability_threshold)
            direct_time = float(graph.tctrl[source, destination].item()) if reliable[source, destination] else math.inf
            start_target(row, source, destination, MODE_NAVIGATE, destination, direct_time)
            if direct_target_selection == "local_successor":
                topo[row, topo_layout.local_candidate_count] = float(candidate_count)
            return
        reachable = torch.isfinite(dist[source])
        reachable[source] = True
        row_candidates = candidates.clone()
        if topo[row, topo_layout.validation_defer_countdown] > 0:
            deferred_source = int(topo[row, topo_layout.pending_source].item()) - 1
            deferred_destination = int(topo[row, topo_layout.pending_destination].item()) - 1
            if 0 <= deferred_source < n_nodes and 0 <= deferred_destination < n_nodes:
                row_candidates[deferred_source, deferred_destination] = False
        validation_edge, validation_score = select_connectivity_probe(
            graph,
            row_candidates,
            reachable,
            confidence_threshold,
            reliability_threshold,
            connectivity_weight,
        )
        # Direct frontier control commands the selected landmark itself; it
        # does not require an already validated route.  Reachability is a
        # prerequisite only for waypoint/common-manager planning.
        observed = graph.node_visits > 0
        eligible = (reachable & observed) if (waypoint_planning or common_manager) else observed
        ids = torch.where(eligible)[0]
        best_node_score = float(scores[ids].max().item()) if ids.numel() else -math.inf
        validation_edge = validation_edge if validation_score >= best_node_score else None
        if validation_edge is not None:
            selected_source, destination = validation_edge
            topo[row, topo_layout.pending_source] = float(selected_source + 1)
            topo[row, topo_layout.pending_destination] = float(destination + 1)
            topo[row, topo_layout.validation_defer_countdown] = 0.0
            if selected_source == source:
                passive_time = float(graph.passive_time[selected_source, destination].item())
                start_target(row, source, destination, MODE_VALIDATE, destination, passive_time)
                return
            else:
                hop = int(next_hop[source, selected_source].item()) if waypoint_planning else selected_source
                if hop >= 0:
                    topo[row, topo_layout.route_available] = 1.0
                    topo[row, topo_layout.plan_hops] = hop_count[source, selected_source]
                    start_target(row, source, hop, MODE_NAVIGATE, selected_source, float(dist[source, hop].item()))
                    return
        if ids.numel():
            frontier = int(ids[torch.argmax(scores[ids])].item())
            if frontier == source:
                start_exploration(row, source, float(scores[frontier].item()))
            else:
                hop = int(next_hop[source, frontier].item()) if waypoint_planning else frontier
                if hop >= 0:
                    topo[row, topo_layout.route_available] = 1.0
                    topo[row, topo_layout.plan_hops] = hop_count[source, frontier]
                    topo[row, topo_layout.frontier_selected] = 1.0
                    topo[row, topo_layout.frontier_score] = scores[frontier]
                    start_target(row, source, hop, MODE_NAVIGATE, frontier, float(dist[source, hop].item()))
                    return
        start_exploration(row, source, float(scores[source].item()) if torch.isfinite(scores[source]) else 0.0)

    target = decode_node(option[:, option_layout.target])
    source = decode_node(option[:, option_layout.source])
    mode = topo[:, topo_layout.mode].long()
    normal_target = (target >= 0) & (target < n_nodes)
    exploring = target == n_nodes
    hit = exclusive & normal_target & (current == target)
    wrong = (
        control_outcome == "first_distinct"
    ) & exclusive & normal_target & (current != source) & (current != target)
    expired = (~hit) & (normal_target | exploring) & (option[:, option_layout.countdown] <= 1.0)
    elapsed = option[:, option_layout.age] + 1.0

    for row in range(option.size(0)):
        current_node = int(current[row].item()) if exclusive[row] else int(source[row].item())
        row_mode = int(mode[row].item())
        passive_event = topo[row, topo_layout.passive_event] > 0
        if row_mode == MODE_EXPLORE and passive_event and edge_exploration:
            pending_source = int(topo[row, topo_layout.passive_source].item()) - 1
            pending_destination = int(topo[row, topo_layout.passive_destination].item()) - 1
            if common_manager:
                # Passive evidence is committed after learner acceptance.  Do
                # not infer reverse reachability or travel back over an
                # unvalidated edge; the normal actionable-probe selector will
                # revisit this directed candidate once its source is current
                # or reachable through reliable edges.
                choose_task(row, pending_destination)
                continue
            previous_confidence = float(graph.passive_confidence[pending_source, pending_destination].item())
            if previous_confidence < 1.0:
                topo[row, topo_layout.discovery] = 1.0
            if previous_confidence + 1.0 >= passive_threshold:
                topo[row, topo_layout.pending_source] = float(pending_source + 1)
                topo[row, topo_layout.pending_destination] = float(pending_destination + 1)
                topo[row, topo_layout.validation_defer_countdown] = 0.0
                reverse_time = float(topo[row, topo_layout.passive_elapsed].item())
                start_target(row, pending_destination, pending_source, MODE_RETURN, pending_source, reverse_time)
            else:
                choose_task(row, pending_destination)
            continue
        if hit[row]:
            option[row, option_layout.target_hit] = 1.0
            option[row, option_layout.completion_elapsed] = elapsed[row]
            if row_mode == MODE_RETURN:
                topo[row, topo_layout.return_success] = 1.0
                destination = int(topo[row, topo_layout.pending_destination].item()) - 1
                passive_time = float(graph.passive_time[current_node, destination].item())
                if passive_time <= 0:
                    passive_time = float(fallback_horizon)
                start_target(row, current_node, destination, MODE_VALIDATE, destination, passive_time)
            elif row_mode == MODE_VALIDATE:
                topo[row, topo_layout.validation_success] = 1.0
                topo[row, topo_layout.validation_defer_countdown] = float(exploration_horizon)
                choose_task(row, current_node)
            elif row_mode == MODE_NAVIGATE:
                final = int(topo[row, topo_layout.final_goal].item()) - 1
                pending_destination = int(topo[row, topo_layout.pending_destination].item()) - 1
                if current_node == final and pending_destination >= 0:
                    passive_time = float(graph.passive_time[current_node, pending_destination].item())
                    start_target(row, current_node, pending_destination, MODE_VALIDATE, pending_destination, passive_time)
                elif current_node == final:
                    topo[row, topo_layout.final_reached] = 1.0
                    start_exploration(row, current_node, float(scores[current_node].item()))
                else:
                    hop = int(next_hop[current_node, final].item()) if waypoint_planning else final
                    if hop >= 0:
                        start_target(row, current_node, hop, MODE_NAVIGATE, final, float(dist[current_node, hop].item()))
                    else:
                        choose_task(row, current_node)
            else:
                choose_task(row, current_node)
            continue
        if wrong[row]:
            # Reuse the unsuccessful-completion pulse. A negative elapsed time
            # under a normal target distinguishes a wrong first outcome from a
            # deadline timeout without changing recurrent-state shape.
            option[row, option_layout.option_expired] = 1.0
            option[row, option_layout.completion_elapsed] = -elapsed[row]
            choose_task(row, current_node)
            continue
        if expired[row]:
            option[row, option_layout.option_expired] = 1.0
            option[row, option_layout.completion_elapsed] = -elapsed[row] if row_mode == MODE_EXPLORE else elapsed[row]
            if row_mode in (MODE_RETURN, MODE_VALIDATE):
                topo[row, topo_layout.validation_timeout] = 1.0
                topo[row, topo_layout.validation_defer_countdown] = float(exploration_horizon)
            choose_task(row, current_node)
            continue
        if target[row] < 0:
            choose_task(row, current_node)
            continue
        option[row, option_layout.age] += 1.0
        option[row, option_layout.countdown] = torch.clamp(option[row, option_layout.countdown] - 1.0, min=0.0)

    option[:, option_layout.persistent_start] = generation
    if target_timing == "immediate":
        behavior_target = option_target_one_hot(option, n_nodes)
        behavior_geometry = topo[:, topo_layout.geometry_start : topo_layout.geometry_end].clone()
        behavior_mode = topo[:, topo_layout.mode].long()
    elif target_timing == "delayed":
        behavior_mode = prev_topological_state[:, topo_layout.mode].long()
    else:
        raise ValueError(f"Unknown HRL target timing: {target_timing}")
    pieces = [behavior_target, behavior_geometry]
    if include_behavior_mode:
        behavior_mode = F.one_hot(
            behavior_mode.clamp(min=0, max=N_MANAGER_MODES - 1), N_MANAGER_MODES
        ).to(option.dtype)
        pieces.append(behavior_mode)
    return option, topo, torch.cat(pieces, dim=-1)


def dg_path_scatter_loss(
    pre_threshold_logits: Tensor,
    rnn_states: Tensor,
    action_features: Tensor,
    topological_offset: int,
    n_nodes: int,
    intercept: float,
    coefficient: float,
    min_displacement: float,
    min_straightness: float,
    temperature: float,
    valids: Tensor,
) -> tuple[Tensor, Tensor]:
    if coefficient <= 0:
        zero = pre_threshold_logits.sum() * 0.0
        return zero, zero.detach()
    layout = TopologicalStateLayout(n_nodes)
    topo = rnn_states[:, topological_offset : topological_offset + layout.size]
    actions = action_features.to(dtype=topo.dtype, device=topo.device)
    zeros = torch.zeros(topo.size(0), dtype=topo.dtype, device=topo.device)
    x, y, _, path, _ = integrate_motion(
        topo[:, layout.episode_x], topo[:, layout.episode_y], topo[:, layout.episode_heading], zeros, zeros, actions
    )
    anchor = topo[:, layout.anchors_start : layout.anchors_end].view(-1, n_nodes, 3)
    displacement = torch.sqrt((x.unsqueeze(1) - anchor[..., 0]).square() + (y.unsqueeze(1) - anchor[..., 1]).square())
    # Straightness uses the segment trace, which resets only at a stable landmark transition.
    segment_x, segment_y = topo[:, layout.segment_x], topo[:, layout.segment_y]
    sx, sy, _, segment_path, _ = integrate_motion(
        segment_x, segment_y, topo[:, layout.segment_heading], topo[:, layout.segment_path], topo[:, layout.segment_turn], actions
    )
    straightness = torch.sqrt(sx.square() + sy.square()) / segment_path.clamp_min(1e-6)
    mask = (anchor[..., 2] > 0) & (displacement >= min_displacement)
    mask &= straightness.unsqueeze(1) >= min_straightness
    mask &= valids.bool().unsqueeze(1)
    smooth_activity = temperature * F.softplus((pre_threshold_logits - intercept) / temperature)
    loss = float(coefficient) * (smooth_activity * mask).sum() / mask.sum().clamp_min(1)
    active = pre_threshold_logits > intercept
    conflict_fraction = (active & mask).sum().float() / active.sum().clamp_min(1)
    return loss, conflict_fraction.detach()


@torch.no_grad()
def update_topological_graph_from_rollout(
    graph,
    option_states: Tensor,
    topological_states: Tensor,
    valid_steps: Tensor,
    *,
    geometry: str = "none",
    pose_steps: int = 5,
    pose_learning_rate: float = 0.05,
) -> dict[str, Tensor]:
    """Apply passive and frontier events exactly once for an accepted rollout."""
    n_nodes = graph.n_nodes
    option_layout = HRLStateLayout(n_nodes)
    topo_layout = TopologicalStateLayout(n_nodes)
    prev_option = option_states[:, :-1].reshape(-1, option_states.shape[-1])
    next_option = option_states[:, 1:].reshape(-1, option_states.shape[-1])
    prev_topo = topological_states[:, :-1].reshape(-1, topological_states.shape[-1])
    next_topo = topological_states[:, 1:].reshape(-1, topological_states.shape[-1])
    valid = valid_steps.bool().reshape(-1)

    passive = (next_topo[:, topo_layout.passive_event] > 0) & valid
    passive_count = passive.sum()
    if passive.any():
        source = next_topo[:, topo_layout.passive_source].long() - 1
        destination = next_topo[:, topo_layout.passive_destination].long() - 1
        accepted = passive & (source >= 0) & (source < n_nodes) & (destination >= 0) & (destination < n_nodes)
        accepted &= source != destination
        edge_index = source[accepted] * n_nodes + destination[accepted]
        old_confidence = graph.passive_confidence.flatten().clone()
        observations = torch.zeros_like(old_confidence)
        observations.scatter_add_(0, edge_index, torch.ones_like(edge_index, dtype=old_confidence.dtype))
        new_confidence = old_confidence + observations

        def weighted_update(buffer: Tensor, values: Tensor):
            weighted = torch.zeros_like(buffer.flatten())
            weighted.scatter_add_(0, edge_index, values[accepted].to(weighted.dtype))
            updated = observations > 0
            result = torch.where(
                updated,
                (old_confidence * buffer.flatten() + weighted) / new_confidence.clamp_min(1e-12),
                buffer.flatten(),
            )
            buffer.copy_(result.view_as(buffer))

        weighted_update(graph.passive_time, next_topo[:, topo_layout.passive_elapsed])
        weighted_update(graph.passive_path_length, next_topo[:, topo_layout.passive_path_length])
        weighted_update(graph.passive_dx, next_topo[:, topo_layout.passive_dx])
        weighted_update(graph.passive_dy, next_topo[:, topo_layout.passive_dy])
        angle = next_topo[:, topo_layout.passive_dtheta]
        weighted_update(graph.passive_dtheta_sin, torch.sin(angle))
        weighted_update(graph.passive_dtheta_cos, torch.cos(angle))
        graph.passive_confidence.copy_(new_confidence.view_as(graph.passive_confidence))

    previous_mode = prev_topo[:, topo_layout.mode].long()
    discovery = (next_topo[:, topo_layout.discovery] > 0) & valid
    exploration_timeout = (
        (previous_mode == MODE_EXPLORE)
        & (next_option[:, option_layout.option_expired] > 0)
        & valid
    )
    attempt = discovery | exploration_timeout
    frontier_node = prev_option[:, option_layout.source].long() - 1
    attempt &= (frontier_node >= 0) & (frontier_node < n_nodes)
    if attempt.any():
        graph.frontier_attempts.scatter_add_(
            0, frontier_node[attempt], torch.ones_like(frontier_node[attempt], dtype=graph.frontier_attempts.dtype)
        )
    discovered_node = frontier_node[discovery & (frontier_node >= 0) & (frontier_node < n_nodes)]
    if discovered_node.numel():
        graph.frontier_discoveries.scatter_add_(
            0, discovered_node, torch.ones_like(discovered_node, dtype=graph.frontier_discoveries.dtype)
        )

    if geometry == "se2":
        _fit_se2_pose_graph(graph, pose_steps, pose_learning_rate)

    return {
        "passive_update_count": passive_count.to(dtype=graph.node_visits.dtype),
        "frontier_attempt_count": attempt.sum().to(dtype=graph.node_visits.dtype),
        "frontier_discovery_count": discovery.sum().to(dtype=graph.node_visits.dtype),
        "edge_probe_success_count": (
            (next_topo[:, topo_layout.validation_success] > 0) & valid
        ).sum().to(dtype=graph.node_visits.dtype),
        "edge_probe_timeout_count": (
            (next_topo[:, topo_layout.validation_timeout] > 0) & valid
        ).sum().to(dtype=graph.node_visits.dtype),
    }


def _fit_se2_pose_graph(graph, steps: int, learning_rate: float) -> None:
    known = graph.passive_confidence > 0
    known.fill_diagonal_(False)
    edge_source, edge_destination = torch.where(known)
    if edge_source.numel() == 0:
        graph.pose_stress.zero_()
        return

    # Initialize newly connected nodes from one observed relative transform.
    with torch.no_grad():
        for source, destination in zip(edge_source.tolist(), edge_destination.tolist()):
            if not graph.pose_valid[source] and not graph.pose_valid[destination]:
                graph.pose_valid[source] = True
                graph.landmark_pose[source] = 0
            if graph.pose_valid[source] and not graph.pose_valid[destination]:
                theta = graph.landmark_pose[source, 2]
                c, s = torch.cos(theta), torch.sin(theta)
                dx, dy = graph.passive_dx[source, destination], graph.passive_dy[source, destination]
                graph.landmark_pose[destination, 0] = graph.landmark_pose[source, 0] + c * dx - s * dy
                graph.landmark_pose[destination, 1] = graph.landmark_pose[source, 1] + s * dx + c * dy
                rel_theta = torch.atan2(
                    graph.passive_dtheta_sin[source, destination],
                    graph.passive_dtheta_cos[source, destination],
                )
                graph.landmark_pose[destination, 2] = graph.landmark_pose[source, 2] + rel_theta
                graph.pose_valid[destination] = True

    pose = graph.landmark_pose.detach().clone()
    weights = graph.passive_confidence[edge_source, edge_destination].detach()
    target_dx = graph.passive_dx[edge_source, edge_destination].detach()
    target_dy = graph.passive_dy[edge_source, edge_destination].detach()
    target_theta = torch.atan2(
        graph.passive_dtheta_sin[edge_source, edge_destination],
        graph.passive_dtheta_cos[edge_source, edge_destination],
    ).detach()
    adjacency = known | known.transpose(0, 1)
    remaining = set(torch.where(graph.pose_valid)[0].tolist())
    components: list[list[int]] = []
    while remaining:
        root = min(remaining)
        stack = [root]
        component: list[int] = []
        remaining.remove(root)
        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = torch.where(adjacency[node])[0].tolist()
            for neighbor in neighbors:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    final_loss = torch.zeros((), device=pose.device)
    with torch.enable_grad():
        for _ in range(max(0, int(steps))):
            pose.requires_grad_(True)
            source_pose = pose[edge_source]
            destination_pose = pose[edge_destination]
            delta = destination_pose[:, :2] - source_pose[:, :2]
            c, s = torch.cos(source_pose[:, 2]), torch.sin(source_pose[:, 2])
            predicted_dx = c * delta[:, 0] + s * delta[:, 1]
            predicted_dy = -s * delta[:, 0] + c * delta[:, 1]
            angular_error = destination_pose[:, 2] - source_pose[:, 2] - target_theta
            per_edge = (predicted_dx - target_dx).square() + (predicted_dy - target_dy).square()
            per_edge = per_edge + 1.0 - torch.cos(angular_error)
            final_loss = (weights * per_edge).sum() / weights.sum().clamp_min(1e-12)
            gradient = torch.autograd.grad(final_loss, pose)[0]
            with torch.no_grad():
                pose = pose - float(learning_rate) * gradient
                for component in components:
                    ids = torch.tensor(component, device=pose.device, dtype=torch.long)
                    gauge = pose[component[0]].clone()
                    translated = pose[ids, :2] - gauge[:2]
                    c0, s0 = torch.cos(-gauge[2]), torch.sin(-gauge[2])
                    x = c0 * translated[:, 0] - s0 * translated[:, 1]
                    y = s0 * translated[:, 0] + c0 * translated[:, 1]
                    pose[ids, 0] = x
                    pose[ids, 1] = y
                    angle = pose[ids, 2] - gauge[2]
                    pose[ids, 2] = torch.atan2(torch.sin(angle), torch.cos(angle))
    with torch.no_grad():
        graph.landmark_pose.copy_(pose)
        graph.pose_stress.copy_(final_loss.detach())
