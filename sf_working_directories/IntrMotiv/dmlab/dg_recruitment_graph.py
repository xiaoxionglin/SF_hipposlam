"""Passive transition evidence and victim selection for DG recruitment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

RECRUITMENT_HISTORY_SIZE = 3
HISTORY_LAST_EXCLUSIVE = 0
HISTORY_AGE = 1
HISTORY_GENERATION = 2


def update_recruitment_history(
    history: Tensor,
    dg_activity: Tensor,
    representation_generation: Tensor,
    max_gap: int,
) -> Tensor:
    """Advance actor history directly from behavior-time DG activity."""
    if history.ndim != 2 or history.size(-1) != RECRUITMENT_HISTORY_SIZE:
        raise ValueError("history must have shape [batch, 3]")
    if dg_activity.ndim != 2 or dg_activity.size(0) != history.size(0):
        raise ValueError("dg_activity must have shape [batch, DG units]")
    if max_gap <= 0:
        raise ValueError("max_gap must be positive")

    generation = representation_generation.to(device=history.device, dtype=history.dtype)
    generation_matches = history[:, HISTORY_GENERATION].eq(generation)
    last_id = history[:, HISTORY_LAST_EXCLUSIVE].where(
        generation_matches, torch.zeros_like(history[:, HISTORY_LAST_EXCLUSIVE])
    )
    age = history[:, HISTORY_AGE].where(generation_matches, torch.zeros_like(history[:, HISTORY_AGE]))

    active = dg_activity > 0
    exclusive = active.sum(dim=-1).eq(1)
    current_id = active.to(dtype=torch.int64).argmax(dim=-1).to(dtype=history.dtype) + 1.0
    next_id = torch.where(exclusive, current_id, last_id)
    aged = torch.where(last_id > 0, (age + 1.0).clamp_max(float(max_gap + 1)), age)
    next_age = torch.where(exclusive, torch.zeros_like(aged), aged)
    return torch.stack((next_id, next_age, generation.expand_as(next_id)), dim=-1)


@dataclass(frozen=True)
class RecruitmentEligibility:
    adjacency: Tensor
    incident_support: Tensor
    isolated: Tensor
    redundant_loser: Tensor
    eligible: Tensor
    victim: int | None
    reason: str | None
    redundant_pair_count: int


@dataclass(frozen=True)
class DirectionalRecruitmentEligibility:
    adjacency: Tensor
    reliability: Tensor
    out_degree: Tensor
    outgoing_confidence: Tensor
    fully_tested: Tensor
    zero_outdegree: Tensor
    bad_source: Tensor
    redundant_loser: Tensor
    eligible: Tensor
    victim: int | None
    reason: str | None
    redundant_pair_count: int


@dataclass(frozen=True)
class PredictiveRecruitmentEligibility:
    predictive_failure: Tensor
    eligible: Tensor
    reliability_gap: Tensor
    supporting_attempts: Tensor
    event_count: int
    context_group_count: int
    victim: int | None
    reason: str | None


def predictive_recruitment_eligibility_from_evidence(
    successes: Tensor,
    attempts: Tensor,
    birth_support: Tensor,
    min_context_attempts: float = 4.0,
    reliability_threshold: float = 0.5,
    birth_threshold: float = 0.25,
) -> PredictiveRecruitmentEligibility:
    """Find context-dependent source failures in persistent decayed evidence."""
    if successes.ndim != 3 or successes.shape != attempts.shape:
        raise ValueError("predictive successes and attempts must be matching rank-three tensors")
    n_nodes = attempts.size(0)
    if attempts.shape != (n_nodes, n_nodes, n_nodes):
        raise ValueError("predictive evidence must have shape [source, goal, context]")
    if birth_support.shape != (n_nodes,):
        raise ValueError("birth_support must have one value per graph vertex")
    if min_context_attempts <= 0 or not 0.0 < reliability_threshold < 1.0:
        raise ValueError("invalid predictive recruitment thresholds")

    reliability = (successes + 1.0) / (attempts + 2.0)
    supported = attempts >= float(min_context_attempts)
    node_index = torch.arange(n_nodes, device=attempts.device)
    valid_group = supported.clone()
    valid_group[node_index, node_index, :] = False
    valid_group[node_index, :, node_index] = False
    high = valid_group & (reliability >= float(reliability_threshold))
    low = valid_group & (reliability < float(reliability_threshold))

    reliability_gap = torch.zeros(n_nodes, device=attempts.device, dtype=attempts.dtype)
    supporting_attempts = torch.zeros_like(reliability_gap)
    predictive_failure = torch.zeros(n_nodes, dtype=torch.bool, device=attempts.device)
    for src in range(n_nodes):
        for goal in range(n_nodes):
            high_contexts = high[src, goal]
            low_contexts = low[src, goal]
            if not bool(high_contexts.any() and low_contexts.any()):
                continue
            gap = reliability[src, goal][high_contexts].max() - reliability[src, goal][low_contexts].min()
            support = attempts[src, goal][valid_group[src, goal]].sum()
            if gap > reliability_gap[src] or (gap == reliability_gap[src] and support > supporting_attempts[src]):
                reliability_gap[src] = gap
                supporting_attempts[src] = support
                predictive_failure[src] = True

    eligible = (birth_support <= float(birth_threshold)) & predictive_failure
    eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten().tolist()
    victim = None
    if eligible_indices:
        victim = int(
            max(
                eligible_indices,
                key=lambda idx: (
                    float(reliability_gap[idx].item()),
                    float(supporting_attempts[idx].item()),
                    -int(idx),
                ),
            )
        )
    return PredictiveRecruitmentEligibility(
        predictive_failure=predictive_failure,
        eligible=eligible,
        reliability_gap=reliability_gap,
        supporting_attempts=supporting_attempts,
        event_count=int((attempts > 0).sum().item()),
        context_group_count=int(valid_group.sum().item()),
        victim=victim,
        reason="predictive" if victim is not None else None,
    )


class PersistentPredictiveRecruitmentEvidence(nn.Module):
    """Checkpointed decayed PRED evidence indexed by source, goal, context."""

    def __init__(self, n_nodes: int):
        super().__init__()
        self.n_nodes = int(n_nodes)
        shape = (self.n_nodes, self.n_nodes, self.n_nodes)
        self.register_buffer("successes", torch.zeros(shape))
        self.register_buffer("attempts", torch.zeros(shape))
        self.register_buffer("update_count", torch.zeros((), dtype=torch.int64))
        self.register_buffer("invalidation_count", torch.zeros((), dtype=torch.int64))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        for name in ("successes", "attempts", "update_count", "invalidation_count"):
            state_dict.setdefault(prefix + name, getattr(self, name).clone())
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    @torch.no_grad()
    def update(
        self,
        source: Tensor,
        target: Tensor,
        context: Tensor,
        success: Tensor,
        half_life_events: float,
    ) -> dict[str, Tensor]:
        inputs = (source, target, context, success)
        if any(value.ndim != 1 for value in inputs) or len({value.numel() for value in inputs}) != 1:
            raise ValueError("predictive events must be aligned one-dimensional tensors")
        if half_life_events <= 0:
            raise ValueError("half_life_events must be positive")
        gamma = float(0.5 ** (1.0 / float(half_life_events)))
        accepted = 0
        decayed_mass = self.attempts.new_zeros(())
        for src, goal, predecessor, hit in zip(
            source.detach().cpu().tolist(),
            target.detach().cpu().tolist(),
            context.detach().cpu().tolist(),
            success.detach().cpu().tolist(),
        ):
            src, goal, predecessor = int(src), int(goal), int(predecessor)
            if not (
                0 <= src < self.n_nodes
                and 0 <= goal < self.n_nodes
                and 0 <= predecessor < self.n_nodes
                and src != goal
                and predecessor != src
            ):
                continue
            before = self.attempts.sum()
            self.attempts.mul_(gamma)
            self.successes.mul_(gamma)
            decayed_mass.add_(before - self.attempts.sum())
            self.attempts[src, goal, predecessor].add_(1.0)
            self.successes[src, goal, predecessor].add_(float(bool(hit)))
            accepted += 1
        self.update_count.add_(accepted)
        return {
            "accepted_count": self.attempts.new_tensor(float(accepted)),
            "decayed_attempt_mass": decayed_mass,
            "supported_context_count": (self.attempts > 0).sum().to(dtype=self.attempts.dtype),
        }

    def eligibility(
        self,
        birth_support: Tensor,
        min_context_attempts: float = 4.0,
        reliability_threshold: float = 0.5,
        birth_threshold: float = 0.25,
    ) -> PredictiveRecruitmentEligibility:
        return predictive_recruitment_eligibility_from_evidence(
            self.successes,
            self.attempts,
            birth_support,
            min_context_attempts,
            reliability_threshold,
            birth_threshold,
        )

    @torch.no_grad()
    def invalidate_node(self, node: int) -> Tensor:
        node = int(node)
        if node < 0 or node >= self.n_nodes:
            raise IndexError(f"DG node {node} is outside [0, {self.n_nodes})")
        invalid = torch.zeros_like(self.attempts, dtype=torch.bool)
        invalid[node, :, :] = True
        invalid[:, node, :] = True
        invalid[:, :, node] = True
        removed = self.attempts.masked_select(invalid).sum()
        self.attempts.masked_fill_(invalid, 0.0)
        self.successes.masked_fill_(invalid, 0.0)
        self.invalidation_count.add_(1)
        return removed


def directional_recruitment_eligibility(
    edge_confidence: Tensor,
    control_attempts: Tensor,
    tctrl: Tensor,
    birth_support: Tensor,
    confidence_threshold: float = 0.5,
    reliability_threshold: float = 0.5,
    attempt_threshold: float = 0.5,
    birth_threshold: float = 0.25,
    redundancy_max_steps: int = 4,
) -> DirectionalRecruitmentEligibility:
    """Classify fully tested failed sources and mutually close control duplicates."""
    matrices = (edge_confidence, control_attempts, tctrl)
    if any(matrix.ndim != 2 for matrix in matrices):
        raise ValueError("control graph inputs must be matrices")
    if edge_confidence.shape != control_attempts.shape or edge_confidence.shape != tctrl.shape:
        raise ValueError("control graph inputs must have matching shapes")
    if edge_confidence.size(0) != edge_confidence.size(1):
        raise ValueError("control graph inputs must be square")
    n_nodes = edge_confidence.size(0)
    if birth_support.shape != (n_nodes,):
        raise ValueError("birth_support must have one value per graph vertex")
    if confidence_threshold < 0 or not 0.0 < reliability_threshold <= 1.0:
        raise ValueError("invalid reliable-edge thresholds")
    if attempt_threshold <= 0 or not 0.0 < birth_threshold < 1.0:
        raise ValueError("attempt and birth thresholds must be positive")
    if redundancy_max_steps <= 0:
        raise ValueError("redundancy_max_steps must be positive")

    reliability = (edge_confidence + 1.0) / (control_attempts + 2.0)
    adjacency = (
        (tctrl > 0) & (edge_confidence >= float(confidence_threshold)) & (reliability >= float(reliability_threshold))
    )
    adjacency = adjacency.clone()
    adjacency.fill_diagonal_(False)
    out_degree = adjacency.sum(dim=1)
    outgoing_confidence = edge_confidence.masked_fill(~adjacency, 0).sum(dim=1)

    tested = control_attempts >= float(attempt_threshold)
    tested = tested.clone()
    tested.fill_diagonal_(True)
    fully_tested = tested.all(dim=1)
    zero_outdegree = out_degree.eq(0)
    mature = birth_support <= float(birth_threshold)
    bad_source = mature & fully_tested & zero_outdegree

    redundant_loser = torch.zeros(n_nodes, dtype=torch.bool, device=edge_confidence.device)
    redundant_pair_count = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if not bool(adjacency[i, j] and adjacency[j, i]):
                continue
            if tctrl[i, j] > redundancy_max_steps or tctrl[j, i] > redundancy_max_steps:
                continue
            redundant_pair_count += 1
            if out_degree[i] < out_degree[j]:
                loser = i
            elif out_degree[j] < out_degree[i]:
                loser = j
            elif outgoing_confidence[i] < outgoing_confidence[j]:
                loser = i
            elif outgoing_confidence[j] < outgoing_confidence[i]:
                loser = j
            else:
                loser = j
            redundant_loser[loser] = True

    eligible_duplicate = mature & redundant_loser
    eligible = bad_source | eligible_duplicate
    victim = None
    reason = None
    bad_indices = torch.nonzero(bad_source, as_tuple=False).flatten().tolist()
    if bad_indices:
        victim = int(min(bad_indices))
        reason = "bad_source"
    else:
        duplicate_indices = torch.nonzero(eligible_duplicate, as_tuple=False).flatten().tolist()
        if duplicate_indices:
            victim = int(
                min(
                    duplicate_indices,
                    key=lambda idx: (
                        int(out_degree[idx].item()),
                        float(outgoing_confidence[idx].item()),
                        -int(idx),
                    ),
                )
            )
            reason = "redundant"

    return DirectionalRecruitmentEligibility(
        adjacency=adjacency,
        reliability=reliability,
        out_degree=out_degree,
        outgoing_confidence=outgoing_confidence,
        fully_tested=fully_tested,
        zero_outdegree=zero_outdegree,
        bad_source=bad_source,
        redundant_loser=redundant_loser,
        eligible=eligible,
        victim=victim,
        reason=reason,
        redundant_pair_count=redundant_pair_count,
    )


def predictive_recruitment_eligibility(
    source: Tensor,
    target: Tensor,
    context: Tensor,
    success: Tensor,
    birth_support: Tensor,
    n_nodes: int,
    min_context_attempts: int = 2,
    reliability_threshold: float = 0.5,
    birth_threshold: float = 0.25,
) -> PredictiveRecruitmentEligibility:
    """Find batch-local source/goal outcomes that depend on predecessor context."""
    inputs = (source, target, context, success)
    if any(value.ndim != 1 for value in inputs) or len({value.numel() for value in inputs}) != 1:
        raise ValueError("predictive events must be aligned one-dimensional tensors")
    if birth_support.shape != (int(n_nodes),):
        raise ValueError("birth_support must have one value per graph vertex")
    if min_context_attempts <= 0 or not 0.0 < reliability_threshold < 1.0:
        raise ValueError("invalid predictive recruitment thresholds")

    device = birth_support.device
    dtype = birth_support.dtype
    reliability_gap = torch.zeros(n_nodes, device=device, dtype=dtype)
    supporting_attempts = torch.zeros(n_nodes, device=device, dtype=dtype)
    groups: dict[tuple[int, int, int], list[int]] = {}
    event_count = 0
    for src, goal, predecessor, hit in zip(
        source.detach().cpu().tolist(),
        target.detach().cpu().tolist(),
        context.detach().cpu().tolist(),
        success.detach().cpu().tolist(),
    ):
        src, goal, predecessor = int(src), int(goal), int(predecessor)
        if not (0 <= src < n_nodes and 0 <= goal < n_nodes and 0 <= predecessor < n_nodes):
            continue
        if src == goal or predecessor == src:
            continue
        key = (src, goal, predecessor)
        attempts, successes = groups.get(key, [0, 0])
        groups[key] = [attempts + 1, successes + int(bool(hit))]
        event_count += 1

    predictive_failure = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    source_goal: dict[tuple[int, int], list[tuple[float, int]]] = {}
    for (src, goal, _), (attempts, successes) in groups.items():
        if attempts < int(min_context_attempts):
            continue
        reliability = (successes + 1.0) / (attempts + 2.0)
        source_goal.setdefault((src, goal), []).append((reliability, attempts))

    for (src, _), contexts in source_goal.items():
        if len(contexts) < 2:
            continue
        reliabilities = [item[0] for item in contexts]
        if max(reliabilities) < reliability_threshold or min(reliabilities) >= reliability_threshold:
            continue
        gap = max(reliabilities) - min(reliabilities)
        attempts = sum(item[1] for item in contexts)
        if gap > float(reliability_gap[src].item()) or (
            gap == float(reliability_gap[src].item()) and attempts > float(supporting_attempts[src].item())
        ):
            reliability_gap[src] = gap
            supporting_attempts[src] = attempts
            predictive_failure[src] = True

    mature = birth_support <= float(birth_threshold)
    eligible = mature & predictive_failure
    eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten().tolist()
    victim = None
    if eligible_indices:
        victim = int(
            max(
                eligible_indices,
                key=lambda idx: (
                    float(reliability_gap[idx].item()),
                    float(supporting_attempts[idx].item()),
                    -int(idx),
                ),
            )
        )
    return PredictiveRecruitmentEligibility(
        predictive_failure=predictive_failure,
        eligible=eligible,
        reliability_gap=reliability_gap,
        supporting_attempts=supporting_attempts,
        event_count=event_count,
        context_group_count=len(groups),
        victim=victim,
        reason="predictive" if victim is not None else None,
    )


def batch_predictive_events(
    option_states: Tensor,
    sequence_core: Tensor,
    valid_steps: Tensor,
    n_nodes: int,
    R: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Extract completed within-batch goal options with a distinct CA3 predecessor."""
    from sf_working_directories.IntrMotiv.dmlab.hrl_controllable_graph import HRLStateLayout

    if option_states.ndim != 3 or option_states.size(1) != valid_steps.size(1) + 1:
        raise ValueError("option_states must have shape [batch, time+1, state]")
    if sequence_core.shape[:2] != option_states.shape[:2] or sequence_core.shape[2] != n_nodes:
        raise ValueError("sequence_core must align with option states and graph nodes")
    if sequence_core.size(-1) < R or valid_steps.shape != (
        option_states.size(0),
        option_states.size(1) - 1,
    ):
        raise ValueError("valid_steps or CA3 history does not match the rollout")

    layout = HRLStateLayout(n_nodes)
    events: list[tuple[int, int, int, bool]] = []
    for stream in range(option_states.size(0)):
        active_option: tuple[int, int, int] | None = None
        for step in range(valid_steps.size(1)):
            if not bool(valid_steps[stream, step]):
                active_option = None
                continue
            prev = option_states[stream, step]
            if prev[layout.option_reset] > 0:
                src = int(round(float(prev[layout.source].item()))) - 1
                goal = int(round(float(prev[layout.target].item()))) - 1
                predecessor = -1
                if 0 <= src < n_nodes and 0 <= goal < n_nodes and src != goal:
                    recent = (sequence_core[stream, step, :, :R] > 0).clone()
                    # The current source can occupy several CA3 trace slots.
                    # PRED asks whether a *distinct* predecessor is uniquely
                    # represented, so the source trace is not a competing
                    # predecessor candidate.
                    recent[src] = False
                    for age in range(R):
                        active = torch.nonzero(recent[:, age], as_tuple=False).flatten()
                        if active.numel() == 1 and int(active.item()) != src:
                            predecessor = int(active.item())
                            break
                active_option = (src, goal, predecessor) if predecessor >= 0 else None

            nxt = option_states[stream, step + 1]
            hit = bool(nxt[layout.target_hit] > 0)
            timeout = bool(nxt[layout.option_expired] > 0 and nxt[layout.completion_elapsed] >= 0)
            if hit or timeout:
                if active_option is not None:
                    events.append((*active_option, hit))
                active_option = None

    if not events:
        empty_int = torch.empty(0, dtype=torch.long, device=option_states.device)
        empty_bool = torch.empty(0, dtype=torch.bool, device=option_states.device)
        return empty_int, empty_int.clone(), empty_int.clone(), empty_bool
    source, target, context, success = zip(*events)
    return (
        torch.tensor(source, dtype=torch.long, device=option_states.device),
        torch.tensor(target, dtype=torch.long, device=option_states.device),
        torch.tensor(context, dtype=torch.long, device=option_states.device),
        torch.tensor(success, dtype=torch.bool, device=option_states.device),
    )


def graph_recruitment_eligibility(
    confidence: Tensor,
    elapsed: Tensor,
    birth_support: Tensor,
    connectivity_threshold: float,
    redundancy_max_steps: int,
) -> RecruitmentEligibility:
    """Classify graph-protected, isolated, and mutually redundant DG rows."""
    if confidence.ndim != 2 or confidence.shape != elapsed.shape or confidence.size(0) != confidence.size(1):
        raise ValueError("confidence and elapsed must be matching square matrices")
    n_nodes = confidence.size(0)
    if birth_support.shape != (n_nodes,):
        raise ValueError("birth_support must have one value per graph vertex")
    if not 0.0 < connectivity_threshold < 1.0:
        raise ValueError("connectivity_threshold must be in (0, 1)")
    if redundancy_max_steps <= 0:
        raise ValueError("redundancy_max_steps must be positive")

    adjacency = (confidence > float(connectivity_threshold)) & (elapsed > 0)
    adjacency = adjacency.clone()
    adjacency.fill_diagonal_(False)
    supported_confidence = torch.where(adjacency, confidence, torch.zeros_like(confidence))
    incident_support = supported_confidence.sum(dim=0) + supported_confidence.sum(dim=1)
    connected = adjacency.any(dim=0) | adjacency.any(dim=1)
    isolated = ~connected
    redundant_loser = torch.zeros(n_nodes, dtype=torch.bool, device=confidence.device)
    redundant_pair_count = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if not bool(adjacency[i, j] and adjacency[j, i]):
                continue
            if elapsed[i, j] > redundancy_max_steps or elapsed[j, i] > redundancy_max_steps:
                continue
            redundant_pair_count += 1
            if incident_support[i] < incident_support[j]:
                redundant_loser[i] = True
            elif incident_support[j] < incident_support[i]:
                redundant_loser[j] = True
            else:
                redundant_loser[j] = True

    mature = birth_support <= float(connectivity_threshold)
    eligible = mature & (isolated | redundant_loser)
    preferred = eligible & isolated
    reason = "isolated"
    if not preferred.any():
        preferred = eligible & redundant_loser
        reason = "redundant"
    victim = None
    if preferred.any():
        masked_support = incident_support.masked_fill(~preferred, torch.inf)
        victim = int(masked_support.argmin().item())
    else:
        reason = None
    return RecruitmentEligibility(
        adjacency=adjacency,
        incident_support=incident_support,
        isolated=isolated,
        redundant_loser=redundant_loser,
        eligible=eligible,
        victim=victim,
        reason=reason,
        redundant_pair_count=redundant_pair_count,
    )


class PassiveRecruitmentGraph(nn.Module):
    """Policy-scoped passive DG transition memory stored in checkpoints."""

    def __init__(self, n_nodes: int):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.register_buffer("confidence", torch.zeros(self.n_nodes, self.n_nodes))
        self.register_buffer("elapsed", torch.zeros(self.n_nodes, self.n_nodes))
        self.register_buffer("birth_support", torch.ones(self.n_nodes))
        self.register_buffer("representation_generation", torch.zeros((), dtype=torch.int64))

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
        for name in ("confidence", "elapsed", "birth_support", "representation_generation"):
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
    def decay_birth_support(self, event_count: Tensor | float | int, half_life_events: float) -> None:
        if half_life_events <= 0:
            raise ValueError("half_life_events must be positive")
        count = torch.as_tensor(event_count, device=self.birth_support.device, dtype=self.birth_support.dtype)
        gamma = torch.pow(self.birth_support.new_tensor(0.5), count / float(half_life_events))
        self.birth_support.mul_(gamma)

    @torch.no_grad()
    def invalidate_node(self, node: int) -> None:
        node = int(node)
        if node < 0 or node >= self.n_nodes:
            raise IndexError(f"DG node {node} is outside [0, {self.n_nodes})")
        self.confidence[node, :] = 0
        self.confidence[:, node] = 0
        self.elapsed[node, :] = 0
        self.elapsed[:, node] = 0
        self.birth_support[node] = 1.0
        self.representation_generation.add_(1)

    @torch.no_grad()
    def update_from_rollout(
        self,
        history_states: Tensor,
        valid_steps: Tensor,
        max_gap: int,
        half_life_events: float,
        decay_birth: bool = True,
    ) -> dict[str, Tensor]:
        """Apply accepted actor-history transitions once in flattened order."""
        if history_states.ndim != 3 or history_states.size(-1) != RECRUITMENT_HISTORY_SIZE:
            raise ValueError("history_states must have shape [batch, time+1, 3]")
        if valid_steps.shape != (history_states.size(0), history_states.size(1) - 1):
            raise ValueError("valid_steps must align with history rollout transitions")
        if max_gap <= 0 or half_life_events <= 0:
            raise ValueError("max_gap and half_life_events must be positive")

        prev = history_states[:, :-1].reshape(-1, RECRUITMENT_HISTORY_SIZE)
        nxt = history_states[:, 1:].reshape(-1, RECRUITMENT_HISTORY_SIZE)
        valid = valid_steps.bool().reshape(-1)
        generation = self.representation_generation.to(device=history_states.device, dtype=history_states.dtype)
        generation_matches = prev[:, HISTORY_GENERATION].eq(generation) & nxt[:, HISTORY_GENERATION].eq(generation)
        source = prev[:, HISTORY_LAST_EXCLUSIVE].round().long() - 1
        destination = nxt[:, HISTORY_LAST_EXCLUSIVE].round().long() - 1
        gap = prev[:, HISTORY_AGE].round().long() + 1
        endpoint_valid = (
            (source >= 0)
            & (source < self.n_nodes)
            & (destination >= 0)
            & (destination < self.n_nodes)
            & source.ne(destination)
            & nxt[:, HISTORY_AGE].eq(0)
        )
        within_gap = (gap >= 1) & (gap <= int(max_gap))
        accepted = valid & generation_matches & endpoint_valid & within_gap
        stale = valid & endpoint_valid & within_gap & ~generation_matches
        over_gap = valid & generation_matches & endpoint_valid & ~within_gap

        gamma = float(0.5 ** (1.0 / float(half_life_events)))
        for index in torch.nonzero(accepted, as_tuple=False).flatten().tolist():
            self.confidence.mul_(gamma)
            if decay_birth:
                self.birth_support.mul_(gamma)
            i = int(source[index].item())
            j = int(destination[index].item())
            old_confidence = self.confidence[i, j].clone()
            new_confidence = old_confidence + 1.0
            self.elapsed[i, j] = (
                old_confidence * self.elapsed[i, j] + gap[index].to(dtype=self.elapsed.dtype)
            ) / new_confidence.clamp_min(1e-12)
            self.confidence[i, j] = new_confidence

        dtype = self.confidence.dtype
        return {
            "accepted_count": accepted.sum().to(dtype=dtype),
            "stale_count": stale.sum().to(dtype=dtype),
            "over_gap_count": over_gap.sum().to(dtype=dtype),
        }
