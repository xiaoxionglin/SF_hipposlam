"""Stored-state DDQN/HER: replay the behavior memory, relabel only the decoder.

This explicit STOP baseline accepts representation lag. It performs no encoder,
recurrent-core, manager, normalization or graph forward during replay.
"""

from dataclasses import replace
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from .ca3_state_readout import contextual_similarity, indexed_contextual_similarity
from .controller_q import continuing_double_q_target
from .controller_snapshot import differentiable_replay, evaluate_replay
from .controller_transition import ReplayRejected, TransitionInput
from .custom_learner import worker_reward_from_magnitude
from .hrl_controllable_graph import HRLStateLayout


def canonical(core, row):
    return row.worker_state[: core.core_output_size].reshape(core.Hippo_n_feature, core.expanded_length)[:, 0]


def raw_ca3(core, row):
    return row.worker_state[: core.core_output_size]


@torch.no_grad()
def contextual_goal_hits(core, candidate_ca3, goal_ca3):
    """Batched semantic contextual hits; DG addresses are not identity labels."""
    graph = core.policy_graph
    if graph is None or not graph.contextual or not bool(graph.calibration_ready):
        count = len(candidate_ca3) if np.asarray(candidate_ca3).ndim > 1 else 1
        return torch.zeros(count, dtype=torch.bool, device=graph.anchor_ca3.device if graph is not None else "cpu")
    candidate = torch.as_tensor(candidate_ca3, device=graph.anchor_ca3.device, dtype=graph.anchor_ca3.dtype)
    goal = torch.as_tensor(goal_ca3, device=graph.anchor_ca3.device, dtype=graph.anchor_ca3.dtype)
    candidate = candidate.reshape(-1, core.core_output_size)
    goal = goal.reshape(-1, core.core_output_size)
    if candidate.shape[0] != goal.shape[0]:
        raise ValueError("contextual candidate and goal batches must have equal length")
    dg = candidate.reshape(-1, core.Hippo_n_feature, core.expanded_length)[:, :, 0]
    real_event = (dg > 0).any(dim=1)
    similarity = contextual_similarity(
        core.state_readout,
        core.innovation_predictor,
        candidate,
        goal,
        space=graph.similarity_space,
    )
    return real_event & (similarity >= graph.recognition_threshold)


@torch.no_grad()
def contextual_goal_hit(core, candidate_ca3, goal_ca3) -> bool:
    """Scalar compatibility wrapper around :func:`contextual_goal_hits`."""
    return bool(contextual_goal_hits(core, candidate_ca3, goal_ca3)[0])


def example_from_replay(learner, key, allow_stale_anchor=False, anchor_snapshot=None):
    _, rows = learner.replay.sequence(key, 0, 2)
    row = rows[0]
    if row.worker_state is None:
        raise ReplayRejected("stored_worker_state_missing")
    if row.real_reward is None:
        raise ReplayRejected("stored_reward_missing")
    if row.terminated or row.truncated:
        if not row.successor_valid:
            raise ReplayRejected("uncertified_terminal_observation")
        successor = replace(
            row, index=row.index + 1, serial=row.serial + 1, worker_state=np.zeros_like(row.worker_state)
        )
    elif len(rows) < 2:
        raise ReplayRejected("missing_successor_context")
    else:
        successor = rows[1]
    generation = int(learner.actor_critic.core.policy_graph.representation_generation)
    if row.generation != generation or successor.generation != generation:
        raise ReplayRejected("stale_structural_generation")
    graph = learner.actor_critic.core.policy_graph
    if graph.contextual and not allow_stale_anchor:
        target = np.flatnonzero(row.condition[: graph.n_nodes] > 0)
        if len(target) == 1:
            node = int(target[0])
            if anchor_snapshot is None:
                selectable = graph.selectable_mask().detach().cpu().numpy()
                anchor_generation = graph.anchor_generation.detach().cpu().numpy()
            else:
                selectable, anchor_generation = anchor_snapshot
            if not bool(selectable[node]) or row.anchor_generation != int(anchor_generation[node]):
                raise ReplayRejected("stale_anchor_generation")
    if successor.worker_state is None:
        raise ReplayRejected("stored_worker_state_missing")
    return TransitionInput((row, successor), 0)


def hindsight_examples(learner, examples):
    core = learner.actor_critic.core
    layout = HRLStateLayout(core.Hippo_n_feature)
    result = []
    contextual_mode = getattr(core, "worker_goal_mode", "target_id") == "state_readout" and core.policy_graph.contextual
    contextual_records = []
    contextual_starts = []
    contextual_goals = []
    for example in examples:
        row = example.rows[0]
        budget = int(row.context[layout.countdown])
        if budget < 1:
            learner.replay.reject("her_budget_expired")
            continue
        _, future = learner.replay.sequence(row.key, 0, budget + 1)
        start = canonical(core, row)
        contextual = contextual_mode
        if contextual and not bool(core.policy_graph.calibration_ready):
            learner.replay.reject("her_contextual_missing_calibration")
            continue
        if contextual:
            contextual_starts.append(raw_ca3(core, row))
            contextual_owner = len(result)
        goals = []
        for candidate in future[1:]:
            if candidate.generation != row.generation or candidate.worker_state is None:
                break
            dg = canonical(core, candidate)
            active = np.flatnonzero(dg > 0)
            if contextual:
                if len(active):
                    learner.replay.reject("her_contextual_candidate")
                    goal_state = raw_ca3(core, candidate).copy()
                    contextual_records.append((contextual_owner, int(active[np.argmax(dg[active])]), goal_state))
                    contextual_goals.append(goal_state)
            elif len(active) == 1 and start[active[0]] <= 0:
                goals.append(int(active[0]))
        # A certified final observation is a future achievement too. Its label
        # was encoded once before fresh DG learning, not reconstructed in replay.
        if future and getattr(core, "worker_goal_mode", "target_id") == "target_id":
            last = future[-1]
            if last.index - row.index < budget and last.generation == row.generation and last.terminal_dg is not None:
                active = np.flatnonzero(last.terminal_dg > 0)
                if len(active) == 1 and start[active[0]] <= 0:
                    goals.append(int(active[0]))
        if contextual:
            # Defer hit testing and goal selection so all action-probe
            # signatures in the HER transaction share one predictor forward.
            result.append((example, goals))
            continue
        if not goals:
            learner.replay.reject("her_no_future_achievement")
            continue
        selected = goals[int(learner.her_rng.integers(len(goals)))]
        if contextual:
            goal, goal_state = selected
        else:
            goal = selected
            endpoint = next(
                (
                    candidate
                    for candidate in future[1:]
                    if candidate.worker_state is not None and canonical(core, candidate)[goal] > 0
                ),
                None,
            )
            goal_state = None if endpoint is None else endpoint.worker_state[: core.core_output_size].copy()
        result.append(
            replace(
                example,
                virtual_goal=goal,
                remaining=budget,
                virtual_goal_state=goal_state,
            )
        )
    if not contextual_mode:
        return result

    if contextual_records:
        graph = core.policy_graph
        starts = torch.as_tensor(
            np.stack(contextual_starts), device=graph.anchor_ca3.device, dtype=graph.anchor_ca3.dtype
        )
        goals = torch.as_tensor(
            np.stack(contextual_goals), device=graph.anchor_ca3.device, dtype=graph.anchor_ca3.dtype
        )
        owners = torch.tensor([record[0] for record in contextual_records], device=graph.anchor_ca3.device)
        similarities = indexed_contextual_similarity(
            core.state_readout,
            core.innovation_predictor,
            starts,
            goals,
            owners,
            space=graph.similarity_space,
        )
        start_dg = starts.reshape(-1, core.Hippo_n_feature, core.expanded_length)[:, :, 0]
        real_events = (start_dg > 0).any(dim=1)
        hits = (real_events[owners] & (similarities >= graph.recognition_threshold)).tolist()
    else:
        hits = []
    goals_by_example = [[] for _ in result]
    for (owner, slot, goal_state), hit in zip(contextual_records, hits):
        if hit:
            learner.replay.reject("her_start_already_achieved_contextual")
        else:
            goals_by_example[owner].append((slot, goal_state))
    contextual_result = []
    for owner, (example, _) in enumerate(result):
        goals = goals_by_example[owner]
        if not goals:
            learner.replay.reject("her_no_future_achievement")
            continue
        goal, goal_state = goals[int(learner.her_rng.integers(len(goals)))]
        contextual_result.append(
            replace(
                example,
                virtual_goal=goal,
                remaining=int(example.rows[0].context[layout.countdown]),
                virtual_goal_state=goal_state,
            )
        )
    return contextual_result


def _q_batch(model, examples):
    # The existing decoder and Q heads own all replay gradients. Stored states
    # are constants even when their generating representation has since changed.
    device = next(model.parameters()).device
    outputs = []
    goal_states = []
    goal_masks = []
    for example in examples:
        for row in example.rows:
            condition = row.condition
            if example.virtual_goal is not None:
                condition = np.zeros_like(condition)
                condition[example.virtual_goal] = 1
            outputs.append(np.concatenate((row.worker_state, condition)))
            goal_states.append(
                example.virtual_goal_state
                if example.virtual_goal_state is not None
                else np.zeros(model.core.core_output_size, dtype=row.worker_state.dtype)
            )
            goal_masks.append(example.virtual_goal_state is not None)
    output_tensor = torch.as_tensor(np.stack(outputs), device=device)
    if getattr(model.core, "readout_mode", "off") == "worker":
        hidden = model.controller_hidden(
            output_tensor,
            torch.as_tensor(np.stack(goal_states), device=device),
            torch.as_tensor(goal_masks, device=device),
        )
    else:
        hidden = model.controller_hidden(output_tensor)
    result = [None] * len(examples)
    for auxiliary in (False, True):
        indices = [i for i, e in enumerate(examples) if (e.virtual_goal is not None) == auxiliary]
        if not indices:
            continue
        positions = torch.tensor([[2 * i, 2 * i + 1] for i in indices], device=device).flatten()
        if auxiliary:
            remaining = torch.tensor(
                [[examples[i].remaining, max(0, examples[i].remaining - 1)] for i in indices], device=device
            ).flatten()
            q = model.controller_q.hindsight(hidden[positions], remaining, float(model.cfg.Hippo_L))
        else:
            q = model.controller_q(hidden[positions])
        for i, value in zip(indices, q.split(2)):
            result[i] = value
    return result


def evaluate_pairs(learner, examples):
    core = learner.actor_critic.core
    n = core.Hippo_n_feature
    layout = HRLStateLayout(n)
    device = learner.device
    results = [None] * len(examples)
    kept = []
    rewards = []
    dones = []
    contextual_hits = {}
    contextual_examples = [
        (i, e)
        for i, e in enumerate(examples)
        if e.virtual_goal is not None
        and getattr(core, "worker_goal_mode", "target_id") == "state_readout"
        and core.policy_graph.contextual
        and e.virtual_goal_state is not None
    ]
    if contextual_examples and bool(core.policy_graph.calibration_ready):
        goals = np.stack([e.virtual_goal_state for _, e in contextual_examples])
        starts = np.stack([raw_ca3(core, e.rows[0]) for _, e in contextual_examples])
        successors = np.stack([raw_ca3(core, e.rows[1]) for _, e in contextual_examples])
        start_hits = contextual_goal_hits(core, starts, goals).tolist()
        successor_hits = contextual_goal_hits(core, successors, goals).tolist()
        contextual_hits = {
            i: (start_hit, successor_hit)
            for (i, _), start_hit, successor_hit in zip(contextual_examples, start_hits, successor_hits)
        }
    for i, e in enumerate(examples):
        row, successor = e.rows
        physical = row.terminated or row.truncated
        reward = row.real_reward
        ended = physical
        if e.virtual_goal is not None:
            contextual = (
                getattr(core, "worker_goal_mode", "target_id") == "state_readout"
                and core.policy_graph.contextual
                and e.virtual_goal_state is not None
            )
            if contextual:
                if not bool(core.policy_graph.calibration_ready):
                    results[i] = "her_contextual_missing_calibration"
                    continue
                start_hit, successor_hit = contextual_hits.get(i, (False, False))
                if start_hit:
                    results[i] = "her_start_already_achieved_contextual"
                    continue
            elif canonical(core, row)[e.virtual_goal] > 0:
                results[i] = "her_start_already_achieved"
                continue
            if e.remaining is None or e.remaining < 1:
                results[i] = "her_budget_expired"
                continue
            if physical and contextual:
                results[i] = "her_terminal_successor_ca3_missing"
                continue
            dg = row.terminal_dg if physical else canonical(core, successor)
            if dg is None:
                results[i] = "stored_terminal_label_missing"
                continue
            magnitude = (row.real_events or {}).get("hrl_control_reward_magnitude")
            if magnitude is None:
                results[i] = "stored_reward_magnitude_missing"
                continue
            active = np.flatnonzero(dg > 0)
            node = int(active[0]) if len(active) == 1 else -1
            if contextual:
                hit = successor_hit
                same_slot = e.virtual_goal in active
                if hit:
                    learner.replay.reject("her_contextual_positive_hit")
                elif same_slot:
                    learner.replay.reject("her_contextual_same_dg_wrong_context")
                wrong = len(active) > 0 and not hit
            else:
                hit = node == e.virtual_goal
                wrong = node >= 0 and node != e.virtual_goal and node != int(row.context[layout.source]) - 1
            reward = worker_reward_from_magnitude(
                torch.tensor([[magnitude]]),
                torch.tensor([[hit]]),
                control_outcome=getattr(learner.cfg, "hrl_control_outcome", "target_hit"),
                wrong_outcome=torch.tensor([[wrong]]),
                n_targets=n,
            ).item()
            ended = physical or hit or e.remaining <= 1
        if reward is None:
            results[i] = "stored_reward_missing"
            continue
        kept.append(i)
        rewards.append(reward)
        dones.append(ended)
    if not kept:
        return results
    if not torch.is_grad_enabled():
        for i in kept:
            results[i] = (torch.zeros((), device=device), {})
        return results
    selected = [examples[i] for i in kept]
    online = differentiable_replay(
        learner.online_snapshot, learner.actor_critic, _q_batch, selected, source_version=learner.controller_version
    )
    target = evaluate_replay(learner.target_snapshot, _q_batch, selected)
    desired = continuing_double_q_target(
        torch.tensor(rewards, device=device),
        torch.tensor(dones, device=device),
        torch.stack([q[1] for q in online]),
        torch.stack([q[1] for q in target]),
        learner.cfg.gamma,
    )
    prediction = torch.stack([q[0, e.rows[0].action] for q, e in zip(online, selected)])
    losses = F.smooth_l1_loss(prediction, desired, reduction="none")
    for i, loss, q in zip(kept, losses, online):
        results[i] = (loss, {"q": q})
    return results
