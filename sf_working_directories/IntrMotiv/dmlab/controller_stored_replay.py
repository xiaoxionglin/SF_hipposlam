"""Stored-state DDQN/HER: replay the behavior memory, relabel only the decoder.

This explicit STOP baseline accepts representation lag. It performs no encoder,
recurrent-core, manager, normalization or graph forward during replay.
"""

from dataclasses import replace
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from .controller_q import continuing_double_q_target
from .controller_snapshot import differentiable_replay, evaluate_replay
from .controller_transition import ReplayRejected, TransitionInput
from .custom_learner import worker_reward_from_magnitude
from .hrl_controllable_graph import HRLStateLayout


def canonical(core, row):
    return row.worker_state[: core.core_output_size].reshape(core.Hippo_n_feature, core.expanded_length)[:, 0]


def example_from_replay(learner, key):
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
    if successor.worker_state is None:
        raise ReplayRejected("stored_worker_state_missing")
    return TransitionInput((row, successor), 0)


def hindsight_examples(learner, examples):
    core = learner.actor_critic.core
    layout = HRLStateLayout(core.Hippo_n_feature)
    result = []
    for example in examples:
        row = example.rows[0]
        budget = int(row.context[layout.countdown])
        if budget < 1:
            learner.replay.reject("her_budget_expired")
            continue
        _, future = learner.replay.sequence(row.key, 0, budget + 1)
        start = canonical(core, row)
        goals = []
        for candidate in future[1:]:
            if candidate.generation != row.generation or candidate.worker_state is None:
                break
            dg = canonical(core, candidate)
            active = np.flatnonzero(dg > 0)
            if len(active) == 1 and start[active[0]] <= 0:
                goals.append(int(active[0]))
        # A certified final observation is a future achievement too. Its label
        # was encoded once before fresh DG learning, not reconstructed in replay.
        if future:
            last = future[-1]
            if last.index - row.index < budget and last.generation == row.generation and last.terminal_dg is not None:
                active = np.flatnonzero(last.terminal_dg > 0)
                if len(active) == 1 and start[active[0]] <= 0:
                    goals.append(int(active[0]))
        if not goals:
            learner.replay.reject("her_no_future_achievement")
            continue
        result.append(replace(example, virtual_goal=goals[int(learner.her_rng.integers(len(goals)))], remaining=budget))
    return result


def _q_batch(model, examples):
    # The existing decoder and Q heads own all replay gradients. Stored states
    # are constants even when their generating representation has since changed.
    device = next(model.parameters()).device
    outputs = []
    for example in examples:
        for row in example.rows:
            condition = row.condition
            if example.virtual_goal is not None:
                condition = np.zeros_like(condition)
                condition[example.virtual_goal] = 1
            outputs.append(np.concatenate((row.worker_state, condition)))
    hidden = model.controller_hidden(torch.as_tensor(np.stack(outputs), device=device))
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
    for i, e in enumerate(examples):
        row, successor = e.rows
        physical = row.terminated or row.truncated
        reward = row.real_reward
        ended = physical
        if e.virtual_goal is not None:
            if canonical(core, row)[e.virtual_goal] > 0:
                results[i] = "her_start_already_achieved"
                continue
            if e.remaining is None or e.remaining < 1:
                results[i] = "her_budget_expired"
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
