"""Snapshot-consistent worker values and original reward/event reconstruction."""

from dataclasses import dataclass

import numpy as np
import torch

from .controller_history import (
    ControllerHistory,
    reconstruct_controller_histories,
    reconstruct_controller_history,
    reconstruction_head,
)
from .goal_conditioned_dg import GoalConditionedDGCore
from .hrl_controllable_graph import HRLStateLayout, current_dg_from_activity, hrl_option_state_size
from .topological_frontier import MODE_NAVIGATE, TopologicalStateLayout


class ReplayRejected(ValueError):
    pass


@dataclass(frozen=True)
class TransitionInput:
    rows: tuple
    burn_in: int
    virtual_goal: int | None = None
    remaining: int | None = None
    virtual_goal_state: np.ndarray | None = None


def assemble_state(core, output, context):
    canonical = torch.cat((output[: core.base_state_size], context), -1).unsqueeze(0)
    if isinstance(core, GoalConditionedDGCore):
        canonical = core.join_worker_state(canonical, output[core.total_output_size :].unsqueeze(0))
    return canonical


def event_signature(context, n):
    layout = HRLStateLayout(n)
    return torch.stack(
        (
            context[..., layout.active_dg],
            context[..., layout.multi_activation],
            context[..., layout.target_hit],
            context[..., layout.option_expired],
            context[..., layout.completion_elapsed].sign(),
        ),
        -1,
    )


def make_history(model, example):
    core = model.core
    device = next(model.parameters()).device
    rows = example.rows
    b = example.burn_in
    obs = {k: torch.as_tensor(np.stack([r.observation[k] for r in rows]), device=device) for k in rows[0].observation}
    conditions = torch.as_tensor(np.stack([r.condition for r in rows]), device=device)
    if example.virtual_goal is not None:
        conditions = torch.zeros_like(conditions)
        conditions[:, example.virtual_goal] = 1
    return ControllerHistory(
        obs,
        torch.tensor([r.index for r in rows], device=device),
        torch.zeros(1, core.total_state_size, device=device),
        conditions,
        rows[0].index == 0,
        b,
    )


@torch.no_grad()
def current_context_rejections(model, examples):
    """Screen the existing current-label veto before expensive finite histories.

    Canonical trace slot zero is exactly the current unconditioned DG activity.
    This uses each private snapshot's encoder and SF normalization; no worker
    history, target choice, reward or replay acceptance rule is approximated.
    """
    from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs

    if model.training:
        raise ValueError("Reconstruction requires an immutable evaluation snapshot")
    if not examples:
        return []
    core = model.core
    device = next(model.parameters()).device
    generation = int(core.policy_graph.representation_generation.item())
    layout = HRLStateLayout(core.Hippo_n_feature)
    results = []
    indices = []
    rows = []
    for i, example in enumerate(examples):
        b = example.burn_in
        history = example.rows
        if b < 0 or len(history) != b + 2:
            raise ValueError("Expected prefix, current and physical successor")
        if any(y.index != x.index + 1 for x, y in zip(history, history[1:])):
            raise ValueError("Replay history has a physical decision gap")
        if history[0].index != 0 and b < core.expanded_length:
            raise ValueError("Incomplete finite washout prefix")
        stale = any(row.generation != generation for row in history)
        results.append("snapshot_structural_generation_incompatible" if stale else None)
        if not stale:
            indices.append(i)
            rows.append(history[b])
    if not rows:
        return results
    obs = {
        key: torch.as_tensor(np.stack([row.observation[key] for row in rows]), device=device)
        for key in rows[0].observation
    }
    head = reconstruction_head(model, obs)
    active, _, count = current_dg_from_activity(head[:, : core.Hippo_n_feature])
    active = active.cpu().tolist()
    count = count.cpu().tolist()
    for i, row, node, n_active in zip(indices, rows, active, count):
        if node + 1 != row.context[layout.active_dg] or (n_active > 1) != bool(row.context[layout.multi_activation]):
            results[i] = "current_recognition_changed"
    return results


def transition_values_batch(model, examples, *, screen_current=False):
    if screen_current:
        results = current_context_rejections(model, examples)
        indices = [i for i, result in enumerate(results) if result is None]
        accepted = transition_values_batch(model, [examples[i] for i in indices])
        for i, result in zip(indices, accepted):
            results[i] = result
        return results
    from .custom_learner import DistanceLearnerReward

    if not examples:
        return []
    histories = [make_history(model, example) for example in examples]
    reconstructed = reconstruct_controller_histories(model, histories)
    prepared = []
    indices = []
    results = [None] * len(examples)
    for index, (example, history) in enumerate(zip(examples, reconstructed)):
        try:
            prepared.append(_prepare_transition(model, example, history))
            indices.append(index)
        except ReplayRejected as exc:
            results[index] = str(exc)
    if not prepared:
        return results
    core = model.core
    n = core.Hippo_n_feature
    device = next(model.parameters()).device
    layout = HRLStateLayout(n)
    with torch.no_grad():
        states = torch.cat([p["state"] for p in prepared])
        currents = torch.stack([p["current"] for p in prepared])
        next_dg = torch.stack([p["next_dg"] for p in prepared])
        hrl = core._split_state(states)[1].clone()
        next_hrl, _ = core._update_hrl(hrl, next_dg, currents[:, : core.core_output_size])
        descriptors = torch.stack([p["contexts"][p["b"] + 1, -core.behavior_goal_state_size :] for p in prepared])
        next_contexts = torch.cat((next_hrl, descriptors), -1)
        signatures = event_signature(next_contexts, n)
        recorded = event_signature(torch.stack([p["contexts"][p["b"] + 1] for p in prepared]), n)
        compatible = (signatures == recorded).all(-1).cpu().tolist()
        physical = [p["rows"][p["b"]].terminated or p["rows"][p["b"]].truncated for p in prepared]
        keep = []
        for i, p in enumerate(prepared):
            if examples[indices[i]].virtual_goal is None and not physical[i] and not compatible[i]:
                results[indices[i]] = "successor_events_changed"
            else:
                keep.append(i)
        if not keep:
            return results
        previous = torch.cat(
            [
                (
                    assemble_state(
                        core,
                        prepared[i]["outputs"][prepared[i]["b"] - 1].detach(),
                        prepared[i]["contexts"][prepared[i]["b"] - 1],
                    )
                    if prepared[i]["b"]
                    else torch.zeros_like(prepared[i]["state"])
                )
                for i in keep
            ]
        )
        successors = torch.cat(
            [assemble_state(core, prepared[i]["next_output"].detach(), next_contexts[i]) for i in keep]
        )
        adapter = object.__new__(DistanceLearnerReward)
        adapter.cfg = model.cfg
        adapter.actor_critic = model
        buff = {
            "rnn_states": torch.stack((previous, states[keep]), 1),
            "rewards": torch.zeros(len(keep), 1, device=device),
            "dones": torch.tensor([[physical[i]] for i in keep], device=device),
        }
        adapter._calculate_reward_components(buff, {"new_rnn_states": successors})
        rewards = buff["rewards"].flatten()
        hits = (next_contexts[:, layout.target_hit] > 0).cpu().tolist()
    for j, i in enumerate(keep):
        p = prepared[i]
        example = examples[indices[i]]
        history = reconstructed[indices[i]]
        hidden = history["hidden"]
        if getattr(model.cfg, "controller_learning", "ddqn") == "shadow":
            hidden = hidden.detach()
        if example.virtual_goal is None:
            q = model.controller_q(hidden)
        else:
            remaining = torch.tensor([example.remaining, max(0, example.remaining - 1)], device=device)
            q = model.controller_q.hindsight(hidden, remaining, float(model.cfg.Hippo_L))
        ended = physical[i] or (example.virtual_goal is not None and (hits[i] or example.remaining <= 1))
        results[indices[i]] = dict(
            q=q,
            reward=rewards[j],
            signature=signatures[i],
            done=ended,
            canonical=p["canonical_dg"].detach(),
            next_canonical=p["next_dg"].detach(),
        )
    return results


def _prepare_transition(model, example, reconstructed=None):
    """Only private snapshots call this; real manager/graph state is never updated."""
    from .custom_learner import DistanceLearnerReward

    core = model.core
    device = next(model.parameters()).device
    rows = example.rows
    b = example.burn_in
    n = core.Hippo_n_feature
    if len(rows) != b + 2:
        raise ValueError("Expected prefix, current and physical successor")
    generation = int(core.policy_graph.representation_generation.item())
    if any(row.generation != generation for row in rows):
        raise ReplayRejected("snapshot_structural_generation_incompatible")
    if reconstructed is None:
        reconstructed = reconstruct_controller_history(model, make_history(model, example))
    contexts = torch.as_tensor(np.stack([r.context for r in rows]), device=device)
    outputs = reconstructed["all_core_outputs"]
    current, next_output = outputs[b : b + 2]
    canonical_dg = current[: core.core_output_size].reshape(n, core.expanded_length)[:, 0]
    next_dg = next_output[: core.core_output_size].reshape(n, core.expanded_length)[:, 0]
    layout = HRLStateLayout(n)
    # Recognition must agree with the context on which the actual manager acted.
    with torch.no_grad():
        active, _, count = current_dg_from_activity(canonical_dg[None])
        # Burn-in is reconstructed under this snapshot with exogenous actual
        # commands, just like actor publication rebuilds. Historical recognition
        # may change; it is not a counterfactual re-plan of the real manager.
        # Validate the action-time context and successor events at the evaluated
        # transition below. Temporal bonuses use the reconstructed history.
        if active.item() + 1 != contexts[b, layout.active_dg].item() or bool(count.item() > 1) != bool(
            contexts[b, layout.multi_activation].item()
        ):
            raise ReplayRejected("current_recognition_changed")
        context = contexts[b].clone()
        if example.virtual_goal is not None:
            goal = example.virtual_goal
            if canonical_dg[goal] > 0:
                raise ReplayRejected("her_start_already_achieved")
            if example.remaining is None or example.remaining < 1:
                raise ReplayRejected("her_budget_expired")
            context[layout.target] = goal + 1
            context[layout.countdown] = example.remaining
            context[layout.selected_deadline] = example.remaining
            context[layout.age] = 0
            topo = TopologicalStateLayout(n)
            offset = hrl_option_state_size(n)
            context[offset + topo.mode] = MODE_NAVIGATE
            context[offset + topo.final_goal] = goal + 1
            context[offset + topo.pending_source] = 0
            context[offset + topo.pending_destination] = 0
        state = assemble_state(core, current.detach(), context)
    return dict(
        state=state,
        current=current,
        next_output=next_output,
        canonical_dg=canonical_dg,
        next_dg=next_dg,
        contexts=contexts,
        outputs=outputs,
        b=b,
        rows=rows,
        reconstructed=reconstructed,
    )


def transition_values(model, example, reconstructed=None):
    from .custom_learner import DistanceLearnerReward

    p = _prepare_transition(model, example, reconstructed)
    core = model.core
    device = next(model.parameters()).device
    n = core.Hippo_n_feature
    layout = HRLStateLayout(n)
    state = p["state"]
    current = p["current"]
    next_output = p["next_output"]
    canonical_dg = p["canonical_dg"]
    next_dg = p["next_dg"]
    contexts = p["contexts"]
    outputs = p["outputs"]
    b = p["b"]
    rows = p["rows"]
    reconstructed = p["reconstructed"]
    with torch.no_grad():
        hrl = core._split_state(state)[1].clone()
        next_hrl, _ = core._update_hrl(hrl, next_dg.detach()[None], current[: core.core_output_size].detach()[None])
        # Descriptor is an actual action-time label. It is not a new graph choice.
        next_context = torch.cat((next_hrl[0], contexts[b + 1, -core.behavior_goal_state_size :]), -1)
        signature = event_signature(next_context, n)
        physical_done = rows[b].terminated or rows[b].truncated
        if example.virtual_goal is None and not physical_done:
            if not torch.equal(signature, event_signature(contexts[b + 1], n)):
                raise ReplayRejected("successor_events_changed")
        previous = assemble_state(core, outputs[b - 1].detach(), contexts[b - 1]) if b else torch.zeros_like(state)
        successor = assemble_state(core, next_output.detach(), next_context)
        # Invoke the parent's exact temporal and control-reward functions, without
        # credit assignment, DG losses, normalizer updates, or graph accumulation.
        adapter = object.__new__(DistanceLearnerReward)
        adapter.cfg = model.cfg
        adapter.actor_critic = model
        buff = {
            "rnn_states": torch.stack((previous[0], state[0]))[None],
            "rewards": torch.zeros(1, 1, device=device),
            "dones": torch.tensor([[physical_done]], device=device),
        }
        adapter._calculate_reward_components(buff, {"new_rnn_states": successor})
        reward = buff["rewards"].reshape(())
        hit = next_context[layout.target_hit] > 0
        ended = physical_done or (example.virtual_goal is not None and (bool(hit) or example.remaining <= 1))
    hidden = reconstructed["hidden"]
    if getattr(model.cfg, "controller_learning", "ddqn") == "shadow":
        hidden = hidden.detach()
    if example.virtual_goal is None:
        q = model.controller_q(hidden)
    else:
        remaining = torch.tensor([example.remaining, max(0, example.remaining - 1)], device=device)
        q = model.controller_q.hindsight(hidden, remaining, float(model.cfg.Hippo_L))
    return {
        "q": q,
        "reward": reward,
        "signature": signature,
        "done": ended,
        "canonical": canonical_dg.detach(),
        "next_canonical": next_dg.detach(),
    }
