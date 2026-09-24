#!/usr/bin/env python3
"""Manifest-driven, frozen-policy target-control intervention evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sf_working_directories.IntrMotiv.dmlab.topological_frontier import (
    GEOMETRY_POLICY_SIZE,
    MODE_NAVIGATE,
    N_MANAGER_MODES,
)
from sf_working_directories.IntrMotiv.evaluation.place_fields import load_policy_env

WORKSPACE_ROOT = pathlib.Path("/work/classic/fr_xl1014-train")
MANIFEST_COLUMNS = (
    "condition",
    "family",
    "schedule",
    "feedback",
    "half_life",
    "seed",
    "target_frames",
    "checkpoint_frames",
    "checkpoint",
    "run_dir",
    "label_suffix",
)


def workspace_path(value: str | pathlib.Path, label: str, *, must_exist: bool = False) -> pathlib.Path:
    path = pathlib.Path(value).expanduser().resolve(strict=False)
    try:
        path.relative_to(WORKSPACE_ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must be under {WORKSPACE_ROOT}, got {path}") from error
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def load_manifest_row(manifest: pathlib.Path, row_index: int) -> dict[str, str]:
    manifest = workspace_path(manifest, "Manifest", must_exist=True)
    with manifest.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError("Intervention input must use the standard place-field manifest columns")
        rows = list(reader)
    if not 0 <= row_index < len(rows):
        raise IndexError(f"Manifest row {row_index} outside [0, {len(rows)})")
    row = dict(rows[row_index])
    workspace_path(row["run_dir"], "Run directory", must_exist=True)
    workspace_path(row["checkpoint"], "Checkpoint", must_exist=True)
    return row


def balanced_target(
    counts: np.ndarray,
    source: int,
    attempts_per_pair: int,
    observed: np.ndarray | None = None,
) -> int | None:
    """Choose the least-sampled alternative with deterministic target-id ties."""
    eligible = [
        target
        for target in range(counts.shape[1])
        if target != source
        and (observed is None or bool(observed[target]))
        and counts[source, target] < attempts_per_pair
    ]
    if not eligible:
        return None
    minimum = min(int(counts[source, target]) for target in eligible)
    return min(target for target in eligible if counts[source, target] == minimum)


def pair_deadline(graph, source: int, target: int, fallback: int = 64) -> int:
    reliability = float(
        ((graph.edge_confidence[source, target] + 1.0) / (graph.control_attempts[source, target] + 2.0)).item()
    )
    confidence = float(graph.edge_confidence[source, target].item())
    tctrl = float(graph.tctrl[source, target].item())
    if tctrl > 0 and confidence >= 0.5 and reliability >= 0.5:
        return max(1, math.ceil(1.2 * tctrl) + 2)
    return int(fallback)


def target_geometry(graph, source: int, target: int, device, dtype) -> torch.Tensor:
    result = torch.zeros(GEOMETRY_POLICY_SIZE, device=device, dtype=dtype)
    if graph is None or not bool(graph.pose_valid[source]) or not bool(graph.pose_valid[target]):
        return result
    source_pose = graph.landmark_pose[source]
    target_pose = graph.landmark_pose[target]
    delta = target_pose[:2] - source_pose[:2]
    cosine, sine = torch.cos(source_pose[2]), torch.sin(source_pose[2])
    dx = cosine * delta[0] + sine * delta[1]
    dy = -sine * delta[0] + cosine * delta[1]
    dtheta = target_pose[2] - source_pose[2]
    return torch.stack((dx / 32.0, dy / 32.0, torch.sin(dtheta), torch.cos(dtheta))).to(device=device, dtype=dtype)


def condition_for_target(actor_critic, core_output: torch.Tensor, source: int, target: int) -> torch.Tensor:
    conditioned = core_output.clone()
    core = actor_critic.core
    n_nodes = int(core.Hippo_n_feature)
    target_start = int(core.target_condition_start)
    conditioned[:, target_start : target_start + n_nodes] = 0
    conditioned[:, target_start + target] = 1
    if getattr(core, "landmark_geometry", "none") == "se2":
        geometry = target_geometry(
            getattr(core, "policy_graph", None), source, target, conditioned.device, conditioned.dtype
        )
        geometry_start = int(core.geometry_condition_start)
        conditioned[:, geometry_start : geometry_start + GEOMETRY_POLICY_SIZE] = geometry
    if bool(getattr(core, "behavior_mode_condition", False)):
        mode_start = int(core.mode_condition_start)
        conditioned[:, mode_start : mode_start + N_MANAGER_MODES] = 0
        conditioned[:, mode_start + MODE_NAVIGATE] = 1
    return conditioned


def counterfactual_action_sensitivity(actor_critic, core_output, source: int) -> float:
    targets = [target for target in range(actor_critic.core.Hippo_n_feature) if target != source]
    conditioned = torch.cat(
        [condition_for_target(actor_critic, core_output, source, target) for target in targets], dim=0
    )
    result = actor_critic.forward_tail(conditioned, values_only=False, sample_actions=False, action_mask=None)
    probabilities = F.softmax(result["action_logits"], dim=-1)
    if probabilities.size(0) < 2:
        return 0.0
    distance = 0.5 * torch.cdist(probabilities, probabilities, p=1)
    upper = torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)
    return float(distance[upper].mean().item())


def position_bin(position: np.ndarray, width: float = 100.0) -> tuple[int, int]:
    return int(math.floor(float(position[0]) / width)), int(math.floor(float(position[1]) / width))


def orientation_bin(rotation: np.ndarray, bins: int = 16) -> int:
    yaw = float(rotation[1])
    if abs(yaw) > 2.0 * math.pi + 1e-6:
        yaw = math.radians(yaw)
    wrapped = yaw % (2.0 * math.pi)
    return min(bins - 1, int(math.floor(wrapped / (2.0 * math.pi) * bins)))


@dataclass
class Trial:
    source: int
    target: int
    source_position: np.ndarray
    source_rotation: np.ndarray
    deadline: int
    elapsed: int = 0
    path_length: float = 0.0
    hit_mask: int = 0
    sensitivity_sum: float = 0.0
    sensitivity_count: int = 0
    last_position: np.ndarray = field(default_factory=lambda: np.zeros(3))


def classify_trial_completion(
    trial: Trial,
    exclusive_node: int,
    first_distinct: bool,
) -> tuple[int, str] | None:
    """Return the frozen-evaluation outcome, or None while the trial continues."""
    distinct = exclusive_node >= 0 and exclusive_node != trial.source
    if first_distinct and distinct:
        return exclusive_node, "first_distinct"
    if exclusive_node == trial.target:
        return exclusive_node, "target_hit"
    if trial.elapsed >= trial.deadline:
        return -1, "timeout"
    return None


def as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def trial_row(
    trial: Trial,
    endpoint: np.ndarray,
    outcome: int,
    completion_reason: str,
    decision: int,
) -> dict[str, object]:
    x_bin, y_bin = position_bin(trial.source_position)
    return {
        "source": trial.source,
        "target": trial.target,
        "source_x": float(trial.source_position[0]),
        "source_y": float(trial.source_position[1]),
        "source_z": float(trial.source_position[2]),
        "source_rot_x": float(trial.source_rotation[0]),
        "source_rot_y": float(trial.source_rotation[1]),
        "source_rot_z": float(trial.source_rotation[2]),
        "source_x_bin": x_bin,
        "source_y_bin": y_bin,
        "source_orientation_bin": orientation_bin(trial.source_rotation),
        "outcome": int(outcome),
        "success": int(outcome == trial.target),
        "completion_reason": completion_reason,
        "elapsed": trial.elapsed,
        "deadline": trial.deadline,
        "path_length": trial.path_length,
        "endpoint_x": float(endpoint[0]),
        "endpoint_y": float(endpoint[1]),
        "endpoint_z": float(endpoint[2]),
        "hit_mask": trial.hit_mask,
        "counterfactual_action_sensitivity": (
            trial.sensitivity_sum / trial.sensitivity_count if trial.sensitivity_count else math.nan
        ),
        "completion_decision": decision,
    }


def add_matched_shuffled_targets(rows: list[dict[str, object]], n_nodes: int) -> None:
    """Attach a deterministic nearest-context target from another same-source trial."""
    for index, row in enumerate(rows):
        candidates = [
            (other_index, other)
            for other_index, other in enumerate(rows)
            if other_index != index and other["source"] == row["source"] and other["target"] != row["target"]
        ]
        if not candidates:
            row["shuffled_target"] = -1
            row["shuffled_success"] = math.nan
            continue

        def context_distance(item):
            other_index, other = item
            orientation_delta = abs(int(other["source_orientation_bin"]) - int(row["source_orientation_bin"]))
            orientation_delta = min(orientation_delta, 16 - orientation_delta)
            return (
                abs(int(other["source_x_bin"]) - int(row["source_x_bin"]))
                + abs(int(other["source_y_bin"]) - int(row["source_y_bin"]))
                + orientation_delta,
                other_index,
            )

        _, donor = min(candidates, key=context_distance)
        shuffled_target = int(donor["target"])
        row["shuffled_target"] = shuffled_target
        row["shuffled_success"] = int(bool(int(row["hit_mask"]) & (1 << shuffled_target)))


def summarize_trials(
    rows: list[dict[str, object]],
    counts: np.ndarray,
    eligible_pairs: np.ndarray,
    decision_cap: int,
) -> dict[str, object]:
    executed = [float(row["success"]) for row in rows]
    shuffled = [float(row["shuffled_success"]) for row in rows if math.isfinite(float(row["shuffled_success"]))]
    executed_rate = float(np.mean(executed)) if executed else math.nan
    shuffled_rate = float(np.mean(shuffled)) if shuffled else math.nan
    lift = executed_rate / shuffled_rate if shuffled and shuffled_rate > 0 else math.inf
    relative_gain = (executed_rate - shuffled_rate) / shuffled_rate if shuffled and shuffled_rate > 0 else math.inf
    return {
        "trial_count": len(rows),
        "ordered_pairs_eligible": int(eligible_pairs.sum()),
        "ordered_pairs_with_one_attempt": int(((counts > 0) & eligible_pairs).sum()),
        "ordered_pairs_complete": int(((counts >= 5) & eligible_pairs).sum()),
        "executed_target_success_rate": executed_rate,
        "matched_shuffled_target_success_rate": shuffled_rate,
        "executed_over_shuffled_lift": lift,
        "executed_over_shuffled_relative_gain": relative_gain,
        "mean_counterfactual_action_sensitivity": (
            float(np.nanmean([row["counterfactual_action_sensitivity"] for row in rows])) if rows else math.nan
        ),
        "decision_cap": decision_cap,
    }


def run_landmark_matched_interventions(
    cfg,
    env,
    env_info,
    actor,
    checkpoint,
    device,
    decision_cap,
    deterministic=False,
    max_sources=16,
    targets_per_source=4,
    repeats=5,
    prefix_cap=512,
    focus_sources=None,
    focus_target=None,
    reward_center=None,
    reward_cell=None,
    physical_horizon=None,
    discovery_multiplier=2,
):
    """Execute different commands from identical engine/prefix states.

    Prefixes are selected by exclusive source observations, never trial success.
    Only observed outgoing passive pairs form the evaluation panel. Both a
    target's own command and all selected alternative commands score that target.
    """
    from sample_factory.algo.utils.make_env import make_env_func_batched

    n = int(cfg.Hippo_n_feature)
    if focus_target is not None and not 0 <= focus_target < n:
        raise ValueError("focus_target is outside the DG capacity")
    if focus_sources is not None and any(not 0 <= source < n for source in focus_sources):
        raise ValueError("focus_sources contains an invalid DG ID")
    graph = actor.core.policy_graph
    before = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    pairs = (graph.passive_confidence > 0).cpu().numpy()
    np.fill_diagonal(pairs, False)
    rows, panel, missing = [], {}, []
    decisions = 0

    def fresh(seed):
        nonlocal env
        env.close()
        ecfg = AttrDict(dict(cfg))
        ecfg.dmlab_use_level_cache = False
        env = make_env_func_batched(
            ecfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=None
        )
        base = env.unwrapped
        if hasattr(base, "reset_on_init"):
            base.reset_on_init = False
        base.seed(seed)
        torch.manual_seed(seed)
        obs, _ = env.reset()
        return obs, torch.zeros(1, get_rnn_size(cfg), device=device)

    def advance(action):
        nonlocal decisions
        if decisions >= decision_cap:
            raise RuntimeError("Matched intervention decision cap exhausted")
        actions = torch.as_tensor([[action]], device=device)
        obs, _, term, trunc, _ = env.step(preprocess_actions(env_info, actions))
        decisions += 1
        return obs, bool(make_dones(term, trunc)[0])

    def encode(obs, state, command=None):
        h = actor.forward_head(prepare_and_normalize_obs(actor, obs))
        if command is not None and getattr(cfg, "dg_goal_input", "none") == "write":
            goal = h.new_zeros((1, n))
            goal[:, command] = 1
            h = torch.cat((h, goal), -1)
        out, nxt = actor.forward_core(h, state)
        if command is not None:
            out = condition_for_target(actor, out, -1, command)
        return h[:, :n], out, nxt

    def signature(obs, state):
        return {
            **{k: v.detach().clone() for k, v in obs.items() if torch.is_tensor(v)},
            "__state": state.detach().clone(),
        }

    def replay(seed, prefix):
        obs, state = fresh(seed)
        for action in prefix:
            _, _, state = encode(obs, state)
            obs, done = advance(action)
            if done:
                raise RuntimeError("Episode boundary in retained prefix")
        return obs, state

    try:
        with torch.no_grad():
            # Bounded discovery; record multiple starts per source where possible.
            for attempt in range(max_sources * repeats * discovery_multiplier):
                if decisions + prefix_cap >= decision_cap // 4:
                    break
                seed = 31000 + attempt
                obs, state = fresh(seed)
                prefix = []
                actions = np.random.default_rng(seed).integers(0, actor.action_space.n, prefix_cap)
                for action in actions:
                    h = actor.forward_head(prepare_and_normalize_obs(actor, obs))
                    active = torch.nonzero(h[0, :n] > 0).flatten()
                    source = int(active.item()) if active.numel() == 1 else -1
                    if source >= 0 and (focus_sources is None or source in focus_sources) and (
                        source in panel or len(panel) < max_sources
                    ):
                        targets = np.flatnonzero(pairs[source]).tolist()
                        if focus_target is not None:
                            if focus_target in targets:
                                targets.remove(focus_target)
                                targets.sort(key=lambda target: -float(graph.passive_confidence[source, target]))
                                targets = [focus_target] + targets[:max(1, targets_per_source - 1)]
                            else:
                                targets = []
                        else:
                            targets = targets[:targets_per_source]
                        if len(targets) >= 2 and len(panel.get(source, [])) < repeats:
                            panel.setdefault(source, []).append((seed, list(prefix), targets, signature(obs, state)))
                            break
                    _, state = actor.forward_core(h, state)
                    obs, done = advance(int(action))
                    prefix.append(int(action))
                    if done:
                        break
                if len(panel) == max_sources and all(len(v) == repeats for v in panel.values()):
                    break
            for source, starts in sorted(panel.items()):
                for rep, (seed, prefix, targets, reference) in enumerate(starts):
                    horizon = max(pair_deadline(graph, source, t) for t in targets)
                    if physical_horizon is not None:
                        horizon = max(horizon, physical_horizon)
                    cost = len(targets) * (len(prefix) + horizon)
                    if decisions + cost > decision_cap:
                        missing.append(dict(source=source, repeat=rep, reason="decision_budget"))
                        continue
                    for command in targets:
                        obs, state = replay(seed, prefix)
                        actual = signature(obs, state)
                        if actual.keys() != reference.keys() or any(
                            not torch.equal(reference[k], actual[k]) for k in reference
                        ):
                            raise RuntimeError(f"Nonreproducible landmark start: source={source}, repeat={rep}")
                        first_times = {t: None for t in targets}
                        elapsed = 0
                        first_other = None
                        done = False
                        rng = torch.Generator(device=device).manual_seed(41000 + seed)
                        start_pos = as_numpy(obs["pos"][0]).copy()
                        last = start_pos.copy()
                        path = 0.0
                        start_distance = (
                            float(np.linalg.norm(start_pos[:2] - reward_center))
                            if reward_center is not None else math.nan
                        )
                        minimum_distance = start_distance
                        physical_contact = False
                        for elapsed in range(1, horizon + 1):
                            _, out, state = encode(obs, state, command)
                            result = actor.forward_tail(out, values_only=False, sample_actions=False)
                            prob = result["action_logits"].softmax(-1)
                            if elapsed == 1:
                                initial_prob = prob[0].cpu().tolist()
                            action = prob.argmax(-1) if deterministic else torch.multinomial(prob, 1, generator=rng)
                            obs, done = advance(int(action.item()))
                            pos = as_numpy(obs["pos"][0]).copy()
                            if reward_center is not None:
                                minimum_distance = min(
                                    minimum_distance, float(np.linalg.norm(pos[:2] - reward_center))
                                )
                            if reward_cell is not None:
                                x0, x1, y0, y1 = reward_cell
                                physical_contact |= bool(x0 <= pos[0] < x1 and y0 <= pos[1] < y1)
                            path += float(np.linalg.norm(pos[:2] - last[:2]))
                            last = pos
                            if done:
                                break
                            h = actor.forward_head(prepare_and_normalize_obs(actor, obs))
                            active = torch.nonzero(h[0, :n] > 0).flatten()
                            hit = int(active.item()) if active.numel() == 1 else -1
                            if hit >= 0 and hit != source and first_other is None:
                                first_other = hit
                            if hit in first_times and first_times[hit] is None:
                                first_times[hit] = elapsed
                        for target in targets:
                            deadline = pair_deadline(graph, source, target)
                            hit_time = first_times[target]
                            hit = hit_time is not None and hit_time <= deadline
                            censored = not hit and done and elapsed < deadline
                            rows.append(
                                dict(
                                    source=source,
                                    repeat=rep,
                                    prefix_seed=seed,
                                    command=command,
                                    target=target,
                                    initial_action_probabilities=json.dumps(initial_prob),
                                    commanded=command == target,
                                    deadline=deadline,
                                    hit=hit,
                                    hit_time=hit_time,
                                    first_distinct=first_other,
                                    censored=censored,
                                    timeout=not hit and not censored,
                                    elapsed=elapsed,
                                    path_length=path,
                                    start_position=json.dumps(start_pos.tolist()),
                                    endpoint=json.dumps(last.tolist()),
                                    exact_start_verified=True,
                                    physical_start_distance=start_distance,
                                    physical_minimum_distance=minimum_distance,
                                    physical_entry_r200=start_distance > 200 and minimum_distance <= 200,
                                    physical_entry_r300=start_distance > 300 and minimum_distance <= 300,
                                    physical_cell_contact=physical_contact,
                                )
                            )
    finally:
        env.close()
    if any(not torch.equal(v, actor.state_dict()[k]) for k, v in before.items()):
        raise RuntimeError("Policy or graph changed during matched intervention")
    differences = []
    for source, starts in panel.items():
        for rep in range(len(starts)):
            for target in starts[rep][2]:
                group = [r for r in rows if r["source"] == source and r["repeat"] == rep and r["target"] == target]
                actual = [float(r["hit"]) for r in group if r["commanded"] and not r["censored"]]
                alternate = [float(r["hit"]) for r in group if not r["commanded"] and not r["censored"]]
                if actual and alternate:
                    differences.append(actual[0] - float(np.mean(alternate)))
    eligible_sources = range(n) if focus_sources is None else focus_sources
    supported_sources = sum(
        int(pairs[source].sum() >= 2 and (focus_target is None or pairs[source, focus_target]))
        for source in eligible_sources
    )
    if len(panel) < min(max_sources, supported_sources):
        missing.append(
            dict(reason="source_discovery_limit", requested=min(max_sources, supported_sources), found=len(panel))
        )
    if supported_sources < max_sources:
        missing.append(
            dict(reason="insufficient_observed_outgoing_pairs", requested=max_sources, supported=supported_sources)
        )
    for source, starts in panel.items():
        if len(starts) < repeats:
            missing.append(
                dict(reason="insufficient_repeated_starts", source=source, requested=repeats, found=len(starts))
            )
    action_distances = []
    for source, starts in panel.items():
        for rep in range(len(starts)):
            commands = {
                r["command"]: np.asarray(json.loads(r["initial_action_probabilities"]))
                for r in rows
                if r["source"] == source and r["repeat"] == rep
            }
            ids = sorted(commands)
            action_distances.extend(
                float(np.abs(commands[a] - commands[b]).sum() / 2) for i, a in enumerate(ids) for b in ids[i + 1 :]
            )
    summary = dict(
        protocol="landmark-matched-commands-v1",
        checkpoint=str(checkpoint),
        decisions=decisions,
        supported_sources=supported_sources,
        source_panel_coverage=len(panel) / supported_sources if supported_sources else 0.0,
        mean_initial_action_total_variation=float(np.mean(action_distances)) if action_distances else None,
        sources_evaluated=len(panel),
        starts_evaluated=sum(len(v) for v in panel.values()),
        max_sources=max_sources,
        targets_per_source=targets_per_source,
        repeats_requested=repeats,
        missing=missing,
        exact_start_verified=bool(rows),
        policy_frozen=True,
        graph_frozen=True,
        paired_arrival_lift=float(np.mean(differences)) if differences else None,
        paired_comparisons=len(differences),
        rows=len(rows),
        focus_sources=focus_sources,
        focus_target=focus_target,
        physical_horizon=physical_horizon,
        discovery_multiplier=discovery_multiplier,
    )
    return pd.DataFrame(rows), summary


def run_interventions(
    run_dir: pathlib.Path,
    checkpoint: pathlib.Path,
    decision_cap: int,
    attempts_per_pair: int,
    deterministic: bool,
    first_distinct: bool = False,
    observed_targets_only: bool = False,
    local_successor_targets_only: bool = False,
    absent_options: dict | None = None,
    landmark_options: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cfg, env, env_info, actor_critic, checkpoint, device = load_policy_env(
        run_dir, decision_cap, deterministic, 0, checkpoint
    )
    if landmark_options is not None:
        return run_landmark_matched_interventions(
            cfg, env, env_info, actor_critic, checkpoint, device, decision_cap, deterministic, **landmark_options
        )
    if getattr(cfg, "dg_goal_input", "none") == "write":
        env.close()
        raise ValueError("DG-conditioned policies require landmark-matched-commands-v1 interventions")
    if getattr(cfg, "intrinsic_goal_mode", "none") == "ca3_absent_target":
        from sf_working_directories.IntrMotiv.evaluation.absent_goal_interventions import run_absent_goal_interventions

        return run_absent_goal_interventions(
            cfg, env, env_info, actor_critic, checkpoint, device, decision_cap, deterministic, **(absent_options or {})
        )
    if not bool(getattr(cfg, "hrl_controllable_graph", False)):
        raise ValueError("Target-control intervention requires a controllable-graph policy")
    n_nodes = int(cfg.Hippo_n_feature)
    graph = getattr(actor_critic.core, "policy_graph", None)
    if graph is None:
        raise ValueError("Target-control intervention requires a policy-buffer graph")
    graph_before = {name: value.clone() for name, value in graph.state_dict().items()}
    counts = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    np.fill_diagonal(counts, attempts_per_pair)
    observed = as_numpy(graph.node_visits >= 1.0).astype(bool)
    if local_successor_targets_only:
        eligible_pairs = as_numpy(graph.passive_confidence > 0).astype(bool)
    else:
        eligible_pairs = np.broadcast_to(observed[None, :], counts.shape).copy()
    np.fill_diagonal(eligible_pairs, False)
    if not observed_targets_only and not local_successor_targets_only:
        eligible_pairs[:] = True
        np.fill_diagonal(eligible_pairs, False)
    rows: list[dict[str, object]] = []
    active_trial: Trial | None = None
    obs, _ = env.reset()
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)

    with torch.no_grad():
        for decision in range(decision_cap):
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
            head_output = actor_critic.forward_head(normalized_obs)
            dg = head_output[:, :n_nodes]
            active_ids = torch.nonzero(dg[0] > 0, as_tuple=False).flatten()
            exclusive_node = int(active_ids.item()) if active_ids.numel() == 1 else -1
            position = as_numpy(obs["pos"][0])
            rotation = as_numpy(obs["rot"][0])
            if active_trial is not None and exclusive_node >= 0:
                active_trial.hit_mask |= 1 << exclusive_node
            completion = (
                classify_trial_completion(active_trial, exclusive_node, first_distinct)
                if active_trial is not None
                else None
            )
            if active_trial is not None and completion is not None:
                outcome, reason = completion
                rows.append(trial_row(active_trial, position, outcome, reason, decision))
                counts[active_trial.source, active_trial.target] += 1
                active_trial = None
            if active_trial is None and exclusive_node >= 0:
                target = balanced_target(
                    counts,
                    exclusive_node,
                    attempts_per_pair,
                    eligible_pairs[exclusive_node] if (observed_targets_only or local_successor_targets_only) else None,
                )
                if target is not None:
                    active_trial = Trial(
                        source=exclusive_node,
                        target=target,
                        source_position=position.copy(),
                        source_rotation=rotation.copy(),
                        deadline=pair_deadline(graph, exclusive_node, target),
                        hit_mask=1 << exclusive_node,
                        last_position=position.copy(),
                    )
            core_output, new_rnn_states = actor_critic.forward_core(head_output, rnn_states)
            if active_trial is None:
                policy_outputs = actor_critic.forward_tail(
                    core_output, values_only=False, sample_actions=True, action_mask=None
                )
            else:
                active_trial.sensitivity_sum += counterfactual_action_sensitivity(
                    actor_critic, core_output, active_trial.source
                )
                active_trial.sensitivity_count += 1
                conditioned = condition_for_target(actor_critic, core_output, active_trial.source, active_trial.target)
                policy_outputs = actor_critic.forward_tail(
                    conditioned, values_only=False, sample_actions=True, action_mask=None
                )
            actions = policy_outputs["actions"]
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)
            obs, _, terminated, truncated, _ = env.step(actions)
            rnn_states = new_rnn_states
            if active_trial is not None:
                next_position = as_numpy(obs["pos"][0])
                active_trial.path_length += float(np.linalg.norm(next_position[:2] - active_trial.last_position[:2]))
                active_trial.last_position = next_position.copy()
                active_trial.elapsed += 1
            dones = make_dones(terminated, truncated).cpu().numpy()
            if bool(dones[0]):
                if active_trial is not None:
                    endpoint = as_numpy(obs["pos"][0])
                    rows.append(trial_row(active_trial, endpoint, -1, "censored_boundary", decision + 1))
                    counts[active_trial.source, active_trial.target] += 1
                    active_trial = None
                rnn_states[0].zero_()
            if np.all(counts[eligible_pairs] >= attempts_per_pair):
                break
    env.close()
    for name, before in graph_before.items():
        after = graph.state_dict()[name]
        if not torch.equal(before, after):
            raise RuntimeError(f"Frozen graph buffer changed during intervention: {name}")
    add_matched_shuffled_targets(rows, n_nodes)
    summary = summarize_trials(rows, counts, eligible_pairs, decision_cap)
    summary.update(
        checkpoint=str(checkpoint),
        n_nodes=n_nodes,
        attempts_per_pair=attempts_per_pair,
        policy_frozen=True,
        graph_frozen=True,
        dg_frozen=True,
        termination="first_distinct" if first_distinct else "target_hit",
        observed_targets_only=observed_targets_only,
        local_successor_targets_only=local_successor_targets_only,
    )
    return pd.DataFrame(rows), summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--decision-cap", type=int, default=100000)
    parser.add_argument("--attempts-per-pair", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.decision_cap <= 0 or args.attempts_per_pair <= 0:
        raise ValueError("decision cap and attempts per pair must be positive")
    row = load_manifest_row(args.manifest, args.row_index)
    output_root = workspace_path(args.out_dir, "Output directory")
    output = output_root / row["label_suffix"]
    output.mkdir(parents=True, exist_ok=True)
    study_manifest = pathlib.Path(args.manifest).with_name("study_manifest.json")
    provenance = json.loads(study_manifest.read_text()) if study_manifest.is_file() else {}
    intervention = provenance.get("intervention", {})
    if int(row["target_frames"]) not in intervention.get("target_frames", [75000000]):
        raise ValueError("Intervention target must match the reviewed manifest provenance")
    trials, summary = run_interventions(
        pathlib.Path(row["run_dir"]),
        pathlib.Path(row["checkpoint"]),
        args.decision_cap,
        args.attempts_per_pair,
        args.deterministic,
        absent_options={
            key: intervention[source]
            for key, source in (
                ("starts", "starts"),
                ("prefix_length", "prefix_decisions"),
                ("horizons", "horizons"),
                ("max_commands", "max_commands"),
            )
            if source in intervention
        },
        landmark_options=(
            {key: intervention[key] for key in ("max_sources", "targets_per_source", "repeats") if key in intervention}
            if intervention.get("evaluation") == "landmark-matched-commands-v1"
            else None
        ),
        first_distinct=bool(intervention.get("terminate_on_first_distinct_exclusive_outcome", False)),
        observed_targets_only=bool(intervention.get("balanced_observed_alternative_targets", False)),
        local_successor_targets_only=bool(intervention.get("balanced_local_successor_targets", False)),
    )
    if provenance:
        summary.update(
            {
                key: provenance[key]
                for key in ("schema", "workflow_version", "study_id", "study_sha256")
                if key in provenance
            }
        )
    summary.update(
        condition=row["condition"],
        seed=int(row["seed"]),
        checkpoint_frames=int(row["checkpoint_frames"]),
        manifest=str(pathlib.Path(args.manifest).resolve()),
        manifest_row=args.row_index,
    )
    trials.to_csv(output / "intervention_trials.csv", index=False)
    (output / "intervention_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
