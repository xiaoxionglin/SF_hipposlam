"""Finite-history reconstruction through the original simple IntrMotiv core.

This operation runs on a private controller snapshot. It does not train the DG
auxiliary, publish actor weights, or treat replay as new manager graph evidence.
The ingress layer must supply one contiguous physical episode per call.
"""

from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sf_working_directories.IntrMotiv.dmlab.custom_core import SimpleSequenceWithBypassCore
from sf_working_directories.IntrMotiv.dmlab.goal_conditioned_dg import GoalConditionedDGCore


@dataclass(frozen=True)
class ControllerHistory:
    observations: dict
    decision_ids: torch.Tensor
    initial_context: torch.Tensor
    # The complete action-time condition in original core output order:
    # target, followed by any enabled geometry/mode fields.
    conditions: torch.Tensor
    episode_start: bool
    burn_in: int


def _validate_history(model, history, memory_only):
    core = model.core
    if type(core) not in (SimpleSequenceWithBypassCore, GoalConditionedDGCore):
        raise ValueError("Finite replay washout is not established for this core")
    if core.context_feedback is not None or core.graph_recruitment or core.context_action_history_size:
        raise ValueError("Contextual/recruitment history needs its own reconstruction contract")
    if model.training:
        raise ValueError("Reconstruction requires an immutable evaluation snapshot")
    steps = history.decision_ids.numel()
    if history.decision_ids.ndim != 1 or steps < 1:
        raise ValueError("Expected a nonempty physical history")
    if not torch.all(history.decision_ids[1:] == history.decision_ids[:-1] + 1):
        raise ValueError("Replay history has a physical decision gap")
    if not 0 <= history.burn_in < steps:
        raise ValueError("Replay must retain TD positions after burn-in")
    if not history.episode_start and (steps if memory_only else history.burn_in) < core.expanded_length:
        raise ValueError("Incomplete finite washout prefix")
    condition_width = core.total_output_size - core.target_condition_start
    if history.conditions.shape != (steps, condition_width):
        raise ValueError("Incomplete action-time worker condition")
    if history.initial_context.shape != (1, core.total_state_size):
        raise ValueError("Invalid initial context layout")
    if any(value.shape[0] != steps for value in history.observations.values()):
        raise ValueError("Observation/history length mismatch")

    return steps


def reconstruction_head(model, observations):
    """Use one fixed GEMM batch shape for snapshot recognition and histories.

    Changing encoder batch geometry can change a near-zero DG activation's
    sign even with identical parameters and buffers. Repeating the final row
    pads the last chunk without introducing invalid instruction observations.
    Evaluation normalization is frozen, so padding cannot affect real rows.
    """
    if model.training:
        raise ValueError("Reconstruction requires an immutable evaluation snapshot")
    normalized = prepare_and_normalize_obs(model, observations)
    count = next(iter(normalized.values())).shape[0]
    width = 256
    heads = []
    for start in range(0, count, width):
        size = min(width, count - start)
        chunk = {key: value[start : start + size] for key, value in normalized.items()}
        if size < width:
            chunk = {
                key: torch.cat((value, value[-1:].expand(width - size, *value.shape[1:])))
                for key, value in chunk.items()
            }
        heads.append(model.forward_head(chunk)[:size])
    return torch.cat(heads)


def reconstruct_controller_histories(model, histories, *, memory_only=False, decode=True):
    """Batch finite histories using the parent's PackedSequence core transport."""
    from torch.nn.utils.rnn import pad_sequence

    if not histories:
        return []
    lengths = [_validate_history(model, h, memory_only) for h in histories]
    core = model.core
    states = torch.cat([h.initial_context.detach().clone() for h in histories])
    if isinstance(core, GoalConditionedDGCore):
        canonical, worker = core.split_worker_state(states)
        canonical = canonical.clone()
        canonical[:, : core.core_output_size] = 0
        states = core.join_worker_state(canonical, torch.zeros_like(worker))
    else:
        states[:, : core.core_output_size] = 0
    observations = {key: torch.cat([h.observations[key] for h in histories]) for key in histories[0].observations}
    head = reconstruction_head(model, observations)
    # STOP already excludes these DG gradients at the controller boundary.
    # Detaching before history reconstruction avoids retaining unused CA3 graphs.
    if getattr(getattr(model, "cfg", None), "ppo_dg_gradient", "joint") == "stop":
        head = torch.cat((head[:, : core.Hippo_n_feature].detach(), head[:, core.Hippo_n_feature :]), -1)
    conditions = [h.conditions.to(head) for h in histories]
    if isinstance(core, GoalConditionedDGCore):
        head = torch.cat((head, torch.cat(conditions)[:, : core.Hippo_n_feature]), -1)
    heads = head.split(lengths)
    packed = pack_padded_sequence(pad_sequence(heads), lengths, enforce_sorted=False)
    outputs, final_states = core(packed, states, replay_conditions=pad_sequence(conditions), replay_padded=True)
    if memory_only:
        return [{"reconstructed_state": final_states[i : i + 1]} for i in range(len(histories))]
    per_history = [outputs[:length, i] for i, length in enumerate(lengths)]
    td_lengths = [length - h.burn_in for length, h in zip(lengths, histories)]
    if decode:
        # One gather has one backward scatter. Concatenating B sliced histories
        # would allocate/scatter the entire padded tensor B times in backward.
        time_index = torch.cat(
            [torch.arange(h.burn_in, length, device=outputs.device) for h, length in zip(histories, lengths)]
        )
        batch_index = torch.repeat_interleave(
            torch.arange(len(histories), device=outputs.device), torch.tensor(td_lengths, device=outputs.device)
        )
        hidden = model.controller_hidden(outputs[time_index, batch_index]).split(td_lengths)
    else:
        hidden = [None] * len(histories)
    return [
        dict(
            hidden=hidden[i],
            core_outputs=out[h.burn_in :],
            all_core_outputs=out,
            head_outputs=heads[i][h.burn_in :],
            decision_ids=h.decision_ids[h.burn_in :],
        )
        for i, (out, h) in enumerate(zip(per_history, histories))
    ]


def reconstruct_controller_history(model, history, *, memory_only=False, decode=True):
    """Reconstruct one physical or virtual history without re-planning its manager."""
    return reconstruct_controller_histories(model, [history], memory_only=memory_only, decode=decode)[0]
