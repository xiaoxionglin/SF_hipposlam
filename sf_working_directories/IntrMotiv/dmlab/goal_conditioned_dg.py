"""Goal-conditioned worker memory with independent, canonical landmark evidence.

Core output keeps the canonical layout intact and appends a worker-memory
payload. Only the policy tail consumes that payload. Encoder objectives and
graph updates therefore never use goal-dependent achievement evidence.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from sf_working_directories.IntrMotiv.dmlab.custom_core import SimpleSequenceWithBypassCore


class GoalConditionedDGCore(SimpleSequenceWithBypassCore):
    def __init__(self, cfg, input_size):
        n = int(cfg.Hippo_n_feature)
        super().__init__(cfg, input_size - n)
        if not self.hrl_enabled or self.hrl_graph_memory != "policy_buffer" or self.hrl_target_timing != "immediate":
            raise ValueError("Goal-conditioned DG requires immediate policy-buffer HRL")
        if self.context_feedback is not None or self.graph_recruitment:
            raise ValueError("Goal-conditioned DG currently excludes contextual feedback and recruitment")
        self.canonical_input_size = input_size - n
        self.canonical_state_size = self.total_state_size
        self.total_state_size += self.core_output_size
        self.dg_goal_modulation = nn.Parameter(torch.zeros(n, 2 * n))
        self.dg_intercept = float(getattr(cfg, "DG_BN_intercept", 2.43))
        self.last_worker_dg_activity = None
        self.last_worker_memory = None

    def split_worker_state(self, state):
        k = self.canonical_state_size - self.behavior_goal_state_size
        canonical = torch.cat((state[..., :k], state[..., k + self.core_output_size :]), -1)
        return canonical, state[..., k : k + self.core_output_size]

    def join_worker_state(self, canonical, worker):
        k = self.canonical_state_size - self.behavior_goal_state_size
        return torch.cat((canonical[..., :k], worker, canonical[..., k:]), -1)

    def _split_state(self, state):
        if state.size(-1) == self.total_state_size:
            state, _ = self.split_worker_state(state)
        return super()._split_state(state)

    def write_activity(self, preactivation, goal):
        scale, bias = (goal @ self.dg_goal_modulation).chunk(2, -1)
        return torch.relu((1 + scale) * preactivation.detach() + bias - self.dg_intercept)

    def advance_worker(self, state, activity):
        shifted = state.roll(1, -1)
        shifted = torch.cat((torch.zeros_like(shifted[..., :1]), shifted[..., 1:]), -1)
        injected = torch.nn.functional.pad(
            activity.unsqueeze(-1).expand(-1, -1, self.R), (0, self.expanded_length - self.R)
        )
        return shifted + injected

    def worker_view(self, output):
        expected = self.total_output_size + self.core_output_size
        if output.size(-1) != expected:
            raise ValueError(f"Worker memory payload missing: expected {expected}, got {output.size(-1)}")
        return torch.cat(
            (output[..., self.total_output_size :], output[..., self.core_output_size : self.total_output_size]), -1
        )

    def forward(self, head_output, rnn_states, *, replay_conditions=None, replay_padded=False):
        canonical_state, worker = self.split_worker_state(rnn_states)
        n = self.Hippo_n_feature
        packed = isinstance(head_output, PackedSequence)
        data = head_output.data if packed else head_output
        replay = data.size(-1) == self.canonical_input_size + 2 * n
        if data.size(-1) not in (self.canonical_input_size + n, self.canonical_input_size + 2 * n):
            raise ValueError("Unexpected goal-conditioned head payload width")
        canonical_data = data[..., : self.canonical_input_size]
        canonical_head = head_output._replace(data=canonical_data) if packed else canonical_data
        canonical_out, canonical_new_state = super().forward(
            canonical_head, canonical_state, replay_conditions=replay_conditions, replay_padded=replay_padded
        )
        worker = worker.reshape(-1, n, self.expanded_length).detach()
        if packed:
            padded_head, lengths = pad_packed_sequence(head_output)
            if replay_padded:
                padded_out = canonical_out
            else:
                padded_out, out_lengths = pad_packed_sequence(canonical_out)
                assert torch.equal(lengths, out_lengths)
            if replay_conditions is not None:
                from .ca3_memory import finite_shift_history

                if worker.count_nonzero():
                    raise ValueError("Finite worker replay requires zero initial traces")
                pre = padded_head[:, :, self.canonical_input_size : self.canonical_input_size + n]
                goals = padded_head[:, :, -n:] if replay else replay_conditions[:, :, :n]
                activity = self.write_activity(pre.reshape(-1, n), goals.reshape(-1, n)).reshape(*pre.shape)
                memories = finite_shift_history(activity, self.R, self.expanded_length).flatten(2)
                batch = torch.arange(len(lengths), device=memories.device)
                final = memories[lengths.to(memories.device) - 1, batch]
                output = torch.cat((padded_out, memories), -1)
                if not replay_padded:
                    output = pack_padded_sequence(output, lengths, enforce_sorted=False)
                self.last_worker_memory = final.detach()
                return output, self.join_worker_state(canonical_new_state, final)
            result = []
            for t in range(padded_head.size(0)):
                valid = (lengths > t).to(worker.device)
                pre = padded_head[t, :, self.canonical_input_size : self.canonical_input_size + n]
                goal = (
                    padded_head[t, :, -n:]
                    if replay
                    else padded_out[t, :, self.target_condition_start : self.target_condition_start + n]
                )
                activity = self.write_activity(pre, goal)
                advanced = self.advance_worker(worker, activity)
                worker = torch.where(valid[:, None, None], advanced, worker)
                canonical_step = padded_out[t]
                if replay:
                    canonical_step = canonical_step.clone()
                    canonical_step[:, self.target_condition_start : self.target_condition_start + n] = goal
                result.append(torch.cat((canonical_step, worker.flatten(1)), -1))
            output = pack_padded_sequence(torch.stack(result), lengths, enforce_sorted=False)
        else:
            pre = data[:, self.canonical_input_size : self.canonical_input_size + n]
            goal = (
                data[:, -n:]
                if replay
                else canonical_out[:, self.target_condition_start : self.target_condition_start + n]
            )
            activity = self.write_activity(pre, goal)
            worker = self.advance_worker(worker, activity)
            self.last_worker_dg_activity = activity.detach()
            if replay:
                canonical_out = canonical_out.clone()
                canonical_out[:, self.target_condition_start : self.target_condition_start + n] = goal
            output = torch.cat((canonical_out, worker.flatten(1)), -1)
        self.last_worker_memory = worker.detach()
        return output, self.join_worker_state(canonical_new_state, worker.flatten(1))
