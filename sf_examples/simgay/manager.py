"""SimGaY manager and completed-trial replay; PyTorch only.

Definitions copied unchanged from the standalone-high-level-bandit fork,
commit bc2bcb22483f82a8dbf133a0a9a0102f0d292c1f. This module does not
include the full navigation core, oracle transfer, or older manager variants.
"""

from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TrialHistory:
    """Completed-trial replay shared by the full core and standalone experiments.

    Each flat record stores the pre-event hidden state, selected mode, reward,
    optional trigger/outcome representations, and a validity flag. The saved
    hidden state is a detached starting point for truncated recurrent replay.
    This object owns no model parameters, keeping learner checkpoint keys intact.
    """

    def __init__(self, K: int, d_H: int, history_len: int, representations: bool = False, d_rep: int = 64):
        if history_len < 1:
            raise ValueError("hl_history_len must be at least one completed trial")
        self.K = K
        self.d_H = d_H
        self.history_len = history_len
        self.representations = representations
        self.d_rep = d_rep
        self.event_width = d_H + K + 2 + (2 * d_rep if representations else 0)
        self.history_size = history_len * self.event_width

    def append_event(
        self, history, h_before, z_before, reward, event_mask, e_trigger: Tensor = None, e_outcome: Tensor = None
    ):
        """Append a completed trial only when an event and a prior mode exist."""
        valid_flag = torch.ones_like(reward[:, None])
        if self.representations:
            if e_trigger is None or e_outcome is None:
                raise ValueError("e_trigger and e_outcome must be provided when representations are enabled")
            entry = torch.cat(
                (
                    h_before.detach(),
                    z_before.detach(),
                    reward[:, None],
                    e_trigger.detach(),
                    e_outcome.detach(),
                    valid_flag,
                ),
                dim=-1,
            )
        else:
            entry = torch.cat((h_before.detach(), z_before.detach(), reward[:, None], valid_flag), dim=-1)
        shifted = torch.cat((history[:, self.event_width :], entry), dim=-1)
        valid_outcome = event_mask & (z_before.sum(dim=-1) > 0.5)
        return torch.where(valid_outcome[:, None], shifted, history)

    def scores_from_history(self, learner, history, fallback_h):
        """Replay saved modes/rewards under current weights without resampling.

        To score the action whose outcome has just arrived, pass the history
        saved before that outcome; append its reward only after scoring it.
        """
        events = history.reshape(-1, self.history_len, self.event_width)
        valid = events[:, :, -1] > 0.5
        first = valid.int().argmax(dim=1)
        batch = torch.arange(events.size(0), device=events.device)
        h = torch.where(valid.any(dim=1, keepdim=True), events[batch, first, : self.d_H], fallback_h)

        for k in range(self.history_len):
            event = events[:, k]
            z = event[:, self.d_H : self.d_H + self.K]
            reward = event[:, self.d_H + self.K]
            if self.representations:
                trig_start = self.d_H + self.K + 1
                trig_end = trig_start + self.d_rep
                out_end = trig_end + self.d_rep
                h = learner.update_state(
                    valid[:, k], reward, h, z, event[:, trig_start:trig_end], event[:, trig_end:out_end]
                )
            else:
                h = learner.update_state(valid[:, k], reward, h, z)
        return learner.head(h)


def compute_log_dict(
    scores_or_logits: Tensor,  # (B, K)  Q-values
    z_index: Tensor,  # (B,)    selected mode index
    h_high_new: Tensor,  # (B, d_H)
    reward_prev: Tensor,  # (B,)
    tau: float,
    chosen_arm: Tensor = None,  # (B,) optional, for logging
    K=4,
    is_policy: bool = False,  # if False, it does Q-learning, if True, it does policy gradient
) -> Dict[str, Tensor]:
    """
    Recommended logging quantities (computed at decision points only).
    High-level process:
        z_k, scores/Q vector, selection entropy, h_k, r_k
    Q-head specific:
        all Q_H(h_k, z), selected-Q vs reward, gap between top two Q-values
    """
    if is_policy:
        pi = F.softmax(scores_or_logits, dim=-1)  # (B, K)
    else:
        pi = F.softmax(scores_or_logits / tau, dim=-1)  # (B, K)

    entropy = -(pi * (pi + 1e-8).log()).sum(dim=-1)  # (B,)  selection entropy

    selected = scores_or_logits.gather(dim=-1, index=z_index.unsqueeze(-1)).squeeze(  # (B,)
        -1
    )  # selected mode's score/Q-value

    top2 = scores_or_logits.topk(k=2, dim=-1).values  # (B, 2)
    gap = top2[:, 0] - top2[:, 1]  # (B,)  gap top1 - top2

    if is_policy:
        log_dict = {"entropy": entropy, "h": h_high_new, "logit_selected": selected, "logit_gap": gap}
    else:
        log_dict = {"entropy": entropy, "h": h_high_new, "q_selected": selected, "q_gap": gap}

    went_right = (chosen_arm == 1.0) if chosen_arm is not None else None
    went_left = (chosen_arm == 2.0) if chosen_arm is not None else None

    for k in range(K):

        if is_policy:
            log_dict[f"logit_mode{k}"] = scores_or_logits[:, k]
        else:
            log_dict[f"qval_mode{k}"] = scores_or_logits[
                :, k
            ]  # if you plot it together with reward_achieved they must overlap.

        log_dict[f"prob_mode{k}"] = pi[:, k]  # predicted probability to select this mode

        went_mode_k = z_index == k

        log_dict[f"selection_rate_mode{k}"] = (
            went_mode_k.float().mean()
        )  # fraction of times mode k was ACTUALLY selected (emprical)

        if went_mode_k.sum() > 0 and chosen_arm is not None:
            log_dict[f"reward_achieved_mode{k}"] = (
                reward_prev[went_mode_k].float().mean()
            )  # average reward achieved when mode k was selected

            prob_R = went_right[went_mode_k].float().mean()
            prob_L = went_left[went_mode_k].float().mean()

            log_dict[f"prob_R_mode{k}"] = prob_R
            log_dict[f"prob_L_mode{k}"] = prob_L

    return log_dict


def q_loss(
    scores: Tensor,  # (T, B, K)
    z_indices: Tensor,  # (T, B)
    rewards: Tensor,  # (T, B)
    decision_mask: Tensor,  # (T, B)  bool — 1 only at decision points
) -> Tensor:
    """
    L_Q = Σ_t d_t (Q_H(h_t, z_t) - r_t)^2  /  (Σ_t d_t + ε)   §19
    """
    q_selected = scores.gather(dim=-1, index=z_indices.unsqueeze(-1)).squeeze(-1)  # (T, B)
    error = (q_selected - rewards) ** 2
    mask = decision_mask.float()
    return (error * mask).sum() / (mask.sum() + 1e-8)


def high_level_policy_loss(
    logits: Tensor, z_indices: Tensor, rewards: Tensor, decision_mask: Tensor, entropy_coef: float = 0.05
) -> Tuple[Tensor, dict]:
    """
    L_H = -(R_k - b_k) * log(pi(z_k|h_k))
    Returns: (total_loss, dict_of_metrics)
    """
    mask_float = (
        decision_mask.float()
    )  # use the mask to multiply all non-decision points by zero, so they don't contribute to the loss
    valid_transitions = mask_float.sum() + 1e-8  # to avoid division by zero in case there are no valid transitions

    log_pi_all = F.log_softmax(logits, dim=-1)  # convert logits to log-probabilities
    pi_all = torch.exp(log_pi_all)  # convert logits to probabilities

    log_pi_selected = log_pi_all.gather(dim=-1, index=z_indices.unsqueeze(-1)).squeeze(
        -1
    )  # get the log-probabiltiy of the mode that was actually selected (z_k) for each batch element
    entropy = -(pi_all * log_pi_all).sum(dim=-1)

    # 1. The Baseline (b_k)
    # Average rewards the HL RNN got in this batch.
    b_k = (rewards * mask_float).sum() / valid_transitions

    # 2. The Advantage (R_k - b_k)
    # Difference between the actual reward and the baseline. This tells us how much better or worse the selected action was compared to the average.
    advantages = rewards - b_k
    advantages = advantages.detach()

    # Losses
    pg_loss = -(advantages * log_pi_selected)
    total_loss = pg_loss - (entropy_coef * entropy)

    final_loss = (total_loss * mask_float).sum() / valid_transitions

    # --- NEW: Create a dictionary of these specific Policy Gradient metrics ---
    metrics = {
        "hl/pg_baseline": b_k.item(),
        "hl/pg_advantage": (advantages * mask_float).sum().item() / valid_transitions.item(),
        "hl/pg_loss": (pg_loss * mask_float).sum().item() / valid_transitions.item(),
    }

    return final_loss, metrics


class HighLevelContextRNN_Learner(nn.Module):
    """
    Unified High-Level RNN that seamlessly handles BOTH Q-Learning and Policy Gradients
    based on the 'is_policy' flag.
    """

    def __init__(
        self,
        K: int = 4,
        d_H: int = 16,
        tau: float = 1.0,
        is_policy: bool = False,
        deterministic: bool = False,
        representations: bool = False,
        d_rep: int = 64,
    ):
        super().__init__()
        self.K = K
        self.d_H = d_H
        self.tau = tau
        self.is_policy = is_policy
        self.deterministic = deterministic
        self.representations = representations  # If True, the model will get representations from decoder 1st layer
        self.d_rep = d_rep  # Dimension of the representation vector

        if self.representations:
            input_dim = K + 1 + (2 * self.d_rep)  # z_prev + prev_reward + 2 * representation size = 2 + 1 + 128 = 131
        else:
            input_dim = K + 1  # z_prev + prev_reward

        self.rnn_cell = nn.RNNCell(input_size=input_dim, hidden_size=d_H, nonlinearity="tanh")

        # This acts as BOTH the QHead and the PolicyHead
        self.head = nn.Linear(d_H, K)

    def sample_mode(self, outputs: Tensor) -> Tuple[Tensor, Tensor]:
        # Policy logits use raw outputs, Q-Learning uses temperature scaling
        if self.is_policy:
            log_pi = F.log_softmax(outputs, dim=-1)
        else:
            log_pi = F.log_softmax(outputs / self.tau, dim=-1)

        if self.deterministic:
            z_index = log_pi.argmax(dim=-1)
        else:
            z_index = torch.distributions.Categorical(logits=log_pi).sample()

        z_onehot = F.one_hot(z_index, num_classes=self.K).float()
        return z_onehot, z_index

    def forward(self, outcome_mask, prev_trial_reward, h_high, z_prev, chosen_arm, e_trigger=None, e_outcome=None):
        # 1. Update State
        h_high_new = self.update_state(outcome_mask, prev_trial_reward, h_high, z_prev, e_trigger, e_outcome)

        # 2. Get Outputs (Q-Scores OR Logits)
        outputs = self.head(h_high_new)

        # 3. Sample and Latch
        z_candidate, z_candidate_index = self.sample_mode(outputs)

        z_new = torch.where(outcome_mask[:, None], z_candidate, z_prev)

        z_index_prev = z_prev.argmax(dim=-1)
        z_index_new = torch.where(outcome_mask, z_candidate_index, z_index_prev)

        new_state = {
            "high_level_h": h_high_new,
            "current_z": z_new,
            "current_z_index": z_index_new,
        }

        log_dict = compute_log_dict(
            outputs,
            z_index_new,
            h_high_new,
            prev_trial_reward,
            self.tau,
            chosen_arm,
            K=self.K,
            is_policy=self.is_policy,
        )

        return h_high_new, z_new, outputs, new_state, log_dict

    def update_state(self, outcome_mask, prev_trial_reward, h_high, z_prev, e_trigger=None, e_outcome=None):
        """Update reward history without choosing a new mode (used during replay)."""
        if self.representations:
            if e_trigger is None or e_outcome is None:
                raise ValueError("representations=True requires both e_trigger and e_outcome")

            rnn_in = torch.cat([z_prev, prev_trial_reward[:, None], e_trigger, e_outcome], dim=-1)
        else:
            rnn_in = torch.cat([z_prev, prev_trial_reward[:, None]], dim=-1)

        h_candidate = self.rnn_cell(rnn_in, h_high)
        h_high_new = torch.where(outcome_mask[:, None], h_candidate, h_high)
        return h_high_new
