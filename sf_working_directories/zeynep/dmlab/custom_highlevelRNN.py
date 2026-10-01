# SimGaY §16 decomposition + §18 forward-pass pseudocode
# Q-style contextual bandit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict


# QHead  —  §16, §9.2

class QHead(nn.Module):
    def __init__(self, d_H: int, K: int, tau: float = 1.0):
        super().__init__()
        self.tau  = tau
        self.head = nn.Linear(d_H, K)

    def forward(self, h: Tensor) -> Tensor:
        """h: (B, d_H)  →  q: (B, K)"""
        q      = self.head(h) # current estimate of expected reward for each mode | THIS IS LEARNED BY THE NETWORK
        return q # output raw expected rewards.


# sample_mode  —  §9, §18 step 4

def sample_mode(scores: Tensor, tau: float = 1.0) -> Tuple[Tensor, Tensor]: #scale the raw values by temperature tau to control exploration vs exploitation
    """
    scores: (B, K)
    → z_onehot: (B, K)   one-hot e(z_k)
    → z_index:  (B,)     integer index  — §17 current_z_index
    """
    K = scores.size(-1)
    log_pi = F.log_softmax(scores / tau, dim=-1) # convert raw Q-values to log-probabilities (distribution) for sampling 

    if torch.is_grad_enabled():
        z_index = torch.distributions.Categorical(logits=log_pi).sample() # from distribution, sample a discrete mode index z_k
    else:
        z_index = log_pi.argmax(dim=-1)
    z_onehot = F.one_hot(z_index, num_classes=K).float() # return binary one-hot vector of the sampled mode index | RNN and DECODER RECIEVE IT
    return z_onehot, z_index 


# Logging helpers  —  §22

def compute_log_dict(
    scores:       Tensor,   # (B, K)  Q-values
    z_index:      Tensor,   # (B,)    selected mode index
    h_high_new:   Tensor,   # (B, d_H)
    reward_prev:  Tensor,   # (B,)
    tau:          float,
    chosen_arm:   Tensor = None, # (B,) optional, for logging
    K=4
) -> Dict[str, Tensor]:
    """
    Recommended logging quantities (computed at decision points only).
    High-level process:
        z_k, scores/Q vector, selection entropy, h_k, r_k
    Q-head specific:
        all Q_H(h_k, z), selected-Q vs reward, gap between top two Q-values
    """
    pi      = F.softmax(scores / tau, dim=-1)               # (B, K)
    entropy = -(pi * (pi + 1e-8).log()).sum(dim=-1)         # (B,)  selection entropy
    q_selected = scores.gather(                              # (B,)
        dim=-1, index=z_index.unsqueeze(-1)
    ).squeeze(-1)
    top2       = scores.topk(k=2, dim=-1).values            # (B, 2)
    q_gap      = top2[:, 0] - top2[:, 1]                   # (B,)  gap top1 - top2

    log_dict = {"z_index": z_index, "scores": scores, "entropy": entropy, "h": h_high_new, "reward_prev": reward_prev,
                "q_selected": q_selected, "q_gap": q_gap, "mode_probs": pi}

    went_right = (chosen_arm == 1.0) if chosen_arm is not None else None
    went_left  = (chosen_arm == 2.0) if chosen_arm is not None else None

    for k in range(K):
        went_mode_k = (z_index == k)
        if went_mode_k.sum() > 0:
            prob_R = went_right[went_mode_k].float().mean()
            prob_L = went_left[went_mode_k].float().mean()

            log_dict[f"prob_R_mode{k}"] = prob_R
            log_dict[f"prob_L_mode{k}"] = prob_L

    return log_dict


# HighLevelContextRNN  —  §16, §18

class HighLevelContextRNN_Stage1(nn.Module):
    """
    Stage 1: Event-driven latching ONLY.
    No learned head. No z_k sampling, random for now. Just verifies that the timing works.
    """
    def __init__(self, K: int = 4, d_H: int = 16):
        super().__init__()
        self.K = K

        # We keep the RNN just to make sure the tensor shapes and updates don't crash,
        # but we ignore its output for now.
        self.rnn_cell = nn.RNNCell(input_size=K + 1, hidden_size=d_H, nonlinearity='tanh')

    def forward(self, outcome_mask, prev_trial_reward, h_high, z_prev, inst_block = None):

        # 1. Background RNN update (just testing mechanics)
        rnn_in = torch.cat([z_prev, prev_trial_reward[:, None]], dim=-1)
        h_candidate = self.rnn_cell(rnn_in, h_high)
        
        h_high_new = torch.where(
            outcome_mask[:, None],
            h_candidate, # update only at outcome events (at trigger)
            h_high,      # keep previous hidden state otherwise
        )

        # MODE SELECTION
        # e.g. [0,0,1,0] -> mode 2 is selected. With randomization that "1" mode is selected randomly
        if inst_block is not None:
            # If inst_block is provided, we can use it to select a mode deterministically
            # For example, if inst_block is 1, we select mode 0; if it's 2, we select mode 1, and so on.
            oracle_idx = torch.clamp(inst_block - 1, 0, 1)  # Ensure the index is within bounds
            z_candidate = F.one_hot(oracle_idx, num_classes=self.K).float()
        else:
            # Otherwise, we can randomly select a mode for testing purposes
            # This case Z has no correlation with where the actual reward is. Low-level policy considers z as "noise". 
            B = z_prev.size(0)
            random_indices = torch.randint(0, self.K, (B,), device=z_prev.device)
            z_candidate = F.one_hot(random_indices, num_classes=self.K).float()

        # 3. Latch the new mode only at outcome events
        # Keep z_prev unless it is at trigger (outcome event) then update to z_candidate
        z_new = torch.where(
            outcome_mask[:, None],
            z_candidate,
            z_prev,
        )

        # Passing the mode to decoder is handeled in the wrapper
        return h_high_new, z_new


class HighLevelContextRNN_QLearning(nn.Module):
    """
    HighLevelContextRNN
    ├── RNNCell  (vanilla tanh)    §2
    └── QHead(K)                   §16

    K   = 4          overcomplete mode set
    d_H = 16         hidden dim (could be 8, 16, 32)
    input_dim = K+1  (z_prev one-hot + reward scalar)  §16
    """

    def __init__(self, K: int = 4, d_H: int = 16, tau: float = 1.0):
        super().__init__()
        self.K   = K
        self.d_H = d_H
        self.tau = tau

        self.rnn_cell    = nn.RNNCell(input_size=K + 1,
                                      hidden_size=d_H,
                                      nonlinearity='tanh')
        
        self.q_head = QHead(d_H=d_H, K=K) # convert hidden state to Q-values and log-probabilities for sampling a new mode

    def forward(
        self, outcome_mask, prev_trial_reward, h_high, z_prev, chosen_arm):

        # 1. Candidate high-level state update  —  §18 step 1
        rnn_in      = torch.cat([z_prev, prev_trial_reward[:,None]], dim=-1)  # (B, K+1)
        h_candidate = self.rnn_cell(rnn_in, h_high)             # (B, d_H)

        # 2. Tick only at outcome events  —  §18 step 2
        h_high_new = torch.where(
            outcome_mask[:, None],
            h_candidate,
            h_high,
        )                                                        # (B, d_H)

        # 3. Produce high-level scores from updated context state  —  §18 step 3
        q_scores = self.q_head(h_high_new)         # (B, K) each

        # 4. Sample a new mode only at decision events  —  §18 step 4
        z_candidate, z_candidate_index = sample_mode(q_scores)                       # (B, K)

        z_new = torch.where(
            outcome_mask[:, None],
            z_candidate,
            z_prev,
        )                                                        # (B, K)

        # §17: persist z_index for logging and loss computation
        z_index_prev = z_prev.argmax(dim=-1)     # (B,)
        z_index_new  = torch.where(
            outcome_mask,
            z_candidate_index,
            z_index_prev,
        )
        
        # 5. z_new goes to existing low-level controller  —  §18 step 5
        # (caller passes z_new into the decoder)

        new_state = {
            "high_level_h": h_high_new,
            "current_z":    z_new,
            "current_z_index": z_index_new,
        }

        # §22: logging quantities (only meaningful at decision points)
        log_dict = compute_log_dict(
            q_scores, z_index_new, h_high_new, prev_trial_reward, self.tau, chosen_arm, K=self.K
        )

        return h_high_new, z_new, q_scores, new_state, log_dict


# Q loss  —  §10 Option C, §19
# compare these raw expected rewards (Q) to actual obtained rewards
def q_loss(
    scores:        Tensor,   # (T, B, K)
    z_indices:     Tensor,   # (T, B)
    rewards:       Tensor,   # (T, B)
    decision_mask: Tensor,   # (T, B)  bool — 1 only at decision points
) -> Tensor:
    """
    L_Q = Σ_t d_t (Q_H(h_t, z_t) - r_t)^2  /  (Σ_t d_t + ε)   §19
    """
    q_selected = scores.gather(
        dim=-1, index=z_indices.unsqueeze(-1)
    ).squeeze(-1)                                   # (T, B)
    error = (q_selected - rewards) ** 2
    mask  = decision_mask.float()
    return (error * mask).sum() / (mask.sum() + 1e-8)