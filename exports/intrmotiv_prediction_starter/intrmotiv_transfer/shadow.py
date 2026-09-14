"""Target-conditioned shadow head extracted verbatim; see PROVENANCE.json."""

import torch
from torch import Tensor, nn

class CA3TargetPredictor(nn.Module):
    """Shadow predictor for target hit probability and conditional hit time."""

    def __init__(self, ca3_size: int, n_targets: int, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(ca3_size + n_targets, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, ca3_state: Tensor, target_onehot: Tensor) -> Tensor:
        return self.network(torch.cat([ca3_state, target_onehot], dim=-1))
