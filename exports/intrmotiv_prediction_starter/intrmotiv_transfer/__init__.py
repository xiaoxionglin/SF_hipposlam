"""Small prediction components; no automatic modification of policy or memory."""

from .labels import future_target_labels
from .losses import shadow_prediction_loss
from .shadow import CA3TargetPredictor
from .state import advance_ca3, advance_action_history, previous_action_onehot, skip_silent_ca3

__all__ = [
    "CA3TargetPredictor", "future_target_labels", "shadow_prediction_loss",
    "advance_ca3", "advance_action_history", "previous_action_onehot", "skip_silent_ca3",
]
