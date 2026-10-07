"""Fixed, position-defined DG events for the four-landmark control test.

Only the environment wrapper calls this module. The policy receives the four
activity values, never the position used to compute them.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


FIELD_CENTERS = np.asarray(
    ((550.0, 550.0), (1450.0, 550.0), (550.0, 1450.0), (1650.0, 1650.0)),
    dtype=np.float64,
)
FIELD_SIGMA = 20.0
FIELD_RADIUS = 2.0 * FIELD_SIGMA
_EDGE_VALUE = np.exp(-2.0)


def prescribed_activity(position: np.ndarray | None, mode: str) -> np.ndarray:
    """Return four nonoverlapping fields with exact zero outside their support."""
    if mode == "zero":
        return np.zeros(4, dtype=np.float32)
    if mode != "gaussian4" or position is None:
        raise ValueError("Prescribed DG requires a finite engine position")
    xy = np.asarray(position, dtype=np.float64)[:2]
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("Prescribed DG requires a finite two-dimensional engine position")
    distance_squared = np.sum((FIELD_CENTERS - xy) ** 2, axis=-1)
    gaussian = np.exp(-distance_squared / (2.0 * FIELD_SIGMA**2))
    return (np.maximum(gaussian - _EDGE_VALUE, 0.0) / (1.0 - _EDGE_VALUE)).astype(np.float32)


def replace_prescribed_channels(activity: Tensor, fields: Tensor) -> Tensor:
    """Keep learned context channels while making the first four DG rows fixed."""
    if activity.ndim != 2 or activity.size(1) < 4 or fields.shape != (activity.size(0), 4):
        raise ValueError("Prescribed DG replacement requires [batch, >=4] and [batch, 4]")
    if not torch.isfinite(fields).all() or (fields < 0).any() or (fields > 1).any():
        raise ValueError("Prescribed DG fields must be finite values in [0, 1]")
    return torch.cat((fields.detach().to(activity), activity[:, 4:]), dim=-1)
