"""Offline place fields must use the verified map's resolution and bounds."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sf_working_directories.IntrMotiv.evaluation.place_fields import (
    compute_place_fields,
    evaluation_grid,
    spatial_details_for_artifact,
)
from sf_working_directories.IntrMotiv.tests.test_corridor_runtime import config as corridor_config
from sf_working_directories.IntrMotiv.tests.test_easy_landmark_runtime import _config as landmark_config


@pytest.mark.parametrize("cfg,expected_grain", [(corridor_config(), 19), (landmark_config(), 9)])
def test_verified_geometry_drives_all_evaluation_maps(cfg, expected_grain):
    grain, bounds, geometry = evaluation_grid(cfg)
    assert grain == expected_grain
    assert geometry["geometry_accessible_mask"].shape == (grain, grain)
    pose = pd.DataFrame(
        {"x": [bounds[0][0] + 50, bounds[0][1] - 50], "y": [bounds[1][0] + 50, bounds[1][1] - 50], "rot_y": [0.0, 0.0]}
    )
    activity = np.ones((2, 1), dtype=np.float32)
    occupancy, rate_maps, _, _ = compute_place_fields(pose, activity, grain, bounds)
    details = spatial_details_for_artifact(pose, activity, grain, bounds)
    assert occupancy.shape == rate_maps.shape[:2] == details["occupancy"].shape == (grain, grain)
    assert int(occupancy.sum()) == 2
    with pytest.raises(ValueError, match="disagrees"):
        evaluation_grid(cfg, grain + 1)


def test_unknown_map_preserves_historical_grid():
    grain, bounds, geometry = evaluation_grid(SimpleNamespace(env="openfield"))
    assert (grain, bounds, geometry) == (19, ((100.0, 2000.0), (100.0, 2000.0)), {})
