import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace

from sf_working_directories.IntrMotiv.evaluation.place_fields import (
    compute_place_fields,
    compute_pre_threshold_maps,
    load_checkpoint_dict,
    optional_graph_arrays,
)
from sf_working_directories.IntrMotiv.evaluation.analyze_place_field_manifest import (
    connected_components_above_half_peak,
    per_unit_rows,
)


def test_pre_threshold_maps_preserve_signed_logits_and_occupancy():
    pose = pd.DataFrame({"x": [150.0, 150.0, 250.0], "y": [150.0, 150.0, 150.0]})
    events = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    logits = np.array([[-1.0, 2.0], [1.0, 4.0], [3.0, -2.0]])

    occupancy, event_maps, _, active = compute_place_fields(pose, events, grain=19)
    logit_occupancy, logit_maps, mean_logits, std_logits = compute_pre_threshold_maps(pose, logits, grain=19)

    assert np.array_equal(logit_occupancy, occupancy)
    assert np.isclose(event_maps[0, 0, 0], 0.5)
    assert np.isclose(logit_maps[0, 0, 0], 0.0)
    assert np.isclose(logit_maps[1, 0, 0], 3.0)
    assert np.allclose(mean_logits, [1.0, 4.0 / 3.0])
    assert np.all(std_logits > 0.0)
    assert np.allclose(active, [1.0 / 3.0, 1.0 / 3.0])


def test_checkpoint_loader_accepts_numpy_scalar_metadata(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"model": {}, "best_performance": np.float64(0.5)}, checkpoint)

    payload = load_checkpoint_dict(checkpoint, torch.device("cpu"))

    assert payload["model"] == {}
    assert payload["best_performance"] == np.float64(0.5)


def test_optional_graph_arrays_preserve_checkpoint_graph_and_assignment_buffers():
    control = SimpleNamespace(
        edge_confidence=torch.ones(2, 2),
        control_attempts=torch.full((2, 2), 2.0),
        tctrl=torch.full((2, 2), 3.0),
    )
    passive = SimpleNamespace(
        confidence=torch.full((2, 2), 4.0),
        elapsed=torch.full((2, 2), 5.0),
        birth_support=torch.tensor([0.25, 1.0]),
    )
    projection = SimpleNamespace(recruitment_row_counts=torch.tensor([1, 3]))
    model = SimpleNamespace(
        core=SimpleNamespace(policy_graph=control, passive_recruitment_graph=passive),
        encoder=SimpleNamespace(DG_projection=projection),
    )

    arrays = optional_graph_arrays(model)

    assert set(arrays) == {
        "control_edge_confidence",
        "control_attempts",
        "control_tctrl",
        "passive_confidence",
        "passive_elapsed",
        "birth_support",
        "recruitment_row_counts",
    }
    assert arrays["control_attempts"].tolist() == [[2.0, 2.0], [2.0, 2.0]]
    assert arrays["recruitment_row_counts"].tolist() == [1, 3]


def test_per_unit_join_counts_four_connected_fields_and_graph_degrees(tmp_path):
    maps = np.zeros((3, 3, 2), dtype=np.float64)
    maps[0, 0, 0] = maps[2, 2, 0] = 1.0
    maps[0, 0, 1] = maps[0, 1, 1] = 1.0
    assert connected_components_above_half_peak(maps[:, :, 0]) == 2
    assert connected_components_above_half_peak(maps[:, :, 1]) == 1

    artifact = tmp_path / "place_fields.npz"
    np.savez_compressed(
        artifact,
        occupancy=np.ones((3, 3)),
        rate_maps=maps,
        spatial_information=np.array([0.4, 0.8]),
        active_fraction=np.array([0.1, 0.2]),
        control_edge_confidence=np.array([[0.0, 1.0], [0.0, 0.0]]),
        control_attempts=np.array([[0.0, 1.0], [1.0, 0.0]]),
        control_tctrl=np.array([[0.0, 3.0], [0.0, 0.0]]),
        birth_support=np.array([0.25, 1.0]),
        recruitment_row_counts=np.array([2, 0]),
    )
    rows = per_unit_rows({"condition": "test"}, artifact)
    assert rows[0]["connected_components_half_peak"] == 2
    assert rows[0]["reliable_out_degree"] == 1
    assert rows[1]["reliable_in_degree"] == 1
    assert rows[0]["recruitment_row_counts"] == 2.0
