import numpy as np

from sf_working_directories.IntrMotiv.evaluation.analyze_place_field_manifest import (
    finite_correlation,
    multilevel_field_structure,
)


def test_offline_multilevel_fields_use_the_online_eight_connected_contract():
    occupancy = np.full((7, 7), 10, dtype=np.int32)
    maps = np.zeros((7, 7, 2), dtype=np.float32)
    maps[2:4, 2:4, 0] = 1.0
    maps[1:3, 1:3, 1] = 1.0
    maps[4:6, 4:6, 1] = 1.0
    structure = multilevel_field_structure(maps, occupancy, np.asarray([0.2, 0.2]))
    assert structure["eligible"].tolist() == [True, True]
    assert structure["component_count"].shape == (3, 2)
    assert structure["mono"][0]
    assert structure["dominant_mass"][:, 0].min() >= 0.8
    assert not structure["mono"][1]


def test_incoming_confidence_field_spread_correlation_is_finite_and_directional():
    incoming = np.asarray([1.0, 2.0, 4.0, 8.0])
    spread = np.asarray([0.1, 0.2, 0.4, 0.8])
    assert finite_correlation(incoming, spread) > 0.9
    assert np.isnan(finite_correlation(np.ones(4), spread))
