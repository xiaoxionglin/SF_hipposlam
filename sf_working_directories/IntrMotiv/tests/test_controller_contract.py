import pytest

from sf_working_directories.IntrMotiv.dmlab.controller_contract import preservation_delta


def test_controller_delta_cannot_hide_frozen_dg_or_reduced_manager():
    parent = {"transfer_freeze_dg": False, "Hippo_n_feature": 64, "hrl_manager_mode": "frontier_waypoint"}
    candidate = dict(parent, controller_learning="ddqn", controller_her=True)
    assert len(preservation_delta(parent, candidate)) == 2
    for field, value in [
        ("transfer_freeze_dg", True),
        ("Hippo_n_feature", 3),
        ("hrl_manager_mode", "frontier_direct"),
        ("controller_typo", True),
    ]:
        with pytest.raises(ValueError, match=field):
            preservation_delta(parent, dict(candidate, **{field: value}))


def test_seed_change_requires_an_explicit_factor_and_no_fields_are_dropped():
    parent = {"seed": 99, "gamma": 0.99}
    with pytest.raises(ValueError, match="seed"):
        preservation_delta(parent, dict(parent, seed=8))
    delta = preservation_delta(
        parent,
        dict(parent, seed=8),
        approved_factors={"seed": "New learner seed; source seed remains 99 in parent manifest"},
    )
    assert delta[0]["classification"] == "explicit_experimental_factor"
    with pytest.raises(ValueError, match="gamma"):
        preservation_delta(parent, {"seed": 99})
