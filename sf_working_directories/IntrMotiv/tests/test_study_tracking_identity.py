import argparse

from sample_factory.cfg.cfg import add_wandb_args
from sf_working_directories.IntrMotiv.dmlab.custom_params import add_hipposlam_env_args


def test_study_tracking_identity_is_saved_as_flat_metadata():
    parser = argparse.ArgumentParser()
    add_hipposlam_env_args(parser)
    add_wandb_args(parser)

    cfg = parser.parse_args(
        [
            "--study_id=example_study",
            "--study_condition=EXAMPLE_ARM",
            "--study_base=WAYPOINT_F64",
            "--wandb_tags=EXAMPLE_ARM",
        ]
    )

    assert cfg.study_id == "example_study"
    assert cfg.study_condition == "EXAMPLE_ARM"
    assert cfg.study_base == "WAYPOINT_F64"
    assert cfg.wandb_tags == ["EXAMPLE_ARM"]


def test_study_tracking_identity_defaults_to_none():
    parser = argparse.ArgumentParser()
    add_hipposlam_env_args(parser)
    cfg = parser.parse_args([])

    assert cfg.study_id is None
    assert cfg.study_condition is None
    assert cfg.study_base is None
