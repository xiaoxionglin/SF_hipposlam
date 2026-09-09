from pathlib import Path

from sf_working_directories.IntrMotiv.dmlab.dmlab_env import dmlab_env_by_name


LEVEL_NAME = "openfield_map2_fixed_loc3_fixedlength_noreward"


def test_fixed_length_level_is_registered():
    spec = dmlab_env_by_name(LEVEL_NAME)

    assert spec.name == LEVEL_NAME
    assert spec.level == LEVEL_NAME


def test_fixed_length_level_disables_goal_termination_only_in_wrapper():
    repo_root = Path(__file__).resolve().parents[3]
    wrapper = repo_root / "deepmindlab_patch" / "game_scripts" / "levels" / f"{LEVEL_NAME}.lua"
    source = wrapper.read_text()

    assert "require 'levels.openfield_map2_fixed_loc3_noreward'" in source
    assert "function api:hasEpisodeFinished" in source
    assert "return false" in source
