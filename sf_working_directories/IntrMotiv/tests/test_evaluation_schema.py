from sf_working_directories.IntrMotiv.evaluation.schema import METRIC_BY_KEY, measurement_scope, resolve_tag
from sf_working_directories.IntrMotiv.evaluation.tensorboard import condition_metadata


def test_explicit_tag_wins_over_legacy_suffix():
    metric = METRIC_BY_KEY["coverage_auc"]
    tags = {"policy_stats/avg_level_coverage_auc", "intrmotiv/exploration/window/coverage_auc"}
    assert resolve_tag(tags, metric) == "intrmotiv/exploration/window/coverage_auc"


def test_fixed_episode_coverage_suffix_and_scope():
    metric = METRIC_BY_KEY["coverage_auc"]
    tag = resolve_tag({"policy_stats/avg_z_00_fixed_coverage_auc"}, metric)
    assert tag == "policy_stats/avg_z_00_fixed_coverage_auc"
    assert measurement_scope(tag) == "physical_episode"


def test_persistence_condition_metadata():
    info = condition_metadata("00_GHRL_F16_L64_T243_HL10000_iter_S123")
    assert info["family"] == "global_fixed_hrl"
    assert info["graph_scope"] == "policy_global"
    assert info["half_life_options"] == 10000
    assert info["update_schedule"] == "iterative"
    assert info["seed"] == 123


def test_anti_collapse_condition_metadata():
    info = condition_metadata("00_GAC_F16_T1p80_G0p03_ERmean_S99")
    assert info["family"] == "global_fixed_hrl"
    assert info["graph_scope"] == "policy_global"
    assert info["dg_anti_collapse_arm"] == "global_prethreshold"
    assert info["dg_threshold"] == 1.8
    assert info["dg_global_punishment_coeff"] == 0.03
    assert info["seed"] == 99

    row = condition_metadata("00_FAR_F16_T2p43_R0p01_ERpunish_S8")
    assert row["family"] == "flat"
    assert row["dg_anti_collapse_arm"] == "row_repulsion"
    assert row["dg_row_repulsion_coeff"] == 0.01
