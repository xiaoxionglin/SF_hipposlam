from collections import deque
from types import SimpleNamespace

from sample_factory.algo.utils.misc import LEARNER_ENV_STEPS, TRAIN_STATS
from sf_working_directories.IntrMotiv.dmlab.reward_summaries import write_intrmotiv_summaries


class RecordingWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))


def test_intrmotiv_metrics_are_grouped_and_removed_from_train_stats():
    writer = RecordingWriter()
    runner = SimpleNamespace(writers={0: writer}, env_steps={0: 999})
    msg = {
        LEARNER_ENV_STEPS: 123,
        TRAIN_STATS: {
            "reward_for_advantage_mean": 0.25,
            "intrinsic_reward_mean": 0.25,
            "intrinsic_reward_sum": 4.0,
            "env_reward_mean": 0.0,
        },
    }

    write_intrmotiv_summaries(runner, msg, 0)

    assert writer.scalars == [
        ("intrmotiv/reward/advantage_mean", 0.25, 123),
        ("intrmotiv/reward/intrinsic_mean", 0.25, 123),
        ("intrmotiv/reward/intrinsic_sum", 4.0, 123),
        ("intrmotiv/reward/environment_mean", 0.0, 123),
    ]

    assert msg[TRAIN_STATS] == {}


def test_intrmotiv_handler_leaves_framework_stats_for_train_namespace():
    writer = RecordingWriter()
    runner = SimpleNamespace(writers={0: writer}, env_steps={0: 77})

    msg = {TRAIN_STATS: {"loss": 1.0}}
    write_intrmotiv_summaries(runner, msg, 0)

    assert writer.scalars == []
    assert msg[TRAIN_STATS] == {"loss": 1.0}


def test_online_spatial_scalars_are_grouped_without_image_payloads():
    writer = RecordingWriter()
    runner = SimpleNamespace(writers={0: writer}, env_steps={0: 77})
    stats = {
        "online_spatial_place_visited_cell_fraction": 0.5,
        "online_spatial_place_active_only_map_cosine": 0.25,
        "online_spatial_trajectory_path_efficiency": 0.8,
    }

    write_intrmotiv_summaries(runner, {LEARNER_ENV_STEPS: 1_000_000, TRAIN_STATS: stats}, 0)

    assert writer.scalars == [
        ("intrmotiv/online/place_field/visited_cell_fraction", 0.5, 1_000_000),
        ("intrmotiv/online/place_field/active_only_map_cosine", 0.25, 1_000_000),
        ("intrmotiv/online/trajectory/path_efficiency", 0.8, 1_000_000),
    ]
    assert stats == {}


def test_dg_usage_summaries_use_the_intrmotiv_namespace():
    writer = RecordingWriter()
    runner = SimpleNamespace(writers={0: writer}, env_steps={0: 77})
    stats = {
        "dg_unit_duty_cycle_min": 0.01,
        "dg_unit_duty_cycle_mean": 0.03,
        "dg_unit_duty_cycle_max": 0.08,
        "dg_usage_entropy": 0.75,
    }

    write_intrmotiv_summaries(runner, {LEARNER_ENV_STEPS: 123, TRAIN_STATS: stats}, 0)

    assert writer.scalars == [
        ("intrmotiv/dg/unit_duty_cycle_min", 0.01, 123),
        ("intrmotiv/dg/unit_duty_cycle_mean", 0.03, 123),
        ("intrmotiv/dg/unit_duty_cycle_max", 0.08, 123),
        ("intrmotiv/dg/usage_entropy", 0.75, 123),
    ]
    assert stats == {}


def test_corrected_event_diagnostics_use_dg_and_encoder_namespaces():
    writer = RecordingWriter()
    runner = SimpleNamespace(writers={0: writer}, env_steps={0: 77})
    stats = {
        "dg_behavior_dominant_event_fraction": 0.02,
        "dg_behavior_multi_onset_event_fraction": 0.25,
        "dg_behavior_non_dominant_onsets_per_event": 0.4,
        "dg_valid_minibatch_unused_unit_count": 3.0,
        "dg_valid_minibatch_unused_unit_fraction": 0.1875,
        "encoder_dominant_event_count": 41.0,
        "encoder_feedback_on_dominant_event_mean": 22.0,
    }

    write_intrmotiv_summaries(runner, {LEARNER_ENV_STEPS: 123, TRAIN_STATS: stats}, 0)

    assert ("intrmotiv/dg/behavior_dominant_event_fraction", 0.02, 123) in writer.scalars
    assert ("intrmotiv/dg/behavior_multi_onset_event_fraction", 0.25, 123) in writer.scalars
    assert ("intrmotiv/dg/valid_minibatch_unused_unit_count", 3.0, 123) in writer.scalars
    assert ("intrmotiv/encoder/dominant_event_count", 41.0, 123) in writer.scalars
    assert ("intrmotiv/encoder/feedback_on_dominant_event_mean", 22.0, 123) in writer.scalars
    assert stats == {}


def test_pbt_objective_uses_coverage_and_rejects_invalid_hrl():
    writers = {0: RecordingWriter(), 1: RecordingWriter()}
    runner = SimpleNamespace(
        writers=writers,
        env_steps={0: 100, 1: 100},
        cfg=SimpleNamespace(with_pbt=True, hrl_pbt_max_silent_fraction=0.5, num_policies=2),
        policy_avg_stats={"z_00_level_coverage_auc": [deque([12.0]), deque([20.0])]},
    )

    valid_stats = {
        "dg_silent_unit_frac": 0.1,
        "hrl_active_target_frac": 0.9,
        "intrinsic_reward_negative_frac": 0.0,
    }
    invalid_stats = dict(valid_stats, dg_silent_unit_frac=0.8)
    write_intrmotiv_summaries(runner, {TRAIN_STATS: valid_stats, LEARNER_ENV_STEPS: 100}, 0)
    write_intrmotiv_summaries(runner, {TRAIN_STATS: invalid_stats, LEARNER_ENV_STEPS: 100}, 1)

    assert runner.policy_avg_stats["intrmotiv_hrl_validity"][0][-1] == 1.0
    assert runner.policy_avg_stats["intrmotiv_hrl_validity"][1][-1] == 0.0
    assert runner.policy_avg_stats["intrmotiv_pbt_objective"][0][-1] == 12.0
    assert runner.policy_avg_stats["intrmotiv_pbt_objective"][1][-1] == 0.0


def test_distance_metric_is_not_exposed_as_a_pbt_objective():
    writer = RecordingWriter()
    runner = SimpleNamespace(
        writers={0: writer},
        env_steps={0: 100},
        cfg=SimpleNamespace(with_pbt=True, num_policies=1),
        policy_avg_stats={},
    )

    write_intrmotiv_summaries(
        runner,
        {LEARNER_ENV_STEPS: 123, TRAIN_STATS: {"distance_metric": 17.25}},
        0,
    )

    assert "distance_metric" not in runner.policy_avg_stats
    assert not any(tag == "intrmotiv/pbt/distance_metric" for tag, _, _ in writer.scalars)
