from types import SimpleNamespace

import numpy as np

from sf_working_directories.IntrMotiv.dmlab.controller_q import exploration_epsilon
from sf_working_directories.IntrMotiv.dmlab.controller_schedule import UpdateClock, replay_rngs


def test_main_budget_identical_with_auxiliary_and_checkpoint_resume():
    clocks = [UpdateClock(), UpdateClock()]
    for accepted in (16000, 16384, 20480, 24576):
        for i, c in enumerate(clocks):
            for _ in range(c.due(accepted)):
                c.finish(256, 256 * i)
    assert clocks[0].completed == clocks[1].completed == 128
    assert clocks[0].main_positions == clocks[1].main_positions == 32768
    assert clocks[1].auxiliary_positions == 32768
    assert clocks[0].target_at == clocks[1].target_at == 100
    resumed = UpdateClock()
    resumed.load_state_dict(clocks[1].state_dict())
    assert resumed.due(28672) == 64
    assert resumed.state_dict() == clocks[1].state_dict()


def test_auxiliary_rng_does_not_perturb_main_selection():
    a, ha = replay_rngs(99)
    b, hb = replay_rngs(99)
    for _ in range(5):
        ha.integers(1000, size=256)
        hb.integers(1000, size=0)
        np.testing.assert_array_equal(a.integers(1000, size=256), b.integers(1000, size=256))


def test_warmup_and_annealing():
    cfg = SimpleNamespace(
        controller_learning_starts=16384, controller_epsilon_decay_decisions=250000, controller_epsilon=0.1
    )
    assert exploration_epsilon(0, cfg) == exploration_epsilon(16384, cfg) == 1
    assert abs(exploration_epsilon(141384, cfg) - 0.55) < 1e-8
    assert abs(exploration_epsilon(999999, cfg) - 0.1) < 1e-8


def test_controller_metrics_use_existing_tensorboard_wandb_route():
    from sample_factory.algo.utils.misc import LEARNER_ENV_STEPS, TRAIN_STATS
    from sf_working_directories.IntrMotiv.dmlab.reward_summaries import write_intrmotiv_summaries

    scalars = []
    writer = SimpleNamespace(add_scalar=lambda *args: scalars.append(args))
    runner = SimpleNamespace(writers={0: writer}, env_steps={0: 0})
    msg = {LEARNER_ENV_STEPS: 64, TRAIN_STATS: {"controller/main_updates": 1, "controller/rejected/stale": 2}}
    write_intrmotiv_summaries(runner, msg, 0)
    assert scalars == [("intrmotiv/controller/main_updates", 1, 64), ("intrmotiv/controller/rejected/stale", 2, 64)]
    assert msg[TRAIN_STATS] == {}


def test_ca3_readout_metrics_use_canonical_intrmotiv_tags():
    from sample_factory.algo.utils.misc import LEARNER_ENV_STEPS, TRAIN_STATS
    from sf_working_directories.IntrMotiv.dmlab.reward_summaries import write_intrmotiv_summaries

    scalars = []
    writer = SimpleNamespace(add_scalar=lambda *args: scalars.append(args))
    runner = SimpleNamespace(writers={0: writer}, env_steps={0: 0})
    msg = {
        LEARNER_ENV_STEPS: 64,
        TRAIN_STATS: {
            "ca3_readout_prediction_loss": 0.25,
            "ca3_readout_action_shuffle_delta": 0.5,
        },
    }
    write_intrmotiv_summaries(runner, msg, 0)
    assert scalars == [
        ("intrmotiv/ca3_readout/prediction_loss", 0.25, 64),
        ("intrmotiv/ca3_readout/action_shuffle_delta", 0.5, 64),
    ]
    assert msg[TRAIN_STATS] == {}
