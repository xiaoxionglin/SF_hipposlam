from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from sf_working_directories.IntrMotiv.dmlab.controller_replay import PhysicalReplay
from sf_working_directories.IntrMotiv.dmlab.controller_snapshot import ControllerSnapshot
from sf_working_directories.IntrMotiv.dmlab.controller_stored_replay import (
    evaluate_pairs,
    example_from_replay,
    hindsight_examples,
)
from sf_working_directories.IntrMotiv.dmlab.controller_transition import TransitionInput, transition_values
from sf_working_directories.IntrMotiv.dmlab.custom_learner import DistanceLearnerReward
from sf_working_directories.IntrMotiv.tests.test_controller_transition import RewardModel, physical_rows


def fixture():
    model = RewardModel(False).eval()
    model.cfg.gamma = 0.99
    model.cfg.ppo_dg_gradient = "stop"
    rows, states = physical_rows(model)
    adapter = object.__new__(DistanceLearnerReward)
    adapter.cfg = model.cfg
    adapter.actor_critic = model
    retained = []
    for i, row in enumerate(rows):
        reward = 0.0
        magnitude = 1.0
        if i + 2 < len(states):
            buff = {
                "rnn_states": torch.stack([states[i][0], states[i + 1][0]])[None],
                "rewards": torch.zeros(1, 1),
                "dones": torch.zeros(1, 1, dtype=torch.bool),
            }
            adapter._calculate_reward_components(buff, {"new_rnn_states": states[i + 2]})
            reward = float(buff["rewards"].item())
            magnitude = float(buff["hrl_control_reward_magnitude"].item())
        retained.append(
            replace(
                row,
                observation={},
                worker_state=states[i + 1][0, : model.core.base_state_size].numpy().copy(),
                real_reward=reward,
                real_events={"hrl_control_reward_magnitude": magnitude},
            )
        )

    def factory():
        return RewardModel(False)

    learner = SimpleNamespace(
        actor_critic=model,
        cfg=model.cfg,
        device=torch.device("cpu"),
        controller_version=0,
        online_snapshot=ControllerSnapshot(factory, model, 0),
        target_snapshot=ControllerSnapshot(factory, model, 0),
        replay=PhysicalReplay(20, 99),
        her_rng=np.random.default_rng(99),
    )
    for row in retained:
        learner.replay.receive(row)
    return learner, rows, retained


def test_stored_batch_matches_frozen_reference_and_never_runs_encoder_or_core():
    learner, raw, rows = fixture()
    reference = transition_values(learner.actor_critic, TransitionInput(tuple(raw[:2]), 0))
    example = TransitionInput(tuple(rows[:2]), 0)
    with patch.object(
        learner.online_snapshot.model, "forward_head", side_effect=AssertionError("encoder forward")
    ), patch.object(
        learner.online_snapshot.model.core, "forward", side_effect=AssertionError("core forward")
    ), patch.object(
        learner.target_snapshot.model, "forward_head", side_effect=AssertionError("target encoder")
    ), patch.object(
        learner.target_snapshot.model.core, "forward", side_effect=AssertionError("target core")
    ):
        result = evaluate_pairs(learner, [example])[0]
        torch.testing.assert_close(result[1]["q"], reference["q"])
        result[0].backward()
    assert learner.actor_critic.scale.grad is None
    assert learner.actor_critic.controller_q.main.weight.grad is not None


def test_her_relabels_decoder_and_reward_with_shared_stored_memory():
    learner, raw, rows = fixture()
    example = TransitionInput(tuple(rows[:2]), 0, virtual_goal=1, remaining=3)
    reference = transition_values(learner.actor_critic, TransitionInput(tuple(raw[:2]), 0, virtual_goal=1, remaining=3))
    result = evaluate_pairs(learner, [example])[0]
    torch.testing.assert_close(result[1]["q"], reference["q"])
    expected = torch.nn.functional.smooth_l1_loss(result[1]["q"][0, rows[0].action], reference["reward"])
    torch.testing.assert_close(result[0], expected)
    result[0].backward()
    assert learner.actor_critic.controller_q.auxiliary.weight.grad is not None
    assert learner.actor_critic.controller_q.main.weight.grad is None
    assert learner.actor_critic.scale.grad is None


@pytest.mark.parametrize("truncated", [False, True])
def test_terminal_main_and_her_use_certified_label_and_never_bootstrap(truncated):
    learner, _, rows = fixture()
    terminal = replace(
        rows[0],
        terminated=not truncated,
        truncated=truncated,
        successor_valid=True,
        terminal_dg=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        terminal_publication=0,
    )
    learner.replay = PhysicalReplay(10, 99)
    learner.replay.receive(terminal)
    main = example_from_replay(learner, terminal.key)
    aux = replace(main, virtual_goal=1, remaining=3)
    with torch.no_grad():
        learner.target_snapshot.model.controller_q.main.bias.fill_(1e6)
    result = evaluate_pairs(learner, [main, aux])
    for value, reward in zip(result, [terminal.real_reward, terminal.real_events["hrl_control_reward_magnitude"]]):
        torch.testing.assert_close(
            value[0], torch.nn.functional.smooth_l1_loss(value[1]["q"][0, terminal.action], torch.tensor(reward))
        )
    # A terminal row is not joined to a subsequent reset episode.
    assert len(main.rows) == 2 and not main.rows[1].worker_state.any()


def test_main_and_her_sampling_preserve_separate_rng_and_exclude_initial_goal():
    learner, _, rows = fixture()
    example = example_from_replay(learner, rows[0].key)
    rng_before = repr(learner.replay.rng.bit_generator.state)
    selected = hindsight_examples(learner, [example] * 8)
    assert selected and all(e.virtual_goal != 0 for e in selected)
    assert repr(learner.replay.rng.bit_generator.state) == rng_before


def test_replay_checkpoint_retains_stored_states_and_terminal_provenance():
    learner, _, rows = fixture()
    saved = learner.replay.state_dict()
    restored = PhysicalReplay(20, 2)
    restored.load_state_dict(saved)
    for key, row in learner.replay.rows.items():
        np.testing.assert_array_equal(restored.rows[key].worker_state, row.worker_state)
        assert restored.rows[key].real_events == row.real_events
    assert restored.session == learner.replay.session + 1
