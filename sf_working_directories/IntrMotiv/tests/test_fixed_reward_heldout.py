"""Heldout results must not label an autoreset pose as a terminal pose."""

from types import SimpleNamespace

import pytest
import torch

from sf_working_directories.IntrMotiv.evaluation import fixed_reward_heldout as heldout


@pytest.mark.parametrize("terminated", [True, False])
def test_autoreset_position_is_not_reported_as_terminal_position(monkeypatch, terminated):
    class Environment:
        unwrapped = None
        last_reset_seed = 70001

        def seed(self, seed):
            self.last_reset_seed = seed

        def reset(self):
            return {"pos": torch.tensor([[100.0, 200.0, 0.0]]), "INSTR": torch.tensor([[1]])}, {}

        def step(self, action):
            # On termination this position belongs to the automatically reset
            # episode. Otherwise it is the actual next observation.
            position = [999.0, 888.0, 0.0] if terminated else [110.0, 220.0, 0.0]
            return (
                {"pos": torch.tensor([position]), "INSTR": torch.tensor([[2]])},
                torch.tensor([10.0 if terminated else 0.0]),
                torch.tensor([terminated]),
                torch.tensor([False]),
                {},
            )

    class Actor:
        core = SimpleNamespace()

        def __call__(self, obs, state):
            return {"actions": torch.tensor([[0]]), "new_rnn_states": state}

    monkeypatch.setattr(heldout, "prepare_and_normalize_obs", lambda actor, obs: obs)
    monkeypatch.setattr(heldout, "preprocess_actions", lambda env_info, action: action)
    monkeypatch.setattr(heldout, "get_rnn_size", lambda cfg: 1)
    env = Environment()
    env.unwrapped = env
    row = heldout.evaluate_episode(
        Actor(), SimpleNamespace(eval_deterministic=False), env, None, 61000, None, None, 1
    )

    assert row["physical_success"] is terminated
    assert row["external_reward"] == (10.0 if terminated else 0.0)
    assert row["decisions_to_termination"] == 1
    assert row["number_instruction"] == 1  # The task cue belongs to the initial episode.
    assert row["terminal"] is terminated
    assert row["terminal_pose_valid"] is (not terminated)
    assert (row["terminal_x"], row["terminal_y"]) == (("", "") if terminated else (110.0, 220.0))
