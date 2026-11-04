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


def test_wrong_cue_is_reproducible_and_does_not_consume_policy_rng():
    before = torch.random.get_rng_state().clone()
    for instruction in range(1, 6):
        for seed in range(61000, 61020):
            wrong = heldout.policy_instruction_for_reset(instruction, seed, "shuffled_cue")
            assert wrong in range(1, 6) and wrong != instruction
            assert wrong == heldout.policy_instruction_for_reset(instruction, seed, "shuffled_cue")
    assert torch.equal(before, torch.random.get_rng_state())


def test_shuffle_keeps_physical_reset_reward_and_policy_seed(monkeypatch):
    class Environment:
        def __init__(self):
            self.unwrapped = self

        def seed(self, seed):
            self.last_reset_seed = seed + 100

        def reset(self):
            self.obs = {"pos": torch.tensor([[100.0, 200.0, 0.0]]), "INSTR": torch.tensor([[3]])}
            return self.obs, {}

        def step(self, action):
            assert self.obs["INSTR"].item() == 3  # Policy intervention must not mutate the environment.
            return self.obs, torch.tensor([10.0]), torch.tensor([True]), torch.tensor([False]), {}

    class Actor:
        core = SimpleNamespace()

        def __call__(self, obs, state):
            self.seen_instruction = obs["INSTR"].item()
            return {"actions": torch.tensor([[0]]), "new_rnn_states": state}

    monkeypatch.setattr(heldout, "prepare_and_normalize_obs", lambda actor, obs: obs)
    monkeypatch.setattr(heldout, "preprocess_actions", lambda env_info, action: action)
    monkeypatch.setattr(heldout, "get_rnn_size", lambda cfg: 1)
    actor = Actor()
    cfg = SimpleNamespace(eval_deterministic=False, reward_instruction_count=5, env_frameskip=4)
    correct = heldout.evaluate_episode(actor, cfg, Environment(), None, 61000, None, None, 1)
    shuffled = heldout.evaluate_episode(actor, cfg, Environment(), None, 61000, None, None, 1, "shuffled_cue")
    for key in ("requested_seed", "engine_seed", "start_x", "start_y", "number_instruction", "reward_center_x", "reward_center_y", "policy_seed"):
        assert correct[key] == shuffled[key]
    assert shuffled["policy_number_instruction"] == actor.seen_instruction != 3
    assert shuffled["number_instruction"] == 3
    assert shuffled["physical_success"] and not shuffled["terminal_pose_valid"]


@pytest.mark.parametrize("protocol", ["zero_ca3", "zero_depth"])
def test_worker_mask_preserves_other_inputs_and_removes_hook_on_failure(protocol):
    core = SimpleNamespace(
        fixed_task_conditioning=True, hrl_enabled=False, core_output_size=4,
        target_condition_start=17, readout_mode="off", dg_goal_modulation=None,
    )
    encoder = SimpleNamespace(depth_encoder=SimpleNamespace(get_out_size=lambda: 10), instructions_lstm_units=3)
    actor = SimpleNamespace(core=core, encoder=encoder, decoder=torch.nn.Identity())
    cfg = SimpleNamespace(depth_sensor=True, dg_goal_input="none")
    # Match the actual D51 layout: CA3, ten depth channels, three source-map
    # channels, then the task vector. A depth intervention must retain both cues.
    value = torch.arange(1.0, 21.0).reshape(1, 20)
    original = value.clone()
    with pytest.raises(RuntimeError, match="probe failure"):
        with heldout.worker_input_intervention(actor, cfg, protocol):
            changed = actor.decoder(value)
            if protocol == "zero_ca3":
                assert torch.count_nonzero(changed[:, :4]) == 0
                assert torch.equal(changed[:, 4:], value[:, 4:])
                assert torch.equal(changed[:, 4:17], value[:, 4:17])  # All 13 bypass channels.
            else:
                assert torch.equal(changed[:, :4], value[:, :4])
                assert torch.count_nonzero(changed[:, 4:14]) == 0
                assert torch.equal(changed[:, 14:], value[:, 14:])
                assert torch.equal(changed[:, 14:17], value[:, 14:17])  # Fixed map identity.
            assert torch.equal(value, original)
            raise RuntimeError("probe failure")
    assert not actor.decoder._forward_pre_hooks
    assert torch.equal(actor.decoder(value), original)


def test_worker_mask_rejects_ambiguous_bypass_layout():
    core = SimpleNamespace(fixed_task_conditioning=True, core_output_size=4, target_condition_start=19)
    encoder = SimpleNamespace(depth_encoder=SimpleNamespace(get_out_size=lambda: 10), instructions_lstm_units=3)
    actor = SimpleNamespace(core=core, encoder=encoder, decoder=torch.nn.Identity())
    with pytest.raises(ValueError, match="depth/instruction"):
        with heldout.worker_input_intervention(actor, SimpleNamespace(depth_sensor=True), "zero_depth"):
            pass
