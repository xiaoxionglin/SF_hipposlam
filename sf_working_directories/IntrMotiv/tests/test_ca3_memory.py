from types import SimpleNamespace

import torch

from sf_working_directories.IntrMotiv.dmlab.ca3_memory import (
    advance_goal, advance_reference_goal, inhibit_reentry, absent_event_gate,
)
from sf_working_directories.IntrMotiv.dmlab.custom_core import FiniteMemoryCore


def test_inhibition_continuity_strength_and_gradient():
    previous = torch.zeros(1, 3, 5, requires_grad=True)
    with torch.no_grad():
        previous[0, 0, 0] = 4
        previous[0, 1, 3] = 2
    raw = torch.tensor([[1., 3., 2.]], requires_grad=True)
    assert torch.equal(inhibit_reentry(raw, previous, "hard"), torch.tensor([[1., 0., 2.]]))
    soft = inhibit_reentry(raw, previous, "trace_subtractive")
    assert torch.equal(soft, torch.tensor([[1., 1., 2.]]))
    soft.sum().backward()
    assert previous.grad is None
    assert torch.equal(raw.grad, torch.ones_like(raw))
    assert inhibit_reentry(torch.ones_like(raw), previous, "trace_subtractive")[0, 1] == 0


def test_absence_is_checked_before_injection():
    progression = torch.tensor([[[5, 5], [0, 5], [1, 0], [0, 1]]])
    dominant = torch.tensor([[[0, 0], [1, 0], [0, 1], [1, 0]]]).bool()
    assert absent_event_gate(progression, dominant, 5).tolist() == [[False, True, True, False]]


def test_goal_samples_only_absent_and_empty_set_is_null():
    previous = torch.zeros(2, 3, 5)
    updated = torch.ones_like(previous)
    updated[0, 2] = 0
    state, goal = advance_goal(torch.zeros(2, 6), previous, updated, 4, .4, 2)
    assert state[:, 0].tolist() == [3, 0]
    assert goal.tolist() == [[0, 0, 1], [0, 0, 0]]
    assert state[:, 2].eq(0).all()


def test_goal_allows_intermediates_and_pays_once_with_latency():
    state = torch.tensor([[3., 0, 0, 0, 0, 3]])
    prev = torch.zeros(1, 3, 5)
    now = prev.clone()
    now[0, 1, 0] = 1
    state, _ = advance_goal(state, prev, now, 4, .4, 2)
    assert state[0, 0] == 3 and state[0, 1] == 1 and state[0, 2] == 0
    nxt = now.roll(1, -1)
    nxt[0, 2, 0] = 1
    state, _ = advance_goal(state, now, nxt, 4, .4, 2)
    assert torch.isclose(state[0, 2], torch.tensor(.3))
    assert state[0, 0] == 3 and state[0, 4] == 1
    state, _ = advance_goal(state, nxt, nxt, 4, .4, 2)
    assert state[0, 2] == 0


def test_goal_ambiguous_entry_timeout_and_boundary_hit():
    prev = torch.zeros(1, 3, 5)
    now = prev.clone()
    now[0, 0, 0], now[0, 1, 0] = 2, 1
    state, _ = advance_goal(torch.tensor([[2., 0, 0, 0, 0, 2]]), prev, now, 4, .4, 2)
    assert state[0, 3] == 1 and state[0, 2] == 0 and state[0, 0] == 2
    now.zero_()
    now[0, 1, 0] = 1
    state, _ = advance_goal(torch.tensor([[2., 3, 0, 0, 0, 2]]), prev, now, 4, .4, 2)
    assert torch.isclose(state[0, 2], torch.tensor(.1))
    state, _ = advance_goal(torch.tensor([[2., 3, 0, 0, 0, 2]]), prev, prev, 4, .4, 2)
    assert state[0, 1] == 4 and state[0, 2] == 0 and state[0, 0] == 2


def test_frozen_reference_goal_persists_and_pays_once():
    torch.manual_seed(1)
    state = torch.zeros(1, 6)
    previous = torch.zeros(1, 3)
    state, goal = advance_reference_goal(state, previous, previous, 4, .4)
    target = int(goal.argmax(-1).item())
    current = previous.clone()
    current[0, target] = 1
    state, _ = advance_reference_goal(state, previous, current, 4, .4)
    assert state[0, 2] > 0 and state[0, 4] == 1
    state, _ = advance_reference_goal(state, previous, current, 4, .4)
    assert state[0, 2] == 0 and state[0, 0] == target + 1


def config(goal=True, inhibition="none"):
    return SimpleNamespace(Hippo_R=2, Hippo_L=3, Hippo_n_feature=3,
        hrl_controllable_graph=False, dg_context_feedback="none",
        intrinsic_goal_mode="ca3_absent_target" if goal else "none",
        intrinsic_goal_horizon=4, intrinsic_goal_reward_max=.4,
        intrinsic_goal_reference_checkpoint=None,
        dg_ca3_reentry_inhibition=inhibition, reward_scale=.1)


def test_goal_core_packed_and_single_match_and_checkpoint():
    core = FiniteMemoryCore(config(), 5)
    sequence = torch.zeros(4, 2, 5)
    sequence[0, :, 0] = 1
    sequence[2, :, 1] = 1
    initial = torch.zeros(2, core.total_state_size)
    torch.manual_seed(44)
    state = initial.clone()
    outputs = []
    for item in sequence:
        out, state = core(item, state)
        outputs.append(out)
    torch.manual_seed(44)
    packed = torch.nn.utils.rnn.pack_padded_sequence(sequence, [4, 4], enforce_sorted=False)
    out, end = core(packed, initial)
    unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(out)
    assert torch.equal(unpacked, torch.stack(outputs))
    assert torch.equal(end, state)
    assert out.data.size(-1) == core.get_out_size()
    clone = FiniteMemoryCore(config(), 5)
    clone.load_state_dict(core.state_dict())
    assert core.policy_graph is None


def test_hard_core_retains_continuity_then_blocks_reentry():
    core = FiniteMemoryCore(config(False, "hard"), 3)
    state = torch.zeros(1, core.total_state_size)
    for raw, expected in (([1., 0, 0], 1), ([1., 0, 0], 1), ([0., 0, 0], 0), ([1., 0, 0], 0)):
        out, state = core(torch.tensor([raw]), state)
        assert out[0, 0] == expected


def test_goal_arrival_is_measured_before_reentry_inhibition():
    cfg = config(True, "trace_subtractive")
    core = FiniteMemoryCore(cfg, 3)
    state = torch.zeros(1, core.total_state_size)
    # Unit zero is remembered but not continuously active, so behavior-level
    # inhibition suppresses its new CA3 injection.
    state[0, 2] = 1
    state[0, core.memory_base_size + core.goal_detector_state_size] = 1
    state[0, -1] = 1
    output, updated = core(torch.tensor([[1., 0, 0]]), state)
    assert output[0, 0] == 0
    assert updated[0, -4] > 0
    assert updated[0, -2] == 1


def test_goal_state_allocation_reset_and_replay_descriptor():
    from sf_working_directories.IntrMotiv.dmlab.train_hipposlam import maybe_overwrite_rnn_size
    from sf_working_directories.IntrMotiv.dmlab.custom_learner import BaseDistanceRecorder
    from sample_factory.utils.attr_dict import AttrDict
    cfg = config()
    cfg.cli_args = {"rnn_size": 0}
    cfg.hrl_goal_conditioning = "target_id_film"
    cfg.hrl_target_timing = "immediate"
    maybe_overwrite_rnn_size(cfg)
    core = FiniteMemoryCore(cfg, 16)
    assert cfg.rnn_size == core.total_state_size
    assert cfg.rnn_persistent_state_size == 1
    learner = BaseDistanceRecorder.__new__(BaseDistanceRecorder)
    learner.cfg = cfg
    learner.actor_critic = SimpleNamespace(core=core)
    states = torch.zeros(3, core.total_state_size)
    states[:, -1] = torch.tensor([1., 3., 0.])
    outputs = torch.randn(3, core.get_out_size())
    replay = learner._override_core_outputs_for_replay(outputs, AttrDict(rnn_states=states))
    assert torch.equal(replay[:, -3:], torch.tensor([[1., 0, 0], [0., 0, 1], [0., 0, 0]]))
    assert torch.equal(outputs[:, :-3], replay[:, :-3])
    assert learner._last_behavior_replay_mismatch == 0


def test_packed_inhibition_gradient_and_unequal_lengths():
    for mode in ("hard", "trace_subtractive"):
        core = FiniteMemoryCore(config(False, mode), 3)
        inputs = torch.rand(5, 2, 3, requires_grad=True)
        packed = torch.nn.utils.rnn.pack_padded_sequence(inputs, [3, 5], enforce_sorted=False)
        output, _ = core(packed, torch.zeros(2, core.total_state_size))
        output.data.sum().backward()
        assert torch.isfinite(inputs.grad).all()
        assert inputs.grad[3:, 0].eq(0).all()
