from types import SimpleNamespace

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sf_working_directories.IntrMotiv.dmlab.goal_conditioned_dg import GoalConditionedDGCore


def make_core(n=3):
    cfg = SimpleNamespace(
        Hippo_R=2,
        Hippo_L=3,
        Hippo_n_feature=n,
        hrl_controllable_graph=True,
        hrl_graph_memory="policy_buffer",
        hrl_target_timing="immediate",
        hrl_manager_mode="frontier_direct",
        dg_context_feedback="none",
        dg_orthogonal_recruitment=False,
        DG_BN_intercept=1.0,
    )
    core = GoalConditionedDGCore(cfg, 2 * n + 2)
    return core


def head(pre, n):
    return torch.cat((torch.relu(pre - 1), torch.zeros(*pre.shape[:-1], 2), pre), -1)


def test_identity_and_detector_invariance():
    c = make_core()
    x = torch.tensor([[2.0, 0.0, 1.5]])
    s = torch.zeros(1, c.total_state_size)
    a, sa = c(head(x, 3), s)
    assert torch.equal(a[:, : c.core_output_size], a[:, c.total_output_size :])
    with torch.no_grad():
        c.dg_goal_modulation.normal_()
    h = head(x, 3)
    h = torch.cat((h, torch.tensor([[0.0, 1.0, 0.0]])), -1)
    b, sb = c(h, s)
    assert torch.equal(a[:, : c.core_output_size], b[:, : c.core_output_size])
    assert torch.equal(
        sa[:, : c.canonical_state_size - c.behavior_goal_state_size],
        sb[:, : c.canonical_state_size - c.behavior_goal_state_size],
    )
    assert not torch.equal(a[:, c.total_output_size :], b[:, c.total_output_size :])


def test_packed_replay_matches_actor_and_gradient_stops_at_base():
    torch.manual_seed(4)
    c = make_core()
    with torch.no_grad():
        c.dg_goal_modulation.normal_(std=0.1)
    x = (torch.rand(4, 2, 3) + 1.5).requires_grad_()
    h = head(x, 3)
    goals = torch.nn.functional.one_hot(torch.tensor([[0, 1], [2, 0], [1, 2], [0, 2]]), 3).float()
    replay = torch.cat((h, goals), -1)
    initial = torch.zeros(2, c.total_state_size)
    state = initial
    outs = []
    for t in range(4):
        out, state = c(replay[t], state)
        outs.append(out)
    packed = pack_padded_sequence(replay, [4, 3], enforce_sorted=False)
    result, newstate = c(packed, initial)
    unpacked, _ = pad_packed_sequence(result)
    for t in range(4):
        assert torch.allclose(unpacked[t, 0], outs[t][0], atol=1e-6)
        if t < 3:
            assert torch.allclose(unpacked[t, 1], outs[t][1], atol=1e-6)
    loss = c.worker_view(result.data).sum()
    loss.backward()
    assert c.dg_goal_modulation.grad.abs().sum() > 0
    assert x.grad is None or x.grad.abs().sum() == 0
    assert torch.allclose(newstate[0], state[0], atol=1e-6)


def test_goal_switch_and_zero_goal_identity_and_reset():
    c = make_core()
    pre = torch.tensor([[2.0, 2.0, 2.0]])
    with torch.no_grad():
        c.dg_goal_modulation[1, 3:] = 2
    assert torch.equal(c.write_activity(pre, torch.zeros(1, 3)), torch.relu(pre - 1))
    a = c.write_activity(pre, torch.tensor([[1.0, 0.0, 0.0]]))
    b = c.write_activity(pre, torch.tensor([[0.0, 1.0, 0.0]]))
    assert (b > a).all()
    h = torch.cat((head(pre, 3), torch.tensor([[0.0, 1.0, 0.0]])), -1)
    z = torch.zeros(1, c.total_state_size)
    o, s = c(h, z)
    o2, s2 = c(h, z)
    assert torch.equal(o, o2) and torch.equal(s, s2)
    restored = make_core()
    restored.load_state_dict(c.state_dict())
    o3, s3 = restored(h, z)
    assert torch.equal(o, o3) and torch.equal(s, s3)


def test_all_capacity_state_widths():
    for n in (16, 32, 64):
        c = make_core(n)
        s = torch.zeros(1, c.total_state_size)
        o, s = c(head(torch.full((1, n), 2.0), n), s)
        assert s.shape[-1] == c.total_state_size
        assert c.worker_view(o).shape[-1] == c.get_out_size()


def test_fixed_observation_panel_separates_worker_from_detector(tmp_path, monkeypatch):
    import numpy as np

    from sf_working_directories.IntrMotiv.evaluation import observation_panel as panel

    c = make_core()
    with torch.no_grad():
        c.dg_goal_modulation[1, 3:] = 1

    class Actor:
        core = c
        encoder = SimpleNamespace(DG_projection=SimpleNamespace(last_pre_threshold_logits=None))

        def forward_head(self, obs):
            self.encoder.DG_projection.last_pre_threshold_logits = obs["pre"]
            return head(obs["pre"], 3)

        def forward_core(self, h, s):
            return self.core(h, s)

    actor = Actor()
    cfg = SimpleNamespace(Hippo_n_feature=3, Hippo_R=2, Hippo_L=3, dg_goal_input="write")
    monkeypatch.setattr(panel, "get_rnn_size", lambda cfg: c.total_state_size)
    monkeypatch.setattr(panel, "prepare_and_normalize_obs", lambda actor, obs: obs)
    path = tmp_path / "panel.npz"
    np.savez(
        path,
        obs_pre=np.full((3, 1, 3), 2.0, dtype=np.float32),
        obs_pos=np.zeros((3, 1, 3)),
        obs_rot=np.zeros((3, 1, 3)),
        dones=np.array([[False], [True], [False]]),
    )
    a = panel.replay_observations(actor, cfg, path, "cpu", goal_id=0, include_worker=True)
    b = panel.replay_observations(actor, cfg, path, "cpu", goal_id=1, include_worker=True)
    assert np.array_equal(a[1], b[1])
    assert np.array_equal(a[2], b[2])
    assert (b[3]["worker_dg_activity"] > a[3]["worker_dg_activity"]).all()


def test_uninterrupted_goal_diagnostics_respect_episode_boundaries():
    import pandas as pd

    from sf_working_directories.IntrMotiv.evaluation.place_fields import goal_behavior_diagnostics

    pose = pd.DataFrame({"agent": [0] * 6, "num_traj": [0, 0, 0, 0, 1, 1]})
    result = goal_behavior_diagnostics(pose, [1, 1, 1, 2, 2, 2], [0, 1, 1, 0, 1, 1])
    assert result["max_uninterrupted_decisions"] == 3
    assert result["max_same_goal_timeouts"] == 2
    assert [x["decisions"] for x in result["segments"]] == [3, 1, 2]


def test_frame_milestones_crossing_and_resume():
    from sf_working_directories.IntrMotiv.dmlab.custom_learner import BaseDistanceRecorder

    saved = []
    fake = SimpleNamespace(
        cfg=SimpleNamespace(checkpoint_frame_targets="100,200,300"),
        env_steps=96,
        save_milestone=lambda: saved.append(fake.env_steps),
    )
    save = BaseDistanceRecorder._save_crossed_frame_targets
    save(fake, 88)
    fake.env_steps = 104
    save(fake, 96)
    fake.env_steps = 112
    save(fake, 104)
    # A restored model already beyond earlier milestones does not resave them.
    fake.env_steps = 208
    save(fake, 200)
    fake.env_steps = 304
    save(fake, 296)
    assert saved == [104, 304]
    fake.cfg.checkpoint_frame_targets = ""
    fake.env_steps = 400
    save(fake, 304)
    assert saved == [104, 304]
