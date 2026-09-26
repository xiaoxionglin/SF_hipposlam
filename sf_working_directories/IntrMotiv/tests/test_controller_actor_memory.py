from copy import deepcopy

import pytest
import torch

from sf_working_directories.IntrMotiv.dmlab.controller_actor_memory import ActorMemory
from sf_working_directories.IntrMotiv.dmlab.controller_history import ControllerHistory, reconstruct_controller_history
from sf_working_directories.IntrMotiv.tests.test_controller_history import SmallHistoryModel


@pytest.mark.parametrize("write", [False, True])
def test_publication_rebuilds_traces_preserves_manager_and_does_not_touch_graph(write):
    torch.manual_seed(62)
    model = SmallHistoryModel(write).eval()
    cache = ActorMemory()
    state = torch.zeros(1, model.core.total_state_size)
    observations = []
    goals = []
    for t in range(8):
        obs = {
            "obs": torch.randn(1, 4),
            "bypass": torch.randn(1, 2),
            "controller_identity": torch.tensor([[7, 0, t, t]]),
        }
        state = cache.before_forward(model, obs, state, 0)
        goal = torch.nn.functional.one_hot(torch.tensor([t % 3]), 3).float()
        head = model.forward_head(obs)
        if write:
            head = torch.cat((head, goal), -1)
        with torch.no_grad():
            _, state = model.forward_core(head, state)
        cache.after_forward(model, goal)
        observations.append(obs)
        goals.append(goal[0])
    old_state = state.clone()
    with torch.no_grad():
        model.dg.linear.weight.mul_(1.7)
        if write:
            model.core.dg_goal_modulation.add_(0.3)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    next_obs = {
        "obs": torch.randn(1, 4),
        "bypass": torch.randn(1, 2),
        "controller_identity": torch.tensor([[7, 0, 8, 8]]),
    }
    rng = torch.get_rng_state().clone()
    rebuilt = cache.before_forward(model, next_obs, state, 1)
    assert torch.equal(rng, torch.get_rng_state())
    assert cache.rebuilds == 1 and cache.failures == 0
    h = ControllerHistory(
        {k: torch.cat([o[k] for o in observations]) for k in ("obs", "bypass")},
        torch.arange(8),
        torch.zeros_like(state),
        torch.stack(goals),
        True,
        0,
    )
    with torch.no_grad():
        expected = reconstruct_controller_history(model, h, memory_only=True)["reconstructed_state"]
    if write:
        old_c, _ = model.core.split_worker_state(old_state)
        new_c, new_w = model.core.split_worker_state(rebuilt)
        exp_c, exp_w = model.core.split_worker_state(expected)
        torch.testing.assert_close(new_w, exp_w)
    else:
        old_c, new_c, exp_c = old_state, rebuilt, expected
    torch.testing.assert_close(new_c[:, : model.core.base_state_size], exp_c[:, : model.core.base_state_size])
    torch.testing.assert_close(
        new_c[:, model.core.base_state_size :], old_c[:, model.core.base_state_size :], rtol=0, atol=0
    )
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)


def test_missing_actor_history_fails_instead_of_zeroing_state():
    model = SmallHistoryModel().eval()
    cache = ActorMemory()
    state = torch.ones(1, model.core.total_state_size)
    obs = {"obs": torch.ones(1, 4), "bypass": torch.zeros(1, 2), "controller_identity": torch.tensor([[0, 0, 4, 4]])}
    with pytest.raises(RuntimeError, match="missing"):
        cache.before_forward(model, obs, state, 1)
    assert torch.all(state == 1) and cache.failures == 1


def test_weights_and_publication_marker_must_agree():
    from types import SimpleNamespace

    model = SmallHistoryModel().eval()
    model.controller_q = SimpleNamespace(publication_version=torch.tensor(2))
    memory = ActorMemory()
    observations = {
        "obs": torch.randn(1, 4),
        "bypass": torch.randn(1, 2),
        "controller_identity": torch.tensor([[0, 0, 0, 0]]),
    }
    with pytest.raises(RuntimeError, match="version mismatch"):
        memory.before_forward(model, observations, torch.zeros(1, model.core.total_state_size), 3)
    assert memory.failures == 1
