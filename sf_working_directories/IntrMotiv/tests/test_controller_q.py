"""Controller return and gradient contracts, using the original worker modules."""

from types import SimpleNamespace

import pytest
import torch

from sf_working_directories.IntrMotiv.dmlab.controller_q import (
    ControllerQHeads,
    bounded_auxiliary_target,
    continuing_double_q_target,
    epsilon_distribution,
)
from sf_working_directories.IntrMotiv.dmlab.custom_actor_critic import TargetFiLMDecoder, controller_core_view
from sf_working_directories.IntrMotiv.dmlab.goal_conditioned_dg import GoalConditionedDGCore


def test_continuing_return_uses_online_selection_target_evaluation_and_physical_boundary():
    reward = torch.tensor([2.5, -3.0, 0.7], requires_grad=True)
    online = torch.tensor([[1.0, 5.0], [9.0, 1.0], [2.0, 3.0]], requires_grad=True)
    target = torch.tensor([[100.0, 7.0], [8.0, 100.0], [99.0, 4.0]], requires_grad=True)
    # The second row is a physical terminal. Rows one/three may cross manager switches.
    actual = continuing_double_q_target(reward, torch.tensor([False, True, False]), online, target, 0.9)
    torch.testing.assert_close(actual, torch.tensor([8.8, -3.0, 4.3]))
    assert not actual.requires_grad


def test_auxiliary_deadline_does_not_end_main_return():
    reward = torch.tensor([2.5, 1.0, -2.0])
    online = torch.ones(3, 2)
    target = torch.full((3, 2), 5.0)
    main = continuing_double_q_target(reward, torch.zeros(3, dtype=torch.bool), online, target, 0.9)
    aux = bounded_auxiliary_target(
        reward, torch.tensor([True, False, False]), torch.tensor([8, 1, 4]), online, target, 0.9
    )
    torch.testing.assert_close(main, reward + 4.5)
    torch.testing.assert_close(aux, torch.tensor([2.5, 1.0, 2.5]))


@pytest.mark.parametrize("epsilon", [0.0, 0.1, 1.0])
def test_epsilon_greedy_preserves_action_mask(epsilon):
    q = torch.tensor([[100.0, -4.0, -2.0], [3.0, 99.0, 0.0]])
    mask = torch.tensor([[False, True, True], [True, False, False]])
    p = epsilon_distribution(q, epsilon, mask)
    torch.testing.assert_close(p.sum(-1), torch.ones(2))
    assert not p[~mask].any()
    assert p[0, 2] >= p[0, 1]
    with pytest.raises(ValueError, match="No valid action"):
        epsilon_distribution(q, epsilon, torch.zeros_like(mask))


@pytest.mark.parametrize("routing", ["stop", "joint"])
def test_auxiliary_trains_shared_worker_without_main_head_gradient(routing):
    torch.manual_seed(41)
    layout = SimpleNamespace(Hippo_n_feature=3, target_condition_start=12, get_out_size=lambda: 15)
    decoder = TargetFiLMDecoder(layout, hidden_size=8)
    heads = ControllerQHeads(8, 4, auxiliary=True)
    features = torch.randn(5, 15, requires_grad=True)
    hidden = decoder(controller_core_view(features, 12, routing))
    loss = heads.hindsight(hidden, torch.tensor([1, 2, 3, 4, 5]), 5).square().mean()
    loss.backward()
    assert all(p.grad is None for p in heads.main.parameters())
    assert sum(p.grad.abs().sum() for p in heads.auxiliary.parameters()) > 0
    assert sum(p.grad.abs().sum() for p in decoder.parameters()) > 0
    assert bool(features.grad[:, :12].abs().sum() > 0) == (routing == "joint")


def test_goal_write_her_rebuilds_worker_history_with_each_network_parameters():
    cfg = SimpleNamespace(
        Hippo_R=2,
        Hippo_L=3,
        Hippo_n_feature=3,
        hrl_controllable_graph=True,
        hrl_graph_memory="policy_buffer",
        hrl_target_timing="immediate",
        hrl_manager_mode="frontier_direct",
        dg_context_feedback="none",
        dg_orthogonal_recruitment=False,
        DG_BN_intercept=1.0,
    )
    online = GoalConditionedDGCore(cfg, 8)
    target = GoalConditionedDGCore(cfg, 8)
    with torch.no_grad():
        online.dg_goal_modulation[1, 3:] = 1.0
        target.dg_goal_modulation[1, 3:] = 2.0
    state_o = torch.zeros(1, online.total_state_size)
    state_t = torch.zeros_like(state_o)
    pre = torch.tensor([[2.0, 1.5, 2.5]], requires_grad=True)
    head = torch.cat((torch.relu(pre - 1.0), torch.zeros(1, 2), pre), -1)
    virtual_goal = torch.tensor([[0.0, 1.0, 0.0]])
    for _ in range(4):
        out_o, state_o = online(torch.cat((head, virtual_goal), -1), state_o)
        with torch.no_grad():
            out_t, state_t = target(torch.cat((head, virtual_goal), -1), state_t)
        torch.testing.assert_close(out_o[:, : online.core_output_size], out_t[:, : target.core_output_size])
    assert not torch.equal(online.worker_view(out_o), target.worker_view(out_t))
    online.worker_view(out_o).sum().backward()
    assert online.dg_goal_modulation.grad.abs().sum() > 0
    assert pre.grad is None or not pre.grad.any()
    assert target.dg_goal_modulation.grad is None


def test_native_sf_q_distribution_has_finite_entropy_and_masks_value():
    import gymnasium as gym

    from sample_factory.utils.attr_dict import AttrDict
    from sf_working_directories.IntrMotiv.dmlab.custom_actor_critic import IntrMotivActorCriticSharedWeights

    actor = IntrMotivActorCriticSharedWeights.__new__(IntrMotivActorCriticSharedWeights)
    torch.nn.Module.__init__(actor)
    actor.cfg = AttrDict(controller_epsilon=0.0)
    actor.action_space = gym.spaces.Discrete(3)
    actor.decoder = torch.nn.Identity()
    actor.controller_q = torch.nn.Identity()
    q = torch.tensor([[100.0, -4.0, -2.0]])
    out = actor._forward_q_tail(q, False, True, torch.tensor([[False, True, True]]))
    assert out["values"].item() == -2.0
    assert out["actions"].item() == 2
    assert torch.isfinite(actor.action_distribution().entropy()).all()
    assert torch.equal(actor.action_distribution().probs, torch.tensor([[0.0, 0.0, 1.0]]))
