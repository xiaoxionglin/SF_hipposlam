from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import torch
from torch import nn
from sample_factory.utils.attr_dict import AttrDict

from sf_working_directories.IntrMotiv.dmlab.custom_actor_critic import (
    IntrMotivActorCriticSharedWeights,
    TargetFiLMDecoder,
    TargetRelativeDecoder,
)
from sf_working_directories.IntrMotiv.dmlab.custom_learner import DistanceLearnerReward


def test_target_relative_adapter_uses_32d_target_and_selected_trace_without_registering_core():
    core = SimpleNamespace(
        Hippo_n_feature=3,
        expanded_length=4,
        core_output_size=12,
        target_condition_start=12,
        get_out_size=lambda: 15,
    )
    decoder = TargetRelativeDecoder(core)
    assert decoder.target_embedding.embedding_dim == 32
    assert decoder.trace_projection[0].out_features == 32
    assert decoder.network[0].in_features == 15 + 64
    assert not any(key.startswith("core.") for key in decoder.state_dict())

    inputs = torch.zeros(2, 15)
    inputs[:, :12] = torch.arange(12, dtype=torch.float32)
    inputs[0, 12] = 1
    inputs[1, 13] = 1
    output = decoder(inputs)
    assert output.shape == (2, 128)
    assert not torch.equal(output[0], output[1])


def _film_core():
    return SimpleNamespace(
        Hippo_n_feature=3,
        target_condition_start=12,
        get_out_size=lambda: 20,
    )


def test_target_id_film_strips_one_hot_and_starts_as_exact_identity_modulation():
    torch.manual_seed(7)
    decoder = TargetFiLMDecoder(_film_core())
    assert decoder.state_layer[0].in_features == 17
    assert decoder.target_modulation.shape == (3, 256)
    assert torch.count_nonzero(decoder.target_modulation).item() == 0

    inputs = torch.randn(3, 20)
    inputs[:, 12:15] = 0
    inputs[0, 12] = 1
    inputs[1, 13] = 1
    # Row 2 has no target and therefore produces exactly zero modulation.
    inputs[1, :12] = inputs[0, :12]
    inputs[1, 15:] = inputs[0, 15:]
    inputs[2] = inputs[0]
    inputs[2, 12:15] = 0

    output = decoder(inputs)
    torch.testing.assert_close(output[0], output[1])
    torch.testing.assert_close(output[0], output[2])
    assert not any(key.startswith("core.") for key in decoder.state_dict())


def test_target_id_film_learns_target_specific_multiplicative_conditioning():
    torch.manual_seed(11)
    decoder = TargetFiLMDecoder(_film_core(), hidden_size=8)
    with torch.no_grad():
        decoder.state_layer[0].weight.zero_()
        decoder.state_layer[0].bias.fill_(1.0)
        decoder.output_layer[0].weight.copy_(torch.eye(8))
        decoder.output_layer[0].bias.zero_()
    inputs = torch.randn(2, 20)
    inputs[:, 12:15] = 0
    inputs[0, 12] = 1
    inputs[1, 13] = 1
    inputs[1, :12] = inputs[0, :12]
    inputs[1, 15:] = inputs[0, 15:]

    decoder(inputs)[0].sum().backward()
    gradient = decoder.target_modulation.grad
    assert torch.count_nonzero(gradient[0]).item() > 0
    assert torch.count_nonzero(gradient[1:]).item() == 0

    with torch.no_grad():
        decoder.target_modulation[1, :8].fill_(0.5)
    output = decoder(inputs)
    assert not torch.equal(output[0], output[1])


def test_target_id_film_checkpoint_round_trip_preserves_modulation():
    first = TargetFiLMDecoder(_film_core(), hidden_size=8)
    second = TargetFiLMDecoder(_film_core(), hidden_size=8)
    with torch.no_grad():
        first.target_modulation[2].copy_(torch.arange(16))
    second.load_state_dict(first.state_dict())
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
from sf_working_directories.IntrMotiv.dmlab.topological_frontier import (
    MODE_EXPLORE,
    MODE_VALIDATE,
    N_MANAGER_MODES,
)


class _ActionParameters(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, 2)

    def forward(self, x, action_mask=None):
        return self.linear(x), None


def _tiny_dual_branch_policy():
    policy = IntrMotivActorCriticSharedWeights.__new__(IntrMotivActorCriticSharedWeights)
    nn.Module.__init__(policy)
    policy.separate_exploration = True
    policy.core = SimpleNamespace(policy_base_output_size=4, mode_condition_start=5)
    policy.decoder = nn.Sequential(nn.Linear(10, 4), nn.ReLU())
    policy.exploration_decoder = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    policy.critic_linear = nn.Linear(4, 1)
    policy.exploration_critic_linear = nn.Linear(4, 1)
    policy.action_parameterization = _ActionParameters(4)
    policy.exploration_action_parameterization = _ActionParameters(4)
    policy.action_space = gym.spaces.Discrete(2)
    policy.last_action_distribution = None
    return policy


def _tiny_goal_specific_policy():
    policy = IntrMotivActorCriticSharedWeights.__new__(IntrMotivActorCriticSharedWeights)
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(Hippo_n_feature=3)
    policy.ppo_dg_gradient = "stop"
    policy.separate_goal_controllers = True
    policy.separate_exploration = False
    policy.core = SimpleNamespace(core_output_size=0, target_condition_start=2)
    policy.goal_decoders = nn.ModuleList(nn.Sequential(nn.Linear(5, 4), nn.ReLU()) for _ in range(3))
    policy.goal_critics = nn.ModuleList(nn.Linear(4, 1) for _ in range(3))
    policy.goal_action_parameterizations = nn.ModuleList(_ActionParameters(4) for _ in range(3))
    policy.action_space = gym.spaces.Discrete(2)
    policy.last_action_distribution = None
    return policy


def test_goal_specific_controller_routes_gradient_only_to_commanded_identity():
    torch.manual_seed(17)
    policy = _tiny_goal_specific_policy()
    core_output = torch.randn(2, 5)
    core_output[:, 2:] = 0
    core_output[:, 3] = 1
    result = policy.forward_tail(core_output, values_only=False, sample_actions=False)
    (result["values"].sum() + result["action_logits"].sum()).backward()
    assert _has_only_zero_or_missing_grad(policy.goal_decoders[0])
    assert _has_nonzero_grad(policy.goal_decoders[1])
    assert _has_only_zero_or_missing_grad(policy.goal_decoders[2])
    assert _has_only_zero_or_missing_grad(policy.goal_critics[0])
    assert _has_nonzero_grad(policy.goal_critics[1])
    assert _has_only_zero_or_missing_grad(policy.goal_critics[2])


def _zero_grad(policy):
    for parameter in policy.parameters():
        parameter.grad = None


def _has_nonzero_grad(module):
    return any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in module.parameters()
    )


def _has_only_zero_or_missing_grad(module):
    return all(
        parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0
        for parameter in module.parameters()
    )


def test_probe_updates_goal_branch_only_and_free_mode_updates_exploration_branch_only():
    torch.manual_seed(3)
    policy = _tiny_dual_branch_policy()
    core_output = torch.randn(2, 10)
    core_output[:, 5 : 5 + N_MANAGER_MODES] = 0
    core_output[0, 5 + MODE_VALIDATE] = 1
    core_output[1, 5 + MODE_EXPLORE] = 1

    result = policy.forward_tail(core_output, values_only=False, sample_actions=False)
    (result["action_logits"][0].sum() + result["values"][0]).backward()
    assert _has_nonzero_grad(policy.decoder)
    assert _has_nonzero_grad(policy.critic_linear)
    assert _has_nonzero_grad(policy.action_parameterization)
    assert _has_only_zero_or_missing_grad(policy.exploration_decoder)
    assert _has_only_zero_or_missing_grad(policy.exploration_critic_linear)
    assert _has_only_zero_or_missing_grad(policy.exploration_action_parameterization)

    _zero_grad(policy)
    result = policy.forward_tail(core_output, values_only=False, sample_actions=False)
    (result["action_logits"][1].sum() + result["values"][1]).backward()
    assert _has_nonzero_grad(policy.exploration_decoder)
    assert _has_nonzero_grad(policy.exploration_critic_linear)
    assert _has_nonzero_grad(policy.exploration_action_parameterization)
    assert _has_only_zero_or_missing_grad(policy.decoder)
    assert _has_only_zero_or_missing_grad(policy.critic_linear)
    assert _has_only_zero_or_missing_grad(policy.action_parameterization)


def test_dual_branch_checkpoint_round_trip_preserves_both_heads():
    first = _tiny_dual_branch_policy()
    second = _tiny_dual_branch_policy()
    second.load_state_dict(first.state_dict())
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])


def test_learner_teacher_forces_target_geometry_and_mode_from_behavior_descriptor():
    learner = object.__new__(DistanceLearnerReward)
    learner.cfg = SimpleNamespace(
        Hippo_n_feature=3,
        hrl_controllable_graph=True,
        hrl_graph_memory="policy_buffer",
        hrl_target_timing="immediate",
        hrl_behavior_mode_condition=True,
        hrl_landmark_geometry="se2",
    )
    learner.actor_critic = SimpleNamespace(
        core=SimpleNamespace(
            behavior_goal_state_size=6,
            target_condition_start=3,
            geometry_condition_start=6,
            mode_condition_start=10,
        )
    )
    behavior = torch.tensor(
        [
            [2.0, 0.25, -0.5, 0.0, 1.0, float(MODE_VALIDATE)],
            [0.0, 0.0, 0.0, 0.0, 0.0, float(MODE_EXPLORE)],
        ]
    )
    minibatch = AttrDict(rnn_states=behavior)
    recomputed = torch.randn(2, 15)
    replayed = learner._override_core_outputs_for_replay(recomputed, minibatch)

    assert torch.equal(replayed[0, 3:6], torch.tensor([0.0, 1.0, 0.0]))
    assert replayed[1, 3:6].eq(0).all()
    assert torch.equal(replayed[:, 6:10], behavior[:, 1:5])
    assert replayed[0, 10 + MODE_VALIDATE].item() == 1
    assert replayed[1, 10 + MODE_EXPLORE].item() == 1
    assert learner._last_behavior_replay_mismatch.item() == 0
