from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sf_working_directories.IntrMotiv.dmlab.controller_snapshot import ControllerSnapshot
from sf_working_directories.IntrMotiv.dmlab.custom_actor_critic import TargetFiLMDecoder
from sf_working_directories.IntrMotiv.dmlab.custom_core import SimpleSequenceWithBypassCore
from sf_working_directories.IntrMotiv.dmlab.custom_encoder import DGProjection_batchnorm_relu
from sf_working_directories.IntrMotiv.dmlab.custom_learner import (
    DistanceLearnerReward,
    classify_task_general_model_key,
    task_general_transfer_metadata,
)
from sf_working_directories.IntrMotiv.dmlab.goal_conditioned_dg import GoalConditionedDGCore


def _core_config():
    return SimpleNamespace(
        Hippo_R=2,
        Hippo_L=3,
        Hippo_n_feature=3,
        hrl_controllable_graph=False,
        fixed_task_conditioning=True,
        hrl_manager_mode="visit_direct",
        dg_context_feedback="none",
        dg_orthogonal_recruitment=False,
    )


def test_fixed_task_condition_is_trainable_graph_free_and_packed_consistent():
    core = SimpleSequenceWithBypassCore(_core_config(), 5)
    assert core.policy_graph is None
    assert core.total_state_size == 14
    assert core.get_out_size() == 17
    initial = torch.zeros(2, core.total_state_size)
    sequence = torch.randn(4, 2, 5)

    state = initial.clone()
    outputs = []
    for step in sequence:
        output, state = core(step, state)
        outputs.append(output)
    assert torch.allclose(torch.stack(outputs)[:, :, -3:], torch.full((4, 2, 3), 1 / 3))

    packed = torch.nn.utils.rnn.pack_padded_sequence(sequence, [4, 4], enforce_sorted=False)
    packed_output, packed_state = core(packed, initial)
    unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output)
    assert torch.allclose(unpacked, torch.stack(outputs))
    assert torch.allclose(packed_state, state)
    unpacked[:, :, -3:].sum().backward()
    assert torch.allclose(core.fixed_task_target.grad, torch.full((3,), 8.0))


def test_flat_goal_mixture_is_normalized_trainable_and_source_aligned():
    cfg = _core_config()
    cfg.fixed_task_goal_mixture = True
    cfg.fixed_task_target_id = 1
    core = SimpleSequenceWithBypassCore(cfg, 5)
    mixture = core.fixed_task_condition()
    assert torch.isclose(mixture.sum(), torch.tensor(1.0))
    assert mixture[1] > 0.99
    out, _ = core(torch.ones(1, 5), torch.zeros(1, core.total_state_size))
    assert torch.allclose(out[0, -3:], mixture)
    out[0, -3].backward()
    assert core.fixed_task_target.grad.abs().sum() > 0


def test_uniform_flat_goal_mixture_has_no_nominated_address():
    cfg = _core_config()
    cfg.fixed_task_goal_mixture = True
    cfg.fixed_task_target_id = 1
    cfg.fixed_task_goal_init = "uniform"
    core = SimpleSequenceWithBypassCore(cfg, 5)
    mixture = core.fixed_task_condition()
    assert torch.allclose(mixture, torch.full((3,), 1 / 3))
    assert core.fixed_task_target.requires_grad


def test_uninformed_jitter_breaks_symmetry_and_matches_seed():
    cfg = _core_config()
    cfg.seed = 42
    cfg.fixed_task_goal_mixture = True
    cfg.fixed_task_goal_init = "uniform_jitter"
    cfg.fixed_task_target_id = 1
    first = SimpleSequenceWithBypassCore(cfg, 5).fixed_task_condition()
    second = SimpleSequenceWithBypassCore(cfg, 5).fixed_task_condition()
    assert torch.allclose(first, second)
    assert torch.isclose(first.sum(), torch.tensor(1.0))
    assert torch.allclose(first, torch.full((3,), 1 / 3), atol=0.01)
    assert not torch.allclose(first, torch.full((3,), 1 / 3), atol=1e-5)
    cfg.seed = 1234
    other = SimpleSequenceWithBypassCore(cfg, 5).fixed_task_condition()
    assert not torch.allclose(first, other)


def test_fresh_flat_goal_effects_are_distinct_and_reward_gradient_reaches_mixture():
    cfg = _core_config()
    cfg.seed = 42
    cfg.fixed_task_goal_mixture = True
    cfg.fixed_task_goal_init = "uniform_jitter"
    core = SimpleSequenceWithBypassCore(cfg, 5)
    decoder = TargetFiLMDecoder(core)
    again = TargetFiLMDecoder(core)
    assert torch.equal(decoder.target_modulation, again.target_modulation)
    assert decoder.target_modulation.norm() > 0
    assert not torch.equal(decoder.target_modulation[0], decoder.target_modulation[1])
    output, _ = core(torch.ones(1, 5), torch.zeros(1, core.total_state_size))
    decoder(output).square().sum().backward()
    assert core.fixed_task_target.grad is not None
    assert core.fixed_task_target.grad.abs().sum() > 0

    cfg.fixed_task_goal_mixture = False
    source_core = SimpleSequenceWithBypassCore(cfg, 5)
    assert torch.count_nonzero(TargetFiLMDecoder(source_core).target_modulation) == 0


def test_goal_write_flat_worker_allows_reward_gradient_into_dg():
    cfg = _core_config()
    cfg.seed = 42
    cfg.fixed_task_goal_mixture = True
    cfg.fixed_task_goal_init = "uniform_jitter"
    cfg.hrl_graph_memory = "policy_buffer"
    cfg.hrl_target_timing = "immediate"
    cfg.ppo_dg_gradient = "joint"
    cfg.DG_BN_intercept = 1.0
    core = GoalConditionedDGCore(cfg, 8)
    pre = torch.full((1, 3), 2.0, requires_grad=True)
    head = torch.cat((torch.ones(1, 3), torch.zeros(1, 2), pre), -1)
    out, _ = core(head, torch.zeros(1, core.total_state_size))
    TargetFiLMDecoder(core)(core.worker_view(out)).square().sum().backward()
    assert pre.grad is not None and pre.grad.abs().sum() > 0
    assert core.fixed_task_target.grad is not None
    assert core.fixed_task_target.grad.abs().sum() > 0


def test_goal_write_replay_uses_current_flat_mixture_instead_of_stored_goal():
    cfg = _core_config()
    cfg.seed = 42
    cfg.fixed_task_goal_mixture = True
    cfg.fixed_task_goal_init = "uniform_jitter"
    cfg.hrl_graph_memory = "policy_buffer"
    cfg.hrl_target_timing = "immediate"
    cfg.ppo_dg_gradient = "joint"
    cfg.DG_BN_intercept = 1.0
    core = GoalConditionedDGCore(cfg, 8)
    decoder = TargetFiLMDecoder(core)
    pre = torch.full((4, 2, 3), 2.0, requires_grad=True)
    stored_goal = torch.zeros(4, 2, 3)
    stored_goal[..., 1] = 1.0
    head = torch.cat((torch.ones(4, 2, 3), torch.zeros(4, 2, 2), pre, stored_goal), -1)
    packed = torch.nn.utils.rnn.pack_padded_sequence(head, [4, 4], enforce_sorted=False)
    output, _ = core(packed, torch.zeros(2, core.total_state_size))
    current = core.fixed_task_condition()
    assert torch.allclose(
        output.data[:, core.target_condition_start : core.target_condition_start + 3],
        current.expand(8, -1),
    )
    decoder(core.worker_view(output.data)).square().sum().backward()
    assert core.fixed_task_target.grad is not None
    assert core.fixed_task_target.grad.abs().sum() > 0
    assert pre.grad is not None and pre.grad.abs().sum() > 0


class _Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False)
        self.batchnorm1d = nn.BatchNorm1d(3, affine=False)
        self.register_buffer("feature_running_mean", torch.zeros(4))
        self.register_buffer("feature_num_batches_tracked", torch.zeros((), dtype=torch.long))
        self.register_buffer("recruitment_count", torch.zeros((), dtype=torch.long))


class _Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.DG_projection = _Projection()
        self.core = nn.Module()
        self.core.state_readout = nn.Linear(3, 2, bias=False)
        self.core.fixed_task_target = nn.Parameter(torch.zeros(3))
        self.decoder = nn.Linear(3, 3)
        self.action_parameterization = nn.Linear(3, 2)
        self.controller_q = nn.Linear(3, 2)
        self.critic_linear = nn.Linear(3, 1)
        self.core.policy_graph = nn.Module()
        self.core.policy_graph.register_buffer("node_visits", torch.zeros(3))
        self.core.policy_graph.register_buffer("edge_confidence", torch.zeros(3, 3))
        self.core.policy_graph.register_buffer("reward_goal_value", torch.zeros(3, 3))
        self.core.policy_graph.register_buffer("reward_goal_count", torch.zeros(3, 3))


def _learner(scope):
    learner = object.__new__(DistanceLearnerReward)
    learner.actor_critic = _Actor()
    learner.cfg = SimpleNamespace(transfer_model_path="source.pth", transfer_scope=scope)
    source = {key: torch.full_like(value, 7) for key, value in learner.actor_critic.state_dict().items()}
    learner.load_checkpoint = lambda paths, device: {"model": source}
    learner.device = torch.device("cpu")
    return learner


def _task_general_learner():
    learner = _learner("task_general")
    learner.cfg = SimpleNamespace(
        transfer_model_path="source.pth",
        transfer_scope="task_general",
        dmlab_navigation_action_set=True,
        dmlab_reduced_action_set=False,
        dmlab_extended_action_set=False,
        Hippo_L=64,
        Hippo_R=8,
        Hippo_n_feature=64,
        ca3_state_readout_dim=16,
        ca3_state_readout_horizon=32,
        ca3_state_readout_mode="worker",
        ca3_worker_goal_mode="state_readout",
        controller_her=True,
        controller_learning="ddqn",
        hrl_controllable_graph=True,
        hrl_goal_conditioning="target_id_film",
        hrl_manager_mode="frontier_waypoint",
    )
    source = {key: torch.full_like(value, 7) for key, value in learner.actor_critic.state_dict().items()}
    learner.load_checkpoint = lambda paths, device: {
        "model": source,
        "task_general_transfer": task_general_transfer_metadata(learner.cfg),
        "optimizer": {"must_not_load": True},
        "env_steps": 123,
    }
    return learner


def test_dg_transfer_loads_only_dg_weights_and_buffers():
    learner = _learner("dg")
    before_decoder = learner.actor_critic.decoder.weight.detach().clone()
    before_critic = learner.actor_critic.critic_linear.weight.detach().clone()
    learner._initialize_transfer_weights()
    state = learner.actor_critic.state_dict()
    assert torch.all(state["encoder.DG_projection.linear.weight"] == 7)
    assert torch.all(state["encoder.DG_projection.batchnorm1d.running_mean"] == 7)
    assert torch.all(state["encoder.DG_projection.feature_running_mean"] == 0)
    assert state["encoder.DG_projection.recruitment_count"] == 0
    assert torch.equal(learner.actor_critic.decoder.weight, before_decoder)
    assert torch.equal(learner.actor_critic.critic_linear.weight, before_critic)


def test_policy_transfer_keeps_critic_fresh_and_freeze_disables_dg_gradients():
    learner = _learner("policy")
    before_critic = learner.actor_critic.critic_linear.weight.detach().clone()
    learner._initialize_transfer_weights()
    assert torch.all(learner.actor_critic.decoder.weight == 7)
    assert torch.all(learner.actor_critic.action_parameterization.weight == 7)
    assert torch.equal(learner.actor_critic.critic_linear.weight, before_critic)
    learner.cfg.transfer_freeze_dg = True
    learner._apply_transfer_freeze()
    assert not any(p.requires_grad for p in learner.actor_critic.encoder.DG_projection.parameters())


def test_task_general_transfer_has_complete_inventory_and_resets_task_specific_state():
    learner = _task_general_learner()
    before_critic = learner.actor_critic.critic_linear.weight.detach().clone()
    before_binding = learner.actor_critic.core.fixed_task_target.detach().clone()
    learner._initialize_transfer_weights()
    state = learner.actor_critic.state_dict()
    assert torch.all(state["encoder.DG_projection.linear.weight"] == 7)
    assert torch.all(state["core.state_readout.weight"] == 7)
    assert torch.all(state["decoder.weight"] == 7)
    assert torch.all(state["action_parameterization.weight"] == 7)
    assert torch.all(state["controller_q.weight"] == 7)
    assert torch.equal(learner.actor_critic.critic_linear.weight, before_critic)
    assert torch.equal(learner.actor_critic.core.fixed_task_target, before_binding)
    inventory = learner.task_general_transfer_inventory
    assert set(inventory["transferred"]) | set(inventory["reset"]) == set(state)
    assert set(inventory["transferred"]).isdisjoint(inventory["reset"])

    target = ControllerSnapshot(_Actor, learner.actor_critic, 0)
    assert torch.equal(target.model.controller_q.weight, learner.actor_critic.controller_q.weight)


def test_task_general_transfer_rejects_action_or_structure_mismatch():
    learner = _task_general_learner()
    checkpoint = learner.load_checkpoint(None, None)
    checkpoint["task_general_transfer"] = dict(checkpoint["task_general_transfer"])
    checkpoint["task_general_transfer"]["action_vectors"] = [[0] * 7] * 8
    learner.load_checkpoint = lambda paths, device: checkpoint
    with pytest.raises(RuntimeError, match="interface mismatch"):
        learner._initialize_transfer_weights()


def test_task_general_inventory_rejects_new_unclassified_module():
    with pytest.raises(RuntimeError, match="Unclassified"):
        classify_task_general_model_key("new_reward_adapter.weight")


def test_fresh_random_dg_can_be_frozen_without_loading_a_source():
    learner = _learner("none")
    learner.cfg.transfer_model_path = None
    learner.cfg.transfer_freeze_dg = True
    initial = learner.actor_critic.encoder.DG_projection.linear.weight.detach().clone()
    learner._initialize_transfer_weights()
    learner._apply_transfer_freeze()
    projection = learner.actor_critic.encoder.DG_projection
    assert torch.equal(projection.linear.weight, initial)
    assert not any(parameter.requires_grad for parameter in projection.parameters())
    assert projection.freeze_running_stats


def test_graph_transfer_preserves_navigation_evidence_but_resets_reward_values():
    learner = _learner("policy")
    learner.cfg.transfer_graph = True
    learner._initialize_transfer_weights()
    graph = learner.actor_critic.core.policy_graph
    assert torch.all(graph.node_visits == 7)
    assert torch.all(graph.edge_confidence == 7)
    assert torch.count_nonzero(graph.reward_goal_value) == 0
    assert torch.count_nonzero(graph.reward_goal_count) == 0


def test_graph_transfer_with_trainable_dg_keeps_source_evidence_and_fresh_reward_values():
    learner = _learner("policy")
    learner.cfg.transfer_graph = True
    learner.cfg.transfer_freeze_dg = False
    learner._initialize_transfer_weights()
    learner._apply_transfer_freeze()
    graph = learner.actor_critic.core.policy_graph
    assert torch.all(graph.node_visits == 7)
    assert torch.all(graph.edge_confidence == 7)
    assert torch.count_nonzero(graph.reward_goal_value) == 0
    assert torch.count_nonzero(graph.reward_goal_count) == 0
    assert any(p.requires_grad for p in learner.actor_critic.encoder.DG_projection.parameters())


def test_frozen_legacy_batchnorm_uses_checkpoint_statistics_without_drift():
    projection = DGProjection_batchnorm_relu(4, 3, intercept=0.0, batchnorm_semantics="legacy_batch")
    projection.train()
    projection.freeze_running_stats = True
    before_mean = projection.batchnorm1d.running_mean.clone()
    before_var = projection.batchnorm1d.running_var.clone()
    output = projection(torch.randn(32, 4) + 10)
    assert torch.isfinite(output).all()
    assert torch.equal(projection.batchnorm1d.running_mean, before_mean)
    assert torch.equal(projection.batchnorm1d.running_var, before_var)
    assert not projection.last_running_stats_updated


def test_frozen_random_dg_calibrates_once_then_keeps_moments_fixed():
    projection = DGProjection_batchnorm_relu(4, 3, intercept=2.43)
    projection.train()
    projection.freeze_running_stats = True
    projection.calibrate_frozen_on_first_batch = True
    with projection.running_stats_update(True):
        first = projection(torch.randn(4096, 4) + 10)
    mean = projection.batchnorm1d.running_mean.clone()
    var = projection.batchnorm1d.running_var.clone()
    assert first.gt(0).any()
    assert int(projection.batchnorm1d.num_batches_tracked) == 1
    with projection.running_stats_update(True):
        projection(torch.randn(128, 4) - 10)
    assert torch.equal(projection.batchnorm1d.running_mean, mean)
    assert torch.equal(projection.batchnorm1d.running_var, var)
    assert not projection.last_running_stats_updated


def test_single_value_ppo_uses_external_reward_when_explicitly_requested():
    learner = object.__new__(DistanceLearnerReward)
    worker = torch.tensor([[0.4, 0.2, 0.0]])
    external = torch.tensor([[0.0, 10.0, 0.0]])
    for source, expected in (("external", external), ("internal", worker), ("legacy", worker)):
        learner.cfg = SimpleNamespace(advantage_reward_source=source)
        batch = {"rewards": worker.clone(), "rewards_worker": worker.clone(), "rewards_external": external.clone()}
        learner._select_ppo_reward(batch)
        assert torch.equal(batch["rewards"], expected)
        assert torch.equal(batch["rewards_worker"], worker)
