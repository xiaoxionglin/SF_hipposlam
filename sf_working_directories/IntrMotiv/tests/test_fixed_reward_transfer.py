from types import SimpleNamespace

import torch
from torch import nn

from sf_working_directories.IntrMotiv.dmlab.custom_core import SimpleSequenceWithBypassCore
from sf_working_directories.IntrMotiv.dmlab.custom_encoder import DGProjection_batchnorm_relu
from sf_working_directories.IntrMotiv.dmlab.custom_learner import DistanceLearnerReward
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


def test_goal_write_flat_worker_allows_reward_gradient_into_dg():
    cfg = _core_config()
    cfg.fixed_task_goal_mixture = True
    cfg.fixed_task_target_id = 1
    cfg.hrl_graph_memory = "policy_buffer"
    cfg.hrl_target_timing = "immediate"
    cfg.ppo_dg_gradient = "joint"
    cfg.DG_BN_intercept = 1.0
    core = GoalConditionedDGCore(cfg, 8)
    pre = torch.full((1, 3), 2.0, requires_grad=True)
    head = torch.cat((torch.ones(1, 3), torch.zeros(1, 2), pre), -1)
    out, _ = core(head, torch.zeros(1, core.total_state_size))
    core.worker_view(out).sum().backward()
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
        self.decoder = nn.Linear(3, 3)
        self.action_parameterization = nn.Linear(3, 2)
        self.critic_linear = nn.Linear(3, 1)


def _learner(scope):
    learner = object.__new__(DistanceLearnerReward)
    learner.actor_critic = _Actor()
    learner.cfg = SimpleNamespace(transfer_model_path="source.pth", transfer_scope=scope)
    source = {key: torch.full_like(value, 7) for key, value in learner.actor_critic.state_dict().items()}
    learner.load_checkpoint = lambda paths, device: {"model": source}
    learner.device = torch.device("cpu")
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
