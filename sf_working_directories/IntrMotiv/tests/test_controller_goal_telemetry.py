from types import SimpleNamespace

import torch

from sample_factory.algo.utils.misc import LEARNER_ENV_STEPS, TRAIN_STATS
from sf_working_directories.IntrMotiv.dmlab.custom_learner import DistanceLearnerReward
from sf_working_directories.IntrMotiv.dmlab.reward_summaries import write_intrmotiv_summaries


def test_goal_write_dashboard_reports_actual_controller_gradient_without_mutation():
    learner = object.__new__(DistanceLearnerReward)
    modulation = torch.nn.Parameter(torch.ones(3, 2))
    (modulation * torch.tensor([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]])).sum().backward()
    learner.actor_critic = SimpleNamespace(core=SimpleNamespace(dg_goal_modulation=modulation))
    learner.cfg = SimpleNamespace(Hippo_n_feature=3)
    before = modulation.detach().clone()
    grad = modulation.grad.clone()
    stats = {}
    learner._record_goal_modulation_summaries(stats)
    assert stats["dg_goal_modulation_gradient_norm"] == grad.norm()
    assert stats["dg_goal_gradient_001"] == 0 and stats["dg_goal_gradient_002"] == 5
    torch.testing.assert_close(modulation, before)
    torch.testing.assert_close(modulation.grad, grad)
    scalars = []
    runner = SimpleNamespace(
        writers={0: SimpleNamespace(add_scalar=lambda *args: scalars.append(args))}, env_steps={0: 0}
    )
    msg = {LEARNER_ENV_STEPS: 100, TRAIN_STATS: stats}
    write_intrmotiv_summaries(runner, msg, 0)
    # Original unmapped metrics retain SF's train/ namespace.
    assert msg[TRAIN_STATS]["dg_goal_modulation_gradient_norm"] > 0


def test_shared_goal_diagnostics_measure_readout_sensitivity_without_gradients():
    learner = object.__new__(DistanceLearnerReward)
    goals = torch.eye(2)
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    calls = []

    def tail(values, **kwargs):
        calls.append((torch.is_grad_enabled(), kwargs))
        return {"action_logits": values, "values": values[:, 0]}

    learner.actor_critic = SimpleNamespace(forward_tail=tail)
    learner._uses_policy_graph = lambda: True
    learner._behavior_targets_from_states = lambda states: goals
    learner._with_worker_target = lambda core, target: core + target
    outputs = SimpleNamespace(
        core_outputs=features, result={"action_logits": features + goals, "values": (features + goals)[:, 0]}
    )
    stats = {}
    learner._record_goal_condition_diagnostics(outputs, SimpleNamespace(rnn_states=None), stats)
    assert stats["goal_condition_target_valid_fraction"] == 1
    assert stats["goal_condition_action_sensitivity"] == 1
    assert stats["goal_condition_action_probability_tv"] > 0
    assert stats["goal_condition_value_span"] == 1
    assert calls == [(False, {"values_only": False, "sample_actions": False})]
    assert all(not value.requires_grad for value in stats.values())
