from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sample_factory.utils.attr_dict import AttrDict
from sf_working_directories.IntrMotiv.dmlab.custom_learner import DistanceLearnerReward


@pytest.mark.parametrize("extra", [False, True])
def test_fresh_auxiliary_objective_has_no_controller_dependency_and_runs_each_component_once(extra):
    calls = []

    def record(name, result):
        def operation(*args):
            calls.append(name)
            return result(*args)

        return operation

    head = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    zero = head.sum() * 0
    learner = SimpleNamespace(
        cfg=SimpleNamespace(
            Hippo_n_feature=2,
            Hippo_R=1,
            Hippo_L=1,
            extra_encoder_losses=extra,
            encoder_grad_coeff=2.0,
            dg_transition_prediction="none",
        ),
        timing=SimpleNamespace(add_time=lambda name: nullcontext()),
        actor_critic=SimpleNamespace(),
        _encoder_loss=record("credit", lambda x, r, m, *a: -(x * r * m).sum() / 2),
        _l1_loss=record("l1", lambda x, *a: x.sum() / 2),
        _extra_encoder_loss=record("extra", lambda *a: (zero + 1, zero + 2, zero + 3, zero)),
        _population_usage_loss=record("population", lambda x: (zero + 4, zero, zero, zero)),
        _anti_collapse_losses=record("anticollapse", lambda *a: (zero + 5, zero + 6, zero + 7, zero + 8, [zero] * 6)),
    )
    outputs = AttrDict(head_outputs=head, core_outputs=head, minibatch_size=2)
    mb = {
        "encoder_credit_activation_mask": torch.ones_like(head, dtype=torch.bool),
        "rewards_encoder": torch.ones_like(head),
        "rnn_states": torch.zeros(2, 2),
        "encoder_dominant_activation_mask": torch.ones_like(head, dtype=torch.bool),
        "encoder_non_dominant_activation_mask": torch.zeros_like(head, dtype=torch.bool),
        "rewards": torch.ones(2),
    }
    stats = {}
    loss = DistanceLearnerReward._calculate_fresh_encoder_loss(
        learner, outputs, mb, torch.ones(2, dtype=torch.bool), 0, 1, stats
    )
    expected = (-5 + (6 if extra else 5) + 4 + 5 + 6 + 7 + 8) * 2
    torch.testing.assert_close(loss, torch.tensor(float(expected)))
    loss.backward()
    torch.testing.assert_close(head.grad, torch.full_like(head, -1.0 if extra else 0.0))
    assert calls == (
        ["credit", "l1", "extra", "population", "anticollapse"]
        if extra
        else ["credit", "l1", "population", "anticollapse"]
    )
    if not extra:
        assert stats["encoder_penalty_loss"].item() == stats["encoder_reward_loss"].item() == 0
