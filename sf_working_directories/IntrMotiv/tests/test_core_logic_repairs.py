from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from sf_working_directories.IntrMotiv.dmlab.custom_actor_critic import IntrMotivActorCriticSharedWeights
from sf_working_directories.IntrMotiv.dmlab.custom_learner import (
    dg_unused_batch_recruitment_loss,
    normalize_dg_projection_rows,
)
from sf_working_directories.IntrMotiv.dmlab.hrl_controllable_graph import source_from_trace
from sf_working_directories.IntrMotiv.dmlab.train_hipposlam import maybe_overwrite_rnn_size
from sf_working_directories.IntrMotiv.dmlab.topological_frontier import (
    GEOMETRY_POLICY_SIZE,
    topological_state_size,
)
from sf_working_directories.IntrMotiv.dmlab.hrl_controllable_graph import hrl_option_state_size


def test_marked_pretrained_layer_is_not_reinitialized():
    actor = IntrMotivActorCriticSharedWeights.__new__(IntrMotivActorCriticSharedWeights)
    nn.Module.__init__(actor)
    actor.cfg = SimpleNamespace(policy_init_gain=1.0, policy_initialization="orthogonal")
    layer = nn.Conv2d(3, 4, 3)
    layer._intrmotiv_preserve_initialization = True
    before_weight = layer.weight.detach().clone()
    before_bias = layer.bias.detach().clone()

    actor.initialize_weights(layer)

    assert torch.equal(layer.weight, before_weight)
    assert torch.equal(layer.bias, before_bias)


def test_unused_batch_recruitment_has_below_threshold_gradient_and_respects_valids():
    logits = torch.full((2, 3), -2.0, requires_grad=True)
    current = torch.zeros_like(logits)
    prior = torch.tensor([[True, False, False], [False, True, False]])
    valids = torch.tensor([True, False])

    loss, unused = dg_unused_batch_recruitment_loss(logits, current, prior, valids, 0.0, 0.5)
    loss.backward()

    assert torch.equal(unused, torch.tensor([False, True, True]))
    assert logits.grad[0, 0].item() == 0.0
    assert torch.all(logits.grad[0, 1:] < 0)
    assert torch.all(logits.grad[1] == 0)


def test_dg_rows_are_unit_norm_after_projection():
    linear = nn.Linear(5, 3, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[3.0, 4.0, 0.0, 0.0, 0.0]]).repeat(3, 1))

    normalize_dg_projection_rows(linear)

    assert torch.allclose(linear.weight.norm(dim=1), torch.ones(3))


def test_trace_source_is_most_recent_not_largest_accumulated_and_empty_is_unknown():
    trace = torch.zeros(2, 3, 6)
    trace[0, 0, 3] = 20.0
    trace[0, 1, 1] = 1.0

    source = source_from_trace(trace)

    assert torch.equal(source, torch.tensor([1, -1]))


def test_unsafe_legacy_objectives_fail_fast():
    try:
        maybe_overwrite_rnn_size(SimpleNamespace(extra_decoder_loss=True))
    except ValueError as error:
        assert "wrong sign" in str(error)
    else:
        raise AssertionError("extra_decoder_loss must fail before training")

    try:
        maybe_overwrite_rnn_size(
            SimpleNamespace(extra_decoder_loss=False, with_pbt=True, pbt_target_objective="distance_metric")
        )
    except ValueError as error:
        assert "not a valid PBT objective" in str(error)
    else:
        raise AssertionError("distance_metric must not be accepted as a PBT objective")


def test_immediate_manager_descriptor_is_in_allocated_and_persistent_rnn_size():
    n = 16
    cfg = SimpleNamespace(
        extra_decoder_loss=False,
        with_pbt=False,
        hrl_manager_mode="control_graph",
        dg_orthogonal_recruitment=False,
        hrl_controllable_graph=True,
        hrl_graph_memory="policy_buffer",
        core_name="BypassSS",
        hrl_action_path_integration=False,
        hrl_motion_policy_input=False,
        hrl_landmark_geometry="none",
        hrl_edge_exploration=True,
        hrl_goal_conditioning="target_trace",
        hrl_target_timing="immediate",
        hrl_exploration_policy="separate",
        hrl_behavior_mode_condition=True,
        hrl_empirical_her=False,
        cli_args={"rnn_size": 0},
        Hippo_R=8,
        Hippo_L=64,
        Hippo_n_feature=n,
    )
    # AttrDict-style dotted lookup used by production resolves to the nested
    # CLI value; emulate it for this focused sizing test.
    class Config(SimpleNamespace):
        def __getattr__(self, name):
            if name == "cli_args.rnn_size":
                return self.cli_args["rnn_size"]
            raise AttributeError(name)

    cfg = Config(**vars(cfg))
    maybe_overwrite_rnn_size(cfg)
    descriptor_size = 1 + GEOMETRY_POLICY_SIZE + 1
    expected = (
        n * (cfg.Hippo_R + cfg.Hippo_L - 1)
        + 13
        + hrl_option_state_size(n)
        + topological_state_size(n)
        + descriptor_size
    )
    assert cfg.rnn_size == expected
    assert cfg.rnn_persistent_state_size == descriptor_size
