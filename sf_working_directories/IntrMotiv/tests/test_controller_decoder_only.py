"""User-approved decoder-only goal conditioning, including relabel equivalence."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_controller_history import SmallHistoryModel, history

from sf_working_directories.IntrMotiv.dmlab.controller_history import reconstruct_controller_history
from sf_working_directories.IntrMotiv.dmlab.custom_actor_critic import TargetFiLMDecoder
from sf_working_directories.IntrMotiv.dmlab.custom_core import SimpleSequenceWithBypassCore
from sf_working_directories.IntrMotiv.dmlab.custom_encoder import DGProjection_batchnorm_relu


@pytest.mark.parametrize("manager,n", [("frontier_direct", 16), ("frontier_waypoint", 64)])
def test_virtual_goal_reuses_physical_memory_and_decoder_relabel_gradients(manager, n):
    torch.manual_seed(99)
    model = SmallHistoryModel(False)
    cfg = SimpleNamespace(
        Hippo_R=8,
        Hippo_L=64,
        Hippo_n_feature=n,
        hrl_controllable_graph=True,
        hrl_graph_memory="policy_buffer",
        hrl_target_timing="immediate",
        hrl_manager_mode=manager,
        dg_context_feedback="none",
        dg_orthogonal_recruitment=False,
    )
    model.dg = DGProjection_batchnorm_relu(4, n, intercept=-2.0)
    model.core = SimpleSequenceWithBypassCore(cfg, n + 2)
    model.decoder = TargetFiLMDecoder(model.core, 8)
    model.eval()
    with torch.no_grad():
        model.decoder.target_modulation.normal_(0, 0.2)
    h = history(model, steps=73, burn=71)
    a = torch.zeros(73, n)
    a[:, 0] = 1
    b = torch.zeros(73, n)
    b[:, 1] = 1
    h = replace(h, conditions=a)
    actual = reconstruct_controller_history(model, h)
    virtual = reconstruct_controller_history(model, replace(h, conditions=b))
    start = model.core.target_condition_start
    torch.testing.assert_close(
        actual["all_core_outputs"][:, :start], virtual["all_core_outputs"][:, :start], rtol=0, atol=0
    )
    # Changing the goal only at the decoder reproduces full virtual-history
    # evaluation, including controller gradients; no virtual memory rebuild.
    relabeled = actual["core_outputs"].clone()
    relabeled[:, start : start + n] = b[h.burn_in :]
    cheap = model.controller_hidden(relabeled)
    torch.testing.assert_close(cheap, virtual["hidden"], rtol=0, atol=0)
    assert not torch.equal(cheap, actual["hidden"])
    params = tuple(model.decoder.parameters())
    g1 = torch.autograd.grad(cheap.sum(), params, retain_graph=True)
    g2 = torch.autograd.grad(virtual["hidden"].sum(), params)
    for x, y in zip(g1, g2):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert not any("dg_goal_modulation" in key for key in model.state_dict())
