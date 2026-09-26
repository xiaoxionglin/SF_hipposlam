from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sf_working_directories.IntrMotiv.dmlab.controller_history import ControllerHistory, reconstruct_controller_history
from sf_working_directories.IntrMotiv.dmlab.custom_actor_critic import TargetFiLMDecoder, controller_core_view
from sf_working_directories.IntrMotiv.dmlab.custom_core import SimpleSequenceWithBypassCore
from sf_working_directories.IntrMotiv.dmlab.custom_encoder import DGProjection_batchnorm_relu
from sf_working_directories.IntrMotiv.dmlab.goal_conditioned_dg import GoalConditionedDGCore


class SmallHistoryModel(nn.Module):
    """Small observation adapter, original DG/core/FiLM implementations."""

    def __init__(self, write=True):
        super().__init__()
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
            DG_BN_intercept=-2.0,
        )
        self.dg = DGProjection_batchnorm_relu(4, 3, intercept=-2.0)
        self.core = GoalConditionedDGCore(cfg, 8) if write else SimpleSequenceWithBypassCore(cfg, 5)
        self.decoder = TargetFiLMDecoder(self.core, 8)
        self.write = write

    def device_for_input_tensor(self, key):
        return torch.device("cpu")

    def type_for_input_tensor(self, key):
        return torch.float32

    def normalize_obs(self, obs):
        return obs

    def forward_head(self, obs):
        pre = self.dg.preactivation(obs["obs"])
        head = torch.cat((torch.relu(pre + 2), obs["bypass"]), -1)
        return torch.cat((head, pre.detach()), -1) if self.write else head

    def forward_core(self, head, state):
        return self.core(head, state)

    def controller_hidden(self, out):
        view = (
            self.core.worker_view(out) if self.write else controller_core_view(out, self.core.core_output_size, "stop")
        )
        return self.decoder(view)


def history(model, steps=7, burn=4, start=False):
    return ControllerHistory(
        {"obs": torch.randn(steps, 4), "bypass": torch.randn(steps, 2)},
        torch.arange(10, 10 + steps),
        torch.zeros(1, model.core.total_state_size),
        torch.tensor([[0.0, 1.0, 0.0]]).repeat(steps, 1),
        start,
        burn,
    )


@pytest.mark.parametrize("write", [False, True])
def test_reconstructed_canonical_trace_matches_physical_forward_after_washout(write):
    model = SmallHistoryModel(write).eval()
    h = history(model)
    state = h.initial_context.clone()
    outputs = []
    # Arbitrary stale memory is deliberately excluded by finite washout.
    state[:, : model.core.core_output_size] = 17.0
    with torch.no_grad():
        heads = model.forward_head(h.observations)
        for t in range(7):
            head = heads[t : t + 1]
            if write:
                head = torch.cat((head, h.conditions[t : t + 1]), -1)
            out, state = model.forward_core(head, state)
            outputs.append(out)
        result = reconstruct_controller_history(model, h)
    expected = torch.cat(outputs)[4:, : model.core.core_output_size]
    torch.testing.assert_close(result["core_outputs"][:, : model.core.core_output_size], expected)


def test_virtual_goal_rebuild_changes_worker_not_canonical_and_retains_history_gradient():
    model = SmallHistoryModel().eval()
    with torch.no_grad():
        model.core.dg_goal_modulation[1, 3:] = 1.0
    h = history(model, steps=4, burn=0, start=True)
    a = reconstruct_controller_history(model, h)
    b = reconstruct_controller_history(model, replace(h, conditions=torch.tensor([[1.0, 0.0, 0.0]]).repeat(4, 1)))
    width = model.core.core_output_size
    torch.testing.assert_close(a["core_outputs"][:, :width], b["core_outputs"][:, :width])
    assert not torch.equal(
        a["core_outputs"][:, model.core.total_output_size :], b["core_outputs"][:, model.core.total_output_size :]
    )
    # Oldest retained worker slot at the final decision depends on earlier writes.
    oldest = a["core_outputs"][-1, model.core.total_output_size :].reshape(3, 4)[:, -1].sum()
    oldest.backward()
    assert model.core.dg_goal_modulation.grad.abs().sum() > 0
    assert model.dg.linear.weight.grad is None or not model.dg.linear.weight.grad.any()


def test_history_rejects_gaps_and_incomplete_nonreset_washout():
    model = SmallHistoryModel().eval()
    h = history(model)
    with pytest.raises(ValueError, match="gap"):
        reconstruct_controller_history(model, replace(h, decision_ids=torch.tensor([1, 2, 3, 8, 9, 10, 11])))
    with pytest.raises(ValueError, match="washout"):
        reconstruct_controller_history(model, replace(h, burn_in=3))
    # At a real episode start, no historical trace is missing.
    assert len(reconstruct_controller_history(model, replace(h, episode_start=True, burn_in=0))["hidden"]) == 7


@pytest.mark.parametrize("width,length,steps", [(1, 4, 9), (2, 4, 9), (8, 71, 83)])
def test_finite_convolution_matches_original_additive_memory_and_gradients(width, length, steps):
    from types import SimpleNamespace

    from sf_working_directories.IntrMotiv.dmlab.ca3_memory import finite_shift_history
    from sf_working_directories.IntrMotiv.dmlab.goal_conditioned_dg import GoalConditionedDGCore

    torch.manual_seed(919)
    activity = torch.rand(steps, 2, 3)
    activity[activity < 0.7] = 0
    a = activity.clone().requires_grad_()
    b = activity.clone().requires_grad_()
    memory = torch.zeros(2, 3, length)
    expected = []
    core = SimpleNamespace(R=width, expanded_length=length)
    for value in a:
        memory = GoalConditionedDGCore.advance_worker(core, memory, value)
        expected.append(memory)
    expected = torch.stack(expected)
    actual = finite_shift_history(b, width, length)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual == 0, expected == 0)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(a.grad, b.grad)
