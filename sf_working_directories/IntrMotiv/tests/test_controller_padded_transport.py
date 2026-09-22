from copy import deepcopy

import pytest
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sf_working_directories.IntrMotiv.tests.test_controller_transition import RewardModel


@pytest.mark.parametrize("write", [False, True])
def test_padded_replay_matches_packed_values_states_and_gradients(write):
    model = RewardModel(write).eval()
    other = deepcopy(model)
    conditions = torch.zeros(3, 2, 3)
    conditions[:, :, 1] = 1
    lengths = [3, 2]
    observations = torch.eye(3).repeat_interleave(2, dim=0)

    def run(m, padded):
        head = m.forward_head({"obs": observations}).reshape(3, 2, -1)
        if write:
            head = torch.cat((head, conditions), -1)
        packet = pack_padded_sequence(head, lengths, enforce_sorted=False)
        out, state = m.core(
            packet, torch.zeros(2, m.core.total_state_size), replay_conditions=conditions, replay_padded=padded
        )
        if not padded:
            out, _ = pad_packed_sequence(out)
        selected = out[torch.tensor(lengths) - 1, torch.arange(2)]
        return selected, state, m.controller_hidden(selected).sum()

    a, sa, la = run(model, True)
    b, sb, lb = run(other, False)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(sa, sb)
    la.backward()
    lb.backward()
    for x, y in zip(model.parameters(), other.parameters()):
        assert (x.grad is None) == (y.grad is None)
        if x.grad is not None:
            torch.testing.assert_close(x.grad, y.grad)


def test_padded_transport_cannot_change_default_acting_api():
    model = RewardModel(False).eval()
    with pytest.raises(ValueError, match="explicit replay conditions"):
        model.core(torch.zeros(1, 16), torch.zeros(1, model.core.total_state_size), replay_padded=True)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_replay_conditions_skip_planning_with_nonzero_initial_ca3(monkeypatch, device):
    model = RewardModel(False).to(device).eval()
    conditions = torch.zeros(3, 2, model.core.hrl_condition_size, device=device)
    conditions[..., 2] = 1
    inputs = torch.rand(3, 2, 16, device=device)
    packet = pack_padded_sequence(inputs, [3, 2], enforce_sorted=False)
    initial = torch.zeros(2, model.core.total_state_size, device=device)
    initial[0, : model.core.core_output_size] = torch.arange(model.core.core_output_size, device=device).float()

    def historical_replanning_is_a_bug(*args, **kwargs):
        raise AssertionError("learner replay must not invoke graph planning")

    monkeypatch.setattr(model.core, "_update_hrl", historical_replanning_is_a_bug)
    output, _ = model.core(packet, initial, replay_conditions=conditions)
    padded, _ = pad_packed_sequence(output)
    valid = torch.arange(3, device=device).unsqueeze(1) < torch.tensor([3, 2], device=device).unsqueeze(0)
    torch.testing.assert_close(
        padded[:, :, -model.core.hrl_condition_size :][valid],
        conditions[valid],
    )
