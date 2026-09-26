from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from sf_working_directories.IntrMotiv.dmlab.controller_q import ControllerQHeads, continuing_double_q_target
from sf_working_directories.IntrMotiv.dmlab.controller_replay import PhysicalDecision
from sf_working_directories.IntrMotiv.dmlab.controller_transition import (
    ReplayRejected,
    TransitionInput,
    transition_values,
)
from sf_working_directories.IntrMotiv.dmlab.custom_actor_critic import TargetFiLMDecoder, controller_core_view
from sf_working_directories.IntrMotiv.dmlab.custom_core import SimpleSequenceWithBypassCore
from sf_working_directories.IntrMotiv.dmlab.goal_conditioned_dg import GoalConditionedDGCore
from sf_working_directories.IntrMotiv.dmlab.hrl_controllable_graph import HRLStateLayout


class RewardModel(nn.Module):
    def __init__(self, write):
        super().__init__()
        self.cfg = SimpleNamespace(
            Hippo_R=2,
            Hippo_L=3,
            Hippo_n_feature=3,
            hrl_controllable_graph=True,
            hrl_graph_memory="policy_buffer",
            hrl_target_timing="immediate",
            hrl_manager_mode="frontier_direct",
            dg_context_feedback="none",
            dg_orthogonal_recruitment=False,
            DG_BN_intercept=0.0,
            reward_scale=0.1,
            encoder_reward_method="encourage",
            hrl_worker_reward_mode="hit_distance",
            hrl_target_hit_reward=1.0,
            hrl_distance_bonus_coeff=0.1,
            controller_learning="ddqn",
        )
        self.core = GoalConditionedDGCore(self.cfg, 19) if write else SimpleSequenceWithBypassCore(self.cfg, 16)
        self.scale = nn.Parameter(torch.ones(3))
        self.decoder = TargetFiLMDecoder(self.core, 8)
        self.controller_q = ControllerQHeads(8, 2, True)
        self.write = write

    def device_for_input_tensor(self, key):
        return torch.device("cpu")

    def type_for_input_tensor(self, key):
        return torch.float32

    def normalize_obs(self, obs):
        return obs

    def forward_head(self, obs):
        pre = obs["obs"] * self.scale
        head = torch.cat((pre.relu(), torch.zeros(len(pre), 13)), -1)
        return torch.cat((head, pre.detach()), -1) if self.write else head

    def forward_core(self, head, state):
        return self.core(head, state)

    def controller_hidden(self, out):
        view = (
            self.core.worker_view(out) if self.write else controller_core_view(out, self.core.core_output_size, "stop")
        )
        return self.decoder(view)


def physical_rows(model):
    state = torch.zeros(1, model.core.total_state_size)
    rows = []
    states = [state]
    with torch.no_grad():
        for t, obs in enumerate(torch.eye(3)):
            out, state = model.forward_core(model.forward_head({"obs": obs[None]}), state)
            canonical = model.core.split_worker_state(state)[0] if model.write else state
            rows.append(
                PhysicalDecision(
                    (0, 0),
                    0,
                    t,
                    t,
                    {"obs": obs.numpy()},
                    0,
                    out[0, model.core.target_condition_start : model.core.total_output_size].numpy(),
                    canonical[0, model.core.base_state_size :].numpy(),
                    0,
                    0,
                    False,
                    False,
                    4,
                )
            )
            states.append(state)
    return rows, states


@pytest.mark.parametrize("write", [False, True])
def test_original_reward_parity_and_virtual_goal_has_positive_learning_signal(write):
    from sf_working_directories.IntrMotiv.dmlab.custom_learner import DistanceLearnerReward

    model = RewardModel(write).eval()
    rows, states = physical_rows(model)
    example = TransitionInput(tuple(rows[:2]), 0)
    main = transition_values(model, example)
    adapter = object.__new__(DistanceLearnerReward)
    adapter.cfg = model.cfg
    adapter.actor_critic = model
    buff = {
        "rnn_states": torch.stack([s[0] for s in states[:2]])[None],
        "rewards": torch.zeros(1, 1),
        "dones": torch.zeros(1, 1, dtype=torch.bool),
    }
    adapter._calculate_reward_components(buff, {"new_rnn_states": states[2]})
    torch.testing.assert_close(main["reward"], buff["rewards"].reshape(()))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    virtual = transition_values(model, replace(example, virtual_goal=1, remaining=3))
    assert virtual["reward"] > 0 and virtual["done"]
    virtual["q"][0, 0].backward()
    assert model.controller_q.auxiliary.weight.grad.abs().sum() > 0
    assert model.controller_q.main.weight.grad is None
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k])


def test_changed_recognition_rejected_and_physical_truncation_stops_main():
    model = RewardModel(True).eval()
    rows, _ = physical_rows(model)
    example = TransitionInput(tuple(rows[:2]), 0)
    ended = replace(rows[0], truncated=True)
    result = transition_values(model, replace(example, rows=(ended, rows[1])))
    assert result["done"]
    with torch.no_grad():
        model.scale[0] = -1
    with pytest.raises(ReplayRejected, match="recognition_changed"):
        transition_values(model, example)


def test_main_target_continues_after_an_actual_goal_switch():
    # Done is physical, regardless of target-hit/manager goal-switch events.
    reward = torch.tensor([1.0])
    online = torch.tensor([[2.0, 5.0]])
    target = torch.tensor([[9.0, 7.0]])
    result = continuing_double_q_target(reward, torch.tensor([False]), online, target, 0.9)
    torch.testing.assert_close(result, torch.tensor([7.3]))


@pytest.mark.parametrize("write", [False, True])
def test_batched_reconstruction_matches_separate_histories_and_gradients(write):
    from copy import deepcopy

    from sf_working_directories.IntrMotiv.dmlab.controller_transition import transition_values_batch

    model = RewardModel(write).eval()
    other = deepcopy(model)
    rows, _ = physical_rows(model)
    examples = [TransitionInput(tuple(rows[:2]), 0), TransitionInput(tuple(rows), 1)]
    separate = [transition_values(other, e) for e in examples]
    batched = transition_values_batch(model, examples)
    for a, b in zip(separate, batched):
        torch.testing.assert_close(a["q"], b["q"])
        torch.testing.assert_close(a["reward"], b["reward"])
        assert torch.equal(a["signature"], b["signature"])
    sum(r["q"][0, 0] for r in separate).backward()
    sum(r["q"][0, 0] for r in batched).backward()
    for a, b in zip(model.parameters(), other.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad)


@pytest.mark.parametrize("write", [False, True])
def test_rebuilt_prefix_keeps_actual_manager_context_and_validates_transition(write):
    from sf_working_directories.IntrMotiv.dmlab.controller_transition import transition_values_batch

    model = RewardModel(write).eval()
    rows, _ = physical_rows(model)
    example = TransitionInput(tuple(rows), 1)
    before = transition_values(model, example)
    recorded = [r.context.copy() for r in rows]
    # The burn-in landmark changes, but current and successor recognition agree.
    # Actor publication already rebuilds exactly this changed worker history.
    with torch.no_grad():
        model.scale[0] = -1
    after = transition_values(model, example)
    batched = transition_values_batch(model, [example])[0]
    assert torch.equal(before["signature"], after["signature"])
    torch.testing.assert_close(after["q"], batched["q"])
    torch.testing.assert_close(after["reward"], batched["reward"])
    for row, context in zip(rows, recorded):
        np.testing.assert_array_equal(row.context, context)
    # A changed current label still invalidates the actual manager context.
    with torch.no_grad():
        model.scale[1] = -1
    with pytest.raises(ReplayRejected, match="current_recognition_changed"):
        transition_values(model, example)


@pytest.mark.parametrize("write", [False, True])
def test_early_current_screen_preserves_rejections_values_and_gradients(write):
    from copy import deepcopy

    from sf_working_directories.IntrMotiv.dmlab.controller_transition import transition_values_batch

    model = RewardModel(write).eval()
    rows, _ = physical_rows(model)
    examples = [TransitionInput(tuple(rows[:2]), 0), TransitionInput(tuple(rows), 1)]
    examples.append(replace(examples[0], virtual_goal=1, remaining=3))
    examples.append(replace(examples[1], virtual_goal=2, remaining=3))
    # One current label fails, while another example has only changed burn-in.
    with torch.no_grad():
        model.scale[0] = -1
    other = deepcopy(model)
    plain = transition_values_batch(model, examples)
    screened = transition_values_batch(other, examples, screen_current=True)
    for a, b in zip(plain, screened):
        if isinstance(a, str):
            assert a == b
        else:
            torch.testing.assert_close(a["q"], b["q"])
            torch.testing.assert_close(a["reward"], b["reward"])
            assert torch.equal(a["signature"], b["signature"]) and a["done"] == b["done"]
    sum(a["q"].sum() for a in plain if not isinstance(a, str)).backward()
    sum(a["q"].sum() for a in screened if not isinstance(a, str)).backward()
    for a, b in zip(model.parameters(), other.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad)


def test_early_screen_skips_rejected_histories_but_keeps_validation(monkeypatch):
    import sf_working_directories.IntrMotiv.dmlab.controller_transition as module

    model = RewardModel(True).eval()
    rows, _ = physical_rows(model)
    example = TransitionInput(tuple(rows[:2]), 0)
    with torch.no_grad():
        model.scale[0] = -1

    def no_history(model, histories):
        assert not histories
        return []

    monkeypatch.setattr(module, "reconstruct_controller_histories", no_history)
    assert module.transition_values_batch(model, [example], screen_current=True) == ["current_recognition_changed"]
    gap = replace(example, rows=(rows[0], replace(rows[1], index=4)))
    with pytest.raises(ValueError, match="gap"):
        module.transition_values_batch(model, [gap], screen_current=True)
    stale = replace(example, rows=tuple(replace(row, generation=1) for row in example.rows))
    assert module.transition_values_batch(model, [stale], screen_current=True) == [
        "snapshot_structural_generation_incompatible"
    ]


def test_learner_screens_only_no_gradient_candidate_search(monkeypatch):
    from copy import deepcopy

    import sf_working_directories.IntrMotiv.dmlab.controller_learner as module

    model = RewardModel(True).eval()
    rows, _ = physical_rows(model)
    learner = object.__new__(module.ControllerLearner)
    learner.actor_critic = model
    learner.cfg = model.cfg
    learner.cfg.controller_td_positions = 256
    learner.cfg.gamma = 0.99
    learner.device = torch.device("cpu")
    learner.controller_version = 0
    learner.online_snapshot = SimpleNamespace(model=deepcopy(model), version=0)
    learner.target_snapshot = SimpleNamespace(model=deepcopy(model), version=0)
    operation = module.transition_values_batch
    calls = []

    def record(model, examples, *, screen_current=False):
        calls.append(screen_current)
        return operation(model, examples, screen_current=screen_current)

    monkeypatch.setattr(module, "transition_values_batch", record)
    examples = [TransitionInput(tuple(rows[:2]), 0)]
    result = learner._evaluate_pairs(examples)
    assert calls == [False, False] and result[0][0].requires_grad
    calls.clear()
    with torch.no_grad():
        result = learner._evaluate_pairs(examples)
    assert calls == [True, True] and not result[0][0].requires_grad
