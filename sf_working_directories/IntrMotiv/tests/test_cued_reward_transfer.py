"""Five-site task contract: invisible sites, cue-specific manager, and flat replay."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sf_working_directories.IntrMotiv.dmlab.custom_core import SimpleSequenceWithBypassCore
from sf_working_directories.IntrMotiv.dmlab.goal_conditioned_dg import GoalConditionedDGCore
from sf_working_directories.IntrMotiv.dmlab.hrl_controllable_graph import HRLStateLayout, PolicyControllableGraph
from sf_working_directories.IntrMotiv.dmlab.topological_frontier import TopologicalStateLayout
from sf_working_directories.IntrMotiv.tests.test_topological_frontier import _manager_step


def _config():
    return SimpleNamespace(
        Hippo_R=2,
        Hippo_L=3,
        Hippo_n_feature=3,
        seed=42,
        reward_instruction_count=5,
        hrl_controllable_graph=False,
        fixed_task_conditioning=True,
        fixed_task_goal_mixture=True,
        fixed_task_goal_init="uniform_jitter",
        hrl_manager_mode="visit_direct",
        dg_context_feedback="none",
        dg_orthogonal_recruitment=False,
        hrl_graph_memory="policy_buffer",
        hrl_target_timing="immediate",
        ppo_dg_gradient="joint",
        DG_BN_intercept=1.0,
    )


def test_five_maps_share_geometry_spawns_and_invisible_goal_model():
    root = Path(__file__).resolve().parents[3]
    level = (root / "deepmindlab_patch/game_scripts/levels/openfield_map2_cued_reward5.lua").read_text()
    maps = [m.splitlines() for m in re.findall(r"\[\[\n([*PG A\n]+?)\n\]\]", level)][-5:]
    assert len(maps) == 5
    sites = [(1, 3), (1, 2), (8, 15), (6, 6), (18, 8)]
    assert "models/goal_transparent.md3" in level
    assert "quantity = 10" in level
    assert "random:uniformInt(1, 5)" in level
    for number, rows in enumerate(maps):
        assert len(rows) == 21 and all(len(row) == 21 for row in rows)
        assert sum(row.count("G") for row in rows) == 1
        assert rows[sites[number][0]][sites[number][1]] == "G"
        assert all(rows[r][c] != "P" for r, c in sites)
        assert [[cell == "*" for cell in row] for row in rows] == [[cell == "*" for cell in row] for row in maps[0]]
        assert [[cell == "P" for cell in row] for row in rows] == [[cell == "P" for cell in row] for row in maps[0]]


def test_reward_manager_credit_is_separate_for_each_instruction():
    graph = PolicyControllableGraph(3, reward_instruction_count=5)
    option = torch.zeros(2, 1, HRLStateLayout(3).size)
    topo = torch.zeros(2, 1, TopologicalStateLayout(3).size)
    option[..., HRLStateLayout(3).source] = 1
    topo[..., TopologicalStateLayout(3).final_goal] = 2
    count = graph.update_reward_goal_values(
        option,
        topo,
        torch.tensor([[10.0], [0.0]]),
        torch.ones(2, 1),
        torch.tensor([[1], [2]]),
    )
    assert count == 2
    assert graph.reward_goal_value[0, 0, 1] == 10
    assert graph.reward_goal_value[1, 0, 1] == 0
    assert graph.reward_goal_count[0, 0, 1] == 1
    assert graph.reward_goal_count[1, 0, 1] == 1


def test_manager_selects_a_different_goal_for_each_cue():
    from sf_working_directories.IntrMotiv.dmlab.hrl_controllable_graph import hrl_option_state_size
    from sf_working_directories.IntrMotiv.dmlab.topological_frontier import topological_state_size

    graph = PolicyControllableGraph(3, reward_instruction_count=5)
    graph.reward_goal_value[0, 0, 1] = 10
    graph.reward_goal_value[1, 0, 2] = 10
    targets = []
    for cue in (1, 2):
        torch.manual_seed(4)
        option, _, _ = _manager_step(
            torch.zeros(1, hrl_option_state_size(3)),
            torch.zeros(1, topological_state_size(3)),
            torch.tensor([[1.0, 0.0, 0.0]]),
            graph,
            direct_target_selection="reward_value",
            edge_exploration=True,
            target_timing="immediate",
            reward_instruction=torch.tensor([cue]),
        )
        targets.append(int(option[0, HRLStateLayout(3).target]))
    assert targets == [2, 3]


def test_flat_goal_write_uses_current_cue_during_packed_replay():
    core = GoalConditionedDGCore(_config(), 13)  # 3 DG + 2 bypass + 5 cue + 3 preactivation
    with torch.no_grad():
        core.fixed_task_target[0, 0] = 3
        core.fixed_task_target[1, 1] = 3
    cue = torch.nn.functional.one_hot(torch.tensor([[0, 1], [0, 1]]), 5).float()
    pre = torch.full((2, 2, 3), 2.0, requires_grad=True)
    stored = torch.zeros(2, 2, 3)
    head = torch.cat((torch.ones(2, 2, 3), torch.zeros(2, 2, 2), cue, pre, stored), -1)
    packed = torch.nn.utils.rnn.pack_padded_sequence(head, [2, 2], enforce_sorted=False)
    output, _ = core(packed, torch.zeros(2, core.total_state_size))
    expected = core.fixed_task_condition(cue)
    actual = output.data[:, core.target_condition_start : core.target_condition_start + 3]
    assert torch.allclose(actual, expected.reshape(4, 3))
    assert not torch.allclose(expected[:, 0], expected[:, 1])
    actual[:, 0].sum().backward()
    assert core.fixed_task_target.grad is not None
    assert core.fixed_task_target.grad.abs().sum() > 0


@pytest.mark.parametrize("edge_exploration", [False, True])
def test_batched_reward_manager_keeps_cues_separate(edge_exploration):
    graph = PolicyControllableGraph(3, reward_instruction_count=5)
    graph.reward_goal_value[0, 0, 1] = 100
    graph.reward_goal_value[1, 0, 2] = 100
    option, _, _ = _manager_step(
        torch.zeros(2, HRLStateLayout(3).size),
        torch.zeros(2, TopologicalStateLayout(3).size),
        torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        graph,
        direct_target_selection="reward_value",
        edge_exploration=edge_exploration,
        target_timing="immediate",
        reward_instruction=torch.tensor([1, 2]),
    )
    assert option[:, HRLStateLayout(3).target].tolist() == [2, 3]
