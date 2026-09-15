"""Device-resident manager regression against pre-vectorization outputs.

The fixture stores inputs and five successive reference transitions, including
all graph buffers. It covers all three selectors, both timing/outcome modes,
geometry, waypoint planning, stale generations, silence and multi-activation.
See data/topological_manager_reference.md for provenance.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sf_working_directories.IntrMotiv.dmlab.hrl_controllable_graph import PolicyControllableGraph
from sf_working_directories.IntrMotiv.dmlab.topological_frontier import advance_topological_manager

FIXTURE = Path(__file__).with_name("data") / "topological_manager_reference.npz"
CASES = (0, 7, 24, 63, 64, 95, 128, 159, 191)
DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
]


def load_case(case, device):
    with np.load(FIXTURE, allow_pickle=False) as saved:
        prefix = f"case{case}/"
        kwargs = json.loads(str(saved[prefix + "kwargs"]))
        values = {
            name[len(prefix) :]: torch.from_numpy(saved[name].copy()).to(device)
            for name in saved.files
            if name.startswith(prefix) and name != prefix + "kwargs"
        }
    graph = PolicyControllableGraph(values["step0/dg"].size(-1)).to(device)
    graph.load_state_dict({name[len("graph/") :]: value for name, value in values.items() if name.startswith("graph/")})
    return values, graph, kwargs


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", CASES)
def test_manager_matches_reference(case, device):
    values, graph, kwargs = load_case(case, device)
    option, topo = values["option"], values["topo"]
    original_graph = {name: value.clone() for name, value in graph.state_dict().items()}
    for step in range(5):
        prefix = f"step{step}/"
        old_option, old_topo = option.clone(), topo.clone()
        result = advance_topological_manager(
            option, topo, values[prefix + "dg"], values[prefix + "actions"], graph, **kwargs
        )
        torch.testing.assert_close(option, old_option)
        torch.testing.assert_close(topo, old_topo)
        for name, value in zip(("option", "topo", "condition"), result):
            assert value.device.type == device
            torch.testing.assert_close(value, values[prefix + name], rtol=1e-5, atol=1e-5)
        option, topo = result[:2]
    for name, value in graph.state_dict().items():
        torch.testing.assert_close(value, original_graph[name])


@pytest.mark.parametrize("device", DEVICES)
def test_manager_has_no_host_scalar_reads_or_dynamic_indices(device):
    values, graph, kwargs = load_case(0, device)
    # Inspect ATen operations, including implicit bool(tensor) and boolean
    # indexing. These synchronize CUDA even when all stored tensors use CUDA.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        advance_topological_manager(
            values["option"], values["topo"], values["step0/dg"], values["step0/actions"], graph, **kwargs
        )
    forbidden = {"aten::_local_scalar_dense", "aten::item", "aten::is_nonzero", "aten::nonzero"}
    for event in profile.events():
        if event.name not in forbidden:
            continue
        parent = event.cpu_parent
        ancestors = []
        while parent is not None:
            ancestors.append(parent.name)
            parent = parent.cpu_parent
        # PyTorch's CPU one_hot validates bounds with scalars internally.
        # CUDA one_hot does not; no exemption is allowed on CUDA.
        assert device == "cpu" and "aten::one_hot" in ancestors, event.name
