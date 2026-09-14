"""Compare manager implementations without environment or training startup."""
import argparse
import importlib.util
import json
import sys
import time

import torch

from sf_working_directories.IntrMotiv.dmlab import topological_frontier as actual
from sf_working_directories.IntrMotiv.dmlab.hrl_controllable_graph import PolicyControllableGraph, hrl_option_state_size

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference', required=True, help='Unmodified topological_frontier.py')
parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
args = parser.parse_args()
spec = importlib.util.spec_from_file_location('reference', args.reference)
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)
torch.set_num_threads(1)
for batch in (32, 256):
    torch.manual_seed(99)
    n = 16
    graph = PolicyControllableGraph(n).to(args.device)
    graph.node_visits.copy_(torch.arange(1, n + 1, device=args.device))
    graph.tctrl.fill_(4)
    graph.edge_confidence.fill_(2)
    graph.control_attempts.fill_(3)
    option = torch.zeros(batch, hrl_option_state_size(n), device=args.device)
    topo = torch.zeros(batch, actual.topological_state_size(n), device=args.device)
    dg = torch.nn.functional.one_hot(torch.arange(batch, device=args.device) % n, n).float()
    action = torch.zeros(batch, 4, device=args.device)
    kwargs = dict(fallback_horizon=64, margin_ratio=.2, margin_steps=2, confidence_threshold=.5,
                  passive_threshold=2, passive_min_displacement=2, passive_max_length=64,
                  use_motion_filter=False, frontier_uncertainty_weight=1, waypoint_planning=False,
                  exploration_horizon=64, edge_exploration=False, target_timing='immediate')
    result = {}
    for name, module in [('reference', reference), ('batched', actual)]:
        state, topology = option.clone(), topo.clone()
        for _ in range(4):
            state, topology, _ = module.advance_topological_manager(state, topology, dg, action, graph, **kwargs)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(64):
            state, topology, _ = module.advance_topological_manager(state, topology, dg, action, graph, **kwargs)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        result[name + '_ms_per_step'] = (time.perf_counter() - start) * 1000 / 64
    result.update(speedup=result['reference_ms_per_step'] / result['batched_ms_per_step'],
                  batch=batch, nodes=n, device=args.device)
    print(json.dumps(result), flush=True)
