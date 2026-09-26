from itertools import product

import torch

from sf_working_directories.IntrMotiv.dmlab.topological_frontier import connectivity_gain


def reachable_pairs(adjacency):
    # Independent graph-search oracle, including cycles and self-reachability.
    n = len(adjacency)
    pairs = set()
    for start in range(n):
        seen = {start}
        todo = [start]
        while todo:
            node = todo.pop()
            for end, edge in enumerate(adjacency[node]):
                if edge and end not in seen:
                    seen.add(end)
                    todo.append(end)
        pairs.update((start, end) for end in seen if end != start)
    return pairs


def check_edges(adjacency, edges):
    before = reachable_pairs(adjacency.tolist())
    for source, destination in edges:
        after = adjacency.clone()
        after[source, destination] = True
        expected = len(reachable_pairs(after.tolist()) - before)
        assert connectivity_gain(adjacency, source, destination) == expected


def test_exact_gain_for_every_three_node_directed_graph_and_edge():
    edges = [(i, j) for i in range(3) for j in range(3) if i != j]
    for bits in product((False, True), repeat=len(edges)):
        adjacency = torch.zeros(3, 3, dtype=torch.bool)
        for (i, j), bit in zip(edges, bits):
            adjacency[i, j] = bit
        check_edges(adjacency, product(range(3), repeat=2))


def test_exact_gain_for_larger_sparse_cyclic_and_acyclic_graphs():
    rng = torch.Generator().manual_seed(119)
    for n in (8, 16, 64):
        for acyclic in (False, True):
            adjacency = torch.rand(n, n, generator=rng) < (1.5 / n)
            if acyclic:
                adjacency = torch.triu(adjacency, diagonal=1)
            edges = torch.randint(n, (40, 2), generator=rng).tolist()
            check_edges(adjacency, edges)
