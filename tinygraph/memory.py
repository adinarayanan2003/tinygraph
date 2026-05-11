from __future__ import annotations

from dataclasses import dataclass

from tinygraph.ir import Graph
from tinygraph.shape import infer_graph_shapes


@dataclass
class MemoryPlan:
    naive_peak_bytes: int
    planned_peak_bytes: int
    tensor_lifetimes: dict[str, tuple[int, int]]


def plan_memory(graph: Graph) -> MemoryPlan:
    infer_graph_shapes(graph)
    lifetimes = _tensor_lifetimes(graph)
    intermediate_names = [name for name in lifetimes if name not in graph.inputs and name not in graph.constants]
    naive_peak = sum(graph.tensor_specs[name].nbytes for name in intermediate_names if name in graph.tensor_specs)

    events: list[tuple[int, int]] = []
    for name in intermediate_names:
        if name not in graph.tensor_specs:
            continue
        start, end = lifetimes[name]
        size = graph.tensor_specs[name].nbytes
        events.append((start, size))
        events.append((end + 1, -size))
    current = 0
    planned_peak = 0
    for _, delta in sorted(events):
        current += delta
        planned_peak = max(planned_peak, current)
    return MemoryPlan(naive_peak, planned_peak, lifetimes)


def _tensor_lifetimes(graph: Graph) -> dict[str, tuple[int, int]]:
    last_use: dict[str, int] = {name: 0 for name in graph.inputs | graph.constants}
    first_def: dict[str, int] = {name: 0 for name in graph.inputs | graph.constants}
    for index, node in enumerate(graph.nodes, start=1):
        for input_name in node.inputs:
            last_use[input_name] = index
        for output in node.outputs:
            first_def[output] = index
            last_use[output] = index
    final_index = len(graph.nodes) + 1
    for output in graph.outputs:
        last_use[output] = final_index
    return {name: (first_def.get(name, 0), last_use.get(name, 0)) for name in set(first_def) | set(last_use)}
