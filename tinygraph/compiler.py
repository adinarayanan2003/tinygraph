from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from tinygraph.ir import Graph
from tinygraph.passes import (
    constant_fold,
    dead_node_elimination,
    fuse_linear,
    fuse_linear_relu,
    identity_op_removal,
)
from tinygraph.shape import infer_graph_shapes
from tinygraph.validation import validate_graph

PassFn = Callable[[Graph], Graph]


PASS_REGISTRY: dict[str, PassFn] = {
    "constant_fold": constant_fold,
    "identity_op_removal": identity_op_removal,
    "fuse_linear_relu": fuse_linear_relu,
    "fuse_linear": fuse_linear,
    "dead_node_elimination": dead_node_elimination,
}

DEFAULT_PASSES = [
    "constant_fold",
    "identity_op_removal",
    "fuse_linear_relu",
    "fuse_linear",
    "dead_node_elimination",
]


@dataclass
class PassReport:
    name: str
    before_nodes: int
    after_nodes: int
    changed: bool
    graph: Graph


@dataclass
class OptimizationResult:
    graph: Graph
    reports: list[PassReport]


def optimize(graph: Graph, passes: list[str] | None = None) -> OptimizationResult:
    current = graph.clone()
    validate_graph(current)
    infer_graph_shapes(current)
    reports: list[PassReport] = []
    for name in passes or DEFAULT_PASSES:
        if name not in PASS_REGISTRY:
            raise ValueError(f"unknown optimization pass {name!r}")
        before = len(current.nodes)
        next_graph = PASS_REGISTRY[name](current)
        validate_graph(next_graph)
        infer_graph_shapes(next_graph)
        after = len(next_graph.nodes)
        reports.append(
            PassReport(
                name=name,
                before_nodes=before,
                after_nodes=after,
                changed=_graph_signature(current) != _graph_signature(next_graph),
                graph=next_graph.clone(),
            )
        )
        current = next_graph
    return OptimizationResult(graph=current, reports=reports)


def _graph_signature(graph: Graph) -> tuple:
    return tuple((node.op, tuple(node.inputs), tuple(node.outputs), tuple(sorted(node.attrs.items()))) for node in graph.nodes)
