from __future__ import annotations

import numpy as np

from tinygraph.ir import Graph, Node
from tinygraph.runtime import execute_node


def constant_fold(graph: Graph) -> Graph:
    out = graph.clone()
    new_nodes: list[Node] = []
    for node in out.nodes:
        if all(name in out.constants for name in node.inputs):
            values = [out.constants[name] for name in node.inputs]
            out.constants[node.outputs[0]] = execute_node(node, values)
            continue
        new_nodes.append(node)
    out.nodes = new_nodes
    return out


def identity_op_removal(graph: Graph) -> Graph:
    out = graph.clone()
    replacements: dict[str, str] = {}
    new_nodes: list[Node] = []
    for node in out.nodes:
        node = _rewrite_inputs(node, replacements)
        replacement = _identity_replacement(node, out.constants)
        if replacement is not None:
            replacements[node.outputs[0]] = replacement
            continue
        new_nodes.append(node)
    out.nodes = [_rewrite_inputs(node, replacements) for node in new_nodes]
    out.outputs = [replacements.get(name, name) for name in out.outputs]
    return out


def fuse_linear_relu(graph: Graph) -> Graph:
    return _fuse_linear_patterns(graph, relu=True)


def fuse_linear(graph: Graph) -> Graph:
    return _fuse_linear_patterns(graph, relu=False)


def _fuse_linear_patterns(graph: Graph, relu: bool) -> Graph:
    out = graph.clone()
    consumers = _consumer_map(out.nodes)
    new_nodes: list[Node] = []
    skip_outputs: set[str] = set()

    for index, node in enumerate(out.nodes):
        if node.outputs[0] in skip_outputs:
            continue
        fused = _try_fuse_linear_relu(out.nodes, index, consumers, set(out.outputs)) if relu else _try_fuse_linear(
            out.nodes,
            index,
            consumers,
            set(out.outputs),
        )
        if fused is None:
            new_nodes.append(node)
            continue
        fused_node, skipped = fused
        new_nodes.append(fused_node)
        skip_outputs.update(skipped)

    out.nodes = new_nodes
    return out


def dead_node_elimination(graph: Graph) -> Graph:
    out = graph.clone()
    needed = set(out.outputs)
    new_nodes: list[Node] = []
    for node in reversed(out.nodes):
        if any(output in needed for output in node.outputs):
            new_nodes.append(node)
            needed.update(node.inputs)
    out.nodes = list(reversed(new_nodes))
    return out


def _rewrite_inputs(node: Node, replacements: dict[str, str]) -> Node:
    rewritten = node.clone()
    rewritten.inputs = [replacements.get(name, name) for name in node.inputs]
    return rewritten


def _identity_replacement(node: Node, constants: dict[str, np.ndarray]) -> str | None:
    if node.op == "add":
        left, right = node.inputs
        if _is_const_value(constants.get(left), 0):
            return right
        if _is_const_value(constants.get(right), 0):
            return left
    if node.op == "mul":
        left, right = node.inputs
        if _is_const_value(constants.get(left), 1):
            return right
        if _is_const_value(constants.get(right), 1):
            return left
    if node.op == "relu":
        source = node.inputs[0]
        value = constants.get(source)
        if value is not None and np.all(value >= 0):
            return source
    return None


def _is_const_value(value: np.ndarray | None, scalar: float) -> bool:
    return value is not None and bool(np.all(value == scalar))


def _consumer_map(nodes: list[Node]) -> dict[str, list[int]]:
    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        for input_name in node.inputs:
            consumers.setdefault(input_name, []).append(index)
    return consumers


def _try_fuse_linear_relu(
    nodes: list[Node],
    index: int,
    consumers: dict[str, list[int]],
    graph_outputs: set[str],
) -> tuple[Node, set[str]] | None:
    matmul = nodes[index]
    if matmul.op != "matmul":
        return None
    matmul_out = matmul.outputs[0]
    matmul_consumers = consumers.get(matmul_out, [])
    if len(matmul_consumers) != 1 or matmul_out in graph_outputs:
        return None
    add_index = matmul_consumers[0]
    add = nodes[add_index]
    if add.op != "add":
        return None
    add_out = add.outputs[0]
    add_consumers = consumers.get(add_out, [])
    if len(add_consumers) != 1 or add_out in graph_outputs:
        return None
    relu = nodes[add_consumers[0]]
    if relu.op != "relu":
        return None

    bias_inputs = [name for name in add.inputs if name != matmul_out]
    if len(bias_inputs) != 1:
        return None
    fused = Node(
        op="fused_linear_relu",
        inputs=[matmul.inputs[0], matmul.inputs[1], bias_inputs[0]],
        outputs=[relu.outputs[0]],
        attrs={},
        name=f"fused_{matmul.name or matmul.outputs[0]}",
    )
    return fused, {matmul_out, add_out, relu.outputs[0]}


def _try_fuse_linear(
    nodes: list[Node],
    index: int,
    consumers: dict[str, list[int]],
    graph_outputs: set[str],
) -> tuple[Node, set[str]] | None:
    matmul = nodes[index]
    if matmul.op != "matmul":
        return None
    matmul_out = matmul.outputs[0]
    matmul_consumers = consumers.get(matmul_out, [])
    if len(matmul_consumers) != 1 or matmul_out in graph_outputs:
        return None
    add = nodes[matmul_consumers[0]]
    if add.op != "add":
        return None
    bias_inputs = [name for name in add.inputs if name != matmul_out]
    if len(bias_inputs) != 1:
        return None
    fused = Node(
        op="fused_linear",
        inputs=[matmul.inputs[0], matmul.inputs[1], bias_inputs[0]],
        outputs=[add.outputs[0]],
        attrs={},
        name=f"fused_{matmul.name or matmul.outputs[0]}",
    )
    return fused, {matmul_out, add.outputs[0]}
