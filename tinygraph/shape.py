from __future__ import annotations

from typing import Any

import numpy as np

from tinygraph.ir import Graph, Node, TensorSpec


class ShapeError(ValueError):
    pass


def infer_graph_shapes(graph: Graph) -> dict[str, TensorSpec]:
    specs = dict(graph.tensor_specs)
    for node in graph.nodes:
        for name in node.inputs:
            if name not in specs:
                raise ShapeError(f"{node.op} reads unknown tensor {name!r}")
        out_spec = infer_node_shape(node, specs)
        specs[node.outputs[0]] = out_spec
    graph.tensor_specs = specs
    return specs


def infer_node_shape(node: Node, specs: dict[str, TensorSpec]) -> TensorSpec:
    if len(node.outputs) != 1:
        raise ShapeError(f"{node.op} must have exactly one output")
    output = node.outputs[0]
    inputs = [specs[name] for name in node.inputs]

    if node.op == "matmul":
        if len(inputs) != 2:
            raise ShapeError("matmul expects two inputs")
        if len(inputs[0].shape) < 2 or len(inputs[1].shape) < 2:
            raise ShapeError("matmul expects tensors with rank >= 2")
        if inputs[0].shape[-1] != inputs[1].shape[-2]:
            raise ShapeError(f"matmul mismatch: {inputs[0].shape} x {inputs[1].shape}")
        try:
            batch_shape = np.broadcast_shapes(inputs[0].shape[:-2], inputs[1].shape[:-2])
        except ValueError as exc:
            raise ShapeError(f"matmul batch dimensions cannot broadcast: {inputs[0].shape} x {inputs[1].shape}") from exc
        return TensorSpec(output, tuple(batch_shape) + (inputs[0].shape[-2], inputs[1].shape[-1]), inputs[0].dtype)

    if node.op in {"add", "mul"}:
        if len(inputs) != 2:
            raise ShapeError(f"{node.op} expects two inputs")
        try:
            shape = np.broadcast_shapes(inputs[0].shape, inputs[1].shape)
        except ValueError as exc:
            raise ShapeError(f"{node.op} cannot broadcast {inputs[0].shape} and {inputs[1].shape}") from exc
        return TensorSpec(output, tuple(shape), inputs[0].dtype)

    if node.op == "relu":
        if len(inputs) != 1:
            raise ShapeError("relu expects one input")
        return TensorSpec(output, inputs[0].shape, inputs[0].dtype)

    if node.op == "gelu":
        if len(inputs) != 1:
            raise ShapeError("gelu expects one input")
        return TensorSpec(output, inputs[0].shape, inputs[0].dtype)

    if node.op == "reshape":
        if len(inputs) != 1:
            raise ShapeError("reshape expects one input")
        target = tuple(int(dim) for dim in node.attrs["shape"])
        if target.count(-1) > 1:
            raise ShapeError("reshape supports at most one inferred -1 dimension")
        known = [dim for dim in target if dim != -1]
        input_size = int(np.prod(inputs[0].shape, dtype=np.int64))
        known_size = int(np.prod(known, dtype=np.int64)) if known else 1
        if -1 in target:
            if known_size == 0 or input_size % known_size != 0:
                raise ShapeError(f"reshape cannot infer {target} from {inputs[0].shape}")
            target = tuple(input_size // known_size if dim == -1 else dim for dim in target)
        elif input_size != known_size:
            raise ShapeError(f"reshape size mismatch: {inputs[0].shape} -> {target}")
        return TensorSpec(output, target, inputs[0].dtype)

    if node.op == "transpose":
        if len(inputs) != 1:
            raise ShapeError("transpose expects one input")
        axes = node.attrs.get("axes")
        if axes is None:
            shape = tuple(reversed(inputs[0].shape))
        else:
            axes = tuple(int(axis) for axis in axes)
            rank = len(inputs[0].shape)
            normalized = tuple(axis if axis >= 0 else rank + axis for axis in axes)
            if sorted(normalized) != list(range(rank)):
                raise ShapeError(f"transpose axes {axes} are invalid for shape {inputs[0].shape}")
            shape = tuple(inputs[0].shape[axis] for axis in normalized)
        return TensorSpec(output, shape, inputs[0].dtype)

    if node.op == "softmax":
        if len(inputs) != 1:
            raise ShapeError("softmax expects one input")
        axis = int(node.attrs.get("axis", -1))
        rank = len(inputs[0].shape)
        normalized = axis if axis >= 0 else rank + axis
        if normalized < 0 or normalized >= rank:
            raise ShapeError(f"softmax axis {axis} out of range for shape {inputs[0].shape}")
        return TensorSpec(output, inputs[0].shape, inputs[0].dtype)

    if node.op == "sum":
        if len(inputs) != 1:
            raise ShapeError("sum expects one input")
        return TensorSpec(output, _sum_shape(inputs[0].shape, node.attrs), inputs[0].dtype)

    if node.op == "layernorm":
        if not 1 <= len(inputs) <= 3:
            raise ShapeError("layernorm expects x, optional gamma, optional beta")
        for param in inputs[1:]:
            if param.shape not in {inputs[0].shape, (inputs[0].shape[-1],)}:
                raise ShapeError(f"layernorm parameter shape {param.shape} is incompatible with {inputs[0].shape}")
        return TensorSpec(output, inputs[0].shape, inputs[0].dtype)

    if node.op in {"fused_linear", "fused_linear_relu"}:
        if len(inputs) != 3:
            raise ShapeError(f"{node.op} expects x, weight, bias")
        matmul_spec = infer_node_shape(Node("matmul", [node.inputs[0], node.inputs[1]], ["_tmp"]), specs)
        bias_spec = specs[node.inputs[2]]
        try:
            shape = np.broadcast_shapes(matmul_spec.shape, bias_spec.shape)
        except ValueError as exc:
            raise ShapeError(f"fused bias shape {bias_spec.shape} cannot broadcast to {matmul_spec.shape}") from exc
        return TensorSpec(output, tuple(shape), matmul_spec.dtype)

    raise ShapeError(f"unsupported op {node.op!r}")


def _sum_shape(shape: tuple[int, ...], attrs: dict[str, Any]) -> tuple[int, ...]:
    axis = attrs.get("axis")
    keepdims = bool(attrs.get("keepdims", False))
    if axis is None:
        return tuple(1 for _ in shape) if keepdims else ()
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    rank = len(shape)
    normalized = tuple(ax if ax >= 0 else rank + ax for ax in axes)
    if any(ax < 0 or ax >= rank for ax in normalized):
        raise ShapeError(f"sum axis {axis!r} out of range for shape {shape}")
    if keepdims:
        return tuple(1 if i in normalized else dim for i, dim in enumerate(shape))
    return tuple(dim for i, dim in enumerate(shape) if i not in normalized)
