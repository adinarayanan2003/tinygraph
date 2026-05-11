from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from tinygraph.ir import Graph, Node
from tinygraph.shape import infer_graph_shapes
from tinygraph.validation import validate_graph


def run(graph: Graph, feeds: Mapping[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Execute a graph with NumPy and return graph outputs."""
    validate_graph(graph)
    infer_graph_shapes(graph)
    env: dict[str, np.ndarray] = {
        name: _runtime_constant(graph, name, value)
        for name, value in graph.constants.items()
    }
    feeds = feeds or {}
    for name, spec in graph.inputs.items():
        if name not in feeds:
            raise ValueError(f"missing feed for input {name!r}")
        value = np.asarray(feeds[name], dtype=spec.dtype)
        if value.shape != spec.shape:
            raise ValueError(f"feed {name!r} shape {value.shape} does not match {spec.shape}")
        env[name] = value

    for node in graph.nodes:
        values = [env[name] for name in node.inputs]
        env[node.outputs[0]] = execute_node(node, values)

    return {name: env[name] for name in graph.outputs}


def execute_node(node: Node, values: list[np.ndarray]) -> np.ndarray:
    if node.op == "matmul":
        return values[0] @ values[1]
    if node.op == "add":
        return values[0] + values[1]
    if node.op == "mul":
        return values[0] * values[1]
    if node.op == "relu":
        return np.maximum(values[0], 0)
    if node.op == "gelu":
        result = 0.5 * values[0] * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (values[0] + 0.044715 * values[0] ** 3)))
        return result.astype(values[0].dtype, copy=False)
    if node.op == "reshape":
        return np.reshape(values[0], tuple(node.attrs["shape"]))
    if node.op == "transpose":
        axes = node.attrs.get("axes")
        return np.transpose(values[0], axes=None if axes is None else tuple(axes))
    if node.op == "softmax":
        axis = int(node.attrs.get("axis", -1))
        shifted = values[0] - np.max(values[0], axis=axis, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=axis, keepdims=True)
    if node.op == "sum":
        axis = node.attrs.get("axis")
        if isinstance(axis, list):
            axis = tuple(axis)
        return np.sum(values[0], axis=axis, keepdims=bool(node.attrs.get("keepdims", False)))
    if node.op == "layernorm":
        x = values[0]
        eps = float(node.attrs.get("eps", 1e-5))
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        y = (x - mean) / np.sqrt(var + eps)
        if len(values) >= 2:
            y = y * values[1]
        if len(values) >= 3:
            y = y + values[2]
        return y
    if node.op == "fused_linear_relu":
        return np.maximum(values[0] @ values[1] + values[2], 0)
    if node.op == "fused_linear":
        return values[0] @ values[1] + values[2]
    raise ValueError(f"unsupported op {node.op!r}")


def _runtime_constant(graph: Graph, name: str, value: np.ndarray) -> np.ndarray:
    metadata = graph.quantization.get(name)
    if not metadata:
        return value.copy()
    if metadata.get("scheme") != "symmetric_int8":
        raise ValueError(f"unsupported quantization scheme {metadata.get('scheme')!r}")
    dtype = metadata.get("original_dtype", "float32")
    return (value.astype("float32") * float(metadata["scale"])).astype(dtype)


def make_random_feeds(graph: Graph, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    feeds: dict[str, np.ndarray] = {}
    for name, spec in graph.inputs.items():
        feeds[name] = rng.normal(size=spec.shape).astype(spec.dtype)
    return feeds
