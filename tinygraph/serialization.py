from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tinygraph.ir import Graph, Node
from tinygraph.shape import infer_graph_shapes


def load_graph(path: str | Path) -> Graph:
    data = json.loads(Path(path).read_text())
    graph = Graph(data.get("name", Path(path).stem))
    for item in data.get("inputs", []):
        graph.input(item["name"], tuple(item["shape"]), item.get("dtype", "float32"))
    for item in data.get("constants", []):
        graph.const(item["name"], item["value"], item.get("dtype", "float32"))
        if "quantization" in item:
            graph.quantization[item["name"]] = dict(item["quantization"])
    for item in data.get("nodes", []):
        graph.nodes.append(
            Node(
                op=item["op"],
                inputs=list(item["inputs"]),
                outputs=list(item["outputs"]),
                attrs=dict(item.get("attrs", {})),
                name=item.get("name"),
            )
        )
    for name in data.get("outputs", []):
        graph.outputs.append(name)
    infer_graph_shapes(graph)
    return graph


def save_graph(graph: Graph, path: str | Path) -> None:
    infer_graph_shapes(graph)
    data: dict[str, Any] = {
        "name": graph.name,
        "inputs": [
            {"name": spec.name, "shape": list(spec.shape), "dtype": spec.dtype}
            for spec in graph.inputs.values()
        ],
        "constants": [
            _constant_payload(graph, name, value)
            for name, value in graph.constants.items()
        ],
        "nodes": [
            {
                "name": node.name,
                "op": node.op,
                "inputs": node.inputs,
                "outputs": node.outputs,
                "attrs": node.attrs,
            }
            for node in graph.nodes
        ],
        "outputs": graph.outputs,
    }
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def graph_summary(graph: Graph) -> str:
    infer_graph_shapes(graph)
    lines = [f"Graph: {graph.name}", "Inputs:"]
    for spec in graph.inputs.values():
        lines.append(f"  - {spec.name}: shape={spec.shape}, dtype={spec.dtype}")
    lines.append("Constants:")
    for name, value in graph.constants.items():
        suffix = ""
        if name in graph.quantization:
            metadata = graph.quantization[name]
            suffix = f", quantized={metadata['scheme']}, scale={float(metadata['scale']):.8f}"
        lines.append(f"  - {name}: shape={value.shape}, dtype={value.dtype}{suffix}")
    lines.append("Nodes:")
    for index, node in enumerate(graph.nodes):
        spec = graph.tensor_specs.get(node.outputs[0])
        shape = spec.shape if spec else "?"
        lines.append(f"  {index:02d}. {node.outputs[0]} = {node.op}({', '.join(node.inputs)}) -> {shape}")
    lines.append(f"Outputs: {', '.join(graph.outputs)}")
    return "\n".join(lines)


def _constant_payload(graph: Graph, name: str, value: np.ndarray) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": name, "dtype": str(value.dtype), "value": value.tolist()}
    if name in graph.quantization:
        payload["quantization"] = graph.quantization[name]
    return payload


def clone_with_random_constants(graph: Graph, seed: int = 0) -> Graph:
    rng = np.random.default_rng(seed)
    out = graph.clone()
    for name, value in out.constants.items():
        out.constants[name] = rng.normal(size=value.shape).astype(value.dtype)
    return out
