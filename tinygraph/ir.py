from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * np.dtype(self.dtype).itemsize


@dataclass(frozen=True)
class Tensor:
    graph: "Graph"
    name: str


@dataclass
class Node:
    op: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)
    name: str | None = None

    def clone(self) -> "Node":
        return Node(
            op=self.op,
            inputs=list(self.inputs),
            outputs=list(self.outputs),
            attrs=dict(self.attrs),
            name=self.name,
        )


class Graph:
    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self.inputs: dict[str, TensorSpec] = {}
        self.constants: dict[str, np.ndarray] = {}
        self.quantization: dict[str, dict[str, Any]] = {}
        self.nodes: list[Node] = []
        self.outputs: list[str] = []
        self.tensor_specs: dict[str, TensorSpec] = {}
        self._next_id = 0

    def input(self, name: str, shape: tuple[int, ...], dtype: str = "float32") -> Tensor:
        self._ensure_new_tensor(name)
        spec = TensorSpec(name=name, shape=tuple(shape), dtype=dtype)
        self.inputs[name] = spec
        self.tensor_specs[name] = spec
        return Tensor(self, name)

    def const(self, name: str, value: Any, dtype: str = "float32") -> Tensor:
        self._ensure_new_tensor(name)
        array = np.asarray(value, dtype=dtype)
        self.constants[name] = array
        self.tensor_specs[name] = TensorSpec(name=name, shape=array.shape, dtype=str(array.dtype))
        return Tensor(self, name)

    def add_node(
        self,
        op: str,
        inputs: list[str | Tensor],
        output: str | None = None,
        attrs: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> Tensor:
        input_names = [tensor.name if isinstance(tensor, Tensor) else tensor for tensor in inputs]
        output_name = output or self.unique_name(op)
        self._ensure_new_tensor(output_name)
        node = Node(op=op, inputs=input_names, outputs=[output_name], attrs=attrs or {}, name=name)
        self.nodes.append(node)
        return Tensor(self, output_name)

    def output(self, tensor: Tensor | str) -> None:
        name = tensor.name if isinstance(tensor, Tensor) else tensor
        if name not in self.tensor_specs and name not in self._node_outputs():
            raise ValueError(f"cannot output unknown tensor {name!r}")
        if name not in self.outputs:
            self.outputs.append(name)

    def unique_name(self, prefix: str) -> str:
        while True:
            self._next_id += 1
            candidate = f"{prefix}_{self._next_id}"
            if candidate not in self.tensor_specs and candidate not in self._node_outputs():
                return candidate

    def clone(self) -> "Graph":
        graph = Graph(self.name)
        graph.inputs = dict(self.inputs)
        graph.constants = {name: value.copy() for name, value in self.constants.items()}
        graph.quantization = {name: dict(metadata) for name, metadata in self.quantization.items()}
        graph.nodes = [node.clone() for node in self.nodes]
        graph.outputs = list(self.outputs)
        graph.tensor_specs = dict(self.tensor_specs)
        graph._next_id = self._next_id
        return graph

    def stats(self) -> dict[str, int]:
        return {
            "inputs": len(self.inputs),
            "constants": len(self.constants),
            "nodes": len(self.nodes),
            "outputs": len(self.outputs),
        }

    def _ensure_new_tensor(self, name: str) -> None:
        if name in self.tensor_specs or name in self._node_outputs():
            raise ValueError(f"duplicate tensor name {name!r}")

    def _node_outputs(self) -> set[str]:
        return {output for node in self.nodes for output in node.outputs}
