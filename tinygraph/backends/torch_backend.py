from __future__ import annotations

import numpy as np

from tinygraph.ir import Graph, Node
from tinygraph.quantization import dequantized_constants
from tinygraph.shape import infer_graph_shapes
from tinygraph.validation import validate_graph


class TorchBackend:
    name = "torch"

    def __init__(self, device: str | None = "auto") -> None:
        self.torch = _import_torch()
        self.device = self._select_device(device or "auto")
        self.device_name = str(self.device)

    def run(self, graph: Graph, feeds: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        torch = self.torch
        validate_graph(graph)
        infer_graph_shapes(graph)
        feeds = feeds or {}
        env = {
            name: torch.as_tensor(value, device=self.device)
            for name, value in dequantized_constants(graph).items()
        }
        for name, spec in graph.inputs.items():
            if name not in feeds:
                raise ValueError(f"missing feed for input {name!r}")
            value = np.asarray(feeds[name], dtype=spec.dtype)
            if value.shape != spec.shape:
                raise ValueError(f"feed {name!r} shape {value.shape} does not match {spec.shape}")
            env[name] = torch.as_tensor(value, device=self.device)

        for node in graph.nodes:
            values = [env[name] for name in node.inputs]
            env[node.outputs[0]] = self._execute_node(node, values)

        self.synchronize()
        return {name: env[name].detach().cpu().numpy() for name in graph.outputs}

    def synchronize(self) -> None:
        torch = self.torch
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

    def _select_device(self, device: str):
        torch = self.torch
        if device == "auto":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        selected = torch.device(device)
        if selected.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("MPS device requested but torch MPS is not available.")
        if selected.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but torch CUDA is not available.")
        return selected

    def _execute_node(self, node: Node, values):
        torch = self.torch
        if node.op == "matmul":
            return values[0] @ values[1]
        if node.op == "add":
            return values[0] + values[1]
        if node.op == "mul":
            return values[0] * values[1]
        if node.op == "relu":
            return torch.relu(values[0])
        if node.op == "gelu":
            return torch.nn.functional.gelu(values[0], approximate="tanh")
        if node.op == "reshape":
            return torch.reshape(values[0], tuple(node.attrs["shape"]))
        if node.op == "transpose":
            axes = node.attrs.get("axes")
            if axes is None:
                axes = tuple(reversed(range(values[0].ndim)))
            return torch.permute(values[0], tuple(axes))
        if node.op == "softmax":
            return torch.softmax(values[0], dim=int(node.attrs.get("axis", -1)))
        if node.op == "sum":
            axis = node.attrs.get("axis")
            if isinstance(axis, list):
                axis = tuple(axis)
            return torch.sum(values[0], dim=axis, keepdim=bool(node.attrs.get("keepdims", False)))
        if node.op == "layernorm":
            x = values[0]
            eps = float(node.attrs.get("eps", 1e-5))
            mean = torch.mean(x, dim=-1, keepdim=True)
            var = torch.mean((x - mean) ** 2, dim=-1, keepdim=True)
            y = (x - mean) / torch.sqrt(var + eps)
            if len(values) >= 2:
                y = y * values[1]
            if len(values) >= 3:
                y = y + values[2]
            return y
        if node.op == "fused_linear":
            return values[0] @ values[1] + values[2]
        if node.op == "fused_linear_relu":
            return torch.relu(values[0] @ values[1] + values[2])
        raise ValueError(f"unsupported op {node.op!r}")


def _import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch backend requested but torch is not installed. "
            "Install with: python3 -m pip install torch"
        ) from exc
    return torch
