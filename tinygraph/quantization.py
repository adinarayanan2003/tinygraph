from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tinygraph.ir import Graph, TensorSpec
from tinygraph.runtime import make_random_feeds, run
from tinygraph.shape import infer_graph_shapes


@dataclass
class QuantizationSummary:
    original_constant_bytes: int
    quantized_constant_bytes: int
    memory_reduction_percent: float
    quantized_constants: list[str]


@dataclass
class QuantizedComparison:
    summary: QuantizationSummary
    max_abs_error: float
    mean_abs_error: float


def quantize_graph(graph: Graph) -> Graph:
    """Return a copy with eligible weight constants stored as symmetric int8."""
    infer_graph_shapes(graph)
    out = graph.clone()
    eligible = _eligible_weight_constants(out)
    for name in sorted(eligible):
        value = out.constants[name]
        if not np.issubdtype(value.dtype, np.floating):
            continue
        quantized, scale = quantize_array_int8(value)
        out.constants[name] = quantized
        out.quantization[name] = {
            "scheme": "symmetric_int8",
            "scale": float(scale),
            "zero_point": 0,
            "original_dtype": str(value.dtype),
            "original_nbytes": int(value.nbytes),
        }
        spec = out.tensor_specs[name]
        out.tensor_specs[name] = TensorSpec(name=spec.name, shape=spec.shape, dtype="int8")
    return out


def quantize_array_int8(value: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(value))) if value.size else 0.0
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    quantized = np.clip(np.round(value / scale), -127, 127).astype("int8")
    return quantized, scale


def dequantize_array(value: np.ndarray, metadata: dict) -> np.ndarray:
    if metadata.get("scheme") != "symmetric_int8":
        raise ValueError(f"unsupported quantization scheme {metadata.get('scheme')!r}")
    dtype = metadata.get("original_dtype", "float32")
    return (value.astype("float32") * float(metadata["scale"])).astype(dtype)


def dequantized_constants(graph: Graph) -> dict[str, np.ndarray]:
    constants: dict[str, np.ndarray] = {}
    for name, value in graph.constants.items():
        metadata = graph.quantization.get(name)
        constants[name] = dequantize_array(value, metadata) if metadata else value.copy()
    return constants


def quantization_summary(original: Graph, quantized: Graph) -> QuantizationSummary:
    original_bytes = sum(int(value.nbytes) for value in original.constants.values())
    quantized_bytes = 0
    for name, value in quantized.constants.items():
        quantized_bytes += int(value.nbytes)
    reduction = 0.0 if original_bytes == 0 else (1.0 - quantized_bytes / original_bytes) * 100.0
    return QuantizationSummary(
        original_constant_bytes=original_bytes,
        quantized_constant_bytes=quantized_bytes,
        memory_reduction_percent=reduction,
        quantized_constants=sorted(quantized.quantization),
    )


def compare_quantized(original: Graph, quantized: Graph, seed: int = 0) -> QuantizedComparison:
    feeds = make_random_feeds(original, seed)
    original_outputs = run(original, feeds)
    quantized_outputs = run(quantized, feeds)
    deltas = [
        np.abs(original_outputs[name].astype("float64") - quantized_outputs[name].astype("float64")).ravel()
        for name in original.outputs
    ]
    merged = np.concatenate(deltas) if deltas else np.array([0.0])
    return QuantizedComparison(
        summary=quantization_summary(original, quantized),
        max_abs_error=float(np.max(merged)),
        mean_abs_error=float(np.mean(merged)),
    )


def _eligible_weight_constants(graph: Graph) -> set[str]:
    eligible: set[str] = set()
    for node in graph.nodes:
        if node.op == "matmul" and len(node.inputs) == 2 and node.inputs[1] in graph.constants:
            eligible.add(node.inputs[1])
        if node.op in {"fused_linear", "fused_linear_relu"} and len(node.inputs) == 3 and node.inputs[1] in graph.constants:
            eligible.add(node.inputs[1])
    return eligible
