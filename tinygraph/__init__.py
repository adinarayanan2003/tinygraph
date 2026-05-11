"""TinyGraph: a compact neural network compiler lab."""

from tinygraph.compiler import OptimizationResult, optimize
from tinygraph.backends import compare_backends, run_with_backend
from tinygraph.ir import Graph, Node, Tensor, TensorSpec
from tinygraph.quantization import QuantizationSummary, QuantizedComparison, compare_quantized, quantize_graph
from tinygraph.runtime import run

__all__ = [
    "Graph",
    "Node",
    "OptimizationResult",
    "QuantizationSummary",
    "QuantizedComparison",
    "Tensor",
    "TensorSpec",
    "compare_quantized",
    "compare_backends",
    "optimize",
    "quantize_graph",
    "run",
    "run_with_backend",
]
