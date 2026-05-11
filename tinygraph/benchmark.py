from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import numpy as np

from tinygraph.backends import run_with_backend
from tinygraph.compiler import optimize
from tinygraph.ir import Graph
from tinygraph.runtime import make_random_feeds, run
from tinygraph.shape import infer_graph_shapes


@dataclass
class BenchResult:
    label: str
    runs: int
    p50_ms: float
    p95_ms: float
    min_ms: float
    node_count: int
    estimated_flops: int
    estimated_bytes: int
    max_abs_delta: float | None = None


def benchmark_graph(
    graph: Graph,
    runs_count: int = 50,
    seed: int = 0,
    backend: str = "numpy",
    device: str | None = "auto",
) -> tuple[BenchResult, BenchResult]:
    infer_graph_shapes(graph)
    feeds = make_random_feeds(graph, seed)
    optimized = optimize(graph).graph
    baseline_output = run_with_backend(graph, feeds, backend=backend, device=device)
    optimized_output = run_with_backend(optimized, feeds, backend=backend, device=device)
    delta = max(
        float(np.max(np.abs(baseline_output[name] - optimized_output[name])))
        for name in graph.outputs
    )
    label_prefix = backend if backend == "numpy" else f"{backend}:{device or 'auto'}"
    baseline = _bench_one(f"{label_prefix}:naive", graph, feeds, runs_count, backend, device)
    optimized_result = _bench_one(f"{label_prefix}:optimized", optimized, feeds, runs_count, backend, device)
    optimized_result.max_abs_delta = delta
    return baseline, optimized_result


def _bench_one(
    label: str,
    graph: Graph,
    feeds: dict[str, np.ndarray],
    runs_count: int,
    backend: str,
    device: str | None,
) -> BenchResult:
    timings: list[float] = []
    run_with_backend(graph, feeds, backend=backend, device=device)
    for _ in range(runs_count):
        start = time.perf_counter()
        run_with_backend(graph, feeds, backend=backend, device=device)
        timings.append((time.perf_counter() - start) * 1000)
    sorted_timings = sorted(timings)
    p95_index = min(len(sorted_timings) - 1, int(len(sorted_timings) * 0.95))
    return BenchResult(
        label=label,
        runs=runs_count,
        p50_ms=statistics.median(sorted_timings),
        p95_ms=sorted_timings[p95_index],
        min_ms=min(sorted_timings),
        node_count=len(graph.nodes),
        estimated_flops=estimate_flops(graph),
        estimated_bytes=estimate_bytes(graph),
    )


def estimate_flops(graph: Graph) -> int:
    infer_graph_shapes(graph)
    total = 0
    for node in graph.nodes:
        output = graph.tensor_specs[node.outputs[0]]
        if node.op in {"add", "mul", "relu", "gelu", "softmax"}:
            total += int(np.prod(output.shape))
        elif node.op == "matmul":
            left = graph.tensor_specs[node.inputs[0]]
            total += _matmul_flops(left.shape, graph.tensor_specs[node.inputs[1]].shape, output.shape)
        elif node.op in {"fused_linear", "fused_linear_relu"}:
            left = graph.tensor_specs[node.inputs[0]]
            total += _matmul_flops(left.shape, graph.tensor_specs[node.inputs[1]].shape, output.shape)
            total += int(np.prod(output.shape))
            if node.op == "fused_linear_relu":
                total += int(np.prod(output.shape))
        elif node.op in {"reshape", "transpose"}:
            total += 0
        elif node.op == "layernorm":
            total += 5 * int(np.prod(output.shape))
        elif node.op == "sum":
            total += int(np.prod(graph.tensor_specs[node.inputs[0]].shape))
    return total


def _matmul_flops(left_shape: tuple[int, ...], right_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> int:
    return int(2 * np.prod(output_shape, dtype=np.int64) * left_shape[-1])


def estimate_bytes(graph: Graph) -> int:
    infer_graph_shapes(graph)
    total = 0
    for node in graph.nodes:
        for name in node.inputs + node.outputs:
            if name in graph.tensor_specs:
                total += graph.tensor_specs[name].nbytes
    return total
