from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from tinygraph.backends.numpy_backend import NumPyBackend
from tinygraph.backends.torch_backend import TorchBackend
from tinygraph.ir import Graph
from tinygraph.runtime import make_random_feeds


@dataclass
class BackendComparison:
    backend: str
    device: str
    available: bool
    p50_ms: float | None = None
    p95_ms: float | None = None
    max_abs_delta: float | None = None
    mean_abs_delta: float | None = None
    error: str | None = None


def create_backend(name: str, device: str | None = None):
    if name == "numpy":
        return NumPyBackend()
    if name == "torch":
        return TorchBackend(device=device)
    raise ValueError(f"unknown backend {name!r}")


def run_with_backend(
    graph: Graph,
    feeds: dict[str, np.ndarray] | None = None,
    backend: str = "numpy",
    device: str | None = None,
) -> dict[str, np.ndarray]:
    return create_backend(backend, device=device).run(graph, feeds or {})


def compare_backends(
    graph: Graph,
    feeds: dict[str, np.ndarray] | None = None,
    backends: Iterable[str] = ("numpy", "torch"),
    device: str | None = "auto",
    runs_count: int = 20,
) -> list[BackendComparison]:
    feeds = feeds or make_random_feeds(graph)
    reference = run_with_backend(graph, feeds, backend="numpy")
    rows: list[BackendComparison] = []
    for backend_name in backends:
        backend_device = None if backend_name == "numpy" else device
        try:
            backend = create_backend(backend_name, device=backend_device)
            output = backend.run(graph, feeds)
            timings = _time_backend(backend, graph, feeds, runs_count)
            max_delta, mean_delta = _output_delta(reference, output)
            rows.append(
                BackendComparison(
                    backend=backend_name,
                    device=getattr(backend, "device_name", "cpu"),
                    available=True,
                    p50_ms=statistics.median(timings),
                    p95_ms=sorted(timings)[min(len(timings) - 1, int(len(timings) * 0.95))],
                    max_abs_delta=max_delta,
                    mean_abs_delta=mean_delta,
                )
            )
        except Exception as exc:
            rows.append(
                BackendComparison(
                    backend=backend_name,
                    device=str(backend_device or "cpu"),
                    available=False,
                    error=str(exc),
                )
            )
    return rows


def _time_backend(backend, graph: Graph, feeds: dict[str, np.ndarray], runs_count: int) -> list[float]:
    timings: list[float] = []
    backend.run(graph, feeds)
    backend.synchronize()
    for _ in range(runs_count):
        start = time.perf_counter()
        backend.run(graph, feeds)
        backend.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    return timings


def _output_delta(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> tuple[float, float]:
    deltas = [
        np.abs(reference[name].astype("float64") - candidate[name].astype("float64")).ravel()
        for name in reference
    ]
    merged = np.concatenate(deltas) if deltas else np.array([0.0])
    return float(np.max(merged)), float(np.mean(merged))
