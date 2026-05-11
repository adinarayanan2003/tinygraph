from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tinygraph.backends import compare_backends, run_with_backend
from tinygraph.benchmark import benchmark_graph
from tinygraph.compiler import optimize
from tinygraph.memory import plan_memory
from tinygraph.quantization import compare_quantized, quantize_graph
from tinygraph.report import write_report
from tinygraph.runtime import make_random_feeds, run
from tinygraph.serialization import graph_summary, load_graph, save_graph


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinygraph")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Print graph structure and inferred shapes.")
    inspect_parser.add_argument("graph")

    run_parser = subparsers.add_parser("run", help="Run a graph with deterministic random inputs.")
    run_parser.add_argument("graph")
    run_parser.add_argument("--optimized", action="store_true")
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    run_parser.add_argument("--device", default="auto")

    optimize_parser = subparsers.add_parser("optimize", help="Optimize a graph and write JSON.")
    optimize_parser.add_argument("graph")
    optimize_parser.add_argument("--out", required=True)

    bench_parser = subparsers.add_parser("bench", help="Benchmark naive vs optimized execution.")
    bench_parser.add_argument("graph")
    bench_parser.add_argument("--runs", type=int, default=50)
    bench_parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    bench_parser.add_argument("--device", default="auto")
    bench_parser.add_argument("--compare-backends", action="store_true")

    report_parser = subparsers.add_parser("report", help="Write a static HTML compiler report.")
    report_parser.add_argument("graph")
    report_parser.add_argument("--out", required=True)
    report_parser.add_argument("--runs", type=int, default=50)
    report_parser.add_argument("--quantize", choices=["int8"], help="Include a quantized graph comparison.")
    report_parser.add_argument("--compare-backends", action="store_true")
    report_parser.add_argument("--device", default="auto")

    quantize_parser = subparsers.add_parser("quantize", help="Write a weight-only INT8 quantized graph.")
    quantize_parser.add_argument("graph")
    quantize_parser.add_argument("--out", required=True)

    compare_parser = subparsers.add_parser("compare", help="Compare an original graph with a quantized graph.")
    compare_parser.add_argument("original")
    compare_parser.add_argument("quantized")
    compare_parser.add_argument("--seed", type=int, default=0)

    memory_parser = subparsers.add_parser("memory", help="Print tensor lifetime memory estimate.")
    memory_parser.add_argument("graph")

    args = parser.parse_args(argv)

    if args.command == "inspect":
        graph = load_graph(args.graph)
        print(graph_summary(graph))
        return 0

    if args.command == "run":
        graph = load_graph(args.graph)
        if args.optimized:
            graph = optimize(graph).graph
        feeds = make_random_feeds(graph, args.seed)
        outputs = run_with_backend(graph, feeds, backend=args.backend, device=args.device)
        for name, value in outputs.items():
            print(f"{name}: shape={value.shape}, dtype={value.dtype}, mean={float(np.mean(value)):.6f}")
        return 0

    if args.command == "optimize":
        graph = load_graph(args.graph)
        result = optimize(graph)
        save_graph(result.graph, args.out)
        for report in result.reports:
            status = "changed" if report.changed else "same"
            print(f"{report.name}: {report.before_nodes} -> {report.after_nodes} ({status})")
        print(f"wrote {args.out}")
        return 0

    if args.command == "bench":
        graph = load_graph(args.graph)
        if args.compare_backends:
            rows = compare_backends(graph, runs_count=args.runs, device=args.device)
            print("backend,device,available,p50_ms,p95_ms,max_abs_delta,mean_abs_delta,error")
            for row in rows:
                print(_backend_row(row))
            return 0
        baseline, optimized = benchmark_graph(graph, runs_count=args.runs, backend=args.backend, device=args.device)
        print("mode,nodes,p50_ms,p95_ms,min_ms,estimated_flops,estimated_bytes,max_abs_delta")
        print(_bench_row(baseline))
        print(_bench_row(optimized))
        return 0

    if args.command == "report":
        graph = load_graph(args.graph)
        path = write_report(
            graph,
            Path(args.out),
            runs=args.runs,
            quantize=args.quantize,
            compare_execution_backends=args.compare_backends,
            backend_device=args.device,
        )
        print(f"wrote {path}")
        return 0

    if args.command == "quantize":
        graph = load_graph(args.graph)
        quantized = quantize_graph(graph)
        save_graph(quantized, args.out)
        summary = compare_quantized(graph, quantized).summary
        print(f"wrote {args.out}")
        print(f"quantized_constants={','.join(summary.quantized_constants)}")
        print(f"original_constant_bytes={summary.original_constant_bytes}")
        print(f"quantized_constant_bytes={summary.quantized_constant_bytes}")
        print(f"memory_reduction_percent={summary.memory_reduction_percent:.2f}")
        return 0

    if args.command == "compare":
        original = load_graph(args.original)
        quantized = load_graph(args.quantized)
        comparison = compare_quantized(original, quantized, seed=args.seed)
        print(f"original_constant_bytes={comparison.summary.original_constant_bytes}")
        print(f"quantized_constant_bytes={comparison.summary.quantized_constant_bytes}")
        print(f"memory_reduction_percent={comparison.summary.memory_reduction_percent:.2f}")
        print(f"max_abs_error={comparison.max_abs_error:.8f}")
        print(f"mean_abs_error={comparison.mean_abs_error:.8f}")
        return 0

    if args.command == "memory":
        graph = load_graph(args.graph)
        result = plan_memory(graph)
        print(f"naive_peak_bytes={result.naive_peak_bytes}")
        print(f"planned_peak_bytes={result.planned_peak_bytes}")
        for name, (start, end) in sorted(result.tensor_lifetimes.items()):
            print(f"{name}: {start}->{end}")
        return 0

    raise AssertionError(f"unhandled command {args.command}")


def _bench_row(result) -> str:
    delta = "" if result.max_abs_delta is None else f"{result.max_abs_delta:.8f}"
    return (
        f"{result.label},{result.node_count},{result.p50_ms:.4f},{result.p95_ms:.4f},"
        f"{result.min_ms:.4f},{result.estimated_flops},{result.estimated_bytes},{delta}"
    )


def _backend_row(row) -> str:
    p50 = "" if row.p50_ms is None else f"{row.p50_ms:.4f}"
    p95 = "" if row.p95_ms is None else f"{row.p95_ms:.4f}"
    max_delta = "" if row.max_abs_delta is None else f"{row.max_abs_delta:.8f}"
    mean_delta = "" if row.mean_abs_delta is None else f"{row.mean_abs_delta:.8f}"
    error = "" if row.error is None else row.error.replace(",", ";")
    return f"{row.backend},{row.device},{row.available},{p50},{p95},{max_delta},{mean_delta},{error}"


if __name__ == "__main__":
    raise SystemExit(main())
