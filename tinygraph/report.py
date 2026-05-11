from __future__ import annotations

from html import escape
from pathlib import Path

from tinygraph.backends import compare_backends
from tinygraph.benchmark import benchmark_graph
from tinygraph.compiler import optimize
from tinygraph.ir import Graph
from tinygraph.memory import plan_memory
from tinygraph.quantization import compare_quantized, quantize_graph
from tinygraph.serialization import graph_summary


def write_report(
    graph: Graph,
    output_path: str | Path,
    runs: int = 50,
    quantize: str | None = None,
    compare_execution_backends: bool = False,
    backend_device: str | None = "auto",
) -> Path:
    optimized = optimize(graph)
    baseline_bench, optimized_bench = benchmark_graph(graph, runs_count=runs)
    memory = plan_memory(optimized.graph)
    quantized_graph = None
    quantized_comparison = None
    if quantize == "int8":
        quantized_graph = quantize_graph(optimized.graph)
        quantized_comparison = compare_quantized(optimized.graph, quantized_graph)
    backend_rows = compare_backends(graph, runs_count=runs, device=backend_device) if compare_execution_backends else []
    html = _html(
        graph,
        optimized.graph,
        baseline_bench,
        optimized_bench,
        optimized.reports,
        memory,
        quantized_graph,
        quantized_comparison,
        backend_rows,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return path


def _html(
    graph: Graph,
    optimized: Graph,
    baseline_bench,
    optimized_bench,
    pass_reports,
    memory,
    quantized_graph,
    quantized_comparison,
    backend_rows,
) -> str:
    passes = "\n".join(
        f"<tr><td>{escape(report.name)}</td><td>{report.before_nodes}</td><td>{report.after_nodes}</td><td>{report.changed}</td></tr>"
        for report in pass_reports
    )
    concepts = "\n".join(
        f"<tr><td>{escape(report.name)}</td><td>{escape(_pass_concept(report.name))}</td></tr>"
        for report in pass_reports
    )
    snapshots = "\n".join(
        f"""
    <section>
      <h2>After {escape(report.name)}</h2>
      <pre>{escape(ascii_graph(report.graph))}</pre>
    </section>
"""
        for report in pass_reports
    )
    quant_section = _quantization_section(quantized_graph, quantized_comparison)
    backend_section = _backend_section(backend_rows)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TinyGraph Report - {escape(graph.name)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; line-height: 1.45; color: #17202a; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #d8dee9; border-radius: 8px; padding: 16px; background: #fbfcfe; }}
    .wide {{ grid-column: 1 / -1; }}
    pre {{ overflow-x: auto; background: #111827; color: #e5e7eb; padding: 12px; border-radius: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px; text-align: left; }}
    th {{ background: #eef2f7; }}
  </style>
</head>
<body>
  <h1>TinyGraph Report: {escape(graph.name)}</h1>
  <div class="grid">
    <section class="card">
      <h2>Benchmark</h2>
      <table>
        <tr><th>Mode</th><th>Nodes</th><th>p50 ms</th><th>p95 ms</th><th>FLOPs</th><th>Bytes</th></tr>
        <tr><td>{baseline_bench.label}</td><td>{baseline_bench.node_count}</td><td>{baseline_bench.p50_ms:.4f}</td><td>{baseline_bench.p95_ms:.4f}</td><td>{baseline_bench.estimated_flops}</td><td>{baseline_bench.estimated_bytes}</td></tr>
        <tr><td>{optimized_bench.label}</td><td>{optimized_bench.node_count}</td><td>{optimized_bench.p50_ms:.4f}</td><td>{optimized_bench.p95_ms:.4f}</td><td>{optimized_bench.estimated_flops}</td><td>{optimized_bench.estimated_bytes}</td></tr>
      </table>
      <p>Max absolute output delta: {optimized_bench.max_abs_delta:.8f}</p>
    </section>
    <section class="card">
      <h2>Memory Plan</h2>
      <table>
        <tr><th>Naive peak bytes</th><th>Planned peak bytes</th></tr>
        <tr><td>{memory.naive_peak_bytes}</td><td>{memory.planned_peak_bytes}</td></tr>
      </table>
    </section>
  </div>
  <h2>Passes</h2>
  <table>
    <tr><th>Pass</th><th>Before nodes</th><th>After nodes</th><th>Changed</th></tr>
    {passes}
  </table>
  <h2>Concepts</h2>
  <table>
    <tr><th>Pass</th><th>What it demonstrates</th></tr>
    {concepts}
  </table>
  {backend_section}
  {quant_section}
  <div class="grid">
    <section>
      <h2>Before</h2>
      <pre>{escape(graph_summary(graph))}</pre>
    </section>
    <section>
      <h2>After</h2>
      <pre>{escape(graph_summary(optimized))}</pre>
    </section>
    <section>
      <h2>Before Diagram</h2>
      <pre>{escape(ascii_graph(graph))}</pre>
    </section>
    <section>
      <h2>After Diagram</h2>
      <pre>{escape(ascii_graph(optimized))}</pre>
    </section>
    {snapshots}
  </div>
</body>
</html>
"""


def _quantization_section(quantized_graph, comparison) -> str:
    if quantized_graph is None or comparison is None:
        return ""
    summary = comparison.summary
    constants = ", ".join(summary.quantized_constants) or "none"
    return f"""
  <h2>INT8 Quantization</h2>
  <div class="grid">
    <section class="card">
      <h2>Storage</h2>
      <table>
        <tr><th>Original constant bytes</th><th>Quantized constant bytes</th><th>Reduction</th></tr>
        <tr><td>{summary.original_constant_bytes}</td><td>{summary.quantized_constant_bytes}</td><td>{summary.memory_reduction_percent:.2f}%</td></tr>
      </table>
      <p>Quantized constants: {escape(constants)}</p>
    </section>
    <section class="card">
      <h2>Output Drift</h2>
      <table>
        <tr><th>Max abs error</th><th>Mean abs error</th></tr>
        <tr><td>{comparison.max_abs_error:.8f}</td><td>{comparison.mean_abs_error:.8f}</td></tr>
      </table>
    </section>
    <section class="wide">
      <h2>Quantized Graph</h2>
      <pre>{escape(graph_summary(quantized_graph))}</pre>
    </section>
    <section class="wide">
      <h2>Quantized Diagram</h2>
      <pre>{escape(ascii_graph(quantized_graph))}</pre>
    </section>
  </div>
"""


def _backend_section(rows) -> str:
    if not rows:
        return ""
    table_rows = "\n".join(
        f"<tr><td>{escape(row.backend)}</td><td>{escape(row.device)}</td><td>{row.available}</td>"
        f"<td>{_fmt(row.p50_ms)}</td><td>{_fmt(row.p95_ms)}</td>"
        f"<td>{_fmt(row.max_abs_delta)}</td><td>{_fmt(row.mean_abs_delta)}</td>"
        f"<td>{escape(row.error or '')}</td></tr>"
        for row in rows
    )
    return f"""
  <h2>Backend Comparison</h2>
  <table>
    <tr><th>Backend</th><th>Device</th><th>Available</th><th>p50 ms</th><th>p95 ms</th><th>Max delta vs NumPy</th><th>Mean delta vs NumPy</th><th>Error</th></tr>
    {table_rows}
  </table>
"""


def _fmt(value) -> str:
    return "" if value is None else f"{value:.8f}"


def ascii_graph(graph: Graph) -> str:
    lines = [f"{graph.name}:"]
    for node in graph.nodes:
        left = ", ".join(_format_tensor(graph, name) for name in node.inputs)
        right = ", ".join(_format_tensor(graph, name) for name in node.outputs)
        attrs = _format_attrs(node.attrs)
        suffix = f" {attrs}" if attrs else ""
        lines.append(f"  ({left})")
        lines.append(f"      |")
        lines.append(f"      v")
        lines.append(f"  [{node.op}{suffix}] -> ({right})")
    if graph.outputs:
        lines.append(f"  outputs: {', '.join(graph.outputs)}")
    return "\n".join(lines)


def _format_tensor(graph: Graph, name: str) -> str:
    if name in graph.quantization:
        return f"{name}:int8*"
    spec = graph.tensor_specs.get(name)
    return f"{name}:{spec.dtype}" if spec else name


def _format_attrs(attrs: dict) -> str:
    if not attrs:
        return ""
    return "{" + ", ".join(f"{key}={value}" for key, value in attrs.items()) + "}"


def _pass_concept(name: str) -> str:
    concepts = {
        "constant_fold": "Compile-time evaluation removes work whose inputs are already known.",
        "identity_op_removal": "No-op algebraic rewrites shrink the graph without changing outputs.",
        "fuse_linear_relu": "Pattern fusion combines matmul, bias add, and activation into one logical op.",
        "fuse_linear": "Pattern fusion combines matmul and bias add when there is no activation.",
        "dead_node_elimination": "Liveness analysis removes nodes that cannot affect graph outputs.",
    }
    return concepts.get(name, "Compiler pass snapshot.")
