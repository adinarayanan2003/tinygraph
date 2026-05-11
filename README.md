# TinyGraph

TinyGraph is a developer-maintained research project for studying how neural network compilers work. It implements a small but complete compiler pipeline: graph IR, shape inference, optimization passes, quantization, backend execution, benchmarking, and HTML reports.

The project is intentionally compact. Its goal is not to replace PyTorch, XLA, TVM, or TensorRT; it is to make the core ideas behind those systems inspectable in a codebase small enough to read end to end.

## Research Focus

TinyGraph explores four questions:

1. How can tensor programs be represented as a graph IR?
2. What compiler passes are useful for neural-network-style graphs?
3. How do optimized and quantized graphs compare numerically with the original graph?
4. How does the same graph behave across NumPy and PyTorch backends?

The current implementation covers MLP-style and attention-style graph fragments, including batched matmul, transpose, softmax, GELU, fused linear ops, INT8 weight quantization, and backend comparison.

## Reports

Generated reports are checked in under `docs/reports/` as reproducible examples.

| Report | What It Shows |
| --- | --- |
| [MLP quantization report](docs/reports/mlp_quant.html) | `matmul + add + relu` fusion, INT8 weight quantization, output drift, memory savings |
| [GELU MLP report](docs/reports/gelu_mlp_quant.html) | `matmul + add` fusion into `fused_linear`, GELU execution, quantization metrics |
| [Attention backend report](docs/reports/attention_backend_quant.html) | attention-style graph, pass snapshots, INT8 weights, NumPy vs PyTorch backend comparison |

To view them after cloning, open the HTML files directly in a browser.

## Implemented System

```text
JSON / Python graph
        |
        v
Graph IR
        |
        v
Validation + shape inference
        |
        v
Compiler passes
        |
        +----> constant folding
        +----> identity removal
        +----> matmul + add fusion
        +----> matmul + add + relu fusion
        +----> dead node elimination
        |
        v
Execution backends
        |
        +----> NumPy reference backend
        +----> PyTorch backend
        |
        v
Benchmarks + reports
```

Core capabilities:

- Graph IR with inputs, constants, nodes, outputs, and tensor specs.
- Shape inference for MLP and attention-style ops.
- NumPy backend as the correctness reference.
- PyTorch backend with `cpu`, `mps`, `cuda`, and `auto` device selection.
- Compiler passes for constant folding, identity removal, fusion, and dead node elimination.
- Weight-only symmetric INT8 quantization with output-drift metrics.
- Benchmarking for runtime, FLOPs, bytes moved, and backend comparison.
- Static HTML reports with pass concepts, pass-by-pass graph snapshots, quantization metrics, and backend tables.

## Supported Ops

| Op | Notes |
| --- | --- |
| `matmul` | Rank >= 2, NumPy-style batched matmul |
| `add`, `mul` | NumPy broadcasting |
| `relu`, `gelu` | Elementwise activations |
| `reshape`, `transpose` | Shape-transform ops |
| `softmax` | Stable softmax over configurable axis |
| `sum` | Axis reduction |
| `layernorm` | Last-axis normalization |
| `fused_linear` | Logical fusion of `matmul + add` |
| `fused_linear_relu` | Logical fusion of `matmul + add + relu` |

See [docs/OPS.md](docs/OPS.md) for shape rules and JSON examples.

## Setup

Use a virtual environment. PyTorch is a project dependency.

```bash
git clone https://github.com/adinarayanan2003/tinygraph.git
cd tinygraph
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Run the test suite:

```bash
python -m unittest discover -s tests
```

Expected result:

```text
Ran 20 tests

OK
```

## Usage

Inspect and run graphs:

```bash
tinygraph inspect examples/mlp.json
tinygraph run examples/mlp.json --optimized
tinygraph run examples/attention_block.json --backend torch --device cpu
```

Optimize and benchmark:

```bash
tinygraph optimize examples/gelu_mlp.json --out reports/gelu_mlp_optimized.json
tinygraph bench examples/attention_block.json --runs 100
tinygraph bench examples/attention_block.json --compare-backends --device cpu
```

Quantize and compare:

```bash
tinygraph quantize examples/mlp.json --out reports/mlp_int8.json
tinygraph compare examples/mlp.json reports/mlp_int8.json
```

Generate reports:

```bash
tinygraph report examples/mlp.json --quantize int8 --out reports/mlp_quant.html
tinygraph report examples/attention_block.json --compare-backends --quantize int8 --out reports/backend_compare.html
```

## Example Graphs

| Example | Purpose |
| --- | --- |
| `examples/linear.json` | Minimal linear layer |
| `examples/mlp.json` | ReLU MLP with `fused_linear_relu` opportunity |
| `examples/gelu_mlp.json` | GELU MLP with `fused_linear` opportunity |
| `examples/attention_block.json` | Attention-style fragment with batched matmul, transpose, softmax, reshape |
| `examples/layernorm.json` | LayerNorm and elementwise ops |

## Repository Layout

```text
tinygraph/
  tinygraph/
    ir.py              # Graph, Node, Tensor, TensorSpec
    shape.py           # Shape inference
    runtime.py         # Compatibility NumPy runtime wrapper
    backends/          # NumPy and PyTorch execution backends
    passes.py          # Compiler optimization passes
    compiler.py        # Pass orchestration and pass snapshots
    benchmark.py       # Runtime and graph metrics
    memory.py          # Tensor lifetime memory estimates
    quantization.py    # INT8 weight quantization and drift metrics
    report.py          # Static HTML report writer
    serialization.py   # JSON graph load/save
    cli.py             # Command line interface
  examples/            # Reproducible graph fixtures
  tests/               # Unit and CLI tests
  docs/                # Design docs and checked-in report artifacts
```

## Documentation

- [System design](SYSTEM_DESIGN.md)
- [Backend design](docs/BACKENDS.md)
- [Supported ops](docs/OPS.md)
- [Quantization notes](docs/QUANTIZATION.md)

## Current Limitations

- No automatic differentiation or training support.
- No ONNX import yet.
- Quantization is weight-only INT8 with per-tensor scale.
- PyTorch backend uses eager execution, not `torch.compile`.
- HTML graph rendering is static, not an interactive graph viewer.

## Roadmap

- Triton lowering for fused kernels.
- ONNX import for small real model graphs.
- Per-channel and activation quantization.
- Interactive graph viewer for pass-by-pass diffs.
- Autodiff and training experiments.
