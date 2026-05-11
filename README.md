# TinyGraph

TinyGraph is a small neural network compiler lab built with Python and NumPy. It is designed to make compiler concepts visible: graph IR, shape inference, node-by-node execution, optimization passes, op fusion, benchmarking, memory planning, and static reports.

## Quick Start

```bash
cd tinygraph
python3 -m tinygraph.cli inspect examples/mlp.json
python3 -m tinygraph.cli run examples/mlp.json
python3 -m tinygraph.cli optimize examples/mlp.json --out reports/mlp_optimized.json
python3 -m tinygraph.cli bench examples/mlp.json --runs 100
python3 -m tinygraph.cli report examples/mlp.json --out reports/mlp.html
python3 -m tinygraph.cli run examples/attention_block.json --optimized
python3 -m tinygraph.cli run examples/attention_block.json --backend torch --device cpu
python3 -m tinygraph.cli bench examples/attention_block.json --compare-backends --device cpu
python3 -m tinygraph.cli quantize examples/mlp.json --out reports/mlp_int8.json
python3 -m tinygraph.cli compare examples/mlp.json reports/mlp_int8.json
python3 -m tinygraph.cli report examples/mlp.json --quantize int8 --out reports/mlp_quant.html
```

Install as a local package if you want the `tinygraph` command:

```bash
python3 -m pip install -e .
tinygraph inspect examples/mlp.json
```

## What It Implements

- A minimal graph IR with inputs, constants, nodes, outputs, and tensor specs.
- Shape inference for MLP and attention-style ops including `matmul`, `reshape`, `transpose`, `softmax`, `gelu`, `layernorm`, and fused linear ops.
- A NumPy runtime for naive graph execution.
- A PyTorch backend with CPU/MPS/CUDA device selection.
- Optimization passes:
  - constant folding
  - identity op removal
  - `matmul + add + relu` fusion
  - `matmul + add` fusion
  - dead node elimination
- Correctness checks comparing naive and optimized outputs.
- Benchmarks for runtime, node count, estimated FLOPs, and estimated bytes moved.
- Backend comparison for NumPy vs PyTorch runtime and output drift.
- Memory planning estimates based on tensor lifetimes.
- Static HTML reports showing benchmark tables, pass concepts, pass-by-pass graph snapshots, and before/after diagrams.
- Weight-only symmetric INT8 quantization with memory savings and output drift metrics.

## Python API

```python
import numpy as np

from tinygraph import Graph, optimize, run
from tinygraph.ops import add, matmul, relu

graph = Graph("demo")
x = graph.input("x", shape=(4, 4))
w = graph.const("w", np.eye(4, dtype="float32"))
b = graph.const("b", np.zeros((1, 4), dtype="float32"))

y = relu(add(matmul(x, w), b))
graph.output(y)

feeds = {"x": np.ones((4, 4), dtype="float32")}
optimized = optimize(graph).graph
outputs = run(optimized, feeds)
print(outputs[y.name])
```

## CLI

```bash
tinygraph inspect examples/mlp.json
tinygraph run examples/mlp.json --optimized
tinygraph run examples/attention_block.json --backend torch --device cpu
tinygraph optimize examples/mlp.json --out reports/mlp_optimized.json
tinygraph bench examples/mlp.json --runs 100
tinygraph bench examples/attention_block.json --compare-backends --device cpu
tinygraph report examples/mlp.json --out reports/mlp.html
tinygraph report examples/attention_block.json --compare-backends --out reports/backend_compare.html
tinygraph quantize examples/mlp.json --out reports/mlp_int8.json
tinygraph compare examples/mlp.json reports/mlp_int8.json
tinygraph report examples/mlp.json --quantize int8 --out reports/mlp_quant.html
tinygraph memory examples/mlp.json
```

## Project Map

```text
tinygraph/
  tinygraph/
    ir.py              # Graph, Node, Tensor, TensorSpec
    shape.py           # Shape inference
    runtime.py         # NumPy interpreter
    passes.py          # Compiler optimization passes
    compiler.py        # Pass orchestration
    benchmark.py       # Runtime and graph metrics
    memory.py          # Tensor lifetime memory estimate
    backends/          # NumPy and PyTorch execution backends
    quantization.py    # INT8 weight quantization and drift metrics
    report.py          # Static HTML report writer
    serialization.py   # JSON graph load/save
    cli.py             # Command line interface
  examples/
    linear.json
    mlp.json
    gelu_mlp.json
    attention_block.json
    layernorm.json
  tests/
```

## Definition of Done

- Example graphs execute with deterministic random inputs.
- Optimized graphs match naive outputs within numeric tolerance.
- Fusion reduces the MLP graph node count.
- Attention-like examples run with batched matmul, transpose, softmax, and reshape.
- PyTorch backend matches NumPy outputs on MLP and attention-like examples.
- INT8 quantized graphs report constant memory savings and bounded output drift.
- Benchmarks and memory planning are available from the CLI.
- `python3 -m unittest discover -s tests` passes.

## Future Work

- Per-channel quantization and activation quantization.
- Triton lowering for fused kernels.
- ONNX import for small feed-forward models.
- Interactive graph visualization.
- Autodiff and training support.
