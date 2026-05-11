# TinyGraph Backends

TinyGraph now separates the graph IR from the execution engine.

```text
Graph IR
   |
   +----> NumPy backend
   |
   +----> PyTorch backend
```

This mirrors a real compiler idea: the front half of the compiler represents and rewrites the program, while the back half chooses how to execute it.

## NumPy Backend

The NumPy backend is the reference implementation. It is simple, deterministic, and used as the correctness baseline for backend comparison.

```bash
python3 -m tinygraph.cli run examples/attention_block.json --backend numpy
```

## PyTorch Backend

The PyTorch backend executes the same TinyGraph IR with Torch tensors. It supports the current v3 op set:

- `matmul`, `add`, `mul`
- `relu`, `gelu`
- `reshape`, `transpose`, `softmax`
- `sum`, `layernorm`
- `fused_linear`, `fused_linear_relu`

```bash
python3 -m tinygraph.cli run examples/attention_block.json --backend torch --device cpu
python3 -m tinygraph.cli run examples/attention_block.json --backend torch --device mps
```

`--device auto` chooses `mps`, then `cuda`, then `cpu`.

## Backend Comparison

Backend comparison runs NumPy as the reference and compares other backends against it.

```bash
python3 -m tinygraph.cli bench examples/attention_block.json --compare-backends --device cpu
python3 -m tinygraph.cli report examples/attention_block.json --compare-backends --out reports/backend_compare.html
```

Metrics:

- backend name
- selected device
- availability
- p50 runtime
- p95 runtime
- max absolute output delta vs NumPy
- mean absolute output delta vs NumPy

## Quantized Graphs

Quantized constants are dequantized before backend math. That means both NumPy and PyTorch execute the same numerical graph, while TinyGraph still reports INT8 storage savings.

## Future Path

The backend split prepares TinyGraph for Triton:

```text
Graph IR
   |
   v
fusion pass
   |
   v
fused op
   |
   +----> NumPy reference
   +----> PyTorch eager
   +----> Triton kernel
```
