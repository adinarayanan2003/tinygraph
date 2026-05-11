# TinyGraph Quantization

TinyGraph v2 adds weight-only symmetric INT8 quantization. The goal is educational: make the memory and accuracy tradeoff visible in the compiler pipeline.

## What Gets Quantized

Only floating-point constants used as matrix weights are quantized:

```text
x -> matmul(x, w) -> ...
```

or after fusion:

```text
x, w, b -> fused_linear_relu -> ...
```

Bias constants and activations stay in floating point.

## Symmetric INT8 Formula

For a floating-point weight array:

```text
scale = max(abs(weight)) / 127
q = clip(round(weight / scale), -127, 127).astype(int8)
```

At runtime:

```text
dequantized = q.astype(float32) * scale
```

If all values are zero, TinyGraph uses `scale = 1.0` so quantization is well-defined.

## Why This Is a Compiler Pass

Quantization changes graph storage and metadata:

```text
fp32 graph
    |
    v
quantize_graph()
    |
    v
int8 constants + scale metadata
    |
    v
runtime dequantization + output drift report
```

This keeps quantization inside the compiler model instead of making it a separate preprocessing script.

## Commands

```bash
python3 -m tinygraph.cli quantize examples/mlp.json --out reports/mlp_int8.json
python3 -m tinygraph.cli compare examples/mlp.json reports/mlp_int8.json
python3 -m tinygraph.cli report examples/mlp.json --quantize int8 --out reports/mlp_quant.html
```

The report includes:

- original constant memory
- quantized constant memory
- memory reduction percentage
- max absolute output error
- mean absolute output error
- original, optimized, and quantized graph views

## Current Limits

- Weight-only INT8.
- Per-tensor scale, not per-channel scale.
- Runtime dequantizes weights before math.
- No activation quantization.
- No hardware-specific int8 kernels yet.
