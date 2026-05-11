# TinyGraph Ops

TinyGraph supports a deliberately small NumPy-backed op set. Every op has shape inference, runtime execution, JSON serialization, and report support.

| Op | Shape Rule | NumPy Equivalent |
| --- | --- | --- |
| `matmul` | Broadcast batch dims, output `(..., m, n)` | `a @ b` |
| `add` | NumPy broadcasting | `a + b` |
| `mul` | NumPy broadcasting | `a * b` |
| `relu` | Same as input | `np.maximum(x, 0)` |
| `gelu` | Same as input | tanh GELU approximation |
| `reshape` | Same element count, one optional `-1` | `np.reshape(x, shape)` |
| `transpose` | Permutes axes, or reverses axes when omitted | `np.transpose(x, axes)` |
| `softmax` | Same as input | stable exp/sum over axis |
| `sum` | Removes or keeps reduced axes | `np.sum(...)` |
| `layernorm` | Same as input | normalize over last axis |
| `fused_linear` | Same as `matmul(x, w) + b` | fused logical op |
| `fused_linear_relu` | Same as `relu(matmul(x, w) + b)` | fused logical op |

## JSON Attributes

`reshape`:

```json
{"op": "reshape", "inputs": ["x"], "outputs": ["y"], "attrs": {"shape": [2, 12]}}
```

`transpose`:

```json
{"op": "transpose", "inputs": ["x"], "outputs": ["y"], "attrs": {"axes": [0, 2, 1]}}
```

`softmax`:

```json
{"op": "softmax", "inputs": ["scores"], "outputs": ["weights"], "attrs": {"axis": -1}}
```

## Example Graphs

- `examples/mlp.json`: ReLU MLP that demonstrates `fused_linear_relu`.
- `examples/gelu_mlp.json`: GELU MLP that demonstrates `fused_linear`.
- `examples/attention_block.json`: simplified attention-like fragment using batched matmul, transpose, softmax, and reshape.
