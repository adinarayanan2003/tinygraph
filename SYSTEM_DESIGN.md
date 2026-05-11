# TinyGraph System Design

TinyGraph is a compact neural network compiler lab. Its purpose is to make the compiler pipeline visible end-to-end: graph construction, validation, shape inference, execution, optimization, benchmarking, memory planning, and report generation.

## 1. System Overview

```text
              +--------------------+
              | Python API / JSON  |
              | graph definition   |
              +---------+----------+
                        |
                        v
              +--------------------+
              | Graph IR           |
              | inputs/constants   |
              | nodes/outputs      |
              +---------+----------+
                        |
                        v
      +-----------------+-----------------+
      |                                   |
      v                                   v
+-------------+                    +--------------+
| Validation  |                    | Shape        |
| pass        |                    | inference    |
+------+------+                    +------+-------+
       |                                  |
       +-----------------+----------------+
                         |
                         v
              +--------------------+
              | Compiler passes    |
              | fold/fuse/DCE      |
              +---------+----------+
                        |
                        v
              +--------------------+
              | NumPy runtime      |
              | node execution     |
              +---------+----------+
                        |
                        v
      +-----------------+-----------------+
      |                                   |
      v                                   v
+-------------+                    +--------------+
| Benchmarks  |                    | HTML report  |
| metrics     |                    | + summaries  |
+-------------+                    +--------------+
```

TinyGraph has one core invariant: every optimization must preserve graph output values within numeric tolerance.

## 2. Major Components

| Component | Responsibility | Main Module |
| --- | --- | --- |
| Graph IR | Stores inputs, constants, nodes, outputs, and tensor specs | `tinygraph/ir.py` |
| Ops API | Builds graph nodes from Python calls | `tinygraph/ops.py` |
| Serialization | Loads and saves JSON graph files | `tinygraph/serialization.py` |
| Validation | Rejects malformed graphs before execution/optimization | `tinygraph/validation.py` |
| Shape Inference | Computes tensor shapes and validates op compatibility, including batched matmul | `tinygraph/shape.py` |
| Runtime | Executes graph nodes with NumPy | `tinygraph/runtime.py` |
| Backends | Dispatches TinyGraph IR to NumPy or PyTorch | `tinygraph/backends/` |
| Compiler | Runs optimization passes in order | `tinygraph/compiler.py` |
| Passes | Implements constant folding, identity removal, linear fusion, DCE | `tinygraph/passes.py` |
| Quantization | Stores eligible weight constants as INT8 plus scale metadata | `tinygraph/quantization.py` |
| Benchmarking | Measures latency and estimates FLOPs/bytes | `tinygraph/benchmark.py` |
| Memory Planning | Estimates tensor lifetimes and peak intermediate memory | `tinygraph/memory.py` |
| Reporting | Emits static HTML reports | `tinygraph/report.py` |
| CLI | Exposes inspect/run/optimize/bench/report/memory commands | `tinygraph/cli.py` |

## 3. Graph IR

TinyGraph uses a simple directed acyclic graph model.

```text
Graph
  name: str
  inputs:    name -> TensorSpec
  constants: name -> ndarray
  nodes:     ordered list[Node]
  outputs:   list[tensor_name]
  specs:     name -> TensorSpec

TensorSpec
  name:  str
  shape: tuple[int, ...]
  dtype: str

Node
  op:      str
  inputs:  list[tensor_name]
  outputs: list[tensor_name]
  attrs:   dict
```

Example MLP graph:

```text
        x
        |
        v
   +----------+       w1
   | matmul   |<------+
   +----+-----+
        |
      h_mm       b1
        |        |
        v        v
      +----------+
      | add      |
      +----+-----+
           |
        h_bias
           |
           v
      +----------+
      | relu     |
      +----+-----+
           |
           h
           |
           v
   +----------+       w2
   | matmul   |<------+
   +----+-----+
        |
     out_mm      b2
        |        |
        v        v
      +----------+
      | add      |
      +----+-----+
           |
        logits
```

## 4. Compiler Pipeline

The compiler pipeline is intentionally explicit. Every pass receives a graph and returns a graph.

```text
load/build graph
      |
      v
+------------+
| validate   |
+-----+------+
      |
      v
+------------+
| infer      |
| shapes     |
+-----+------+
      |
      v
+-------------------+
| constant folding  |
+---------+---------+
          |
          v
+-------------------+
| identity removal  |
+---------+---------+
          |
          v
+-------------------+
| linear+relu fusion|
+---------+---------+
          |
          v
+-------------------+
| linear fusion     |
+---------+---------+
          |
          v
+-------------------+
| dead node elim    |
+---------+---------+
          |
          v
 optimized graph
```

Current default pass order:

| Order | Pass | Purpose |
| ---: | --- | --- |
| 1 | `constant_fold` | Executes const-only subgraphs at compile time |
| 2 | `identity_op_removal` | Removes no-op add/mul/relu cases |
| 3 | `fuse_linear_relu` | Replaces `matmul + add + relu` with `fused_linear_relu` |
| 4 | `fuse_linear` | Replaces `matmul + add` with `fused_linear` |
| 5 | `dead_node_elimination` | Removes nodes not needed by graph outputs |

## 5. Fusion Example

Before optimization:

```text
x ----+
      v
   matmul <---- w
      |
      v
    add  <----- b
      |
      v
    relu
      |
      v
      y
```

After optimization:

```text
x ----+
      |
w ----+----> fused_linear_relu ----> y
      |
b ----+
```

Behavioral contract:

```text
fused_linear_relu(x, w, b) == relu(matmul(x, w) + b)
```

The optimized graph must produce numerically equivalent outputs to the original graph.

## 6. Runtime Execution

The default runtime is a simple NumPy environment-based interpreter. The backend layer can also execute the same graph with PyTorch tensors.

```text
inputs + constants
       |
       v
+---------------------+
| env: tensor -> array |
+----------+----------+
           |
           v
for node in graph.nodes:
  values = env[node.inputs]
  output = execute_node(node, values)
  env[node.output] = output
           |
           v
return env[graph.outputs]
```

Supported runtime ops:

| Op | NumPy Behavior |
| --- | --- |
| `matmul` | `a @ b` |
| `add` | `a + b` |
| `mul` | `a * b` |
| `relu` | `maximum(x, 0)` |
| `gelu` | tanh GELU approximation |
| `reshape` | `np.reshape(...)` |
| `transpose` | `np.transpose(...)` |
| `softmax` | stable exponentiation and normalization |
| `sum` | `np.sum(...)` |
| `layernorm` | normalize over last axis |
| `fused_linear` | `x @ w + b` |
| `fused_linear_relu` | `maximum(x @ w + b, 0)` |

Quantized constants are dequantized by the runtime before execution. This keeps v2 focused on compiler representation and accuracy/memory tradeoffs rather than hardware-specific INT8 kernels.

## 6.1 Backend Execution

TinyGraph separates graph representation from execution backend.

```text
             +-------------+
             | Graph IR    |
             +------+------+
                    |
       +------------+------------+
       |                         |
       v                         v
 +-----------+             +-------------+
 | NumPy     |             | PyTorch     |
 | reference |             | tensor exec |
 +-----------+             +-------------+
```

The NumPy backend remains the correctness reference. The PyTorch backend executes the same op set on `cpu`, `mps`, or `cuda`, with `auto` selecting `mps`, then `cuda`, then `cpu`.

## 7. Benchmarking and Metrics

Benchmarking compares naive execution against optimized execution.

```text
              +----------------+
              | original graph |
              +--------+-------+
                       |
       +---------------+---------------+
       |                               |
       v                               v
+-------------+                 +--------------+
| run naive   |                 | optimize     |
+------+------+                 +------+-------+
       |                               |
       v                               v
+-------------+                 +--------------+
| timings     |                 | run optimized|
+------+------+                 +------+-------+
       |                               |
       +---------------+---------------+
                       |
                       v
             +-------------------+
             | compare outputs   |
             | report metrics    |
             +-------------------+
```

Metrics:

| Metric | Meaning |
| --- | --- |
| `node_count` | Number of graph nodes after pass pipeline |
| `p50_ms` | Median runtime over repeated executions |
| `p95_ms` | Tail runtime over repeated executions |
| `estimated_flops` | Rough static FLOP estimate from inferred shapes |
| `estimated_bytes` | Rough tensor bytes touched by graph nodes |
| `max_abs_delta` | Maximum absolute difference between naive and optimized outputs |

## 8. Memory Planning

TinyGraph estimates tensor lifetimes to explain peak intermediate memory.

```text
node index: 0      1        2       3       4       5
            | matmul | add | relu | matmul | add | output

h_mm          [------]
h_bias                 [-----]
h                             [-------------]
out_mm                                  [-----]
logits                                          [------]
```

Memory planner flow:

```text
infer shapes
    |
    v
find first definition for each tensor
    |
    v
find last use for each tensor
    |
    v
create allocation/free events
    |
    v
scan events to estimate peak live bytes
```

Memory metrics:

| Metric | Meaning |
| --- | --- |
| `naive_peak_bytes` | Sum of all intermediate tensor sizes |
| `planned_peak_bytes` | Peak bytes if tensors are freed after last use |
| `tensor_lifetimes` | First definition and last use per tensor |

## 9. CLI and User Workflows

```text
             +----------------+
             | examples/*.json|
             +-------+--------+
                     |
       +-------------+-------------+
       |             |             |
       v             v             v
   inspect          run         optimize
       |             |             |
       v             v             v
 graph summary   output stats   optimized json
                     |
       +-------------+-------------+
       |                           |
       v                           v
     bench                       report
       |                           |
       v                           v
 CSV-style metrics             HTML artifact
```

CLI commands:

| Command | Output |
| --- | --- |
| `inspect` | Graph structure and inferred shapes |
| `run` | Output tensor shape, dtype, and mean |
| `optimize` | Optimized JSON graph and pass summary |
| `bench` | CSV-style benchmark metrics |
| `report` | Static HTML report |
| `memory` | Tensor lifetimes and peak memory estimates |
| `quantize` | INT8 quantized graph JSON |
| `compare` | Memory savings and output drift for original vs quantized graphs |

Backend commands:

```bash
tinygraph run examples/attention_block.json --backend torch --device cpu
tinygraph bench examples/attention_block.json --compare-backends --device cpu
tinygraph report examples/attention_block.json --compare-backends --out reports/backend_compare.html
```

## 10. Static Report Structure

```text
+--------------------------------------------------+
| TinyGraph Report: graph name                     |
+------------------------+-------------------------+
| Benchmark table        | Memory plan             |
+------------------------+-------------------------+
| Pass report table                                |
+--------------------------------------------------+
| Before graph summary   | After graph summary     |
+------------------------+-------------------------+
```

The report is intentionally static HTML so it can be opened directly in a browser or committed as an artifact.

## 11. Test Strategy

```text
          +--------------------+
          | unit tests         |
          | op + shape logic   |
          +----------+---------+
                     |
                     v
          +--------------------+
          | pass tests         |
          | output equivalence |
          +----------+---------+
                     |
                     v
          +--------------------+
          | CLI smoke tests    |
          | user workflows     |
          +----------+---------+
                     |
                     v
          +--------------------+
          | report generation  |
          | artifact validity  |
          +--------------------+
```

Acceptance requirements:

| Area | Requirement |
| --- | --- |
| Runtime | Example graphs execute successfully |
| Optimization | Optimized outputs match naive outputs |
| Fusion | MLP graph node count decreases |
| CLI | Inspect/run/optimize/bench/report commands work |
| Report | HTML report is generated and includes benchmark data, pass concepts, and pass snapshots |
| Memory | Memory command reports tensor lifetimes and peaks |

## 12. Quantization Pipeline

TinyGraph v2 adds weight-only symmetric INT8 quantization.

```text
original graph
      |
      v
find eligible weight constants
      |
      v
quantize weights to int8
      |
      v
save scale metadata
      |
      v
run graph with runtime dequantization
      |
      v
compare output drift + memory savings
```

Only matrix weight constants are quantized. Activations and bias constants stay in floating point.

```text
FP32 weight:
  w = [[...]]

Quantized storage:
  w: int8 array
  metadata:
    scheme: symmetric_int8
    scale: float
    zero_point: 0
    original_dtype: float32
```

Quantization report:

| Metric | Meaning |
| --- | --- |
| Original constant bytes | Total constant storage before quantization |
| Quantized constant bytes | Total constant storage after quantization |
| Memory reduction | Percent storage reduction for constants |
| Max absolute error | Largest output difference vs FP32 |
| Mean absolute error | Average output difference vs FP32 |

## 13. Planned Extension Roadmap

```text
v1: NumPy compiler lab
 |
 +--> v2: quantization pass
 |       - quantize constants
 |       - compare fp32/int8 outputs
 |       - report accuracy drift
 |
 +--> v3: Triton lowering
 |       - lower fused ops to custom kernels
 |       - compare NumPy/PyTorch/Triton execution
 |
 +--> v4: ONNX import
 |       - load small feed-forward graphs
 |       - run TinyGraph pass pipeline
 |
 +--> v5: interactive visualizer
         - clickable graph nodes
         - pass-by-pass diffs
         - tensor shape overlays
```

Extension matrix:

| Extension | Adds | Depends On |
| --- | --- | --- |
| Quantization | Precision and compression learning | IR, runtime, benchmark |
| Triton Lowering | GPU systems learning | Fusion, shape inference |
| ONNX Import | Real model interoperability | Validation, serialization |
| Interactive UI | Portfolio polish | Report generation |
| Autodiff | Training fundamentals | Runtime, op definitions |

## 14. Design Boundaries

TinyGraph v1 intentionally does not implement:

- automatic differentiation
- training loops
- ONNX import
- GPU execution
- dynamic shapes
- multi-output nodes
- control flow
- activation quantization
- hardware-native INT8 kernels

These are good future topics, but keeping v1 small makes the compiler pipeline easier to understand and verify.

## 15. Landscape and Growth Plan

TinyGraph is intentionally small right now. The current codebase is an educational compiler skeleton, not a production ML compiler. That is the point of v1: make the important compiler ideas visible before adding the harder systems layers.

```text
What TinyGraph is today:

  tiny educational compiler skeleton

What TinyGraph can become:

  mini PyTorch / XLA / TVM-style compiler lab
```

### 15.1 Where TinyGraph Fits

Modern ML systems generally follow this shape:

```text
PyTorch / TensorFlow / JAX
        |
        v
Graph capture
"what operations is the model doing?"
        |
        v
Intermediate Representation, IR
"represent the model as a graph"
        |
        v
Compiler passes
"remove useless work, fuse ops, plan memory"
        |
        v
Lowering
"turn graph ops into backend-specific code"
        |
        v
Kernels / hardware
"CPU, GPU, TPU, accelerator"
```

TinyGraph currently teaches the middle of that stack:

```text
Graph IR -> shape inference -> optimization passes -> NumPy execution
```

Real systems use the same broad ideas at much larger scale:

| System | Role in the Landscape | TinyGraph Equivalent |
| --- | --- | --- |
| PyTorch `torch.compile` | Captures Python model execution and compiles it through backends | Future graph capture / frontend |
| TorchInductor | PyTorch compiler backend for optimized generated code | Future lowering/backend layer |
| XLA | Compiler stack for TensorFlow/JAX-style graphs and accelerators | Compiler pipeline inspiration |
| TVM | Deep learning compiler for many model/hardware targets | Long-term compiler stack inspiration |
| MLIR | General compiler infrastructure with dialects and lowering | Long-term IR/lowering inspiration |
| Triton | Python-like language for writing efficient GPU kernels | Future fused-kernel backend |

TinyGraph should not try to compete with these systems. It should help explain their core mechanics by rebuilding a tiny version.

### 15.2 What We Have Built

Current TinyGraph v1 can represent and optimize small feed-forward graphs.

Example:

```text
matmul -> add -> relu
```

can become:

```text
fused_linear_relu
```

That sounds small, but it is a foundational compiler idea. Real ML compilers fuse operations because separate operations create extra scheduling overhead and often write intermediate tensors to memory. Fusing ops keeps more work together and can reduce memory traffic.

Current capability map:

| Area | Current Status |
| --- | --- |
| Graph IR | Implemented |
| JSON graph format | Implemented |
| Python graph-building API | Implemented |
| Shape inference | Implemented for MLP and attention-style ops |
| NumPy runtime | Implemented |
| Validation | Implemented |
| Constant folding | Implemented |
| Identity removal | Implemented |
| Linear/ReLU fusion | Implemented |
| Dead node elimination | Implemented |
| Benchmarks | Implemented |
| Static HTML report | Implemented |
| Memory lifetime estimate | Implemented |
| Weight-only INT8 quantization | Implemented |
| Quantization drift metrics | Implemented |
| Richer ops: reshape, transpose, softmax, gelu | Implemented |
| Linear fusion without activation | Implemented |
| Pass-by-pass graph snapshots | Implemented |
| PyTorch backend | Implemented |
| Backend comparison metrics | Implemented |
| Tests | Implemented |

### 15.3 Why It Still Feels Small

TinyGraph v1 is small because it has only enough machinery to demonstrate the core compiler loop.

```text
define graph
    |
validate + infer shapes
    |
run naive graph
    |
optimize graph
    |
run optimized graph
    |
compare correctness and performance
```

What is missing before it feels like a serious ML systems project:

| Missing Layer | Why It Matters |
| --- | --- |
| Even more ops | Real neural nets still need conv2d, dropout, masking, embeddings, etc. |
| Better optimizer | Real compilers need robust pattern matching and many rewrite rules |
| Interactive visualization | Current reports are static; clickable pass diffs would teach more |
| Deeper quantization | Per-channel scales, activation quantization, and real INT8 kernels |
| Hardware-specific lowering | PyTorch eager exists; Triton kernels would teach direct kernel lowering |
| ONNX import | Lets TinyGraph operate on small real model graphs |

### 15.4 Growth Roadmap

TinyGraph should grow in stages. Each stage teaches a distinct deep-tech concept.

```text
v1: educational compiler skeleton
 |
v2: quantization + graph reports
 |
 v3: richer graph + more ops
 |
 v4: PyTorch backend                    <- current
 |
 v5: Triton fused kernels
 |
 v6: deeper quantization
 |
 v7: ONNX import for small real models
```

#### v3: Richer Graph and Visualization

Goal: make TinyGraph handle more realistic model fragments and make transformations easier to see.

Added:

- `reshape`
- `transpose`
- `softmax`
- `gelu`
- `matmul + bias` fusion
- pass-by-pass graph snapshots
- graph diagrams in the HTML report

Result:

```text
TinyGraph becomes a clear compiler teaching tool instead of only a code skeleton.
```

#### v4: PyTorch Backend

Goal: separate graph representation from execution backend.

Added:

- backend interface
- NumPy backend
- PyTorch backend
- CPU/MPS/CUDA device selection
- backend comparison metrics
- same graph, multiple runtimes

Architecture:

```text
             +-------------+
             | Graph IR    |
             +------+------+
                    |
       +------------+------------+
       |                         |
       v                         v
 +-----------+             +-------------+
 | NumPy     |             | PyTorch     |
 | backend   |             | backend     |
 +-----------+             +-------------+
```

Core learning:

```text
IR is separate from runtime.
The same graph can target different execution engines.
```

#### v5: Triton Fused Kernels

Goal: lower selected fused ops to GPU kernels.

Add:

- Triton implementation of `fused_linear_relu`
- benchmark against NumPy/PyTorch
- shape constraints for kernel lowering
- fallback when Triton/GPU is unavailable

Architecture:

```text
Graph IR
   |
   v
fusion pass
   |
   v
fused_linear_relu node
   |
   +---------> NumPy fallback
   |
   +---------> PyTorch backend
   |
   +---------> Triton kernel
```

Core learning:

```text
compiler pass creates a fused op
lowering maps fused op to a hardware-specific implementation
```

#### v6: Deeper Quantization Workbench

Goal: connect TinyGraph to efficient inference.

Add:

- per-channel INT8 quantization
- optional INT4 experiment
- activation quantization experiment
- quantization-aware benchmark presets
- side-by-side benchmark table

Core learning:

```text
lower precision -> less memory
lower precision -> possible output drift
compiler pass -> transforms graph/constants
```

Quantization should be implemented as a compiler pass, not as a separate script:

```text
original graph
      |
      v
quantize constants pass
      |
      v
quantized graph
      |
      v
runtime + accuracy drift report
```

#### v7: ONNX Import

Goal: make TinyGraph run on small real model graphs.

Add:

- import simple ONNX graphs
- support a small ONNX op subset
- compare TinyGraph output with reference output
- document unsupported ops clearly

Core learning:

```text
real model format -> TinyGraph IR -> TinyGraph passes -> backend execution
```

### 15.5 Best Next Step

The next step that will make the project feel more like a serious compiler lab is:

```text
Triton fused kernels
```

Why:

- Quantization now exists as the first inference-efficiency feature.
- Richer ops and pass-by-pass diagrams now exist.
- PyTorch now gives a real tensor runtime.
- Triton is the next step from backend dispatch to hardware-specific kernel lowering.

Recommended next milestone:

```text
Input:
  existing TinyGraph examples

Add kernel backend:
  Triton fused_linear / fused_linear_relu

Add CLI:
  tinygraph bench --backend triton

Add report:
  NumPy vs PyTorch vs Triton comparison table
```

This turns TinyGraph from:

```text
clear miniature ML compiler with realistic graph fragments
```

into:

```text
mini compiler with hardware-specific fused kernels
```
