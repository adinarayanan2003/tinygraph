from __future__ import annotations

from tinygraph.ir import Graph


SUPPORTED_OPS = {
    "matmul",
    "add",
    "mul",
    "relu",
    "gelu",
    "reshape",
    "transpose",
    "softmax",
    "sum",
    "layernorm",
    "fused_linear",
    "fused_linear_relu",
}


def validate_graph(graph: Graph) -> None:
    defined = set(graph.inputs) | set(graph.constants)
    seen_outputs: set[str] = set()

    if not graph.outputs:
        raise ValueError("graph must declare at least one output")

    for node in graph.nodes:
        if node.op not in SUPPORTED_OPS:
            raise ValueError(f"unsupported op {node.op!r}")
        if len(node.outputs) != 1:
            raise ValueError(f"{node.op} must have exactly one output")
        for output in node.outputs:
            if output in defined or output in seen_outputs:
                raise ValueError(f"duplicate tensor name {output!r}")
            seen_outputs.add(output)
        for input_name in node.inputs:
            if input_name not in defined:
                raise ValueError(f"{node.op} reads undefined tensor {input_name!r}")
        defined.update(node.outputs)

    for output in graph.outputs:
        if output not in defined:
            raise ValueError(f"graph output {output!r} is undefined")
