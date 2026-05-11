from __future__ import annotations

from tinygraph.ir import Tensor


def matmul(a: Tensor, b: Tensor, name: str | None = None) -> Tensor:
    return a.graph.add_node("matmul", [a, b], output=name)


def add(a: Tensor, b: Tensor, name: str | None = None) -> Tensor:
    return a.graph.add_node("add", [a, b], output=name)


def mul(a: Tensor, b: Tensor, name: str | None = None) -> Tensor:
    return a.graph.add_node("mul", [a, b], output=name)


def relu(x: Tensor, name: str | None = None) -> Tensor:
    return x.graph.add_node("relu", [x], output=name)


def gelu(x: Tensor, name: str | None = None) -> Tensor:
    return x.graph.add_node("gelu", [x], output=name)


def reshape(x: Tensor, shape: tuple[int, ...], name: str | None = None) -> Tensor:
    return x.graph.add_node("reshape", [x], output=name, attrs={"shape": shape})


def transpose(x: Tensor, axes: tuple[int, ...] | None = None, name: str | None = None) -> Tensor:
    return x.graph.add_node("transpose", [x], output=name, attrs={"axes": axes})


def softmax(x: Tensor, axis: int = -1, name: str | None = None) -> Tensor:
    return x.graph.add_node("softmax", [x], output=name, attrs={"axis": axis})


def sum(x: Tensor, axis: int | tuple[int, ...] | None = None, keepdims: bool = False, name: str | None = None) -> Tensor:
    attrs = {"axis": axis, "keepdims": keepdims}
    return x.graph.add_node("sum", [x], output=name, attrs=attrs)


def layernorm(
    x: Tensor,
    gamma: Tensor | None = None,
    beta: Tensor | None = None,
    eps: float = 1e-5,
    name: str | None = None,
) -> Tensor:
    inputs = [x]
    if gamma is not None:
        inputs.append(gamma)
    if beta is not None:
        inputs.append(beta)
    return x.graph.add_node("layernorm", inputs, output=name, attrs={"eps": eps})
