from __future__ import annotations

import numpy as np

from tinygraph.ir import Graph
from tinygraph.runtime import run


class NumPyBackend:
    name = "numpy"
    device_name = "cpu"

    def run(self, graph: Graph, feeds: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        return run(graph, feeds)

    def synchronize(self) -> None:
        return None
