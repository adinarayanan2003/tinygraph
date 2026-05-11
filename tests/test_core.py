import unittest

import numpy as np

from tinygraph import Graph, optimize, run
from tinygraph.ops import add, matmul, relu
from tinygraph.runtime import make_random_feeds
from tinygraph.serialization import load_graph


class CoreTests(unittest.TestCase):
    def test_python_api_executes_graph(self):
        graph = Graph("api")
        x = graph.input("x", (2, 2))
        w = graph.const("w", np.eye(2, dtype="float32"))
        b = graph.const("b", np.ones((1, 2), dtype="float32"))
        y = relu(add(matmul(x, w), b))
        graph.output(y)

        out = run(graph, {"x": np.array([[-2, 1], [3, -4]], dtype="float32")})
        np.testing.assert_allclose(out[y.name], np.array([[0, 2], [4, 0]], dtype="float32"))

    def test_mlp_fusion_preserves_output(self):
        graph = load_graph("examples/mlp.json")
        feeds = make_random_feeds(graph, seed=42)
        naive = run(graph, feeds)
        result = optimize(graph)
        optimized = run(result.graph, feeds)

        self.assertLess(len(result.graph.nodes), len(graph.nodes))
        self.assertTrue(any(node.op == "fused_linear_relu" for node in result.graph.nodes))
        np.testing.assert_allclose(naive["logits"], optimized["logits"], rtol=1e-6, atol=1e-6)

    def test_constant_folding_removes_const_only_nodes(self):
        graph = Graph("const_fold")
        c1 = graph.const("c1", np.array([[1, 2]], dtype="float32"))
        c2 = graph.const("c2", np.array([[3, 4]], dtype="float32"))
        y = add(c1, c2, name="y")
        graph.output(y)

        result = optimize(graph, passes=["constant_fold"])
        self.assertEqual(result.graph.nodes, [])
        np.testing.assert_allclose(result.graph.constants["y"], np.array([[4, 6]], dtype="float32"))

    def test_invalid_shape_fails(self):
        graph = Graph("bad")
        x = graph.input("x", (2, 3))
        w = graph.const("w", np.ones((4, 2), dtype="float32"))
        y = matmul(x, w)
        graph.output(y)

        with self.assertRaises(ValueError):
            run(graph, {"x": np.ones((2, 3), dtype="float32")})


if __name__ == "__main__":
    unittest.main()
