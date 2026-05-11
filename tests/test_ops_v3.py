import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tinygraph import Graph, optimize, run
from tinygraph.ops import gelu, matmul, reshape, softmax, transpose
from tinygraph.runtime import make_random_feeds
from tinygraph.serialization import load_graph, save_graph


class OpsV3Tests(unittest.TestCase):
    def test_new_ops_shape_and_runtime(self):
        graph = Graph("ops")
        x = graph.input("x", (2, 3, 4))
        xt = transpose(x, axes=(0, 2, 1), name="xt")
        flat = reshape(xt, (2, 12), name="flat")
        y = softmax(gelu(flat, name="g"), axis=-1, name="y")
        graph.output(y)

        feeds = {"x": np.linspace(-1, 1, 24, dtype="float32").reshape(2, 3, 4)}
        out = run(graph, feeds)["y"]
        self.assertEqual(out.shape, (2, 12))
        self.assertEqual(out.dtype, np.dtype("float32"))
        np.testing.assert_allclose(np.sum(out, axis=-1), np.ones((2,), dtype="float32"), rtol=1e-6, atol=1e-6)

    def test_batched_matmul_attention_example_runs(self):
        graph = load_graph("examples/attention_block.json")
        feeds = make_random_feeds(graph, seed=4)
        out = run(graph, feeds)["flat"]
        self.assertEqual(out.shape, (2, 12))

    def test_new_op_attributes_round_trip(self):
        graph = load_graph("examples/attention_block.json")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "attention.json"
            save_graph(graph, path)
            loaded = load_graph(path)
        transpose_node = next(node for node in loaded.nodes if node.op == "transpose")
        reshape_node = next(node for node in loaded.nodes if node.op == "reshape")
        softmax_node = next(node for node in loaded.nodes if node.op == "softmax")
        self.assertEqual(transpose_node.attrs["axes"], [0, 2, 1])
        self.assertEqual(reshape_node.attrs["shape"], [2, 12])
        self.assertEqual(softmax_node.attrs["axis"], -1)

    def test_fused_linear_for_matmul_add_without_relu(self):
        graph = load_graph("examples/gelu_mlp.json")
        feeds = make_random_feeds(graph, seed=9)
        original = run(graph, feeds)
        result = optimize(graph)
        optimized = run(result.graph, feeds)

        self.assertTrue(any(node.op == "fused_linear" for node in result.graph.nodes))
        self.assertTrue(any(report.name == "fuse_linear" and report.changed for report in result.reports))
        np.testing.assert_allclose(original["logits"], optimized["logits"], rtol=1e-6, atol=1e-6)

    def test_python_api_batched_matmul(self):
        graph = Graph("batched")
        a = graph.input("a", (2, 3, 4))
        b = graph.input("b", (2, 4, 5))
        y = matmul(a, b, name="y")
        graph.output(y)

        feeds = {
            "a": np.ones((2, 3, 4), dtype="float32"),
            "b": np.ones((2, 4, 5), dtype="float32"),
        }
        self.assertEqual(run(graph, feeds)["y"].shape, (2, 3, 5))


class V3CliTests(unittest.TestCase):
    def test_new_examples_and_report_snapshots(self):
        with TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "attention.html"
            subprocess.run(
                [sys.executable, "-m", "tinygraph.cli", "run", "examples/attention_block.json", "--optimized"],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tinygraph.cli",
                    "report",
                    "examples/gelu_mlp.json",
                    "--out",
                    str(report_path),
                    "--runs",
                    "3",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            text = report_path.read_text()
            self.assertIn("After fuse_linear", text)
            self.assertIn("Concepts", text)
            self.assertIn("fused_linear", text)


if __name__ == "__main__":
    unittest.main()
