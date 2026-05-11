import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tinygraph.backends import compare_backends, run_with_backend
from tinygraph.quantization import quantize_graph
from tinygraph.runtime import make_random_feeds, run
from tinygraph.serialization import load_graph


class BackendTests(unittest.TestCase):
    def test_numpy_backend_matches_runtime_run(self):
        graph = load_graph("examples/attention_block.json")
        feeds = make_random_feeds(graph, seed=11)
        direct = run(graph, feeds)
        backend = run_with_backend(graph, feeds, backend="numpy")
        np.testing.assert_allclose(direct["flat"], backend["flat"])

    def test_torch_backend_matches_numpy_on_examples(self):
        for path in ["examples/mlp.json", "examples/gelu_mlp.json", "examples/attention_block.json"]:
            graph = load_graph(path)
            feeds = make_random_feeds(graph, seed=12)
            numpy_out = run_with_backend(graph, feeds, backend="numpy")
            torch_out = run_with_backend(graph, feeds, backend="torch", device="cpu")
            for name in graph.outputs:
                np.testing.assert_allclose(numpy_out[name], torch_out[name], rtol=1e-5, atol=1e-5)

    def test_torch_backend_runs_quantized_graph(self):
        graph = quantize_graph(load_graph("examples/mlp.json"))
        feeds = make_random_feeds(graph, seed=13)
        numpy_out = run_with_backend(graph, feeds, backend="numpy")
        torch_out = run_with_backend(graph, feeds, backend="torch", device="cpu")
        np.testing.assert_allclose(numpy_out["logits"], torch_out["logits"], rtol=1e-5, atol=1e-5)

    def test_compare_backends_reports_torch_cpu(self):
        graph = load_graph("examples/attention_block.json")
        rows = compare_backends(graph, runs_count=2, device="cpu")
        by_backend = {row.backend: row for row in rows}
        self.assertTrue(by_backend["numpy"].available)
        self.assertTrue(by_backend["torch"].available)
        self.assertLess(by_backend["torch"].max_abs_delta, 1e-4)


class BackendCliTests(unittest.TestCase):
    def test_backend_cli_and_report(self):
        with TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "backend_report.html"
            run_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tinygraph.cli",
                    "run",
                    "examples/attention_block.json",
                    "--backend",
                    "torch",
                    "--device",
                    "cpu",
                    "--optimized",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("flat: shape=(2, 12)", run_result.stdout)

            bench_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tinygraph.cli",
                    "bench",
                    "examples/attention_block.json",
                    "--compare-backends",
                    "--device",
                    "cpu",
                    "--runs",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("torch,cpu,True", bench_result.stdout)

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tinygraph.cli",
                    "report",
                    "examples/attention_block.json",
                    "--compare-backends",
                    "--device",
                    "cpu",
                    "--out",
                    str(report_path),
                    "--runs",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            text = report_path.read_text()
            self.assertIn("Backend Comparison", text)
            self.assertIn("torch", text)


if __name__ == "__main__":
    unittest.main()
