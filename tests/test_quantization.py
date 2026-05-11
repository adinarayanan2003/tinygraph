import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tinygraph.quantization import compare_quantized, quantize_array_int8, quantize_graph
from tinygraph.runtime import make_random_feeds, run
from tinygraph.serialization import load_graph, save_graph


class QuantizationTests(unittest.TestCase):
    def test_int8_quantization_math_is_zero_safe(self):
        quantized, scale = quantize_array_int8(np.zeros((2, 2), dtype="float32"))
        self.assertEqual(scale, 1.0)
        self.assertEqual(quantized.dtype, np.dtype("int8"))
        np.testing.assert_array_equal(quantized, np.zeros((2, 2), dtype="int8"))

    def test_quantized_mlp_stays_close_to_fp32(self):
        graph = load_graph("examples/mlp.json")
        quantized = quantize_graph(graph)
        feeds = make_random_feeds(graph, seed=7)
        original_outputs = run(graph, feeds)
        quantized_outputs = run(quantized, feeds)

        self.assertEqual(sorted(quantized.quantization), ["w1", "w2"])
        self.assertEqual(quantized.constants["w1"].dtype, np.dtype("int8"))
        np.testing.assert_allclose(original_outputs["logits"], quantized_outputs["logits"], atol=0.01)

    def test_quantized_graph_serializes_metadata(self):
        graph = load_graph("examples/mlp.json")
        quantized = quantize_graph(graph)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "mlp_int8.json"
            save_graph(quantized, path)
            raw = json.loads(path.read_text())
            w1 = next(item for item in raw["constants"] if item["name"] == "w1")
            self.assertEqual(w1["dtype"], "int8")
            self.assertEqual(w1["quantization"]["scheme"], "symmetric_int8")

            loaded = load_graph(path)
            self.assertEqual(loaded.quantization["w1"]["scheme"], "symmetric_int8")
            comparison = compare_quantized(graph, loaded, seed=7)
            self.assertLess(comparison.max_abs_error, 0.01)
            self.assertGreater(comparison.summary.memory_reduction_percent, 50.0)


if __name__ == "__main__":
    unittest.main()
