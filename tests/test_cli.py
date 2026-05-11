import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class CliTests(unittest.TestCase):
    def test_inspect_and_run(self):
        inspect_result = subprocess.run(
            [sys.executable, "-m", "tinygraph.cli", "inspect", "examples/linear.json"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("Graph: linear", inspect_result.stdout)

        run_result = subprocess.run(
            [sys.executable, "-m", "tinygraph.cli", "run", "examples/linear.json"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("y: shape=(2, 2)", run_result.stdout)

    def test_optimize_and_report(self):
        with TemporaryDirectory() as tmp:
            optimized_path = Path(tmp) / "optimized.json"
            report_path = Path(tmp) / "report.html"
            quantized_path = Path(tmp) / "mlp_int8.json"
            quant_report_path = Path(tmp) / "quant_report.html"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tinygraph.cli",
                    "optimize",
                    "examples/mlp.json",
                    "--out",
                    str(optimized_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(optimized_path.exists())

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tinygraph.cli",
                    "report",
                    "examples/mlp.json",
                    "--out",
                    str(report_path),
                    "--runs",
                    "3",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("TinyGraph Report", report_path.read_text())

            quantize_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tinygraph.cli",
                    "quantize",
                    "examples/mlp.json",
                    "--out",
                    str(quantized_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("quantized_constants=w1,w2", quantize_result.stdout)

            compare_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tinygraph.cli",
                    "compare",
                    "examples/mlp.json",
                    str(quantized_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("max_abs_error=", compare_result.stdout)

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tinygraph.cli",
                    "report",
                    "examples/mlp.json",
                    "--quantize",
                    "int8",
                    "--out",
                    str(quant_report_path),
                    "--runs",
                    "3",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report_text = quant_report_path.read_text()
            self.assertIn("INT8 Quantization", report_text)
            self.assertIn("Quantized Diagram", report_text)


if __name__ == "__main__":
    unittest.main()
