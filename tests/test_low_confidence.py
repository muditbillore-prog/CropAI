"""Tests for count_low_confidence in app.py, the dashboard's low-confidence rule.

The function is read out of app.py without importing it, because importing the
dashboard would start Streamlit.
"""
import ast
import unittest
from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _load_count_low_confidence():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "count_low_confidence":
            module = ast.Module(body=[node], type_ignores=[])
            namespace = {}
            exec(compile(module, str(APP_PATH), "exec"), namespace)
            return namespace["count_low_confidence"]
    raise AssertionError("count_low_confidence not found in app.py")


count_low_confidence = _load_count_low_confidence()


class CountLowConfidenceTests(unittest.TestCase):
    def test_counts_values_below_the_default_threshold(self):
        self.assertEqual(count_low_confidence([95.0, 89.99, 50.0, 10.0]), 3)

    def test_a_value_equal_to_the_threshold_is_not_low(self):
        self.assertEqual(count_low_confidence([90.0]), 0)
        self.assertEqual(count_low_confidence([89.999]), 1)

    def test_empty_input_gives_zero(self):
        self.assertEqual(count_low_confidence([]), 0)

    def test_accepts_a_custom_threshold(self):
        self.assertEqual(count_low_confidence([70.0, 80.0, 95.0], threshold=75.0), 1)

    def test_all_high_and_all_low(self):
        self.assertEqual(count_low_confidence([91.0, 99.9]), 0)
        self.assertEqual(count_low_confidence([1.0, 2.0, 3.0]), 3)


if __name__ == "__main__":
    unittest.main()
