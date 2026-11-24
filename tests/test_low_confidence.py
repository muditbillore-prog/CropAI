"""Tests for count_low_confidence in cropai_helpers.py, the dashboard's low-confidence rule."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cropai_helpers import count_low_confidence  # noqa: E402


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
