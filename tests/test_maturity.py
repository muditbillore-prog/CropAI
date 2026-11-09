"""Tests for calculate_maturity in app.py.

The function is read out of app.py and run with a fixed "today" so results do
not change from day to day. Importing app.py directly would start the Streamlit
dashboard, so it is not imported here.
"""
import ast
import datetime
import unittest
from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
TODAY = datetime.date(2026, 6, 1)


class _FixedDate(datetime.date):
    @classmethod
    def today(cls):
        return TODAY


def _load_calculate_maturity():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "calculate_maturity":
            module = ast.Module(body=[node], type_ignores=[])
            namespace = {"date": _FixedDate, "timedelta": datetime.timedelta}
            exec(compile(module, str(APP_PATH), "exec"), namespace)
            return namespace["calculate_maturity"]
    raise AssertionError("calculate_maturity not found in app.py")


calculate_maturity = _load_calculate_maturity()


def planted(days_ago):
    return TODAY - datetime.timedelta(days=days_ago)


class CalculateMaturityTests(unittest.TestCase):
    def test_returns_expected_keys(self):
        result = calculate_maturity(planted(10), 100)
        self.assertEqual(
            set(result),
            {"duration_days", "maturity_date", "remaining", "progress", "stage"},
        )

    def test_mid_cycle_values(self):
        result = calculate_maturity(planted(50), 100)
        self.assertEqual(result["duration_days"], 100)
        self.assertEqual(result["maturity_date"], planted(50) + datetime.timedelta(days=100))
        self.assertEqual(result["remaining"], 50)
        self.assertEqual(result["progress"], 0.5)
        self.assertEqual(result["stage"], "Vegetative growth")

    def test_stage_boundaries(self):
        cases = [
            (0, "Early establishment"),
            (14, "Early establishment"),
            (15, "Vegetative growth"),
            (59, "Vegetative growth"),
            (60, "Reproductive growth"),
            (79, "Reproductive growth"),
            (80, "Ripening / grain filling"),
            (99, "Ripening / grain filling"),
            (100, "Expected maturity window reached"),
        ]
        for days_elapsed, expected_stage in cases:
            with self.subTest(days_elapsed=days_elapsed):
                result = calculate_maturity(planted(days_elapsed), 100)
                self.assertEqual(result["stage"], expected_stage)

    def test_future_planting_date_clamps_to_zero(self):
        result = calculate_maturity(planted(-10), 100)
        self.assertEqual(result["progress"], 0.0)
        self.assertEqual(result["stage"], "Early establishment")
        self.assertEqual(result["remaining"], 110)

    def test_past_maturity_clamps_to_one(self):
        result = calculate_maturity(planted(130), 100)
        self.assertEqual(result["progress"], 1.0)
        self.assertEqual(result["remaining"], -30)
        self.assertEqual(result["stage"], "Expected maturity window reached")

    def test_decimal_duration_is_truncated(self):
        result = calculate_maturity(planted(0), 120.7)
        self.assertEqual(result["duration_days"], 120)
        self.assertEqual(result["maturity_date"], TODAY + datetime.timedelta(days=120))

    def test_duration_limits_allowed_by_the_ui(self):
        for duration in (70, 180):
            with self.subTest(duration=duration):
                result = calculate_maturity(planted(0), duration)
                self.assertEqual(result["duration_days"], duration)
                self.assertEqual(result["maturity_date"], TODAY + datetime.timedelta(days=duration))
                self.assertEqual(result["progress"], 0.0)


if __name__ == "__main__":
    unittest.main()
