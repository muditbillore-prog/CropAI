"""Tests for the time-ordered split planner in make_time_ordered_split.py.

plan_class() cuts one variety's images, sorted by capture time, into train,
validation and test blocks with a buffer of dropped images between them. These
tests use synthetic timestamps, so they need neither the dataset nor the model.
"""
import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "make_time_ordered_split.py"
spec = importlib.util.spec_from_file_location("make_time_ordered_split", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
plan_class = module.plan_class

BASE = 1_700_000_000.0


def make_items(count, per_second=1):
    """count images, per_second of them sharing each second, sorted by time."""
    return [
        (BASE + i // per_second + (i % per_second) * 0.1, Path(f"img{i:04d}.jpg"))
        for i in range(count)
    ]


def seconds(part):
    return {int(t) for t, _ in part}


class PlanClassTests(unittest.TestCase):
    def test_block_sizes_one_image_per_second(self):
        for buffer, expected in [(3, (75, 9, 10, 6)), (6, (70, 9, 9, 12))]:
            with self.subTest(buffer=buffer):
                train, val, test, dropped = plan_class("x", make_items(100), buffer)
                self.assertEqual((len(train), len(val), len(test), len(dropped)), expected)

    def test_every_image_is_used_at_most_once_and_none_are_lost(self):
        for per_second in (1, 3):
            with self.subTest(per_second=per_second):
                items = make_items(120, per_second)
                train, val, test, dropped = plan_class("x", items, 3)
                used = [p.name for part in (train, val, test, dropped) for _, p in part]
                self.assertEqual(len(used), len(set(used)))
                self.assertEqual(sorted(used), sorted(p.name for _, p in items))

    def test_blocks_are_in_time_order_and_share_no_second(self):
        for per_second, buffer in [(1, 3), (3, 3), (3, 6)]:
            with self.subTest(per_second=per_second, buffer=buffer):
                train, val, test, _ = plan_class("x", make_items(120, per_second), buffer)
                self.assertLess(max(seconds(train)), min(seconds(val)))
                self.assertLess(max(seconds(val)), min(seconds(test)))
                self.assertFalse(seconds(train) & seconds(val))
                self.assertFalse(seconds(val) & seconds(test))
                self.assertFalse(seconds(train) & seconds(test))

    def test_buffer_drops_at_least_two_buffers_of_images(self):
        for buffer in (3, 6, 10):
            with self.subTest(buffer=buffer):
                _, _, _, dropped = plan_class("x", make_items(150), buffer)
                self.assertGreaterEqual(len(dropped), 2 * buffer)

    def test_raises_when_all_images_share_one_second(self):
        items = [(BASE + i * 0.01, Path(f"img{i}.jpg")) for i in range(10)]
        with self.assertRaises(RuntimeError):
            plan_class("x", items, 3)

    def test_raises_when_there_are_too_few_images(self):
        with self.assertRaises(RuntimeError):
            plan_class("x", make_items(8), 3)

    def test_is_deterministic(self):
        items = make_items(120, 3)
        self.assertEqual(plan_class("x", items, 6), plan_class("x", items, 6))


if __name__ == "__main__":
    unittest.main()
