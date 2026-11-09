"""Smoke tests for the trained rice classifier (models/best.pt).

Synthetic images are used because the training dataset is not part of the
repository. The predict call mirrors the one used by the dashboard.
"""
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best.pt"
EXPECTED_CLASSES = 33


def synthetic_image(height, width, seed=0):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


def predict(model, image):
    return model.predict(source=image, imgsz=224, device="cpu", verbose=False)[0]


class RiceClassifierSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL_PATH.is_file():
            raise AssertionError(f"Model file not found: {MODEL_PATH}")
        cls.model = YOLO(str(MODEL_PATH))

    def test_model_is_a_33_class_classifier(self):
        self.assertEqual(self.model.task, "classify")
        self.assertEqual(len(self.model.names), EXPECTED_CLASSES)
        for name in self.model.names.values():
            self.assertTrue(isinstance(name, str) and name.strip())

    def test_probabilities_are_valid(self):
        probs = predict(self.model, synthetic_image(480, 640)).probs
        values = probs.data.cpu().numpy()
        self.assertEqual(len(values), EXPECTED_CLASSES)
        self.assertTrue(np.all(values >= 0.0))
        self.assertTrue(np.all(values <= 1.0))
        self.assertAlmostEqual(float(values.sum()), 1.0, places=3)

    def test_top5_is_consistent_with_probabilities(self):
        probs = predict(self.model, synthetic_image(480, 640)).probs
        values = probs.data.cpu().numpy()
        top5 = list(probs.top5)
        self.assertEqual(len(top5), 5)
        self.assertEqual(len(set(top5)), 5)
        self.assertEqual(top5[0], int(values.argmax()))
        confidences = [float(values[i]) for i in top5]
        self.assertEqual(confidences, sorted(confidences, reverse=True))
        self.assertEqual(probs.top1, top5[0])

    def test_accepts_non_square_and_small_images(self):
        for height, width in [(480, 640), (60, 100), (700, 300)]:
            with self.subTest(height=height, width=width):
                probs = predict(self.model, synthetic_image(height, width)).probs
                self.assertEqual(len(probs.data), EXPECTED_CLASSES)

    def test_prediction_is_deterministic(self):
        image = synthetic_image(224, 224, seed=1)
        first = predict(self.model, image).probs.top1
        second = predict(self.model, image).probs.top1
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
