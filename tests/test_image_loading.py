"""Tests for load_uploaded_image in app.py, which opens images uploaded in the dashboard.

The function is read out of app.py without importing it, because importing the
dashboard would start Streamlit.
"""
import ast
import io
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _load_function():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "load_uploaded_image":
            module = ast.Module(body=[node], type_ignores=[])
            namespace = {"Image": Image}
            exec(compile(module, str(APP_PATH), "exec"), namespace)
            return namespace["load_uploaded_image"]
    raise AssertionError("load_uploaded_image not found in app.py")


load_uploaded_image = _load_function()


def png_bytes(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


class LoadUploadedImageTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_colour_image_is_returned_as_rgb(self):
        data = png_bytes(self.rng.integers(0, 256, size=(120, 200, 3), dtype=np.uint8))
        image = load_uploaded_image(io.BytesIO(data))
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (200, 120))

    def test_grayscale_image_is_converted_to_rgb(self):
        data = png_bytes(self.rng.integers(0, 256, size=(80, 80), dtype=np.uint8))
        image = load_uploaded_image(io.BytesIO(data))
        self.assertEqual(image.mode, "RGB")

    def test_text_file_raises_a_friendly_error(self):
        with self.assertRaises(ValueError) as context:
            load_uploaded_image(io.BytesIO(b"this is not an image"))
        self.assertIn("could not be read as an image", str(context.exception))

    def test_empty_file_raises_a_friendly_error(self):
        with self.assertRaises(ValueError):
            load_uploaded_image(io.BytesIO(b""))

    def test_truncated_png_raises_a_friendly_error(self):
        data = png_bytes(self.rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            load_uploaded_image(io.BytesIO(data[: len(data) // 2]))


if __name__ == "__main__":
    unittest.main()
