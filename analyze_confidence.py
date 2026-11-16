"""Measure how well a model's confidence tracks whether it is right.

Runs a model over the test split of a dataset the same way the dashboard does
(imgsz=224, CPU) and reports accuracy by confidence level and for a chosen
threshold. The defaults evaluate the deployed model on the original split,
which overlaps with its training data, so the results are optimistic; for a
fairer view use a model and a split it was not trained on.
"""
import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

parser = argparse.ArgumentParser(description="Accuracy by confidence level.")
parser.add_argument("--model", default="models/best.pt",
                    help="model weights, relative to the project folder (default: the deployed model)")
parser.add_argument("--dataset", default="rice_classification_clean_v1",
                    help="folder name under datasets/ (default: the original random split)")
parser.add_argument("--threshold", type=float, default=0.9,
                    help="confidence below which a prediction is flagged as uncertain (default: 0.9)")
args = parser.parse_args()

MODEL_PATH = PROJECT_ROOT / args.model
TEST_DIR = PROJECT_ROOT / "datasets" / args.dataset / "test"
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not TEST_DIR.is_dir():
    raise FileNotFoundError(f"Test dataset folder not found: {TEST_DIR}")

model = YOLO(str(MODEL_PATH))
images = sorted(p for p in TEST_DIR.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
if not images:
    raise RuntimeError(f"No test images found in {TEST_DIR}")
unknown = {p.parent.name for p in images} - set(model.names.values())
if unknown:
    raise RuntimeError(f"Test folders not in the model's classes: {sorted(unknown)}")

print(f"Model: {MODEL_PATH}")
print(f"Test set: {TEST_DIR}")
conf, correct = [], []
results = model.predict(source=[str(p) for p in images], imgsz=224, batch=8,
                        device="cpu", stream=True, verbose=False)
for path, r in zip(images, results):
    conf.append(float(r.probs.top1conf))
    correct.append(model.names[r.probs.top1] == path.parent.name)
if len(conf) != len(images):
    raise RuntimeError(f"Processed {len(conf)} of {len(images)} images.")
conf, correct = np.array(conf), np.array(correct)


def rate(mask):
    return f"{correct[mask].mean():.1%} ({int(correct[mask].sum())}/{int(mask.sum())})" if mask.any() else "n/a"


print(f"\nImages: {len(conf)}, accuracy: {rate(np.ones(len(conf), dtype=bool))}")
print(f"Mean confidence: all {conf.mean():.1%} | when correct {conf[correct].mean():.1%} | "
      f"when wrong {conf[~correct].mean() if (~correct).any() else float('nan'):.1%}")
edges = [0.0, 0.5, 0.7, 0.9, 1.0001]
print(f"\n{'confidence':12} {'images':>7} {'correct':>8} {'accuracy':>9}")
for lo, hi in zip(edges, edges[1:]):
    m = (conf >= lo) & (conf < hi)
    print(f"{lo:.0%}-{min(hi, 1.0):.0%}".ljust(12) +
          f" {int(m.sum()):7d} {int(correct[m].sum()):8d} {(f'{correct[m].mean():.1%}' if m.any() else '-'):>9}")

flagged = conf < args.threshold
print(f"\nThreshold {args.threshold:.0%}:")
print(f"  flagged as uncertain: {int(flagged.sum())} of {len(conf)} ({flagged.mean():.1%})")
print(f"  accuracy at or above the threshold: {rate(~flagged)}")
print(f"  accuracy below the threshold:       {rate(flagged)}")
print(f"  errors below the threshold: {int((~correct & flagged).sum())} of {int((~correct).sum())}; "
      f"confident errors (at or above): {int((~correct & ~flagged).sum())}")
