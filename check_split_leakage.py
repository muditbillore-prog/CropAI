"""Measure how much the held-out test split resembles the training split.

For every test image this finds the most similar training image of the same
variety (64-bit difference hash, Hamming distance) and reports the model's
accuracy by similarity bucket. Small distances mean near-identical images.
Writes runs/split_leakage_report.csv (runs/ is git-ignored).
"""
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
DATASET_DIR = PROJECT_ROOT / "datasets" / "rice_classification_clean_v1"
OUTPUT_CSV = PROJECT_ROOT / "runs" / "split_leakage_report.csv"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
BUCKETS = [(0, 5, "0-5 (near-identical)"), (6, 10, "6-10"),
           (11, 15, "11-15"), (16, 64, ">15 (distinct)")]

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
for split in ("train", "test"):
    if not (DATASET_DIR / split).is_dir():
        raise FileNotFoundError(f"Missing {split} folder: {DATASET_DIR / split}")


def list_images(split):
    return sorted(p for p in (DATASET_DIR / split).rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def dhash(path):
    img = Image.open(path).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


train_hashes = {}
for p in list_images("train"):
    train_hashes.setdefault(p.parent.name, []).append((p.name, dhash(p)))
train_matrix = {c: np.array([h for _, h in v]) for c, v in train_hashes.items()}

test_images = list_images("test")
if not test_images:
    raise RuntimeError("No test images found.")

model = YOLO(str(MODEL_PATH))
print(f"Scoring {len(test_images)} test images ...")
results = model.predict(source=[str(p) for p in test_images], imgsz=224,
                        batch=8, device="cpu", stream=True, verbose=False)

rows = []
for path, result in zip(test_images, results):
    true_name = path.parent.name
    predicted = model.names[result.probs.top1]
    distances = (train_matrix[true_name] != dhash(path)).sum(axis=1)
    j = int(distances.argmin())
    rows.append({
        "image": f"{true_name}/{path.name}", "true_variety": true_name,
        "predicted_variety": predicted, "correct": int(predicted == true_name),
        "nearest_train_image": train_hashes[true_name][j][0],
        "hash_distance": int(distances[j]),
    })

total_correct = sum(r["correct"] for r in rows)
print(f"\nOverall: {total_correct}/{len(rows)} correct ({total_correct / len(rows):.2%})")
print("\nAccuracy by distance to the nearest same-variety training image:")
for low, high, label in BUCKETS:
    subset = [r for r in rows if low <= r["hash_distance"] <= high]
    if subset:
        right = sum(r["correct"] for r in subset)
        print(f"  {label:22} images={len(subset):4}  correct={right:4}  accuracy={right / len(subset):.2%}")
    else:
        print(f"  {label:22} images=   0")

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved {len(rows)} rows to {OUTPUT_CSV.relative_to(PROJECT_ROOT).as_posix()}")
