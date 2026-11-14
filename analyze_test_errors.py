"""List the held-out test images that the trained rice classifier gets wrong.

Runs a model over the test split of a dataset the same way the dashboard does
(imgsz=224, CPU), prints a per-variety accuracy table and the most common
confusions, and saves every misclassified image to a CSV (runs/ is git-ignored).
The defaults reproduce the original evaluation (models/best.pt on the v1 split).
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

parser = argparse.ArgumentParser(description="List misclassified test images.")
parser.add_argument("--model", default="models/best.pt",
                    help="model weights, relative to the project folder (default: the deployed model)")
parser.add_argument("--dataset", default="rice_classification_clean_v1",
                    help="folder name under datasets/ (default: the original random split)")
parser.add_argument("--output", default="runs/test_misclassifications.csv",
                    help="CSV for the misclassified images, relative to the project folder")
args = parser.parse_args()

MODEL_PATH = PROJECT_ROOT / args.model
TEST_DIR = PROJECT_ROOT / "datasets" / args.dataset / "test"
OUTPUT_CSV = PROJECT_ROOT / args.output

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not TEST_DIR.is_dir():
    raise FileNotFoundError(f"Test dataset folder not found: {TEST_DIR}")

model = YOLO(str(MODEL_PATH))

images = sorted(
    p for p in TEST_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
)
if not images:
    raise RuntimeError(f"No test images found in {TEST_DIR}")

unknown = {p.parent.name for p in images} - set(model.names.values())
if unknown:
    raise RuntimeError(f"Test folders not in the model's classes: {sorted(unknown)}")

print(f"Model: {MODEL_PATH}")
print(f"Test set: {TEST_DIR}")
print(f"Running {len(images)} test images ...")
results = model.predict(
    source=[str(p) for p in images],
    imgsz=224, batch=8, device="cpu", stream=True, verbose=False,
)

errors = []
per_class = Counter()
per_class_correct = Counter()
confusions = Counter()
top1_correct = top5_correct = processed = 0
for path, result in zip(images, results):
    processed += 1
    true_name = path.parent.name
    values = result.probs.data
    order = values.argsort(descending=True).tolist()
    ranked = [model.names[i] for i in order]
    true_rank = ranked.index(true_name) + 1
    per_class[true_name] += 1
    top1_correct += int(true_rank == 1)
    top5_correct += int(true_rank <= 5)
    if true_rank == 1:
        per_class_correct[true_name] += 1
    else:
        confusions[(true_name, ranked[0])] += 1
        errors.append({
            "image": path.relative_to(TEST_DIR).as_posix(),
            "true_variety": true_name,
            "predicted_variety": ranked[0],
            "predicted_confidence_pct": round(float(values[order[0]]) * 100, 2),
            "true_variety_rank": true_rank,
            "true_variety_confidence_pct": round(float(values[order[true_rank - 1]]) * 100, 2),
        })

if processed != len(images):
    raise RuntimeError(f"Processed {processed} of {len(images)} images.")

print(f"\nImages: {processed}")
print(f"Top-1 correct: {top1_correct} ({top1_correct / processed:.2%})")
print(f"Top-5 correct: {top5_correct} ({top5_correct / processed:.2%})")
print(f"Misclassified: {len(errors)}")

weak = sorted(
    (per_class_correct[c] / per_class[c], c)
    for c in per_class if per_class_correct[c] < per_class[c]
)
print(f"\nVarieties with at least one error: {len(weak)} of {len(per_class)}")
for accuracy, name in weak:
    print(f"  {name:26} {per_class_correct[name]:3}/{per_class[name]:3} correct  ({accuracy:.0%})")

print("\nMost common confusions (true -> predicted):")
for (true_name, predicted), count in confusions.most_common(10):
    print(f"  {true_name} -> {predicted}: {count}")

if len(errors) <= 10:
    for e in errors:
        print(
            f"\n  {e['image']}\n"
            f"    true: {e['true_variety']} (rank {e['true_variety_rank']}, "
            f"{e['true_variety_confidence_pct']}%)\n"
            f"    predicted: {e['predicted_variety']} ({e['predicted_confidence_pct']}%)"
        )
else:
    print(f"\n({len(errors)} errors: see the CSV for the full list)")

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
fields = ["image", "true_variety", "predicted_variety", "predicted_confidence_pct",
          "true_variety_rank", "true_variety_confidence_pct"]
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(errors)
print(f"\nSaved {len(errors)} rows to {OUTPUT_CSV.relative_to(PROJECT_ROOT).as_posix()}")
