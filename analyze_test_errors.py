"""List the held-out test images that the trained rice classifier gets wrong.

Runs models/best.pt over the test split the same way the dashboard does
(imgsz=224, CPU), prints every misclassified image and saves the list to
runs/test_misclassifications.csv (runs/ is git-ignored).
"""
import csv
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
TEST_DIR = PROJECT_ROOT / "datasets" / "rice_classification_clean_v1" / "test"
OUTPUT_CSV = PROJECT_ROOT / "runs" / "test_misclassifications.csv"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

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

print(f"Running {len(images)} test images through {MODEL_PATH.name} ...")
results = model.predict(
    source=[str(p) for p in images],
    imgsz=224, batch=8, device="cpu", stream=True, verbose=False,
)

errors = []
top1_correct = top5_correct = processed = 0
for path, result in zip(images, results):
    processed += 1
    true_name = path.parent.name
    values = result.probs.data
    order = values.argsort(descending=True).tolist()
    ranked = [model.names[i] for i in order]
    true_rank = ranked.index(true_name) + 1
    top1_correct += int(true_rank == 1)
    top5_correct += int(true_rank <= 5)
    if true_rank != 1:
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
for e in errors:
    print(
        f"\n  {e['image']}\n"
        f"    true: {e['true_variety']} (rank {e['true_variety_rank']}, "
        f"{e['true_variety_confidence_pct']}%)\n"
        f"    predicted: {e['predicted_variety']} ({e['predicted_confidence_pct']}%)"
    )

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
fields = ["image", "true_variety", "predicted_variety", "predicted_confidence_pct",
          "true_variety_rank", "true_variety_confidence_pct"]
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(errors)
print(f"\nSaved {len(errors)} rows to {OUTPUT_CSV.relative_to(PROJECT_ROOT).as_posix()}")
