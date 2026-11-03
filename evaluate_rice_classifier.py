
from pathlib import Path
from ultralytics import YOLO

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "rice_classification_clean_v1"
)

# Safety checks
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not (DATASET_DIR / "test").is_dir():
    raise FileNotFoundError("Test dataset folder not found.")

print("Loading trained CropAI rice classifier...")
model = YOLO(str(MODEL_PATH))

print("\nEvaluating on unseen test images...")

results = model.val(
    data=str(DATASET_DIR),
    split="test",
    imgsz=224,
    batch=8,
    device="cpu",
    workers=0,
    plots=True,
)

print("\n========== CROP AI TEST RESULTS ==========")
print(f"Top-1 Accuracy: {results.top1:.2%}")
print(f"Top-5 Accuracy: {results.top5:.2%}")
print("Evaluation completed.")
