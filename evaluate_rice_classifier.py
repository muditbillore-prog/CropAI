
import argparse
from pathlib import Path
from ultralytics import YOLO

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Evaluate the CropAI rice classifier on a test split.")
parser.add_argument("--model", default="models/best.pt",
                    help="model weights, relative to the project folder (default: the deployed model)")
parser.add_argument("--dataset", default="rice_classification_clean_v1",
                    help="folder name under datasets/ (default: the original random split)")
parser.add_argument("--name", default=None,
                    help="optional name for the output folder under runs/classify/")
args = parser.parse_args()

MODEL_PATH = PROJECT_ROOT / args.model

DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / args.dataset
)

# Safety checks
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not (DATASET_DIR / "test").is_dir():
    raise FileNotFoundError("Test dataset folder not found.")

print(f"Loading CropAI rice classifier: {MODEL_PATH}")
print(f"Test dataset: {DATASET_DIR}")
model = YOLO(str(MODEL_PATH))

print("\nEvaluating on unseen test images...")

extra = {"name": args.name} if args.name else {}

results = model.val(
    data=str(DATASET_DIR),
    split="test",
    imgsz=224,
    batch=8,
    device="cpu",
    workers=0,
    plots=True,
    **extra,
)

print("\n========== CROP AI TEST RESULTS ==========")
print(f"Top-1 Accuracy: {results.top1:.2%}")
print(f"Top-5 Accuracy: {results.top5:.2%}")
print("Evaluation completed.")
