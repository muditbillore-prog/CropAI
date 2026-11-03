
from pathlib import Path
from ultralytics import YOLO

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_ROOT / "datasets" / "rice_classification_clean_v1"
PROJECT_DIR = PROJECT_ROOT / "runs" / "rice_classification"

RUN_NAME = "rice_variety_clean_v1"

# Safety checks
for split in ("train", "val", "test"):
    if not (DATASET_DIR / split).is_dir():
        raise FileNotFoundError(
            f"Missing {split} folder: {DATASET_DIR / split}"
        )

def get_classes(split):
    return {
        p.name for p in (DATASET_DIR / split).iterdir()
        if p.is_dir()
    }

train_classes = get_classes("train")
val_classes = get_classes("val")
test_classes = get_classes("test")

if not train_classes:
    raise RuntimeError("No training classes found.")

if train_classes != val_classes or train_classes != test_classes:
    raise RuntimeError("Train/val/test class folders do not match.")

if len(train_classes) != 33:
    raise RuntimeError(
        f"Expected 33 rice varieties, found {len(train_classes)}."
    )

# Protect existing training results
run_dir = PROJECT_DIR / RUN_NAME

if run_dir.exists():
    raise FileExistsError(
        f"Training output already exists: {run_dir}\n"
        "Choose a new RUN_NAME to protect previous results."
    )

print("Dataset:", DATASET_DIR)
print("Rice varieties:", len(train_classes))
print("Loading pretrained classification model...")

model = YOLO("yolo11n-cls.pt")

print("\nStarting clean rice-variety training on CPU...")
print("Old model weights will not be overwritten.")

model.train(
    data=str(DATASET_DIR),
    epochs=20,
    imgsz=224,
    batch=8,
    device="cpu",
    workers=0,
    patience=5,
    project=str(PROJECT_DIR),
    name=RUN_NAME,
    seed=42,
    plots=True,
    exist_ok=False,
)

best_model = run_dir / "weights" / "best.pt"

print("\nTraining finished!")
print("Best model:", best_model)
print("Training results:", run_dir)
