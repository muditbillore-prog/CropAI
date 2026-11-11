
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import csv
import random
import shutil

# ---------- PATHS ----------
# Paths are relative to this script, so the project works from any location.
PROJECT_ROOT = Path(__file__).resolve().parent

ROOT = PROJECT_ROOT / "dataset" / "Primitive_Rice_OnField_new"
OUTPUT = PROJECT_ROOT / "datasets" / "rice_classification_clean_v1"
REPORT = PROJECT_ROOT / "reports" / "rice_cleaning_manifest.csv"

EXTENSIONS = {".jpg", ".jpeg", ".png"}
SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

# ---------- SAFETY CHECKS ----------
if not ROOT.exists():
    raise FileNotFoundError(f"Original dataset not found: {ROOT}")

if OUTPUT.exists():
    raise FileExistsError(
        f"Output already exists: {OUTPUT}\n"
        "Nothing was changed. Rename the output folder or "
        "choose a new version before running again."
    )

if REPORT.exists():
    raise FileExistsError(
        f"Report already exists: {REPORT}\n"
        "Rename the report before running again."
    )

# ---------- GROUP EXACT DUPLICATES ----------
groups = defaultdict(list)

for file in sorted(ROOT.rglob("*")):
    if not file.is_file() or file.suffix.lower() not in EXTENSIONS:
        continue

    relative = file.relative_to(ROOT)

    if len(relative.parts) < 2:
        continue

    variety = relative.parts[0]
    digest = hashlib.sha256(file.read_bytes()).hexdigest()

    groups[digest].append({
        "path": file,
        "relative": str(relative),
        "variety": variety,
        "digest": digest,
    })

classes = sorted({
    item["variety"]
    for items in groups.values()
    for item in items
})

# ---------- CLASSIFY DUPLICATE GROUPS ----------
kept_candidates = []
manifest = []
conflict_groups = 0
same_label_duplicate_groups = 0

for digest, items in groups.items():
    labels = {item["variety"] for item in items}

    if len(labels) > 1:
        conflict_groups += 1

        for item in items:
            manifest.append({
                "source": item["relative"],
                "variety": item["variety"],
                "sha256": digest,
                "status": "QUARANTINED_CONFLICTING_LABELS",
                "split": "",
                "output": "",
            })
        continue

    # Same image and same label: retain one deterministic copy.
    items = sorted(items, key=lambda x: x["relative"])
    keeper = items[0]
    kept_candidates.append(keeper)

    manifest.append({
        "source": keeper["relative"],
        "variety": keeper["variety"],
        "sha256": digest,
        "status": "KEPT",
        "split": "",
        "output": "",
    })

    if len(items) > 1:
        same_label_duplicate_groups += 1

        for duplicate in items[1:]:
            manifest.append({
                "source": duplicate["relative"],
                "variety": duplicate["variety"],
                "sha256": digest,
                "status": "DUPLICATE_SAME_LABEL",
                "split": "",
                "output": "",
            })

# ---------- STRATIFIED SPLIT ----------
by_class = defaultdict(list)

for item in kept_candidates:
    by_class[item["variety"]].append(item)

rng = random.Random(SEED)
assignments = []
split_counts = Counter()
class_counts = Counter()

for variety in classes:
    items = by_class[variety][:]
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    for index, item in enumerate(items):
        if index < n_train:
            split = "train"
        elif index < n_train + n_val:
            split = "val"
        else:
            split = "test"

        assignments.append((item, split))
        split_counts[split] += 1
        class_counts[(variety, split)] += 1

# ---------- COPY TO NEW DATASET ----------
try:
    for split in ("train", "val", "test"):
        for variety in classes:
            (OUTPUT / split / variety).mkdir(
                parents=True, exist_ok=True
            )

    source_to_manifest = {
        row["source"]: row
        for row in manifest
        if row["status"] == "KEPT"
    }

    for item, split in assignments:
        destination = OUTPUT / split / item["variety"] / item["path"].name

        # Protect against any unexpected filename collision.
        if destination.exists():
            destination = destination.with_name(
                f"{item['path'].stem}_{item['digest'][:8]}"
                f"{item['path'].suffix}"
            )

        shutil.copy2(item["path"], destination)

        row = source_to_manifest[item["relative"]]
        row["split"] = split
        row["output"] = str(destination)

    REPORT.parent.mkdir(parents=True, exist_ok=True)

    with REPORT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source", "variety", "sha256",
                "status", "split", "output"
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)

except Exception:
    print(
        "\nERROR: Cleaning did not finish successfully.\n"
        "Original images were not modified. Check the new output "
        "folder before retrying; do not blindly rerun."
    )
    raise

# ---------- SUMMARY ----------
print("\n===== CLEAN DATASET SUMMARY =====")
print("Original unique image contents:", len(groups))
print("Original image files:", sum(len(x) for x in groups.values()))
print("Varieties found:", len(classes))
print("Conflicting duplicate groups excluded:", conflict_groups)
print("Same-label duplicate groups deduplicated:",
      same_label_duplicate_groups)
print("Images retained:", len(kept_candidates))
print("Train images:", split_counts["train"])
print("Validation images:", split_counts["val"])
print("Test images:", split_counts["test"])
print("Dataset:", OUTPUT)
print("Manifest:", REPORT)

print("\nPer-variety split counts:")
for variety in classes:
    print(
        f"{variety:25} "
        f"train={class_counts[(variety, 'train')]:3} "
        f"val={class_counts[(variety, 'val')]:3} "
        f"test={class_counts[(variety, 'test')]:3}"
    )
