"""Basic validation of the cleaned rice dataset.

Checks that every variety has images in the train, val and test splits, that no
exact duplicate image (SHA-256) appears in more than one split, and that the
cleaning manifest agrees with the dataset on disk (one KEPT row per image).
It only reads files. The exit code is 1 when issues are found.

It does not detect near-duplicates (the same photo re-saved or taken a second
apart); see check_split_leakage.py for that.
"""
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT / "datasets" / "rice_classification_clean_v1"
REPORT = PROJECT_ROOT / "reports" / "rice_cleaning_manifest.csv"
SPLITS = ("train", "val", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png"}

errors = []
split_hashes = defaultdict(set)
class_counts = defaultdict(dict)
total = 0

if not ROOT.exists():
    raise FileNotFoundError(f"Dataset missing: {ROOT}")

for split in SPLITS:
    split_dir = ROOT / split

    if not split_dir.exists():
        errors.append(f"Missing split folder: {split_dir}")
        continue

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        files = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ]
        class_counts[class_dir.name][split] = len(files)
        total += len(files)

        for file in files:
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            split_hashes[digest].add(split)

leaks = {
    digest: splits
    for digest, splits in split_hashes.items()
    if len(splits) > 1
}

kept_rows = None
if not REPORT.exists():
    errors.append(f"Missing manifest: {REPORT}")
else:
    with REPORT.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    statuses = defaultdict(int)
    for row in rows:
        statuses[row["status"]] += 1

    print("Manifest rows:", len(rows))
    print("Manifest statuses:", dict(statuses))
    kept_rows = statuses.get("KEPT", 0)

print("\n===== VALIDATION =====")
print("Classes:", len(class_counts))
print("Total cleaned images:", total)
print("Exact images leaking across splits:", len(leaks))

for class_name in sorted(class_counts):
    counts = class_counts[class_name]
    missing = [s for s in SPLITS if counts.get(s, 0) == 0]
    if missing:
        errors.append(
            f"{class_name}: missing images in {', '.join(missing)}"
        )

if kept_rows is not None and kept_rows != total:
    errors.append(
        f"Manifest lists {kept_rows} kept images but the dataset contains {total}."
    )

if leaks:
    errors.append("Exact duplicate content exists across splits.")

if errors:
    print("\nISSUES FOUND:")
    for error in errors:
        print("-", error)
    sys.exit(1)

print(
    "\nBasic validation passed (exact duplicates only; near-duplicates are "
    "not checked, see check_split_leakage.py)."
)
