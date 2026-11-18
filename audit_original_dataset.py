"""Audit the raw rice dataset for duplicate images.

Groups images by content hash and reports how many contents are duplicated and
how many identical images appear under different variety folders (conflicting
labels). It only reads files.
"""
from collections import defaultdict
from pathlib import Path
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT / "dataset" / "Primitive_Rice_OnField_new"
EXTENSIONS = {".jpg", ".jpeg", ".png"}

if not ROOT.exists():
    raise FileNotFoundError(f"Original dataset not found: {ROOT}")

groups = defaultdict(list)

for file in ROOT.rglob("*"):
    if not file.is_file() or file.suffix.lower() not in EXTENSIONS:
        continue

    digest = hashlib.md5(file.read_bytes()).hexdigest()
    relative = file.relative_to(ROOT)

    if len(relative.parts) >= 2:
        variety = relative.parts[0]
        groups[digest].append((variety, str(relative)))

duplicates = [items for items in groups.values() if len(items) > 1]
conflicts = [items for items in duplicates
             if len({variety for variety, _ in items}) > 1]

print("===== ORIGINAL DATASET AUDIT =====")
print("Unique image contents:", len(groups))
print("Duplicate groups:", len(duplicates))
print("Groups in different variety folders:", len(conflicts))

for items in conflicts[:20]:
    print("\nDuplicate image group:")
    for variety, path in items:
        print(f"  {variety:20} | {path}")
