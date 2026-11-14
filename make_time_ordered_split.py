"""Build a time-ordered train/val/test split of the rice dataset (v2).

Photos of each variety were taken in one short sitting, so a random split puts
near-identical shots in both train and test. Here each variety's photos are
sorted by the capture time in the file name and cut into consecutive blocks:
train first, then val, then test, with a few images dropped between blocks and
cuts placed only between different seconds. The source dataset is never
modified. By default this is a dry run; use --write to create the new dataset.

This measures generalisation within the same sitting. It cannot measure
performance on other days, fields, cameras or lighting.
"""
import argparse
import csv
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE = PROJECT_ROOT / "datasets" / "rice_classification_clean_v1"
OUTPUT = PROJECT_ROOT / "datasets" / "rice_classification_time_split_v2"
MANIFEST = PROJECT_ROOT / "reports" / "time_split_manifest.csv"
STAMP = re.compile(r"(\d{8})_?(\d{6})(\d{3})?")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
TRAIN_RATIO, VAL_RATIO = 0.80, 0.10


def timestamp(name):
    m = STAMP.search(name)
    if not m:
        raise ValueError(f"No timestamp in file name: {name}")
    ms = int(m.group(3) or 0)
    base = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return base.timestamp() + ms / 1000


def collect():
    by_class = defaultdict(list)
    for split in ("train", "val", "test"):
        for p in sorted((SOURCE / split).rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                by_class[p.parent.name].append((timestamp(p.name), p))
    for items in by_class.values():
        items.sort(key=lambda x: (x[0], x[1].name))
    return by_class


def plan_class(name, items, buffer):
    n = len(items)
    secs = [int(t) for t, _ in items]
    boundaries = [i for i in range(1, n) if secs[i] != secs[i - 1]]

    def snap(target, low):
        candidates = [b for b in boundaries if b >= low]
        if not candidates:
            raise RuntimeError(f"{name}: no valid cut point")
        return min(candidates, key=lambda b: (abs(b - target), b))

    usable = n - 2 * buffer
    n_val = max(1, round(VAL_RATIO * usable))
    cut1 = snap(round(TRAIN_RATIO * usable), 1)
    val_start = cut1 + buffer
    while val_start < n and secs[val_start] == secs[cut1 - 1]:
        val_start += 1
    cut2 = snap(val_start + n_val, val_start + 1)
    test_start = cut2 + buffer
    while test_start < n and secs[test_start] == secs[cut2 - 1]:
        test_start += 1
    if not (0 < cut1 < val_start < cut2 < test_start < n):
        raise RuntimeError(f"{name}: could not place all three blocks")
    return items[:cut1], items[val_start:cut2], items[test_start:], items[cut1:val_start] + items[cut2:test_start]


def dhash(path):
    img = Image.open(path).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--buffer", type=int, default=3, help="images dropped at each block boundary")
    ap.add_argument("--write", action="store_true", help="create the v2 dataset (default: dry run)")
    ap.add_argument("--quiet", action="store_true", help="hide the per-variety table")
    args = ap.parse_args()

    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE}")
    by_class = collect()
    plans = {c: plan_class(c, items, args.buffer) for c, items in sorted(by_class.items())}

    # Invariants: each image used at most once, blocks in strict time order.
    seen = set()
    for c, (tr, va, te, dropped) in plans.items():
        for part in (tr, va, te, dropped):
            for _, p in part:
                assert p not in seen, f"{c}: image assigned twice: {p.name}"
                seen.add(p)
        assert int(tr[-1][0]) < int(va[0][0]) and int(va[-1][0]) < int(te[0][0]), f"{c}: blocks overlap in time"
    assert len(seen) == sum(len(v) for v in by_class.values()), "image count mismatch"

    totals = {"train": 0, "val": 0, "test": 0, "dropped": 0}
    if not args.quiet:
        print(f"{'variety':26} {'n':>4} {'train':>6} {'val':>4} {'test':>5} {'drop':>5}")
    for c, (tr, va, te, dr) in plans.items():
        for k, part in zip(totals, (tr, va, te, dr)):
            totals[k] += len(part)
        if not args.quiet:
            print(f"{c:26} {len(by_class[c]):4d} {len(tr):6d} {len(va):4d} {len(te):5d} {len(dr):5d}")
    print(f"\nbuffer={args.buffer} | images {sum(totals.values())}: train {totals['train']}, "
          f"val {totals['val']}, test {totals['test']}, dropped {totals['dropped']}")
    smallest = min(len(te) for _, _, te, _ in plans.values())
    print(f"smallest test set for any variety: {smallest} images")

    distances, gaps = [], []
    for c, (tr, va, te, dr) in plans.items():
        train_h = np.array([dhash(p) for _, p in tr])
        train_t = np.array([t for t, _ in tr])
        for t, p in te:
            distances.append(int((train_h != dhash(p)).sum(axis=1).min()))
            gaps.append(float(np.abs(train_t - t).min()))
    d = np.array(distances)
    print(f"test-to-train time gap: min {min(gaps):.1f}s, median {float(np.median(gaps)):.1f}s")
    for lo, hi, label in [(0, 5, "0-5 (near-identical)"), (6, 10, "6-10"), (11, 15, "11-15"), (16, 64, ">15 (distinct)")]:
        print(f"  nearest train image, hash distance {label:22}: {int(((d >= lo) & (d <= hi)).sum())}")

    if not args.write:
        print("\nDry run only - nothing was written. Use --write to create the dataset.")
        return

    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists: {OUTPUT}\nNothing was changed.")
    if MANIFEST.exists():
        raise FileExistsError(f"Manifest already exists: {MANIFEST}\nNothing was changed.")
    rows = []
    for c, (tr, va, te, dr) in plans.items():
        for split, part in (("train", tr), ("val", va), ("test", te)):
            (OUTPUT / split / c).mkdir(parents=True, exist_ok=True)
            for t, p in part:
                shutil.copy2(p, OUTPUT / split / c / p.name)
                rows.append({"variety": c, "file": p.name, "timestamp": t, "split": split})
        for t, p in dr:
            rows.append({"variety": c, "file": p.name, "timestamp": t, "split": "DROPPED_BUFFER"})
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variety", "file", "timestamp", "split"])
        w.writeheader()
        w.writerows(rows)
    written = sum(1 for _ in OUTPUT.rglob("*") if _.is_file())
    assert written == totals["train"] + totals["val"] + totals["test"], "written file count mismatch"
    print(f"\nWrote {written} images to {OUTPUT}\nManifest: {MANIFEST}")


if __name__ == "__main__":
    main()
