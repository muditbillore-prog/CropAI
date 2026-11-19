"""Compare two rice classifiers on frames sampled from videos.

Frames are sampled at evenly spaced timestamps, as the dashboard does, and both
models predict on the same frames. No labels are needed, so this shows how often
the models agree and how confident each is, not which one is correct.
"""
import argparse
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}

parser = argparse.ArgumentParser(description="Compare two models on sampled video frames.")
parser.add_argument("--folder", required=True, help="folder containing the videos")
parser.add_argument("--pattern", default="*", help="file name pattern, for example 'vidssave*'")
parser.add_argument("--model-a", default="models/best.pt",
                    help="first model, relative to the project folder (default: the deployed model)")
parser.add_argument("--model-b",
                    default="runs/rice_classification/rice_variety_time_split_v2/weights/best.pt",
                    help="second model, relative to the project folder (default: the time-split model)")
parser.add_argument("--frames", type=int, default=10, help="frames sampled per video (default: 10)")
parser.add_argument("--threshold", type=float, default=0.9,
                    help="confidence below which a prediction counts as uncertain (default: 0.9)")
args = parser.parse_args()

paths = {"A": PROJECT_ROOT / args.model_a, "B": PROJECT_ROOT / args.model_b}
for label, path in paths.items():
    if not path.is_file():
        raise FileNotFoundError(f"Model {label} not found: {path}")
videos = sorted(p for p in Path(args.folder).glob(args.pattern) if p.suffix.lower() in VIDEO_SUFFIXES)
if not videos:
    raise RuntimeError(f"No videos matching '{args.pattern}' in {args.folder}")
models = {label: YOLO(str(path)) for label, path in paths.items()}
print(f"Model A: {args.model_a}")
print(f"Model B: {args.model_b}")


def predict(model, rgb):
    result = model.predict(source=Image.fromarray(rgb), imgsz=224, device="cpu", verbose=False)[0]
    return model.names[result.probs.top1], float(result.probs.top1conf)


def sample_frames(video, count):
    """Return ([(seconds, rgb frame)], number of timestamps requested)."""
    with tempfile.NamedTemporaryFile(suffix=video.suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        shutil.copyfile(video, tmp_path)
        cap = cv2.VideoCapture(str(tmp_path))
        try:
            if not cap.isOpened():
                return [], count
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
            stamps = [duration * i / count for i in range(count)] if duration > 0 else []
            frames = []
            for t in stamps:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, bgr = cap.read()
                if ok:
                    frames.append((t, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            return frames, len(stamps)
        finally:
            cap.release()
    finally:
        tmp_path.unlink(missing_ok=True)


def mode(names):
    name, count = Counter(names).most_common(1)[0]
    return f"{name} ({count}/{len(names)})"


all_rows = []
for index, video in enumerate(videos, 1):
    frames, requested = sample_frames(video, args.frames)
    shown = video.name.encode("ascii", "replace").decode()[:55]
    print(f"\n=== video {index}: {shown} ({len(frames)} of {requested} frames read)")
    rows = [(t, predict(models["A"], rgb), predict(models["B"], rgb)) for t, rgb in frames]
    if not rows:
        continue
    all_rows += rows
    print(f"{'time':>6}  {'model A':30}  {'model B':30}  same")
    for t, a, b in rows:
        print(f"{t:5.1f}s  {a[0][:20]:20} {a[1]:7.1%}  {b[0][:20]:20} {b[1]:7.1%}  {'yes' if a[0] == b[0] else 'no'}")
    agree = sum(a[0] == b[0] for _, a, b in rows)
    print(f"agree on {agree}/{len(rows)} frames | most common: A {mode([a[0] for _, a, _ in rows])}, "
          f"B {mode([b[0] for _, _, b in rows])}")

if all_rows:
    n = len(all_rows)
    agree = sum(a[0] == b[0] for _, a, b in all_rows)
    print(f"\n===== ALL VIDEOS: {n} frames =====")
    print(f"agreement: {agree}/{n} ({agree / n:.1%})")
    for label, pick in (("A", 1), ("B", 2)):
        conf = np.array([row[pick][1] for row in all_rows])
        print(f"model {label}: mean confidence {conf.mean():.1%}, "
              f"below {args.threshold:.0%}: {int((conf < args.threshold).sum())}/{n}")
