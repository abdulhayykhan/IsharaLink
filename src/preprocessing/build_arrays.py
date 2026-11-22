"""
build_arrays.py -- Data Lead Step 6. Turn the raw clips into model-ready arrays.

Iterates data/processed/labels.csv IN ORDER and runs preprocess_clip_dir on each
clip, producing:
    data/processed/X.npy       (N, 60, 126) float32   (git-ignored, large)
    data/processed/y.npy       (N,)          int16     integer label ids
    data/processed/camera.npy  (N,)          int8      0=mobile 1=laptop 2=webcam
    data/processed/cameras.json  camera code -> name

Row i of every array corresponds to row i of labels.csv and to split index i.

Reading ~385k tiny .npy files is slow on Windows, so clips are processed in
parallel across CPU cores (multiprocessing).

Run from repo root:  python -u -I src/preprocessing/build_arrays.py
"""
import os
import sys
import csv
import json
import time
import numpy as np
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling import under -I
import preprocess as P  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_ROOT = os.path.join(
    REPO_ROOT, "data", "raw",
    "PakistanSignLanguageDatasetV2", "PakistanSignLanguageDatasetV2",
)
PROC = os.path.join(REPO_ROOT, "data", "processed")
LABELS = os.path.join(PROC, "labels.csv")
CAMERAS = ["mobile_data", "laptop_data", "webcam_data"]
CAM_CODE = {c: i for i, c in enumerate(CAMERAS)}
N_WORKERS = min(10, (os.cpu_count() or 4))


def process_one(rel_path):
    """Worker: dataset-relative clip path -> (60,126) float32."""
    clip_dir = os.path.join(DATASET_ROOT, *rel_path.split("/"))
    return P.preprocess_clip_dir(clip_dir)


def main():
    with open(LABELS, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    n = len(rows)
    paths = [r["path"] for r in rows]
    print(f"building arrays for {n} clips using {N_WORKERS} workers ...", flush=True)

    X = np.zeros((n, P.TARGET_FRAMES, P.F), dtype=np.float32)
    y = np.array([int(r["label_id"]) for r in rows], dtype=np.int16)
    cam = np.array([CAM_CODE[r["camera"]] for r in rows], dtype=np.int8)

    bad = 0
    t0 = time.time()
    with mp.Pool(processes=N_WORKERS) as pool:
        for i, arr in enumerate(pool.imap(process_one, paths, chunksize=16)):
            if arr.shape != (P.TARGET_FRAMES, P.F) or np.isnan(arr).any():
                bad += 1
                arr = np.nan_to_num(arr)[: P.TARGET_FRAMES]
            X[i] = arr
            if (i + 1) % 500 == 0 or (i + 1) == n:
                rate = (i + 1) / (time.time() - t0)
                eta = (n - (i + 1)) / rate
                print(f"  {i + 1}/{n}  ({rate:.0f} clips/s, ETA {eta:.0f}s)", flush=True)

    np.save(os.path.join(PROC, "X.npy"), X)
    np.save(os.path.join(PROC, "y.npy"), y)
    np.save(os.path.join(PROC, "camera.npy"), cam)
    with open(os.path.join(PROC, "cameras.json"), "w", encoding="utf-8") as fh:
        json.dump(CAMERAS, fh, indent=2)

    print("\n=== DONE ===", flush=True)
    print(f"X: {X.shape} {X.dtype}   y: {y.shape} {y.dtype}   camera: {cam.shape} {cam.dtype}")
    print(f"total time: {time.time() - t0:.0f}s   rows patched: {bad}")
    print(f"label range: {int(y.min())}..{int(y.max())}  |  NaNs in X: {int(np.isnan(X).sum())}")
    print(f"saved to {PROC}")


if __name__ == "__main__":
    main()
