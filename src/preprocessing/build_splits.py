"""
build_splits.py -- Data Lead Step 5. Build the three evaluation splits.

Indices refer to ROW ORDER in data/processed/labels.csv, which is the same order
build_arrays.py writes X.npy / y.npy / camera.npy. So split index i == X[i].

Splits (saved to data/processed/splits.json):
  1. random          stratified 70/15/15 by word  (labelled "optimistic")
  2. leave_one_camera_out  3 folds: train on two cameras, test on the third
  3. wlpsl           test-only placeholder, filled by Step 7 (wlpsl_extract.py)

Run from repo root:  python -I src/preprocessing/build_splits.py
"""
import os
import csv
import json
import random
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LABELS = os.path.join(REPO_ROOT, "data", "processed", "labels.csv")
OUT = os.path.join(REPO_ROOT, "data", "processed", "splits.json")
CAMERAS = ["mobile_data", "laptop_data", "webcam_data"]
SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def load_rows():
    with open(LABELS, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def stratified_random(rows):
    by_label = defaultdict(list)
    for i, r in enumerate(rows):
        by_label[int(r["label_id"])].append(i)
    rng = random.Random(SEED)
    train, val, test = [], [], []
    for label, idxs in by_label.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = max(1, round(n * TEST_FRAC))
        n_val = max(1, round(n * VAL_FRAC))
        test += idxs[:n_test]
        val += idxs[n_test:n_test + n_val]
        train += idxs[n_test + n_val:]
    return sorted(train), sorted(val), sorted(test)


def leave_one_camera_out(rows):
    folds = []
    for held in CAMERAS:
        test = [i for i, r in enumerate(rows) if r["camera"] == held]
        train = [i for i, r in enumerate(rows) if r["camera"] != held]
        folds.append({"held_out": held, "train": train, "test": test})
    return folds


def check_disjoint(a, b, name):
    overlap = set(a) & set(b)
    assert not overlap, f"{name}: {len(overlap)} overlapping indices!"


def main():
    rows = load_rows()
    n = len(rows)

    tr, va, te = stratified_random(rows)
    check_disjoint(tr, va, "random train/val")
    check_disjoint(tr, te, "random train/test")
    check_disjoint(va, te, "random val/test")
    assert len(tr) + len(va) + len(te) == n

    folds = leave_one_camera_out(rows)
    for f in folds:
        check_disjoint(f["train"], f["test"], f"loco[{f['held_out']}]")
        assert len(f["train"]) + len(f["test"]) == n

    splits = {
        "meta": {
            "n_samples": n,
            "index_ref": "row order of data/processed/labels.csv == X.npy order",
            "seed": SEED,
        },
        "random": {"train": tr, "val": va, "test": te},
        "leave_one_camera_out": folds,
        "wlpsl": {"note": "test-only; filled by src/preprocessing/wlpsl_extract.py (Step 7)",
                  "test": []},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(splits, fh)

    print(f"random split      : train={len(tr)} val={len(va)} test={len(te)} (total {n})")
    for f in folds:
        print(f"leave-one-cam-out : held={f['held_out']:12s} train={len(f['train'])} test={len(f['test'])}")
    print("overlap checks    : all zero (asserts passed)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
