"""
build_dataset.py  -- Data Lead (Zainab) pipeline, Step 3 / Step 5 / Step 6.

Scans the raw WordLevel PSL dataset and produces the reproducible label
artifacts the rest of the team depends on:

    data/processed/classes.json   ordered word list (index == label_id)  [CONTRACT]
    data/processed/labels.csv     one row per clip: path, word, camera, label_id
    data/processed/clip_counts.csv  inventory table (per word per camera) for the report

Vocabulary cleaning rule (documented in DATA_NOTES.md):
  A word is KEPT only if it
    1. appears in ALL three cameras (needed for leave-one-camera-out eval),
    2. is not a junk/test class, and
    3. has at least MIN_CLIPS_PER_CAMERA clips in every camera.
Everything else is dropped and reported with a reason in clip_counts.csv.

Run from repo root:  python -I src/preprocessing/build_dataset.py
"""
import os
import csv
import json
from collections import Counter

# --- config ------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_ROOT = os.path.join(
    REPO_ROOT, "data", "raw",
    "PakistanSignLanguageDatasetV2", "PakistanSignLanguageDatasetV2",
)
CAMERAS = ["mobile_data", "laptop_data", "webcam_data"]
JUNK_WORDS = {"test2", "test_word", "s"}   # obvious recording-test artifacts
MIN_CLIPS_PER_CAMERA = 15                   # drop words with a broken camera

OUT_DIR = os.path.join(REPO_ROOT, "data", "processed")


def _subdirs(path):
    return sorted(d.name for d in os.scandir(path) if d.is_dir())


def scan():
    """Return clip_counts[camera][word] = n_clips and the set of all words."""
    clip_counts = {}
    all_words = set()
    for cam in CAMERAS:
        cdir = os.path.join(DATASET_ROOT, cam)
        clip_counts[cam] = {}
        for w in _subdirs(cdir):
            all_words.add(w)
            clip_counts[cam][w] = len(_subdirs(os.path.join(cdir, w)))
    return clip_counts, all_words


def select_vocab(clip_counts, all_words):
    """Apply the cleaning rule. Returns (kept_sorted, dropped[word]=reason)."""
    kept, dropped = [], {}
    for w in sorted(all_words):
        counts = {cam: clip_counts[cam].get(w, 0) for cam in CAMERAS}
        if w in JUNK_WORDS:
            dropped[w] = "junk/test class"
        elif any(counts[cam] == 0 for cam in CAMERAS):
            missing = [cam for cam in CAMERAS if counts[cam] == 0]
            dropped[w] = "absent in " + ",".join(missing)
        elif any(counts[cam] < MIN_CLIPS_PER_CAMERA for cam in CAMERAS):
            low = [f"{cam}={counts[cam]}" for cam in CAMERAS if counts[cam] < MIN_CLIPS_PER_CAMERA]
            dropped[w] = "too few clips: " + ",".join(low)
        else:
            kept.append(w)
    return kept, dropped


def write_classes(kept):
    path = os.path.join(OUT_DIR, "classes.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(kept, fh, indent=2, ensure_ascii=False)
    return path


def write_labels(kept, clip_counts):
    label_id = {w: i for i, w in enumerate(kept)}
    path = os.path.join(OUT_DIR, "labels.csv")
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["path", "word", "camera", "label_id"])
        for cam in CAMERAS:
            cdir = os.path.join(DATASET_ROOT, cam)
            for w in kept:
                wdir = os.path.join(cdir, w)
                if not os.path.isdir(wdir):
                    continue
                for clip in _subdirs(wdir):
                    rel = f"{cam}/{w}/{clip}"      # relative to DATASET_ROOT
                    wr.writerow([rel, w, cam, label_id[w]])
                    n += 1
    return path, n


def write_counts(clip_counts, all_words, kept, dropped):
    path = os.path.join(OUT_DIR, "clip_counts.csv")
    kept_set = set(kept)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["word", "mobile", "laptop", "webcam", "total", "kept", "reason"])
        for w in sorted(all_words):
            counts = [clip_counts[cam].get(w, 0) for cam in CAMERAS]
            kept_flag = w in kept_set
            wr.writerow([w, *counts, sum(counts), int(kept_flag),
                         "" if kept_flag else dropped.get(w, "")])
    return path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    clip_counts, all_words = scan()
    kept, dropped = select_vocab(clip_counts, all_words)

    cpath = write_classes(kept)
    lpath, nrows = write_labels(kept, clip_counts)
    npath = write_counts(clip_counts, all_words, kept, dropped)

    print(f"raw words found        : {len(all_words)}")
    print(f"vocabulary kept        : {len(kept)}")
    print(f"dropped                : {len(dropped)}")
    for w, r in dropped.items():
        print(f"   - {w:20s} {r}")
    print(f"total clips (labels)   : {nrows}")
    # per-camera kept totals
    for cam in CAMERAS:
        tot = sum(clip_counts[cam].get(w, 0) for w in kept)
        print(f"   {cam:12s} clips kept: {tot}")
    print(f"\nwrote: {cpath}")
    print(f"wrote: {lpath}")
    print(f"wrote: {npath}")


if __name__ == "__main__":
    main()
