"""
make_test_cases.py -- generate tests/preprocess_cases.json (Data Lead, Step 4).

Picks real dataset clips, reconstructs the raw browser payload from them, runs
preprocess_window(), and saves {input frames -> expected (60,126) output} so the
backend can assert its imported preprocess_window matches Zainab's exactly.

Run from repo root:  python -I src/preprocessing/make_test_cases.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # allow sibling import under -I
import preprocess as P  # noqa: E402  (sibling module in this folder)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_ROOT = os.path.join(
    REPO_ROOT, "data", "raw",
    "PakistanSignLanguageDatasetV2", "PakistanSignLanguageDatasetV2",
)
CAMERAS = ["mobile_data", "laptop_data", "webcam_data"]
OUT = os.path.join(REPO_ROOT, "tests", "preprocess_cases.json")
ROUND = 6
TOL = 1e-5


def reconstruct_browser(seq):
    """(T,126) dataset vectors -> browser frames [{left, right}] (null if hand absent)."""
    frames = []
    for v in seq:
        g = v.reshape(P.NUM_HANDS, P.NUM_POINTS, P.COORDS)
        left = None if not np.any(g[0]) else g[0].tolist()
        right = None if not np.any(g[1]) else g[1].tolist()
        frames.append({"left": left, "right": right})
    return frames


def presence(seq):
    g = seq.reshape(-1, P.NUM_HANDS, P.NUM_POINTS, P.COORDS)
    l = float(np.mean([np.any(g[t, 0]) for t in range(len(g))])) if len(g) else 0.0
    r = float(np.mean([np.any(g[t, 1]) for t in range(len(g))])) if len(g) else 0.0
    return l, r


def iter_clip_dirs(limit=400):
    for cam in CAMERAS:
        cdir = os.path.join(DATASET_ROOT, cam)
        for w in sorted(d.name for d in os.scandir(cdir) if d.is_dir()):
            wdir = os.path.join(cdir, w)
            for c in sorted((d.name for d in os.scandir(wdir) if d.is_dir()),
                            key=lambda x: int(x) if x.isdigit() else 0):
                yield cam, w, os.path.join(wdir, c)
                limit -= 1
                if limit <= 0:
                    return


def find_clips():
    two_handed = one_handed = None
    for cam, w, cdir in iter_clip_dirs():
        seq = P.load_clip_frames(cdir)
        if seq.shape[0] < 20:
            continue
        l, r = presence(seq)
        if two_handed is None and l > 0.6 and r > 0.6:
            two_handed = (cam, w, cdir, seq)
        if one_handed is None and ((l < 0.1 and r > 0.6) or (r < 0.1 and l > 0.6)):
            one_handed = (cam, w, cdir, seq)
        if two_handed and one_handed:
            break
    return two_handed, one_handed


def main():
    two, one = find_clips()
    cases = []

    def add_case(name, note, browser_frames, expected, consistency=None):
        exp = np.round(expected, ROUND)
        assert exp.shape == (P.TARGET_FRAMES, P.F), exp.shape
        assert not np.isnan(exp).any()
        cases.append({
            "name": name,
            "note": note,
            "frames": browser_frames,
            "expected_shape": list(exp.shape),
            "expected": exp.tolist(),
        })
        print(f"  {name:18s} frames_in={len(browser_frames):3d} -> {exp.shape} "
              f"nan={bool(np.isnan(exp).any())}"
              + (f" consistency(max|diff|)={consistency:.2e}" if consistency is not None else ""))

    # Case 1: two-handed, full length -> also check browser-path == training-path
    cam, w, cdir, seq = two
    bf = reconstruct_browser(seq)
    out_browser = P.preprocess_window(bf)
    out_train = P.preprocess_clip_dir(cdir)
    diff = float(np.max(np.abs(out_browser - out_train)))
    assert diff < TOL, f"browser vs training mismatch: {diff}"
    add_case("two_handed", f"{cam}/{w} two-handed clip, {seq.shape[0]} frames", bf, out_browser, diff)

    # Case 2: one-handed, full length -> consistency check too
    cam, w, cdir, seq = one
    bf = reconstruct_browser(seq)
    out_browser = P.preprocess_window(bf)
    out_train = P.preprocess_clip_dir(cdir)
    diff = float(np.max(np.abs(out_browser - out_train)))
    assert diff < TOL, f"browser vs training mismatch: {diff}"
    add_case("one_handed", f"{cam}/{w} one-handed clip, {seq.shape[0]} frames", bf, out_browser, diff)

    # Case 3: short input (30 frames) to exercise upsampling to 60
    cam, w, cdir, seq = two
    short = seq[np.rint(np.linspace(0, seq.shape[0] - 1, 30)).astype(int)]
    bf = reconstruct_browser(short)
    out_browser = P.preprocess_window(bf)
    add_case("short_resample", f"{cam}/{w} subsampled to 30 frames -> resampled to 60", bf, out_browser)

    payload = {
        "description": "Golden preprocessing cases. Load 'frames', call preprocess_window(frames), "
                       "compare to 'expected' with np.allclose(..., atol=1e-5).",
        "output_shape": [P.TARGET_FRAMES, P.F],
        "tolerance": TOL,
        "feature_layout": "126 = [left 21x(x,y,z)] + [right 21x(x,y,z)]; missing hand = zeros",
        "cases": cases,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    size_kb = os.path.getsize(OUT) / 1024
    print(f"\nwrote {OUT}  ({len(cases)} cases, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
