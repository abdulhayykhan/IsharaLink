"""
preprocess.py -- IsharaLink shared preprocessing (Data Lead: Zainab).

THE single copy of preprocessing, used by BOTH:
  * training-array building (preprocess_clip_dir, reads dataset .npy frames)
  * the live FastAPI server (preprocess_window, reads the raw browser payload)
so the browser and the training data can never disagree.

Pipeline (both paths): assemble (T,126) -> resample to 60 frames -> wrist-relative
+ hand-size normalization -> (60, 126) float32.

Feature layout (confirmed from the dataset authors' MediaPipe Holistic code):
  index [0:63]   = LEFT  hand  (21 landmarks x [x,y,z])
  index [63:126] = RIGHT hand  (21 landmarks x [x,y,z])
  a missing hand is all zeros.
"""
import os
import numpy as np

# --- layout constants (CONTRACT) ---------------------------------------------
NUM_HANDS = 2
NUM_POINTS = 21
COORDS = 3
HAND_LEN = NUM_POINTS * COORDS          # 63
F = NUM_HANDS * HAND_LEN                 # 126
TARGET_FRAMES = 60
WRIST = 0                               # landmark 0
MID_FINGER_MCP = 9                      # landmark 9, used for hand-size scale
EPS = 1e-6

# If a frontend test shows hands landing in the wrong slot, set MIRROR = True to
# swap the browser's left/right assignment (see DATA_NOTES.md §3f).
MIRROR = False


# --- frame assembly ----------------------------------------------------------
def _hand_vec(points):
    """One hand -> (63,) float32. None/missing -> zeros. Defensive on bad shapes."""
    if points is None:
        return np.zeros(HAND_LEN, dtype=np.float32)
    arr = np.asarray(points, dtype=np.float32)
    if arr.shape == (NUM_POINTS, COORDS):
        return arr.reshape(-1)
    flat = arr.reshape(-1)[:HAND_LEN]
    v = np.zeros(HAND_LEN, dtype=np.float32)
    v[: flat.size] = flat
    return v


def frames_to_array(raw_frames):
    """Browser payload -> (T, 126). raw_frames: list of {"left":..., "right":...}."""
    rows = []
    for fr in raw_frames:
        left = fr.get("left")
        right = fr.get("right")
        if MIRROR:
            left, right = right, left
        rows.append(np.concatenate([_hand_vec(left), _hand_vec(right)]))
    if not rows:
        return np.zeros((0, F), dtype=np.float32)
    return np.stack(rows).astype(np.float32)


# --- resampling --------------------------------------------------------------
def resample(seq, target=TARGET_FRAMES):
    """(T,126) -> (target,126) by nearest-index sampling.

    Nearest (not linear) so we never blend a present hand with a missing
    (all-zero) neighbour into a half-strength ghost hand.
    """
    seq = np.asarray(seq, dtype=np.float32)
    n = seq.shape[0]
    if n == 0:
        return np.zeros((target, F), dtype=np.float32)
    if n == target:
        return seq
    idx = np.rint(np.linspace(0, n - 1, target)).astype(int)
    idx = np.clip(idx, 0, n - 1)
    return seq[idx].astype(np.float32)


# --- normalization -----------------------------------------------------------
def normalize(seq):
    """(T,126) -> (T,126): per hand, per frame, wrist-relative + hand-size scale.

    A present hand's wrist becomes the origin; coordinates are divided by the
    wrist->middle-finger-MCP (x,y) distance. A missing (all-zero) hand stays zero.
    """
    seq = np.asarray(seq, dtype=np.float32).reshape(-1, NUM_HANDS, NUM_POINTS, COORDS)
    out = seq.copy()
    for t in range(seq.shape[0]):
        for h in range(NUM_HANDS):
            pts = seq[t, h]
            if not np.any(pts):          # missing hand -> keep zeros
                continue
            wrist = pts[WRIST]
            rel = pts - wrist
            dx = pts[MID_FINGER_MCP, 0] - pts[WRIST, 0]
            dy = pts[MID_FINGER_MCP, 1] - pts[WRIST, 1]
            scale = float((dx * dx + dy * dy) ** 0.5)
            if scale < EPS:
                scale = 1.0
            out[t, h] = rel / scale
    return out.reshape(-1, F).astype(np.float32)


# --- public entry points -----------------------------------------------------
def preprocess_window(raw_frames):
    """Live API path: raw browser frames -> (60, 126) float32."""
    seq = frames_to_array(raw_frames)
    seq = resample(seq, TARGET_FRAMES)
    seq = normalize(seq)
    return seq.astype(np.float32)


def load_clip_frames(clip_dir):
    """Read a dataset clip folder of per-frame .npy -> (T, 126), frames in order."""
    files = [f for f in os.listdir(clip_dir) if f.endswith(".npy")]
    files.sort(key=lambda x: int(x[:-4]) if x[:-4].isdigit() else 0)
    frames = []
    for f in files:
        a = np.load(os.path.join(clip_dir, f))   # allow_pickle=False (default)
        frames.append(np.asarray(a, dtype=np.float32).reshape(-1)[:F])
    if not frames:
        return np.zeros((0, F), dtype=np.float32)
    return np.stack(frames).astype(np.float32)


def preprocess_clip_dir(clip_dir):
    """Training path: dataset clip folder -> (60, 126) float32.

    Uses the SAME resample + normalize as preprocess_window, so training arrays
    and live inference are produced identically.
    """
    seq = load_clip_frames(clip_dir)
    seq = resample(seq, TARGET_FRAMES)
    seq = normalize(seq)
    return seq.astype(np.float32)
