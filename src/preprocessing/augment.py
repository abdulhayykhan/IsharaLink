"""
augment.py -- Data Lead Step 8. Training-time augmentation for landmark sequences.

augment(seq) applies small random scale, in-plane rotation, coordinate jitter and a
time shift to a normalized (60,126) sequence, leaving missing (all-zero) hands
untouched and keeping the shape unchanged.

**Training data ONLY.** Never call this on validation or test samples.

Run a self-test:  python -I src/preprocessing/augment.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling import under -I
from preprocess import NUM_HANDS, NUM_POINTS, COORDS, F, TARGET_FRAMES  # noqa: E402


def augment(seq, rng=None,
            scale_range=0.10, max_rot_rad=0.17, jitter_std=0.01, max_time_shift=3):
    """(60,126) -> augmented (60,126) float32. Missing hands stay zero."""
    if rng is None:
        rng = np.random.default_rng()
    g = np.asarray(seq, dtype=np.float32).reshape(-1, NUM_HANDS, NUM_POINTS, COORDS).copy()
    T = g.shape[0]

    scale = 1.0 + rng.uniform(-scale_range, scale_range)
    ang = rng.uniform(-max_rot_rad, max_rot_rad)
    ca, sa = np.cos(ang), np.sin(ang)

    for t in range(T):
        for h in range(NUM_HANDS):
            pts = g[t, h]
            if not np.any(pts):                 # missing hand -> leave zeros
                continue
            pts *= scale
            x = pts[:, 0].copy()
            y = pts[:, 1].copy()
            pts[:, 0] = ca * x - sa * y          # rotate in the x-y plane
            pts[:, 1] = sa * x + ca * y
            pts += rng.normal(0.0, jitter_std, size=pts.shape).astype(np.float32)
            g[t, h] = pts

    out = g.reshape(T, F)
    k = int(rng.integers(-max_time_shift, max_time_shift + 1))
    if k != 0:
        out = np.roll(out, k, axis=0)            # small temporal shift
    return out.astype(np.float32)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # a fake one-handed normalized sample: right hand present, left hand missing
    seq = np.zeros((TARGET_FRAMES, F), dtype=np.float32)
    seq[:, F // 2:] = rng.normal(0, 0.5, size=(TARGET_FRAMES, F // 2)).astype(np.float32)
    aug = augment(seq, rng)
    changed = float(np.mean(np.abs(aug - seq)) > 0)
    left_still_zero = bool(np.all(aug[:, : F // 2] == 0))
    print(f"shape in/out : {seq.shape} -> {aug.shape}")
    print(f"values changed : {bool(changed)}")
    print(f"missing (left) hand stayed zero : {left_still_zero}")
    print(f"no NaN : {not bool(np.isnan(aug).any())}")
