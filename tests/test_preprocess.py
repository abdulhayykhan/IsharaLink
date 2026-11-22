"""
test_preprocess.py -- Unit tests for preprocessing pipeline.

Verifies:
  1. preprocess_window matches golden expected outputs (tolerance 1e-5).
  2. Resampling handles variable input lengths (e.g. 30 frames -> 60 frames).
  3. Feature shape is always (60, 126) with zero NaNs.
"""
import json
from pathlib import Path
import numpy as np
import pytest

from src.preprocessing.preprocess import preprocess_window, TARGET_FRAMES, F

ROOT_DIR = Path(__file__).resolve().parents[1]


def test_preprocess_golden_cases():
    """Verify all golden cases in tests/preprocess_cases.json."""
    cases_path = ROOT_DIR / "tests" / "preprocess_cases.json"
    with open(cases_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for case in data["cases"]:
        name = case["name"]
        raw_frames = case["frames"]
        expected = np.array(case["expected"], dtype=np.float32)

        actual = preprocess_window(raw_frames)
        assert actual.shape == (TARGET_FRAMES, F), f"Shape mismatch on {name}"
        assert not np.isnan(actual).any(), f"NaNs found in {name}"
        assert np.allclose(actual, expected, atol=1e-5), f"Failed numerical check on {name}"


def test_preprocess_short_clip():
    """Subsampled input should resample cleanly to 60 frames."""
    dummy_frame = {
        "left": [[0.1, 0.2, 0.0] for _ in range(21)],
        "right": None
    }
    short_frames = [dummy_frame for _ in range(15)]
    out = preprocess_window(short_frames)
    assert out.shape == (60, 126)
    assert not np.isnan(out).any()


def test_preprocess_all_missing_hands():
    """Missing hands should produce an all-zeros array without errors."""
    empty_frames = [{"left": None, "right": None} for _ in range(30)]
    out = preprocess_window(empty_frames)
    assert out.shape == (60, 126)
    assert np.all(out == 0.0)
