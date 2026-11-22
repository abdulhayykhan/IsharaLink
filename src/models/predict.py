"""Step 8: predict_window(array) for Abdul Hayy's backend.

    from src.models.predict import predict_window, load_assets
    load_assets()                       # once, at server startup
    result = predict_window(array)      # array: (60, F) float array from preprocess_window()

Returns:
    {"label": str, "confidence": float,
     "top3": [{"label": str, "p": float} x3],
     "status": "ok" | "unknown"}

status is "unknown" when confidence < threshold in config.json.
This file is self-contained on purpose (only numpy + tensorflow + json).

Self-test on a real dataset sample (from repo root):
    python src/models/predict.py            # real data
    python src/models/predict.py --dummy    # dummy model
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_STATE: dict = {}


def _resolve_paths():
    model = Path(os.environ.get("ISHARA_MODEL", ROOT / "models" / "ishara_model.keras"))
    classes = Path(os.environ.get("ISHARA_CLASSES", ROOT / "data" / "processed" / "classes.json"))
    cfg_env = os.environ.get("ISHARA_CONFIG")
    if cfg_env:
        config = Path(cfg_env)
    else:
        config = model.parent / "config.json"
        if not config.exists():
            config = ROOT / "config.json"
    return model, classes, config


def load_assets(force: bool = False) -> None:
    """Load model, classes and config once (safe to call many times)."""
    if _STATE and not force:
        return
    import tensorflow as tf

    model_path, classes_path, config_path = _resolve_paths()
    for p in (model_path, classes_path, config_path):
        if not Path(p).exists():
            raise FileNotFoundError(f"predict.py: required file not found: {p}")
    model = tf.keras.models.load_model(model_path)
    with open(classes_path, encoding="utf-8") as f:
        classes = json.load(f)
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    n_out = int(model.output_shape[-1])
    if len(classes) != n_out:
        raise ValueError(f"classes.json has {len(classes)} words but the model outputs {n_out}.")
    _STATE.update(model=model, classes=classes,
                  threshold=float(config["threshold"]),
                  input_shape=tuple(int(d) for d in model.input_shape[1:]))


def predict_window(array) -> dict:
    """Predict the word for one preprocessed window of shape (60, F)."""
    load_assets()
    x = np.asarray(array, dtype="float32")
    expected = _STATE["input_shape"]
    if x.shape != expected:
        raise ValueError(f"Expected array of shape {expected}, got {x.shape}.")
    if not np.isfinite(x).all():
        raise ValueError("Input array contains NaN or infinite values.")

    probs = _STATE["model"](x[None, ...], training=False).numpy()[0]
    classes = _STATE["classes"]
    order = np.argsort(-probs)[:3]
    top3 = [{"label": classes[int(i)], "p": round(float(probs[i]), 4)} for i in order]
    confidence = float(probs[order[0]])
    status = "ok" if confidence >= _STATE["threshold"] else "unknown"
    return {"label": top3[0]["label"], "confidence": round(confidence, 4),
            "top3": top3, "status": status}


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import get_dirs, load_data  # only used by this self-test

    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--n", type=int, default=20, help="how many test samples to check")
    args = ap.parse_args()

    if args.dummy:
        mdir, _ = get_dirs(True)
        os.environ["ISHARA_MODEL"] = str(mdir / "ishara_model.keras")
        os.environ["ISHARA_CLASSES"] = str(mdir / "classes.json")
        os.environ["ISHARA_CONFIG"] = str(mdir / "config.json")

    d = load_data(args.dummy)
    X, y, classes, te = d["X"], d["y"], d["classes"], d["te"]
    k = min(args.n, len(te))
    right = 0
    for n_i, i in enumerate(te[:k]):
        r = predict_window(X[i])
        right += r["label"] == classes[y[i]]
        if n_i < 3:
            print(f"true={classes[y[i]]!r} ->", r)
    print(f"predict_window correct on {right}/{k} test samples")
