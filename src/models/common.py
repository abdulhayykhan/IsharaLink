"""Shared helpers for the IsharaLink model-lead scripts (Bushra, Part 2).

Used by: train.py, baselines.py, experiments.py, evaluate.py, cnn1d.py.
predict.py is intentionally self-contained (it does NOT import this file)
so Abdul Hayy's backend can import it without any path tricks.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]          # repo root
DATA_DIR = ROOT / "data" / "processed"
SEED = 42
CAMERA_NAMES = {0: "mobile", 1: "laptop", 2: "webcam"}


# --------------------------------------------------------------------------
# Paths / seeds
# --------------------------------------------------------------------------
def get_dirs(dummy: bool = False):
    """Return (models_dir, results_dir). Dummy runs never touch the real ones."""
    suffix = "_dummy" if dummy else ""
    models_dir = ROOT / f"models{suffix}"
    results_dir = ROOT / f"results{suffix}"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return models_dir, results_dir


def set_seed(seed: int = SEED):
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def stratified_split_indices(y, seed: int = SEED):
    """Stratified 70/15/15 split -> (train, val, test) index arrays."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=seed)
    return np.sort(tr), np.sort(va), np.sort(te)


def make_dummy(n_per_class: int = 10, n_classes: int = 62, T: int = 60, F: int = 126, seed: int = 0):
    """Random but learnable arrays with the same shapes as the real data."""
    rng = np.random.default_rng(seed)
    N = n_per_class * n_classes
    y = np.repeat(np.arange(n_classes), n_per_class).astype("int64")
    centers = rng.normal(0, 1, (n_classes, 1, F)).astype("float32")
    X = (centers[y] + 0.8 * rng.normal(size=(N, T, F))).astype("float32")
    cam = (np.arange(N) % 3).astype("int64")
    classes = [f"word_{i}" for i in range(n_classes)]
    return X, y, cam, classes


def load_data(dummy: bool = False) -> dict:
    """Load X, y, camera, classes and the random split.

    Returns a dict with keys: X, y, cam, classes, tr, va, te, dummy.
    Real data comes from data/processed/ (made by Zainab).
    """
    if dummy:
        X, y, cam, classes = make_dummy()
        tr, va, te = stratified_split_indices(y)
    else:
        needed = ["X.npy", "y.npy", "camera.npy", "classes.json", "splits.json"]
        missing = [f for f in needed if not (DATA_DIR / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing in {DATA_DIR}: {missing}. Copy Zainab's processed files there, "
                "or run the script with --dummy to test the code on random data."
            )
        X = np.load(DATA_DIR / "X.npy").astype("float32")
        y = np.load(DATA_DIR / "y.npy").astype("int64")
        cam = np.load(DATA_DIR / "camera.npy").astype("int64")
        with open(DATA_DIR / "classes.json", encoding="utf-8") as f:
            classes = json.load(f)
        with open(DATA_DIR / "splits.json", encoding="utf-8") as f:
            splits = json.load(f)
        tr = np.asarray(splits["random"]["train"], dtype="int64")
        va = np.asarray(splits["random"]["val"], dtype="int64")
        te = np.asarray(splits["random"]["test"], dtype="int64")

    if X.ndim != 3:
        raise ValueError(f"X must be (N, T, F), got {X.shape}")
    if not (len(X) == len(y) == len(cam)):
        raise ValueError(f"Row counts differ: X={len(X)} y={len(y)} camera={len(cam)}")
    if np.isnan(X).any():
        raise ValueError("X contains NaN; ask Zainab to fix preprocessing.")
    if y.min() < 0 or y.max() >= len(classes):
        raise ValueError("y has label ids outside classes.json range.")
    if set(tr) & set(te) or set(tr) & set(va) or set(va) & set(te):
        raise ValueError("Random split indices overlap!")
    return dict(X=X, y=y, cam=cam, classes=list(classes), tr=tr, va=va, te=te, dummy=dummy)


def load_wlpsl(dummy: bool = False, n_classes: int = 62, shape=(60, 126)):
    """Return (X_wlpsl, y_wlpsl) or None if the files are not there."""
    if dummy:
        rng = np.random.default_rng(1)
        Xw = rng.normal(0, 1, (50, *shape)).astype("float32")
        yw = rng.integers(0, n_classes, 50).astype("int64")
        return Xw, yw
    xp, yp = DATA_DIR / "X_wlpsl.npy", DATA_DIR / "y_wlpsl.npy"
    if not (xp.exists() and yp.exists()):
        return None
    Xw = np.load(xp).astype("float32")
    yw = np.load(yp).astype("int64")
    if len(Xw) != len(yw) or len(Xw) == 0:
        return None
    return Xw, yw


def split_train_val(idx, y, val_frac: float = 0.10, seed: int = SEED):
    """Split index array idx into (train, val); stratified when possible."""
    from sklearn.model_selection import train_test_split

    idx = np.asarray(idx)
    try:
        a, b = train_test_split(idx, test_size=val_frac, stratify=y[idx], random_state=seed)
    except ValueError:  # a class with too few samples
        a, b = train_test_split(idx, test_size=val_frac, random_state=seed)
    return np.sort(a), np.sort(b)


def resample_time(X, T: int):
    """Linearly resample (N, t, F) -> (N, T, F). Used for the sequence-length experiment."""
    X = np.asarray(X, dtype="float32")
    t = X.shape[1]
    if T == t:
        return X
    pos = np.linspace(0, t - 1, T)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (pos - lo).astype("float32")[None, :, None]
    return (X[:, lo, :] * (1 - w) + X[:, hi, :] * w).astype("float32")


# --------------------------------------------------------------------------
# Augmentation (TRAINING DATA ONLY)
# --------------------------------------------------------------------------
def get_augment_fn(dummy: bool = False):
    """Import Zainab's augment(); in dummy mode fall back to a simple jitter."""
    sys.path.insert(0, str(ROOT / "src" / "preprocessing"))
    try:
        from augment import augment  # type: ignore

        return augment
    except ImportError:
        if not dummy:
            raise ImportError(
                "Could not import augment from src/preprocessing/augment.py. "
                "Merge Zainab's data branch (git merge origin/dev) or run without --augment."
            )
        rng = np.random.default_rng(0)

        def _fallback(seq):
            return (seq * rng.uniform(0.95, 1.05) + rng.normal(0, 0.01, seq.shape)).astype("float32")

        return _fallback


def maybe_augment(Xtr, ytr, enabled: bool, dummy: bool = False):
    """Return Xtr, ytr plus one augmented copy of the training set (if enabled)."""
    if not enabled:
        return Xtr, ytr
    aug = get_augment_fn(dummy)
    Xa = np.stack([np.asarray(aug(s), dtype="float32") for s in Xtr])
    if Xa.shape != Xtr.shape:
        raise ValueError(f"augment() changed the shape: {Xtr.shape} -> {Xa.shape}")
    return np.concatenate([Xtr, Xa]), np.concatenate([ytr, ytr])


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def get_optimizer(name: str, lr: float | None = None):
    import tensorflow as tf

    name = name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr or 0.001)
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr or 0.01)
    if name in ("sgd_momentum", "sgd+momentum"):
        return tf.keras.optimizers.SGD(learning_rate=lr or 0.01, momentum=0.9)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr or 0.001)
    raise ValueError(f"Unknown optimizer {name!r}")


def build_rnn(cell: str, n_classes: int, seq_len: int = 60, n_features: int = 126,
              units: int = 64, dropout: float = 0.3, l2: float = 0.0,
              head_activation: str | None = None, head_units: int = 64,
              optimizer: str = "adam", lr: float | None = None):
    """GRU(units) / LSTM(units) -> Dropout -> [Dense(head_activation)] -> softmax.

    Loss is sparse categorical cross-entropy (= categorical cross-entropy for
    integer labels, so no one-hot encoding is needed).
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    cell = cell.lower()
    if cell not in ("gru", "lstm"):
        raise ValueError("cell must be 'gru' or 'lstm'")
    reg = tf.keras.regularizers.l2(l2) if l2 and l2 > 0 else None
    Rnn = layers.GRU if cell == "gru" else layers.LSTM

    inp = layers.Input((seq_len, n_features))
    x = Rnn(units, kernel_regularizer=reg)(inp)
    x = layers.Dropout(dropout)(x)
    if head_activation:
        x = layers.Dense(head_units, activation=head_activation, kernel_regularizer=reg)(x)
    out = layers.Dense(n_classes, activation="softmax", kernel_regularizer=reg)(x)
    model = tf.keras.Model(inp, out, name=f"{cell}_model")
    model.compile(optimizer=get_optimizer(optimizer, lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def fit_model(model, Xtr, ytr, Xva, yva, epochs: int = 40, patience: int = 5,
              batch_size: int = 32, verbose: int = 2) -> dict:
    """Train with early stopping on validation loss; returns the history dict."""
    import tensorflow as tf

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs,
                     batch_size=batch_size, callbacks=cbs, verbose=verbose)
    return hist.history


def predict_proba(model, X, batch_size: int = 256):
    return model.predict(X, batch_size=batch_size, verbose=0)


# --------------------------------------------------------------------------
# Metrics / plots
# --------------------------------------------------------------------------
def accuracy(y_true, probs) -> float:
    return float((np.argmax(probs, axis=1) == np.asarray(y_true)).mean())


def topk_accuracy(y_true, probs, k: int = 3) -> float:
    y_true = np.asarray(y_true)
    k = min(k, probs.shape[1])
    top = np.argsort(-probs, axis=1)[:, :k]
    return float((top == y_true[:, None]).any(axis=1).mean())


def save_curves(history: dict, path: Path, title: str):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["loss"], label="train")
    ax[0].plot(history["val_loss"], label="validation")
    ax[0].set_title(f"{title} - loss")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].legend()
    ax[1].plot(history["accuracy"], label="train")
    ax[1].plot(history["val_accuracy"], label="validation")
    ax[1].set_title(f"{title} - accuracy")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj):
    # allow_nan=False -> never write NaN/Infinity (invalid JSON for browsers/other tools)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
