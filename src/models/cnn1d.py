"""Step 9 (optional, only if ahead): small 1D-CNN over time as a 4th model (Lab 11).

Usage:  python src/models/cnn1d.py [--dummy] [--epochs 40]
Output: results/cnn1d.csv, results/curves_cnn1d.png, models/cnn1d.keras
(Does NOT overwrite ishara_model.keras.)
"""
from __future__ import annotations

import argparse

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from common import (accuracy, fit_model, get_dirs, load_data, maybe_augment, predict_proba,
                    save_curves, set_seed, topk_accuracy, SEED)


def build_cnn(n_classes, T=60, F=126):
    inp = layers.Input((T, F))
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    m = tf.keras.Model(inp, out, name="cnn1d")
    m.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--augment", action="store_true")
    args = ap.parse_args()

    models_dir, results_dir = get_dirs(args.dummy)
    d = load_data(args.dummy)
    X, y, tr, va, te = d["X"], d["y"], d["tr"], d["va"], d["te"]
    Xtr, ytr = maybe_augment(X[tr], y[tr], args.augment, args.dummy)

    set_seed(SEED)
    m = build_cnn(len(d["classes"]), X.shape[1], X.shape[2])
    hist = fit_model(m, Xtr, ytr, X[va], y[va], args.epochs, args.patience, 32)
    save_curves(hist, results_dir / "curves_cnn1d.png", "1D-CNN")
    pv, pt = predict_proba(m, X[va]), predict_proba(m, X[te])
    row = dict(model="1D-CNN", val_acc=accuracy(y[va], pv), test_acc=accuracy(y[te], pt),
               test_top3=topk_accuracy(y[te], pt, 3), epochs_run=len(hist["loss"]),
               params=int(m.count_params()))
    pd.DataFrame([row]).to_csv(results_dir / "cnn1d.csv", index=False)
    m.save(models_dir / "cnn1d.keras")
    print(row)


if __name__ == "__main__":
    main()
