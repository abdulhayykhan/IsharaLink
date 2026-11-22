"""Steps 1, 3, 4: train GRU and LSTM with identical settings and compare.

Usage (from repo root):
    python src/models/train.py --dummy --epochs 1      # Step 1 check (random data)
    python src/models/train.py                          # real data, GRU + LSTM
    python src/models/train.py --augment                # also use augment() on training data

Outputs (results/ and models/):
    curves_gru.png, curves_lstm.png      loss/accuracy curves
    comparison_random_split.csv          Baseline vs GRU vs LSTM table
    best.json                            which cell won + its settings
    models/gru.keras, lstm.keras, ishara_model.keras (= the better one)
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from common import (build_rnn, fit_model, get_dirs, load_data, maybe_augment,
                    predict_proba, accuracy, topk_accuracy, save_curves, set_seed,
                    write_json, SEED)

import numpy as np

# Settings from the guide: GRU/LSTM(64), Dropout(0.3), Adam lr=0.001, batch 32.
CONFIG = dict(units=64, dropout=0.3, l2=0.0, head_activation=None, head_units=64,
              optimizer="adam", lr=0.001)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true", help="use random data")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--augment", action="store_true", help="augment training data only")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    models_dir, results_dir = get_dirs(args.dummy)
    data = load_data(args.dummy)
    X, y, classes = data["X"], data["y"], data["classes"]
    tr, va, te = data["tr"], data["va"], data["te"]
    n_classes, T, F = len(classes), X.shape[1], X.shape[2]
    print(f"X={X.shape} classes={n_classes} train/val/test={len(tr)}/{len(va)}/{len(te)}")

    if args.dummy:  # predict.py needs a classes file next to the dummy model
        write_json(models_dir / "classes.json", classes)

    Xtr, ytr = maybe_augment(X[tr], y[tr], args.augment, args.dummy)
    Xva, yva, Xte, yte = X[va], y[va], X[te], y[te]
    print(f"training rows used: {len(Xtr)} (augment={args.augment})")

    rows, models = [], {}
    for cell in ("gru", "lstm"):
        print(f"\n===== {cell.upper()} =====")
        set_seed(args.seed)
        model = build_rnn(cell, n_classes, T, F, **CONFIG)
        if cell == "gru":
            model.summary()
        t0 = time.time()
        hist = fit_model(model, Xtr, ytr, Xva, yva, args.epochs, args.patience, args.batch_size)
        secs = time.time() - t0
        save_curves(hist, results_dir / f"curves_{cell}.png", cell.upper())
        pv, pt = predict_proba(model, Xva), predict_proba(model, Xte)
        rows.append(dict(model=cell.upper(),
                         val_acc=accuracy(yva, pv),
                         test_acc=accuracy(yte, pt),
                         test_top3=topk_accuracy(yte, pt, 3),
                         epochs_run=len(hist["loss"]),
                         params=int(model.count_params()),
                         train_seconds=round(secs, 1)))
        model.save(models_dir / f"{cell}.keras")
        models[cell] = model
        print(rows[-1])

    table = pd.DataFrame(rows)

    # Add baseline rows (from baselines.py) so the table is Baseline vs GRU vs LSTM.
    base_csv = results_dir / "baselines.csv"
    if base_csv.exists():
        table = pd.concat([pd.read_csv(base_csv), table], ignore_index=True)
    else:
        print("NOTE: run baselines.py first to include the baseline rows in the table.")
    table.to_csv(results_dir / "comparison_random_split.csv", index=False)
    print("\nBaseline vs GRU vs LSTM (random split - optimistic):")
    print(table.to_string(index=False))

    best_row = max(rows, key=lambda r: (r["val_acc"], r["test_acc"]))
    best_cell = best_row["model"].lower()
    models[best_cell].save(models_dir / "ishara_model.keras")
    write_json(results_dir / "best.json", dict(
        cell=best_cell, val_acc=best_row["val_acc"], test_acc=best_row["test_acc"],
        augment=bool(args.augment), config=CONFIG, seq_len=T, n_features=F))
    print(f"\nBest model by validation accuracy: {best_cell.upper()} "
          f"-> saved as {models_dir / 'ishara_model.keras'}")


if __name__ == "__main__":
    main()
