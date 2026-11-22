"""Step 5: one-thing-at-a-time experiments on the better of GRU/LSTM.

  (a) optimizer:       SGD, SGD+momentum, RMSprop, Adam        (Lab 5)
  (b) regularization:  dropout 0 / 0.3 / 0.5  x  L2 off / on   (Lab 9)
  (c) head activation: none, ReLU, tanh, sigmoid                (Lab 2)
  (d) sequence length: 30, 45, 60                               (Lab 12)

Usage:
    python src/models/experiments.py [--dummy] [--epochs 15] [--patience 3]
    python src/models/experiments.py --only optimizer activation

Needs results/best.json (made by train.py). Conclusions are chosen on
VALIDATION accuracy (never on the test set).
Outputs: results/exp_<name>.csv and results/experiment_conclusions.txt
"""
from __future__ import annotations

import argparse

import pandas as pd

from common import (build_rnn, fit_model, get_dirs, load_data, predict_proba, accuracy,
                    read_json, resample_time, set_seed, SEED)

import tensorflow as tf  # noqa: E402


def run_one(cell, base_cfg, data, epochs, patience, batch_size, seq_len=60, **override):
    cfg = {**base_cfg, **override}
    X, y = data["X"], data["y"]
    tr, va, te = data["tr"], data["va"], data["te"]
    Xs = resample_time(X, seq_len)
    set_seed(SEED)
    tf.keras.backend.clear_session()
    model = build_rnn(cell, len(data["classes"]), seq_len, X.shape[2], **cfg)
    hist = fit_model(model, Xs[tr], y[tr], Xs[va], y[va], epochs, patience, batch_size, verbose=0)
    return dict(val_acc=accuracy(y[va], predict_proba(model, Xs[va])),
                test_acc=accuracy(y[te], predict_proba(model, Xs[te])),
                best_val_loss=float(min(hist["val_loss"])),
                epochs_run=len(hist["loss"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["optimizer", "regularization", "activation", "seqlen"])
    args = ap.parse_args()

    _, results_dir = get_dirs(args.dummy)
    best_path = results_dir / "best.json"
    if not best_path.exists():
        raise FileNotFoundError("results/best.json not found - run train.py first.")
    best = read_json(best_path)
    cell, base = best["cell"], dict(best["config"])
    data = load_data(args.dummy)
    want = lambda name: args.only is None or name in args.only  # noqa: E731
    print(f"Experiments on {cell.upper()} with base config {base}")

    conclusions = []

    def finish(name, label, rows):
        df = pd.DataFrame(rows)
        df.to_csv(results_dir / f"exp_{name}.csv", index=False)
        print(f"\n--- {label} ---")
        print(df.to_string(index=False))
        top = df.sort_values("val_acc", ascending=False).iloc[0]
        line = (f"{label}: best on validation = {top['setting']} "
                f"(val_acc={top['val_acc']:.3f}, test_acc={top['test_acc']:.3f}).")
        conclusions.append(line)
        print("Conclusion:", line)

    if want("optimizer"):
        rows = []
        for opt in ("sgd", "sgd_momentum", "rmsprop", "adam"):
            print("running optimizer", opt)
            r = run_one(cell, base, data, args.epochs, args.patience, args.batch_size,
                        optimizer=opt, lr=None)
            rows.append(dict(setting=opt, **r))
        finish("optimizer", "(a) Optimizer", rows)

    if want("regularization"):
        rows = []
        for dr in (0.0, 0.3, 0.5):
            for l2 in (0.0, 1e-4):
                print("running dropout", dr, "l2", l2)
                r = run_one(cell, base, data, args.epochs, args.patience, args.batch_size,
                            dropout=dr, l2=l2)
                rows.append(dict(setting=f"dropout={dr}, L2={'on(1e-4)' if l2 else 'off'}", **r))
        finish("regularization", "(b) Dropout / L2", rows)

    if want("activation"):
        rows = []
        for act in (None, "relu", "tanh", "sigmoid"):
            print("running head activation", act)
            r = run_one(cell, base, data, args.epochs, args.patience, args.batch_size,
                        head_activation=act)
            rows.append(dict(setting=act or "none (no hidden dense)", **r))
        finish("activation", "(c) Dense-head activation", rows)

    if want("seqlen"):
        rows = []
        for T in (30, 45, 60):
            print("running sequence length", T)
            r = run_one(cell, base, data, args.epochs, args.patience, args.batch_size, seq_len=T)
            rows.append(dict(setting=f"T={T}", **r))
        finish("seqlen", "(d) Sequence length", rows)

    out = results_dir / "experiment_conclusions.txt"
    with open(out, "a", encoding="utf-8") as f:
        f.write("\n".join(conclusions) + "\n")
    print(f"\nConclusions appended to {out}")


if __name__ == "__main__":
    main()
