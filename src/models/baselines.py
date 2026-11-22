"""Step 2: baselines without temporal memory (Labs 3, 4, 6).

Mean-pool every sequence over time -> (N, F), then train
  * a single-layer Perceptron (scikit-learn)
  * a small dense MLP (scikit-learn)

Usage:  python src/models/baselines.py [--dummy]
Output: results/baselines.csv  (read later by train.py for the comparison table)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import get_dirs, load_data, SEED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true")
    args = ap.parse_args()

    _, results_dir = get_dirs(args.dummy)
    d = load_data(args.dummy)
    X, y, tr, va, te = d["X"], d["y"], d["tr"], d["va"], d["te"]

    P = X.mean(axis=1)  # (N, F): average over time
    Ptr, Pva, Pte = P[tr], P[va], P[te]
    ytr, yva, yte = y[tr], y[va], y[te]

    models = {
        "Perceptron (mean-pool)": make_pipeline(
            StandardScaler(), Perceptron(max_iter=1000, tol=1e-3, random_state=SEED)),
        "MLP 128 (mean-pool)": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, early_stopping=True,
                          n_iter_no_change=10, random_state=SEED)),
    }

    rows = []
    for name, clf in models.items():
        clf.fit(Ptr, ytr)
        rows.append(dict(model=name,
                         val_acc=float(clf.score(Pva, yva)),
                         test_acc=float(clf.score(Pte, yte))))
        print(rows[-1])

    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "baselines.csv", index=False)
    print("\nBaseline accuracy table (random split):")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
