"""Steps 6, 7 (+ config for 8): real evaluation and rejection threshold.

Evaluations on the best model (settings from results/best.json):
  1. random split (optimistic)              - uses models/ishara_model.keras
  2. leave-one-camera-out, 3 folds          - retrains 3 models (train on 2 cameras, test on 3rd)
  3. WLPSL on overlapping words             - uses models/ishara_model.keras (skipped if files missing)
For each: accuracy, top-3, macro-F1, per-class F1 CSV, confusion matrix (PNG+CSV),
5 most confused word pairs.

Then picks the rejection threshold on the VALIDATION set and writes config.json.

Usage:
    python src/models/evaluate.py [--dummy] [--loco-epochs 40] [--skip-loco]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from common import (CAMERA_NAMES, ROOT, accuracy, build_rnn, fit_model, get_dirs, load_data,
                    load_wlpsl, maybe_augment, predict_proba, read_json, set_seed,
                    split_train_val, topk_accuracy, write_json, SEED)

import tensorflow as tf  # noqa: E402


# --------------------------------------------------------------------------
def report(name, y_true, probs, classes, results_dir):
    """Compute metrics, save per-class F1 / confusion matrix / top pairs. Returns summary."""
    n = len(classes)
    y_true = np.asarray(y_true)
    y_pred = probs.argmax(axis=1)
    labels = np.arange(n)

    present = np.unique(y_true)
    summary = dict(evaluation=name, n_test=int(len(y_true)),
                   accuracy=accuracy(y_true, probs),
                   top3=topk_accuracy(y_true, probs, 3),
                   macro_f1=float(f1_score(y_true, y_pred, labels=present,
                                           average="macro", zero_division=0)))

    _, _, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    pd.DataFrame(dict(word=classes, f1=f1, support=support)).to_csv(
        results_dir / f"per_class_f1_{name}.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(results_dir / f"confusion_{name}.csv")

    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    size = max(8, n * 0.18)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(classes, rotation=90, fontsize=5)
    ax.set_yticklabels(classes, fontsize=5)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix (row-normalized) - {name}")
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    fig.savefig(results_dir / f"confusion_{name}.png", dpi=150)
    plt.close(fig)

    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.argsort(-off, axis=None)[:5]
    pairs = []
    for flat in order:
        i, j = np.unravel_index(flat, off.shape)
        if off[i, j] > 0:
            pairs.append(dict(true=classes[i], predicted=classes[j], count=int(off[i, j])))
    pd.DataFrame(pairs, columns=["true", "predicted", "count"]).to_csv(
        results_dir / f"top_confused_pairs_{name}.csv", index=False)

    print(f"\n[{name}] n={summary['n_test']} acc={summary['accuracy']:.4f} "
          f"top3={summary['top3']:.4f} macroF1={summary['macro_f1']:.4f}")
    print("  most confused pairs (true -> predicted : count):")
    for p in pairs:
        print(f"    {p['true']} -> {p['predicted']} : {p['count']}")
    if not pairs:
        print("    (no errors)")
    return summary


# --------------------------------------------------------------------------
def choose_threshold(conf, correct, target_acc=0.95, min_accept=0.70):
    """Pick a confidence threshold on validation data.

    Rule: among thresholds that still accept >= min_accept of samples, take the
    lowest one whose accuracy-on-accepted >= target_acc; if none reaches the
    target, take the one with the highest accuracy-on-accepted.
    """
    grid = [float(t) for t in np.round(np.arange(0.30, 0.981, 0.05), 2)]

    def build_table(thresholds):
        rows = []
        for t in thresholds:
            mask = conf >= t
            rows.append(dict(threshold=float(t),
                             accept_rate=float(mask.mean()),
                             accuracy_on_accepted=(float(correct[mask].mean()) if mask.any()
                                                   else float("nan")),
                             n_accepted=int(mask.sum())))
        return pd.DataFrame(rows)

    table = build_table(grid)
    cand = table[(table.accept_rate >= min_accept) & table.accuracy_on_accepted.notna()]
    if cand.empty:
        # Model is not confident enough for the fixed grid (e.g. barely trained):
        # use the confidence quantile that accepts ~min_accept of validation samples.
        q = round(float(np.floor(float(np.quantile(conf.astype('float64'), 1.0 - min_accept)) * 1000) / 1000), 3)
        print(f"WARNING: no grid threshold keeps {min_accept:.0%} of samples; "
              f"falling back to the validation-confidence quantile {q}. "
              "The model is probably under-trained.")
        table = build_table(sorted(set(grid + [q])))
        return q, table
    ok = cand[cand.accuracy_on_accepted >= target_acc]
    if not ok.empty:
        chosen = float(ok.iloc[0].threshold)
    else:
        chosen = float(cand.sort_values("accuracy_on_accepted", ascending=False,
                                        kind="stable").iloc[0].threshold)
    return chosen, table


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--loco-epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--skip-loco", action="store_true")
    args = ap.parse_args()

    models_dir, results_dir = get_dirs(args.dummy)
    best_path, model_path = results_dir / "best.json", models_dir / "ishara_model.keras"
    if not (best_path.exists() and model_path.exists()):
        raise FileNotFoundError("Run train.py first (needs best.json and ishara_model.keras).")
    best = read_json(best_path)
    cell, cfg = best["cell"], best["config"]

    data = load_data(args.dummy)
    X, y, cam, classes = data["X"], data["y"], data["cam"], data["classes"]
    tr, va, te = data["tr"], data["va"], data["te"]
    n, T, F = len(classes), X.shape[1], X.shape[2]
    model = tf.keras.models.load_model(model_path)

    rows = []

    # 1) random split ------------------------------------------------------
    p_te = predict_proba(model, X[te])
    rows.append(report("random", y[te], p_te, classes, results_dir))

    # 2) leave-one-camera-out ---------------------------------------------
    if not args.skip_loco:
        pooled_y, pooled_p, fold_rows = [], [], []
        for c in sorted(np.unique(cam)):
            cname = CAMERA_NAMES.get(int(c), str(c))
            print(f"\n===== LOCO fold: test camera = {cname} =====")
            te_idx = np.where(cam == c)[0]
            tr_all = np.where(cam != c)[0]
            tr_idx, va_idx = split_train_val(tr_all, y, 0.10, SEED)
            Xtr, ytr = maybe_augment(X[tr_idx], y[tr_idx], best.get("augment", False), args.dummy)
            set_seed(SEED)
            tf.keras.backend.clear_session()
            m = build_rnn(cell, n, T, F, **cfg)
            fit_model(m, Xtr, ytr, X[va_idx], y[va_idx], args.loco_epochs,
                      args.patience, args.batch_size, verbose=2)
            p = predict_proba(m, X[te_idx])
            fold_rows.append(report(f"loco_{cname}", y[te_idx], p, classes, results_dir))
            pooled_y.append(y[te_idx])
            pooled_p.append(p)
        rows.extend(fold_rows)
        rows.append(dict(evaluation="loco_mean",
                         n_test=int(sum(r["n_test"] for r in fold_rows)),
                         accuracy=float(np.mean([r["accuracy"] for r in fold_rows])),
                         top3=float(np.mean([r["top3"] for r in fold_rows])),
                         macro_f1=float(np.mean([r["macro_f1"] for r in fold_rows]))))
        # pooled predictions of all 3 folds -> one confusion matrix / pair list
        report("loco_pooled", np.concatenate(pooled_y), np.concatenate(pooled_p),
               classes, results_dir)
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(model_path)  # reload the packaged model
    else:
        print("LOCO skipped.")

    # 3) WLPSL ---------------------------------------------------------------
    wl = load_wlpsl(args.dummy, n, (T, F))
    if wl is None:
        print("\nWLPSL files (X_wlpsl.npy / y_wlpsl.npy) not found in data/processed - skipped. "
              "Report: unseen-signer generalization was not measured.")
    else:
        Xw, yw = wl
        rows.append(report("wlpsl", yw, predict_proba(model, Xw), classes, results_dir))

    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "three_evaluation.csv", index=False)
    print("\n=== Three-evaluation table ===")
    print(table.to_string(index=False))

    # one-sentence gap ---------------------------------------------------------
    acc = {r["evaluation"]: r["accuracy"] for r in rows}
    parts = [f"random split {acc['random']*100:.1f}%"]
    if "loco_mean" in acc:
        parts.append(f"leave-one-camera-out {acc['loco_mean']*100:.1f}%")
    if "wlpsl" in acc:
        parts.append(f"WLPSL {acc['wlpsl']*100:.1f}%")
    gap = ("Accuracy falls as the test data gets less similar to the training data: "
           + ", ".join(parts) + ".")
    (results_dir / "gap_summary.txt").write_text(gap + "\n", encoding="utf-8")
    print("\nGap:", gap)

    # 4) rejection threshold on VALIDATION -----------------------------------
    p_va = predict_proba(model, X[va])
    conf = p_va.max(axis=1)
    correct = (p_va.argmax(axis=1) == y[va]).astype(float)
    thr, thr_table = choose_threshold(conf, correct)
    thr_table.to_csv(results_dir / "threshold_table.csv", index=False)
    print("\n=== Threshold table (validation) ===")
    print(thr_table.to_string(index=False))
    row = thr_table[thr_table.threshold == thr].iloc[0]
    print(f"\nChosen threshold = {thr}: accept rate {row.accept_rate:.3f}, "
          f"accuracy on accepted {row.accuracy_on_accepted:.3f}")

    # sanity check on the test split (not used for choosing)
    conf_te = p_te.max(axis=1)
    m_te = conf_te >= thr
    chk = dict(threshold=thr, test_accept_rate=float(m_te.mean()),
               test_accuracy_on_accepted=float((p_te.argmax(1) == y[te])[m_te].mean()) if m_te.any() else None)
    write_json(results_dir / "threshold_test_check.json", chk)
    print("Test-split check:", chk)

    # config.json (merge with anything already there) -----------------------
    cfg_path = models_dir / "config.json"
    config = read_json(cfg_path) if cfg_path.exists() else {}
    config.update(dict(
        threshold=thr,
        n_classes=n, seq_len=T, n_features=F,
        model_type=cell,
        model_file="models/ishara_model.keras",
        classes_file="data/processed/classes.json",
        threshold_val_accept_rate=float(row.accept_rate),
        threshold_val_accuracy_on_accepted=float(row.accuracy_on_accepted),
    ))
    write_json(cfg_path, config)
    print(f"Saved {cfg_path}")
    if not args.dummy:
        write_json(ROOT / "config.json", config)  # copy at repo root as well
        print(f"Saved {ROOT / 'config.json'}")


if __name__ == "__main__":
    main()
