# Model Lead (Bushra) - step by step

All commands run from the **repo root** (`IsharaLink/`), on branch `model`, with the venv active
(`.\.venv\Scripts\Activate.ps1` on Windows). Python 3.10 or 3.11.

## 0. Put the files in place
```
src/models/common.py  train.py  baselines.py  experiments.py  evaluate.py  predict.py  cnn1d.py
data/processed/       X.npy y.npy camera.npy classes.json splits.json  (+ X_wlpsl.npy y_wlpsl.npy later)
```
Never commit `.npy` files. DO commit: `results/*.csv`, `results/*.png`, `models/config.json`, `config.json`.

## Step 1 - Training skeleton (random data)
```
python src/models/train.py --dummy --epochs 1
```
**Done when:** it finishes without errors (writes only to `models_dummy/` and `results_dummy/`, never to the real folders).

## Step 2 - Baselines (Perceptron + MLP on mean-pooled sequences)
```
python src/models/baselines.py
```
**Done when:** `results/baselines.csv` has the accuracy table.

## Steps 3 + 4 - GRU and LSTM (same settings, fair comparison)
```
python src/models/train.py
```
GRU(64) / LSTM(64) -> Dropout(0.3) -> softmax, Adam lr=0.001, batch 32, early stopping on val loss.
**Done when:** `results/curves_gru.png`, `results/curves_lstm.png`, and `results/comparison_random_split.csv`
(Baseline vs GRU vs LSTM) exist. The better model is saved as `models/ishara_model.keras`
and recorded in `results/best.json`.
Optional: `--augment` trains on original + augmented copies (uses Zainab's `augment.py`, training data only).

## Step 5 - Experiments (one change at a time, on the better of GRU/LSTM)
```
python src/models/experiments.py
```
Runs optimizer, dropout x L2, dense-head activation, sequence length (30/45/60).
**Done when:** `results/exp_optimizer.csv`, `exp_regularization.csv`, `exp_activation.csv`, `exp_seqlen.csv`
and `results/experiment_conclusions.txt` (one line each, chosen on validation accuracy).
If one experiment crashes or you want to repeat it: `--only optimizer` (or `regularization`, `activation`, `seqlen`).

> If an experiment finds a clearly better setting, edit `CONFIG` in `train.py`, re-run `train.py`,
> then continue with Step 6 so the packaged model uses it.

## Steps 6 + 7 - Real evaluation and rejection threshold
```
python src/models/evaluate.py
```
Produces: random split, leave-one-camera-out (3 folds, retrains 3 models), WLPSL (skipped
automatically if `X_wlpsl.npy` is missing). Accuracy, top-3, macro-F1, per-class F1, confusion matrices (PNG + CSV),
5 most confused pairs, the one-sentence gap (`results/gap_summary.txt`), the accept-rate table
(`results/threshold_table.csv`) and the threshold in `models/config.json` (+ copy at repo root `config.json`).
**Done when:** `results/three_evaluation.csv` and `config.json` exist.

## Step 8 - Package for Abdul Hayy
```
python src/models/predict.py
```
**Done when:** it prints `predict_window correct on N/20 test samples` with a high N.
Abdul Hayy uses:
```python
from src.models.predict import load_assets, predict_window
load_assets()                      # once at startup
predict_window(array_60_by_126)    # -> {"label","confidence","top3":[...],"status":"ok"|"unknown"}
```
`classes.json` is read from `data/processed/`, config from `models/config.json`.

## Step 9 - Optional
```
python src/models/cnn1d.py
```

## Report material
Tables are in `results/`. Say clearly: the random split is optimistic; cross-camera and WLPSL are the honest numbers.
