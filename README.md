# IsharaLink: Real-Time Pakistan Sign Language Recognition in the Browser

IsharaLink is an end-to-end deep learning web application that recognizes Pakistan Sign Language (PSL) words in real time directly from camera feeds using MediaPipe hand tracking and recurrent neural networks (GRU/LSTM).

---

## 📌 Course & Team Details
* **Course:** Artificial Neural Networks & Deep Learning (ANN & DL), BS Artificial Intelligence, DUET Karachi
* **Deliverable:** Lab 13, Open-Ended Lab No. 2
* **Instructor:** Engr. Bushra Shaikh
* **Team Roles & Branches:**
  * **Zainab Siddiq** (24F-AI-006) — Data Lead (`data`)
  * **Bushra Zafar** (24F-AI-009) — Model Lead (`model`)
  * **Abdul Hayy Khan** (24F-AI-016) — Backend Lead (`backend`)
  * **Muhammad Fasih** (24F-AI-051) — Frontend Lead (`frontend`)

---

## 📁 Repository Structure

```
IsharaLink/
|-- .gitignore
|-- environment.yml
|-- requirements.txt
|-- README.md
|-- docs/                          # Project proposals, execution guides
|-- data/
|   |-- raw/                       # Downloaded raw datasets (git-ignored)
|   `-- processed/                 # 60-frame normalized arrays & split indices
|-- notebooks/                     # Exploratory analysis & experiment notebooks
|-- src/
|   |-- preprocessing/             # Resampling, wrist-normalization, data extraction
|   `-- models/                    # Baseline, GRU, LSTM architectures & training
|-- backend/                       # FastAPI application & model serving routes
|-- frontend/                      # Web interface (HTML, CSS, JS + MediaPipe)
|-- models/                        # Saved model weights & configs
|-- tests/                         # Golden tests & preprocessing verification
`-- report/                        # Final report, diagrams & metric tables
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/abdulhayykhan/IsharaLink.git
cd IsharaLink
git checkout <your-branch>
```

### 2. Environment Setup (Python 3.10 or 3.11)

#### Option A: Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate isharalink
```

#### Option B: Pip / Virtualenv
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

Verify the environment:
```bash
python -c "import tensorflow, mediapipe, fastapi, sklearn; print('Setup successful!')"
```

---

## 🌿 Git Branching & Workflow

* **`main`**: Production-ready code. **Protected — no direct pushes.**
* **`dev`**: Integration branch for combining tested features from member branches.
* **Member Branches**:
  * `data` (Zainab)
  * `model` (Bushra)
  * `backend` (Abdul Hayy)
  * `frontend` (Fasih)

### Daily Workflow
Work on your assigned branch only:
```bash
git checkout <your-branch>
git add <files>
git commit -m "tag: clear message"
git push origin <your-branch>
```

To incorporate recent changes from `dev`:
```bash
git fetch origin
git merge origin/dev
```

When a milestone is complete, open a **Pull Request (PR)** from your branch into `dev`.

---

## 📜 Interface Contract

The browser captures raw MediaPipe hand landmarks and transmits them to the FastAPI backend. The server normalizes coordinates and resamples clips to exactly 60 frames.

### `POST /predict`
* **Request:**
```json
{
  "frames": [
    {
      "left": [[0.12, 0.45, 0.01], ...],
      "right": null
    }
  ]
}
```
* **Response:**
```json
{
  "label": "shukriya",
  "confidence": 0.94,
  "top3": [
    {"label": "shukriya", "p": 0.94},
    {"label": "salam", "p": 0.04},
    {"label": "madad", "p": 0.01}
  ],
  "status": "ok"
}
```
*(If confidence falls below the validation threshold, `"status": "unknown"` is returned).*

---

## ⚠️ Core Rules
1. **No direct pushes to `main`.** All merges happen via PR into `dev`, then to `main` at code freeze.
2. **Never commit raw datasets.** Keep `data/raw/` in `.gitignore`. Commit only processed indices (`classes.json`, `splits.json`), small CSV metric tables, and visual plots.
3. **TensorFlow/Keras only** for model training and deployment.
4. Always run `git status` before committing.
5. Post metric tables in the team group after every model training run.
