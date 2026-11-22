# DATA_NOTES.md — IsharaLink Data Lead (Zainab)

Day-1 dataset audit for the WordLevel Pakistan Sign Language dataset. This file is
the single source of truth for how the data is laid out and how we preprocess it.
**Abdul Hayy (backend) and Fasih (frontend) must read this before building.**

---

## 1. Dataset identity & license

- **Name:** Dynamic Word-Level Pakistan Sign Language Dataset
- **Source:** Kaggle — `mohib123456/dynamic-word-level-pakistan-sign-language-dataset`
- **Authors' project repo:** https://github.com/MohibUllahKhanSherwani/SignSpeak_FYP
- **License:** **Apache-2.0** (permits use/redistribution with attribution — cite the authors in the report).
- **Download size:** ~274 MB, extracts to `data/raw/PakistanSignLanguageDatasetV2/PakistanSignLanguageDatasetV2/`.
- **Privacy:** only landmark coordinates are published — no video/images.

**Citation (report):** Mohib Ullah Khan Sherwani et al., *Dynamic Word-Level Pakistan
Sign Language Dataset* (SignSpeak FYP), Kaggle, Apache-2.0.

---

## 2. On-disk structure

```
<camera>/<word>/<clip_index>/<frame>.npy
   e.g. laptop_data/absolutely/0/0.npy
```

- **3 cameras:** `mobile_data`, `laptop_data`, `webcam_data`
- **Clip** = one recording of a sign = a folder of per-frame `.npy` files
- **Frame** = one `.npy`, shape **`(126,)`, dtype float64**
- **424,776** frame files + 2 irrelevant `stats.json` (recording time only)
- Clip folders are named `0,1,2,…` — **plain repetition indices, no signer ID**

---

## 3. The six Day-1 questions (answered)

### (a) License & citation
Apache-2.0. See §1.

### (b) Feature layout — how many values per frame, hands only?
**126 = 2 hands × 21 landmarks × 3 coords (x, y, z).**
- **Hands only** — no pose, no face.
- **Order (confirmed from authors' `mediapipe_utils.py`):**
  - indices **`[0:63]` = LEFT hand** (`results.left_hand_landmarks`)
  - indices **`[63:126]` = RIGHT hand** (`results.right_hand_landmarks`)
  - built as `np.concatenate([lh, rh])`, each hand `[x0,y0,z0, x1,y1,z1, …]` for 21 points.
- **The authors used MediaPipe _Holistic_** (`mp.solutions.holistic`), **not** the standalone Hand Landmarker. Left/right come from Holistic's attributes, not a handedness label string.

### (c) How is a missing hand encoded?
**Zeros.** Authors use `np.zeros(21*3)` when a hand is absent. **No NaN** anywhere (verified across 900 sampled frames). A one-handed sign therefore has one 63-value half all-zero.

### (d) Raw or normalized coordinates?
**Raw MediaPipe coordinates — the authors applied NO normalization.**
Measured ranges on real hands: `x ≈ 0.01–0.89`, `y ≈ 0.05–1.04`, `z ≈ −0.72–0.10`.
So x,y are image-normalized (≈0–1) and z is MediaPipe's wrist-relative depth.
→ **We add our own normalization** (see §5): wrist-relative + hand-size scaling, applied by the single shared `preprocess_window()` so training data and the live browser can never disagree.

### (e) Is there a signer identifier?
**No.** Clips are only numbered repetitions; there is no signer/subject label.
**Consequence:** we **cannot** build a signer-independent split or claim performance
on unseen signers from this dataset alone. The random split is therefore *optimistic*
(same person's repetitions can land on both sides). This is why we also do
leave-one-camera-out and the WLPSL cross-dataset probe.

### (f) Are the landmarks mirrored (selfie view)?
**No flip is applied by the authors** (`collect_data_gui.py` feeds the raw camera
frame to Holistic; only BGR↔RGB conversion). Whether Holistic's "left"/"right"
matches the signer's physical hands is **not provable from the repo**.
→ `preprocess.py` exposes a `MIRROR` config flag so Fasih can flip hand assignment
if his one-handed-sign test (frontend Step 4) lands in the wrong slot.

---

## 4. Vocabulary cleaning (→ `classes.json`)

The raw dataset has **75 word folders**, but they are not clean. Cleaning rule:
a word is **kept only if** it (1) appears in **all three cameras**, (2) is not a
junk/test class, and (3) has ≥15 clips in every camera.

- **Kept: 62 words** → `data/processed/classes.json` (alphabetical; **index = label_id**, this ordering is a CONTRACT).
- **Dropped: 13** (see `data/processed/clip_counts.csv` for the full table):
  - junk/test: `s`, `test2`, `test_word`
  - not in every camera: `bird, come, doctor, give, help, hungry, need, want, where, who`
- ⚠ **`left_hand` is kept** (present in all 3 cameras, normal counts) but **the team should confirm it is a genuine sign and not a collection artifact.**

### Clip balance (kept vocabulary)
~30 mobile / ~50 laptop / ~20 webcam per word ≈ **100 clips/word**; **6,424 clips total**
(mobile 2,084 · laptop 3,100 · webcam 1,240). A handful of words have 100–122.

---

## 5. Preprocessing decisions (→ `preprocess_window()`)

Single Python function used by **both** training-array building and the live API, so
the browser and the training data use identical preprocessing.

1. **Assemble 126-vector per frame** in dataset order: `left → [0:63]`, `right → [63:126]`; missing hand = zeros.
2. **Resample to exactly 60 frames** (linear index resampling). Dataset clips are
   mostly already 60, but a failed capture can be shorter, so this is enforced.
3. **Normalize per hand (wrist-relative + hand-size scale):** subtract the wrist
   landmark (index 0) from all 21 points, then divide by the hand scale (wrist→middle-finger-MCP distance). A missing (all-zero) hand stays all-zero.
4. Output shape **`(60, 126)`**, float32, no NaN. **F = 126.**

### For Fasih (frontend) — match these or live predictions will fail
- Use MediaPipe **Holistic** hand landmarks (left_hand / right_hand), **not** Hand Landmarker alone.
- `min_detection_confidence=0.4`, `min_tracking_confidence=0.4`.
- Send **raw** landmarks (no normalization in the browser — the server does it).
- Browser payload per the interface contract: `frames:[{left:[[x,y,z]*21]|null, right:…}]`.

---

## 6. Limitations (for the report)

- **Unknown signers / no signer split** → generalization to new people is unmeasured here (random split is optimistic).
- **Hands-only landmarks** → signs that rely on face/body position will confuse (expect confusable pairs).
- **Small, cleaned vocabulary (62 words).**
- **Camera/lighting variation** only partially covered (3 cameras).
- **Mobile-camera caveat:** the authors' repo hints the mobile split may have been
  collected by a separate app; its layout *looks* identical in our inspection but was not independently re-derived.
- **WLPSL cross-dataset probe: DEFERRED** this iteration (time-boxed per the plan).
  Unseen-signer / cross-recording generalization is therefore evaluated via
  **leave-one-camera-out only**; `splits.json` keeps an empty `wlpsl` slot and
  `src/preprocessing/wlpsl_extract.py` can be run later to fill it.
- **Experimental prototype**, not a replacement for interpreters.

---

## 7. Contracts I own (don't change without telling the team)
- `classes.json` **word order** (= label ids)
- Raw browser payload format (`frames` of `left`/`right`)
- Preprocessed output shape **`(60, 126)`**
