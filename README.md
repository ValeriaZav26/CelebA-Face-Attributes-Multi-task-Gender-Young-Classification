# CelebA Face Attributes: Multi-task Gender & Young Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Model](https://img.shields.io/badge/Backbone-MobileNetV2-green)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow)

A single neural network with two outputs ("heads") predicts two binary attributes from a face photo at the same time: **gender** (`gender`, 1 = male) and **"youth"** (`young`). The model is built with **transfer learning** (MobileNetV2 pretrained on ImageNet) followed by **fine-tuning**.

> **Note:** `Young` in CelebA is a subjective binary appearance attribute, not a real age.

## Table of Contents

1. [Task Overview](#task-overview)
2. [Repository Structure](#repository-structure)
3. [Data](#data)
4. [Pipeline](#pipeline)
5. [Model Architecture](#model-architecture)
6. [Results and Metrics](#results-and-metrics)
7. [Interpretability and Error Analysis](#interpretability-and-error-analysis)
8. [Installation and Replication](#installation-and-replication)
9. [Limitations](#limitations)
10. [Ethical Considerations](#ethical-considerations)

---

## Task Overview

This is an educational project that covers the full cycle of an applied computer vision task:

- preparing annotations and a stratified data split;
- checking for data leakage;
- a `tf.data` pipeline with augmentations;
- a multi-task model: a shared backbone and two independent heads;
- transfer learning and safe fine-tuning;
- honest evaluation on a held-out test set against a baseline;
- error analysis and a bias check across groups;
- a class-weighting experiment for the imbalanced `young` attribute.

**Value of the project:** a reproducible example of multi-task classification on imbalanced data with correct quality evaluation (accuracy, F1, ROC-AUC vs. baseline).

## Repository Structure

```text
.
├── Gender-Young-Classification.ipynb   # main notebook (stages 0–9)
└── README.md
```

Training artifacts are not stored in the repository and are saved to Google Drive under `MyDrive/DS/outputs/`:

| File | Description |
|---|---|
| `best.keras` | best weights after training the heads (backbone frozen) |
| `best_ft.keras` | best weights after fine-tuning |

## Data

**Source:** the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset — `img_align_celeba` (photos) and `list_attr_celeba.txt` (attribute annotations). The data is stored on Google Drive in the `MyDrive/DS` folder.

| Parameter | Value |
|---|---|
| Rows in the annotation file | 202,599 |
| Photos available on Drive and used in the project | 10,839 |
| Sample size cap (`MAX_IMAGES`) | 40,000 (not reached) |
| Original image size | 178×218 |
| Size after preprocessing | 128×128 |

**Attributes used (out of 40):**

- target: `Male` → `gender`, `Young` → `young`;
- auxiliary (only for error analysis by group): `Eyeglasses`, `Wearing_Hat`, `Smiling`.

Labels are converted from `-1/1` to `0/1`.

**Split** 80% / 10% / 10% with **stratification** by the (gender × young) combination, `random_state = 42`:

| Subset | Size |
|---|---|
| train | 8,671 |
| val | 1,084 |
| test | 1,084 |

**Class imbalance:**

- share of men ≈ 0.42, share of `young` ≈ 0.78;
- among women `young` = 88.2%, among men — 63.6%;
- baseline accuracy (predicting the majority class): `gender` = 0.579, `young` = 0.778.

**Data quality control:** the file name is taken from the `file` column, not from the index, so that labels and photos do not get mixed up. An `assert` checks that train/val/test do not overlap by file name. The match between labels and images is also checked visually.

## Pipeline

| Stage | Description |
|---|---|
| **0. Setup** | Mounting Google Drive, importing libraries, fixing the seed (`SEED = 42`), finding `.jpg` files. |
| **1. Annotations and split** | Reading `list_attr_celeba.txt`, converting labels, selecting columns, stratified split, leakage check, parallel copying of photos (16 threads) to the local Colab disk, initial analysis (class shares, baseline). |
| **2. `tf.data` pipeline** | Cropping a 178×178 square (without distorting face proportions), resizing to 128×128, batch size 64, `prefetch`, `cache` for val/test. Augmentations for train only: `RandomFlip("horizontal")`, `RandomRotation(0.05)`, `RandomZoom(0.1)`, `RandomContrast(0.1)`. |
| **3. Model** | MobileNetV2 (ImageNet, frozen) and two heads. See [Model Architecture](#model-architecture). |
| **4. Sanity check** | The "overfit on 64 photos" test (no augmentations or dropout): accuracy = 1.0 for both attributes, i.e. the data, labels and paths are correct. |
| **5. Training the heads** | Backbone frozen. Adam, `lr = 1e-3`, `dropout = 0.3`, up to 15 epochs, `ModelCheckpoint` on `val_loss`, `EarlyStopping` (`patience = 5`, `restore_best_weights=True`). |
| **6. Fine-tuning** | Loading `best.keras`, unfreezing the last 40 layers of MobileNetV2, keeping all `BatchNormalization` layers frozen, recompiling the model, `lr = 1e-5`, up to 10 epochs, `EarlyStopping` (`patience = 3`). |
| **7. Test evaluation** | Accuracy, F1, ROC-AUC vs. baseline; comparison of `frozen` and `finetuned`; confusion matrices and ROC curves. |
| **8. Error analysis** | Accuracy by slices (gender, glasses, hat, smile), visualization of the most confident errors. |
| **9. Class weights for `young`** | An experiment with class weighting for the imbalanced attribute. |

**Safe fine-tuning rules** followed in the project:

1. a very small `lr` (1e-5), so as not to destroy the pretrained features;
2. not all layers are unfrozen, only the last N (N = 40);
3. BatchNorm layers stay frozen;
4. after changing `trainable`, the model is always recompiled.

## Model Architecture

```text
photo 128×128×3 (0..255) → Rescaling (−1..1) → MobileNetV2 (ImageNet, frozen)
    → GlobalAveragePooling2D → Dropout
          /                                \
   "gender" head                      "young" head
   Dense(128, relu) → Dropout         Dense(128, relu) → Dropout
   Dense(1, sigmoid)                  Dense(1, sigmoid)
```

- **Loss:** `binary_crossentropy` for each head.
- **Training metric:** `BinaryAccuracy` (`acc`).
- **Optimizer:** Adam.
- Each head outputs a probability from 0 to 1; the classification threshold is 0.5.
- Pixel normalization (−1..1) is done inside the model by the `Rescaling` layer.

## Results and Metrics

Evaluation on the **test set (1,084 photos)**, which was used neither for training nor for checkpoint selection.

| Model | Attribute | Accuracy | F1 | ROC-AUC | Baseline acc |
|---|---|---|---|---|---|
| frozen | gender | 0.899 | 0.882 | 0.971 | 0.579 |
| frozen | young | 0.826 | 0.894 | 0.843 | 0.779 |
| finetuned | gender | 0.946 | 0.936 | 0.986 | 0.579 |
| finetuned | young | 0.853 | 0.909 | 0.866 | 0.779 |
| frozen + weights | gender | _TBD_ | _TBD_ | _TBD_ | 0.579 |
| frozen + weights | young | _TBD_ | _TBD_ | _TBD_ | 0.779 |

> The `frozen + weights` rows are to be filled in after running stage 9.

**Conclusions:**

- Fine-tuning improved all metrics:
  - `gender`: accuracy 0.899 → 0.946, ROC-AUC 0.971 → 0.986;
  - `young`: accuracy 0.826 → 0.853, ROC-AUC 0.843 → 0.866.
- For `gender` the model clearly outperforms the baseline (0.946 vs. 0.579).
- For `young` the baseline is already high (0.779), so the accuracy gain should be assessed together with F1 and ROC-AUC, not by accuracy alone.
- The sanity check on 64 photos (accuracy = 1.0) confirmed that the data and the pipeline are correct.

### Class-Weight Experiment (Stage 9)

`young = 1` occurs in ~77% of photos, so the model can reach high accuracy by almost always answering "young". Class weights penalize errors on the rare class (`not young`) more heavily:

```text
W_not_young = n / (2 · n_not_young)
W_young     = n / (2 · n_young)
```

The weights are passed as `sample_weight` **for train only**. Expected trade-off: accuracy may drop slightly, while F1 for the `not young` class should improve.

## Interpretability and Error Analysis

For an image classification task, diagnostics of errors are used instead of SHAP and feature importance:

- **Confusion matrices** for `gender` and `young`.
- **ROC curves** with the random-classifier diagonal.
- **Accuracy by group slices:** gender, glasses (`Eyeglasses`), hat (`Wearing_Hat`), smile (`Smiling`). A single accuracy number hides *where* the model fails, while slices reveal possible bias.
- **Most confident errors:** 16 photos on which the model was wrong with the highest confidence, for each head.

**Observation:** the model most often classifies women with short haircuts as men.

## Installation and Replication

The project is designed to run in **Google Colab**; no local installation is required.

### 1. Clone the repository

```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
```

### 2. Prepare the data

Download CelebA and upload it to Google Drive into the `MyDrive/DS` folder so that you get the following structure:

```text
MyDrive/DS/
├── img_align_celeba/        # .jpg photos (any folder nesting is allowed)
└── list_attr_celeba.txt     # attribute annotations
```

### 3. Run in Colab

1. Open `Deeplearning_simple.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Enable GPU: **Runtime → Change runtime type → GPU**.
3. Run the cells sequentially from top to bottom (stages 0–9). The first cell mounts Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

The dependencies (TensorFlow, pandas, scikit-learn, matplotlib, tqdm) are preinstalled in Colab.

### 4. Paths and parameters

Paths are set in the stage 0 cell:

```python
DRIVE   = "/content/drive/MyDrive/DS"
IMG_SRC = f"{DRIVE}/img_align_celeba"
ATTR    = f"{DRIVE}/list_attr_celeba.txt"
LOCAL   = "/content/data"            # local copy of photos (faster training)
OUT     = "/content/outputs"
BACKUP  = f"{DRIVE}/outputs"         # backup on Drive (best.keras, best_ft.keras)
```

The `MAX_IMAGES` parameter (default `40000`) caps the sample size to fit the time budget; `None` means "take everything found".

### 5. Reproducibility

- Fixed seed: `SEED = 42` (`tf.keras.utils.set_random_seed`).
- Stratified split with `random_state = SEED`.
- Re-running with the same data and library versions should give close results. Bit-for-bit identical results on GPU are not guaranteed.

### 6. Local run (optional)

Create `requirements.txt`:

```text
tensorflow
pandas
numpy
scikit-learn
matplotlib
tqdm
```

Install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For a local run, remove the Drive mounting (`google.colab`) in the notebook and replace the paths in the stage 0 cell with local ones.

## Limitations

- **No real age:** `Young` is a subjective binary annotation.
- **Class imbalance** in `young` (≈ 0.78 positives).
- **`Male` is a binary label of appearance**, not a person's gender as such.
- Photos are mostly **frontal**; quality may be lower on rotated faces and in poor lighting.
- **The split is by photo, not by person:** one celebrity may end up in both train and test, so the metrics are slightly optimistic.
- A **subsample** of CelebA is used (10,839 photos out of 202,599 annotation rows).

## Ethical Considerations

Gender and "young" are the annotator's opinion of appearance, not a fact. **The model must not be used for decisions about people** (hiring, access, surveillance, etc.). The project is intended solely for education and demonstration of methods.

## License

Specify the repository license (e.g. MIT). The CelebA dataset is distributed under its own terms intended for non-commercial research; please review them on the [dataset page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
