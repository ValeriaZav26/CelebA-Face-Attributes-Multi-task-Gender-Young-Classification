# Deep Learning Multi-Task Classification (Gender & Age Group: Young / Not Young) based on Human Photos

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZNVHgIs3TQ_s)  [![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)  [![Framework: TensorFlow/Keras](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-orange.svg)](https://www.tensorflow.org/)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset & Preprocessing](#-dataset--preprocessing)
3. [Model Architecture](#-model-architecture)
4. [Training & Hyperparameters](#-training--hyperparameters)
5. [Evaluation & Results](#-evaluation--results)
6. [Project Structure](#-project-structure)
7. [How to Run (Quick Start)](#-how-to-run-quick-start)
8. [Future Improvements](#-future-improvements)

---

## 🌟 Project Overview

This repository contains an end-to-end Deep Learning pipeline designed for **Multi-Task Image Classification** using facial photographs. Given a human face image, the model simultaneously predicts two core demographic attributes:
1. **Gender** (`Male` / `Female`)
2. **Age Group** (`Young` / `Not Young`)

### Key Features:
* **Multi-Task Learning (MTL):** Joint optimization of feature representations for predicting gender and age group concurrently, improving generalization.
* **Robust Stratified Splitting:** Ensures balanced target distribution across training, validation, and test sets.
* **Optimized Data Pipeline:** Utilizes parallelized file copying (`ThreadPoolExecutor`) from Google Drive to local high-speed SSD storage in Google Colab for rapid I/O operations.

---

## 📊 Dataset & Preprocessing

* **Dataset:** [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
* **Annotation Parsing:** Extracted binary attributes (`Male`, `Young`, along with auxiliary attributes like `Eyeglasses`, `Wearing_Hat`, `Smiling` for downstream error analysis).
* **Preprocessing Pipeline:**
  * Mapping `-1 / 1` raw annotation labels to binary `0 / 1` targets.
  * Filtering out missing image instances dynamically based on available files.
  * Stratified splitting to preserve joint target distribution proportions.

---

## 🏗️ Model Architecture

* **Backbone:** Pretrained Convolutional Neural Network (CNN) backbone capable of extracting high-level facial representations.
* **Multi-Task Heads:** Dual classification heads branching from shared feature representations to predict:
  * `gender` (Sigmoid / Binary Cross-Entropy)
  * `young` (Sigmoid / Binary Cross-Entropy)
* **Loss Optimization:** Combined multi-task loss function balancing performance weights across both objectives.

---

## ⚙️ Training & Hyperparameters

| Hyperparameter / Setting | Value |
| :--- | :--- |
| **Framework** | TensorFlow / Keras |
| **Random Seed** | `42` (for complete reproducibility) |
| **Hardware** | Google Colab (T4 / V100 GPU) |
| **Data Split Ratio** | 80% Train, 10% Validation, 10% Test |
| **Parallel Workers** | 16 threads for local data staging |

---

## 📈 Evaluation & Results

Model performance is evaluated on the held-out test set across both classification tasks.

| Task | Metric | Baseline | Model Performance |
| :--- | :--- | :--- | :--- |
| **Gender Classification** | Accuracy / F1-Score | ~0.579 | *Optimized via MTL* |
| **Age Group Classification** | Accuracy / F1-Score | ~0.778 | *Optimized via MTL* |

---

## 📂 Project Structure

```text
├── outputs/                  # Saved models, checkpoints, and logs
├── data/                     # Local working directory for fast image loading
└── notebook.ipynb            # Original Google Colab Jupyter Notebook
```

---

## 🚀 How to Run (Quick Start)

You can run this project interactively in Google Colab with a single click:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZNVHgIs3TQ_s)

### Requirements & Installation
If running locally, ensure you have the required dependencies installed:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn tqdm
```

---

## 🔮 Future Improvements

* **Backbone Upgrade:** Transition from standard architectures to modern state-of-the-art backbones like EfficientNetV2 or ConvNeXt for higher feature fidelity.
* **Data Augmentation:** Implement advanced spatial and color augmentations using `Albumentations` to enhance model robustness against real-world lighting and pose variations.
* **Test-Time Augmentation (TTA):** Apply multi-crop and flip evaluations during inference to boost predictive stability.
* **Fairness & Bias Audit:** Leverage auxiliary attributes (`Eyeglasses`, `Wearing_Hat`, `Smiling`) to audit subgroup performance and mitigate demographic biases.