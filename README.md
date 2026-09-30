# Deep Learning Multi-Task Classification (Gender & Age Group: Young / Not Young) based on Human Photos

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZNVHgIs3TQ_s)  [![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)  [![Framework: TensorFlow/Keras](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-orange.svg)](https://www.tensorflow.org/)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset & Preprocessing](#-dataset--preprocessing)
3. [Model Architecture](#-model-architecture)
4. [Training & Hyperparameters](#-training--hyperparameters)
5. [Evaluation & Results](#-evaluation--results)
6. [Conclusions & Key Findings](#-conclusions--key-findings)
7. [Project Structure](#-project-structure)
8. [How to Run (Quick Start)](#-how-to-run-quick-start)
9. [Future Improvements](#-future-improvements)

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

The multi-task model was evaluated on the held-out test split, measuring overall accuracy, loss convergence, and task-specific metrics compared to baseline expectations.

| Task | Baseline Accuracy | Model Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gender Classification** | ~57.9% | **~94.2%** | 0.94 | 0.94 | 0.94 |
| **Age Group Classification** | ~77.8% | **~85.6%** | 0.86 | 0.85 | 0.85 |

* **Loss Convergence:** Both training and validation losses for gender and age heads decreased steadily over epochs without significant overfitting, stabilized by early stopping.
* **Generalization:** Shared feature extraction prevented the model from overfitting to single-label artifacts, producing robust embeddings across different demographic intersections.

---

## 💡 Conclusions & Key Findings

1. **Synergy in Multi-Task Learning:** Training the network to predict gender and age simultaneously yielded performance gains compared to training separate single-task models. Facial features learned for age estimation (e.g., skin texture, facial structure contours) provided helpful structural context for gender classification, and vice versa.
2. **Data Pipeline Efficiency:** Utilizing local SSD staging with `ThreadPoolExecutor` eliminated data starvation bottlenecks on Google Colab, cutting total epoch training time significantly.
3. **Robustness Against Visual Noise:** Auxiliary attribute analysis revealed that heavy accessories (e.g., hats, sunglasses) temporarily degraded age prediction confidence more than gender prediction, highlighting areas for targeted data augmentation in future iterations.

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
