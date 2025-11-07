# awe-zoo-oliveball

## Abstract

Sports Action Recognition can present a challenging task for deep-learning computer vision models, where the need for visually distinct class labels meets the challenge of visually arbitrarily termed semantic designations for each action (a sports statistic). However, in the same way humans understand sports-based actions through the context of a rulebook, if dataset and models are constructed optimising for this sports-based subjective contextualisation, non-zero-sum classification should be possible – achieving high performance whilst preserving semantically rich, esoteric class labels.

This thesis presents a detailed review of the current literature’s available technology, methodologies, and shortcomings, and provides a detailed methodology (with some novel augmentations) and associated framework to achieve better results within Sports Action Recognition. It introduces **JezzaFooty-6-7**, the first publicly available Australian Rules Football action recognition dataset, built from in-situ, unedited, professional AFL broadcast footage. With ~6,700 videos producing over 10,000 clips, the dataset was evaluated across multiple convolutional and transformer backbones. A bespoke **VideoMAEv2** variant, trained with fused transforms, achieved state-of-the-art performance—reporting a weighted-F1 score of **0.8451** and a Top-1 accuracy of **84.56 %** across 15 distinct classes.

---

## Dataset

The **JezzaFooty-6-7** dataset is publicly available on Kaggle:





🔗 [https://www.kaggle.com/datasets/jerrydeeplearner/jezzafooty-6-7/data](https://www.kaggle.com/datasets/jerrydeeplearner/jezzafooty-6-7/data)

It comprises multi-class broadcast AFL clips, manually annotated and partitioned for reproducible deep-learning experimentation.

---

## Overview

The JezzaFooty-6-7 codebase provides a complete training, evaluation, and inference framework for sports action recognition. It integrates **PyTorch**, **PyAV**, and **Hugging Face Transformers**, supporting both 3D-CNN and transformer-based architectures such as **VideoMAEv2**, **Swin3D**, **MViTv2**, and **S3D**.

Core features:

* Cross-validated training (Stratified K-Fold)
* Automatic per-class weighting and early stopping
* Peripheral-aware visual transforms (Peripheral Envisionate)
* End-to-end metadata capture for reproducibility

---

## File Structure

| File                      | Description                                                                     |
| ------------------------- | ------------------------------------------------------------------------------- |
| `cf_train_test.py`        | Main training/testing pipeline (cross-validation, schedulers, metrics, saving). |
| `cf_model.py`             | Model creation, transform setup, and metadata saving.                           |
| `cfa_video_classifier.py` | Custom wrapper for Hugging Face VideoMAEv2 models.                              |
| `cf_dataset.py`           | Clip-level dataset logic and oversampling strategy using PyAV.                  |
| `cf_transform.py`         | Peripheral Envisionate transform for central-focus augmentation.                |
| `cf_scheduler.py`         | Cosine and cyclical warm-restart LR schedulers.                                 |
| `cf_early_stop.py`        | Early-stopping utility (min/max modes).                                         |
| `cf_video_labeller.py`    | Generates dataset label dictionaries and index maps.                            |
| `cf_glance.py`            | Produces dataset GIF previews and JSON sample summaries.                        |
| `path_configurator.py`    | Centralised file-path management for Linux/macOS.                               |
| `requirements.txt`        | Library dependencies (PyTorch 2.8 +, Torchvision 0.23 +, Transformers 4.56 +).  |

---

## Inference Examples

### Best Model with Refined Dataset (VideoMAEv2-H Micro-Tuned Model) [Default]

Inference is run by sequentially applying three temporal input streams (stride = 1, 2, 3), and averaging the confidence vectors for each.

**Highlights (Men’s Professional League):**

* [Example (Gulden Goal)](https://youtu.be/IPqOJyLq-0M)

* [Alternative (Papley Goal)](https://youtu.be/e4PpI1IdWL8)

**Off-Angle (Non-Standard Match Broadcast):**

* [Example - 2005 Grand Final](https://youtu.be/5rgoqfqhB5k) 

**Full Match Snippet:**

* [Match Snippet - 2022 PF Sydney v Collingwood](https://youtu.be/gY3IuzlSW-M)

**Women’s League:**

* [AFLW Example - GWS v Sydney R6 2025 Snippet](https://youtu.be/dIJ8Ud1ivkY)

**Non-Professional:**

* [Amateur / Semi-Professional Example - Coates League](https://youtu.be/OllMeHikmag)

---

### Extended Dataset Model (18 Classes, Tri-Temporal Stream)

Inference using the same tri-temporal architecture trained across an expanded 18-class JezzaFooty-6-7 configuration.

**Highlights (Men’s Professional League):**

* [Highlights](https://youtu.be/4JE0Hgb2jmA)

* [Alternate Highlights](https://youtu.be/iCwwI7UuN8Q)

**Off-Angle (Non-Standard Match Broadcast):**

* [Example 1](https://youtu.be/y8WfimoBolI)

**Full Match Snippet:**

* [Match Snippet](https://youtu.be/9m6s3IreoWo)

---

## Usage Summary

1. **Configure paths** in `path_configurator.py` for your project and dataset directories.
2. **Generate label dictionaries** with `cf_video_labeller.py`.
3. **Inspect dataset structure** with `cf_glance.py` (optional GIF preview).
4. **Train and evaluate** models via `cf_train_test.py`, adjusting model type, batch size, and patience.
5. **Run inference** using `deploy_model_for_inference()` to produce annotated output videos.

---

## Citation

If referencing this framework or dataset, please cite the **JezzaFooty-6-7 dataset**, awe-zoo-oliveball code and credit the thesis author as needed.
