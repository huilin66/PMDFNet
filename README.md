# Segformer for Mars Landslide Segmentation Challenge

This repository is built based on [M3LSNet](https://github.com/tingul4/M3LSNet), and contains the solution for the **1st Mars Landslide Segmentation (Mars-LS) Challenge**. It implements Dual_Segformer with ConvNext backbonefor the challenges of Martian terrain segmentation.

## 📂 Repository Structure

```
DualSegformerConvNext/
├── losses/                         # Loss Functions Definition
├── networks/                       # Model Definition
├── augmentations.py                # Online Data Augmentation
├── data_prepare.py                 # Data Preparation
├── dataset.py                      # Dataset Definition
├── generate_submission_aug.py      # Inference & Submission Zipping
├── generate_submission_voting.py   # Inference & Submission Zipping
├── generate_submission.py          # Inference & Submission Zipping
├── log_write.py                    # Training Log Writing
├── result_ana.py                   # Result Visualization and Analysis
├── train.py                        # Main Training Loop (A100 Optimized)
└── outputs/                        # Structured Logs & Checkpoints
    └── YYYYMMDD_HHMMSS/
        ├── checkpoints/            # best.pth, last.pth
        ├── logs/                   # Training logs
        └── tensorboard/            # Visualization   
```

---

## 🛠️ Usage
You can use the following commands to run the code:
### 0. Installation
```bash
conda create -n dualsegformer python=3.10
conda activate dualsegformer
pip install -r requirements.txt
```

### 1. Data Preparation
```bash
# Update the paths marked as TODO to configure your dataset.
python data_prepare.py
```
---
### 2. Training model

```bash
# train models and model ckpts will be saved to `outputs/CurrentTimestamp/`.*
python train.py
```
---
### 3. Inference
```bash
# Update the paths marked as TODO to configure your dataset.
python generate_submission_voting.py
```
---



