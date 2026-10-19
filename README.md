# PMDFNet: Physically Motivated Dual-Stream Fusion for Multimodal Landslide Segmentation

[中文说明 / Chinese README](README.zh-CN.md)

This repository provides the paper-level training and reproducibility pipeline for **PMDFNet** on the **converted MMLSv2** dataset. Competition submission code, legacy training code, the SKG branch, and historical analysis utilities are archived under `archive/legacy/` and are not part of the current paper-training workflow.

The project and repository are named PMDFNet. Python modules and historical results may retain `DualSegFormer`/`dual_segformer_*` as compatibility identifiers; these identifiers do not change the legacy checkpoints or reproducibility entry points.

## Repository structure

```text
PMDFNet/
├── train_ablation.py                 # Configuration-driven paper experiment trainer
├── exp_train.sh                      # Main entry point for paper experiments
├── scripts/                          # Training launch scripts
│   ├── train_paper_experiments.sh    # Paper protocol entry point
│   ├── train_paper_experiments.py
│   └── ...                           # Current launch helpers
├── configs/                          # Training and protocol configurations
├── networks/                         # Model architectures
├── losses/                           # Loss functions
├── dataset.py                        # Dataset definitions
├── augmentations.py                  # Training augmentations
├── reproducibility.py                # Random seeds and deterministic settings
├── doc/                              # Current experiment plan and results
├── tools/                            # Current data, evaluation, and validation tools
│   ├── data/                         # MMLSv2 mapping
│   ├── evaluation/                   # Frozen-checkpoint validation and testing
│   └── validation/                   # Data and VRAM smoke tests
└── archive/legacy/                   # Legacy training, SKG, submission, and analysis code
```

## Installation

```bash
conda create -n pmdfnet python=3.10
conda activate pmdfnet
pip install -r requirements.txt
```

## Dataset mapping

The mapping script is `tools/data/map_mmlsv2_to_mars.py`. It converts the MMLSv2 channel order to the legacy competition format, preserves the original data, and writes a mapped dataset:

```bash
python tools/data/map_mmlsv2_to_mars.py \
  --source-root /path/to/mmlsv2 \
  --output-root /path/to/mmlsv2_mapped_mars_ls
```

## Frozen-model evaluation

Run test evaluation only after the model, hyperparameters, and checkpoint-selection rule have been frozen:

```bash
CHECKPOINT=/path/to/best.pth \
DATA_ROOT=/path/to/mmlsv2_mapped_mars_ls \
  bash tools/evaluation/evaluate_reproduction_test.sh
```

Historical result aggregation and competition-submission utilities are located under `archive/legacy/tools/` and are not part of the current paper protocol.
