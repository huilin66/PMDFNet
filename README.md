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

## Paper experiments

The formal protocol is defined in `configs/mmlsv2_paper_protocol.json`. It contains the M1–M4 baselines, the E1 extended baselines M6–M11 (UPerNet/FPN/MANet/U-Net++ with ResNet50, OCRNet with HRNet18, and M3LSNet), and the key ablations (A1/A2/A5/B2/B3/C2 and L1/L2/L3). The fixed random seeds are `41 42 43`. All configurations use the CE-only training objective (`common.loss=ce`), except for the explicitly overridden loss ablations L2/L3.

The main paper model M5 corresponds to **L1** in the protocol (PMDFNet with CE-only training) and is trained once. A3/A4/B1/C1 are display aliases for M4/L1 and are not trained separately. The `paper_id_map` section records the mapping between paper IDs and run IDs.

The additional baselines are constructed as follows: M6–M9 use the SMP branch of `train_ablation.build_model` (`arch` = `upernet` / `fpn` / `manet` / `unetplusplus`, encoder `tu-resnet50`); M10 uses `networks.OCRNet.SMP_OCRNet` (`ocrnet` with `tu-hrnet_w18`, including a 0.4-weight auxiliary loss); and M11 uses `networks.get_model` (`model_name=m3lsnet`, `pretrain=false`).

Before the first run in a new environment, verify that the additional baselines can be constructed. On a machine without PyTorch, this check cannot be completed locally:

```bash
python - <<'PY'
import json, torch
import train_ablation as ta
proto = json.load(open("configs/mmlsv2_paper_protocol.json", encoding="utf-8"))
common = proto["common"]
for exp in proto["experiments"]:
    if exp["id"] not in {"M6", "M7", "M8", "M9", "M10", "M11"}:
        continue
    m = dict(common); m.update({k: v for k, v in exp.items() if k in ta.DEFAULTS})
    m["data_root"] = "/tmp"; m["pretrain"] = exp.get("pretrain", common["pretrain"])
    for k in ta.DEFAULTS:
        m.setdefault(k, ta.DEFAULTS[k])
    cfg = ta.RunConfig(**{k: m[k] for k in ta.DEFAULTS})
    model = ta.build_model(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(2, cfg.in_channels, cfg.input_size, cfg.input_size))
    logits = out[0] if isinstance(out, tuple) else out
    n = sum(p.numel() for p in model.parameters())
    print(f"{exp['id']:>4} {type(model).__name__:<22} params={n/1e6:7.2f}M out={tuple(logits.shape)}")
PY
```

Run the three experiment groups separately when needed:

```bash
DATA_ROOT=/path/to/mmlsv2_mapped_mars_ls bash scripts/run_core.sh       # M1-M4
DATA_ROOT=/path/to/mmlsv2_mapped_mars_ls bash scripts/run_ablations.sh  # Ablations + L1/L2/L3
DATA_ROOT=/path/to/mmlsv2_mapped_mars_ls bash scripts/run_extended.sh   # M6-M11
```

`DATA_ROOT` must be provided explicitly. The scripts no longer embed a remote-host path or conflict with the evaluation scripts' defaults. Alternatively, provide `MARS_DATA_ROOT`, `MAPPED_DATA_ROOT`, `MMLSV2_DATA_ROOT`, or a `.env` file in the repository root.

The three groups are mutually exclusive. The following command therefore trains exactly 57 runs (19 experiments × 3 seeds) without duplication:

```bash
DATA_ROOT=/path/to/mmlsv2_mapped_mars_ls bash scripts/run_all.sh
```

Temporary outputs are written to `outputs_experiments/pmdfnet_mmlsv2` and copied to the dataset root's `outputs/` directory after completion. The formal configurations save only `best.pth` and `last.pth`; `best.pth` is selected by validation-set global mIoU.

The combined entry point can also be used:

```bash
bash exp_train.sh paper_core
bash exp_train.sh paper_ablations
bash exp_train.sh paper_extended
bash exp_train.sh paper_all
```

Run the VRAM smoke test with:

```bash
bash exp_train.sh vram_smoke
```

Results are written to `outputs_experiments/smoke/vram/vram_smoke_results.csv` and `vram_smoke_results.json`. This test uses random inputs only; it does not read the dataset, train a model, or save checkpoints.

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
