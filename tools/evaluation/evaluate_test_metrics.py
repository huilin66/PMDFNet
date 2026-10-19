"""Evaluate a frozen checkpoint on the labeled public test split.

The legacy MarsSegDataset intentionally returns only ``(image, image_name)``
for ``split="test"`` because the original competition test labels were
hidden.  The mapped MMLSv2 release contains public test masks, so this script
loads those masks explicitly while keeping the same Mars test normalization
and metric definition used by train_ablation.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import MARS_MEAN_TEST, MARS_STD_TEST  # noqa: E402
from networks import get_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a selected checkpoint on the labeled public test split."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model-name",
        default="",
        help="Legacy networks.get_model identifier; use --run-config for paper runs.",
    )
    parser.add_argument(
        "--run-config",
        default="",
        help="Training run config.json used to reconstruct the exact trained model.",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--staging-output-dir",
        default="",
        help="Optional project-side run directory to update alongside output-dir.",
    )
    parser.add_argument("--device", default=os.environ.get("TRAIN_DEVICE", "cuda:0"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


class LabeledTestDataset(Dataset):
    """MARS-formatted test images paired with public test masks by stem."""

    def __init__(
        self,
        root_dir: str,
        size: int = 128,
        input_channels: list[int] | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.size = size
        self.input_channels = list(range(7)) if input_channels is None else list(input_channels)
        if not self.input_channels:
            raise ValueError("input_channels must contain at least one source channel index")
        invalid = [channel for channel in self.input_channels if channel < 0 or channel >= 7]
        if invalid:
            raise ValueError(f"Channel indices must be in [0, 6], got {invalid}")
        self.images_dir = self.root_dir / "test" / "images"
        self.masks_dir = self.root_dir / "test" / "masks"
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Missing test image directory: {self.images_dir}")
        if not self.masks_dir.is_dir():
            raise FileNotFoundError(f"Missing test mask directory: {self.masks_dir}")

        self.image_files = sorted(self.images_dir.glob("*.tif"))
        mask_files = list(self.masks_dir.glob("*.tif"))
        self.mask_by_stem = {path.stem: path for path in mask_files}
        missing = [path.name for path in self.image_files if path.stem not in self.mask_by_stem]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"Missing test masks for {len(missing)} images; examples: {preview}"
            )
        if not self.image_files:
            raise RuntimeError(f"No test images found in {self.images_dir}")

        self.mean = np.asarray(MARS_MEAN_TEST, dtype=np.float32).reshape(7, 1, 1)
        self.std = np.asarray(MARS_STD_TEST, dtype=np.float32).reshape(7, 1, 1)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int):
        image_path = self.image_files[index]
        mask_path = self.mask_by_stem[image_path.stem]

        image = tifffile.imread(str(image_path)).astype(np.float32)
        if image.ndim == 3 and image.shape[-1] == 7:
            image = image.transpose(2, 0, 1)
        if image.ndim != 3 or image.shape[0] != 7:
            raise ValueError(f"Expected 7-channel test image, got {image.shape}: {image_path}")
        image[image < -100000] = 0.0
        image = (image - self.mean) / (self.std + 1e-8)
        image = image[self.input_channels]

        mask = tifffile.imread(str(mask_path)).astype(np.int64)
        mask = np.squeeze(mask)
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D test mask, got {mask.shape}: {mask_path}")
        if tuple(mask.shape) != tuple(image.shape[-2:]):
            raise ValueError(
                f"Image/mask shape mismatch: {image.shape[-2:]} vs {mask.shape} "
                f"for {image_path.name}"
            )
        mask = (mask > 0).astype(np.int64)
        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()


def confusion_counts(preds: torch.Tensor, targets: torch.Tensor) -> tuple[int, int, int, int]:
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    tp = int(((preds == 1) & (targets == 1)).sum().item())
    fp = int(((preds == 1) & (targets == 0)).sum().item())
    fn = int(((preds == 0) & (targets == 1)).sum().item())
    tn = int(((preds == 0) & (targets == 0)).sum().item())
    return tp, fp, fn, tn


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou_fg = tp / (tp + fp + fn + eps)
    iou_bg = tn / (tn + fp + fn + eps)
    return {
        "miou": float((iou_fg + iou_bg) / 2),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou_fg": float(iou_fg),
        "iou_bg": float(iou_bg),
    }


def mean_metric_dict(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([metrics[key] for metrics in metric_dicts]))
        for key in metric_dicts[0]
    }


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError:
        # Be tolerant of checkpoints produced under DataParallel.
        stripped = {
            key.removeprefix("module."): value for key, value in checkpoint.items()
        }
        model.load_state_dict(stripped)


def build_evaluation_model(
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, str, str, list[int]]:
    """Build the exact training model when a run config is supplied."""
    if args.run_config:
        import train_ablation

        config_path = Path(args.run_config).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        merged = dict(train_ablation.DEFAULTS)
        merged.update(payload)
        config = train_ablation.RunConfig(
            **{key: merged[key] for key in train_ablation.DEFAULTS}
        )
        model = train_ablation.build_model(config)
        input_channels = train_ablation.parse_channels(config.input_channels)
        return model, config.experiment_name, str(config_path), input_channels

    if not args.model_name:
        raise ValueError("Provide either --run-config or --model-name")
    return get_model(args.model_name, in_channels=7, num_classes=2), args.model_name, "", list(range(7))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def update_results_json(path: Path, evaluation: dict) -> None:
    results = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            results = json.load(f)
    results["test"] = evaluation
    write_json(path, results)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else checkpoint_path.parent.parent
    )
    output_dirs = [output_dir]
    if args.staging_output_dir:
        staging_dir = Path(args.staging_output_dir).expanduser().resolve()
        if staging_dir not in output_dirs:
            output_dirs.append(staging_dir)

    requested_device = args.device
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    model, resolved_model_name, run_config_path, input_channels = build_evaluation_model(args)
    dataset = LabeledTestDataset(str(data_root), input_channels=input_channels)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = model.to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    batch_metrics = []
    total_tp = total_fp = total_fn = total_tn = 0
    with torch.inference_mode():
        for images, targets in tqdm(loader, desc="Test", unit="batch"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            output = model(images)
            if isinstance(output, tuple):
                output = output[0]
            predictions = torch.argmax(output, dim=1)

            tp, fp, fn, tn = confusion_counts(predictions, targets)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            batch_metrics.append(metrics_from_counts(tp, fp, fn, tn))

    batch_average_metrics = mean_metric_dict(batch_metrics)
    global_pixel_metrics = metrics_from_counts(
        total_tp, total_fp, total_fn, total_tn
    )
    evaluation = {
        "status": "completed",
        "evaluation_type": "public_test_metrics",
        "split": "test",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(data_root),
        "model_name": resolved_model_name,
        "run_config": run_config_path,
        "input_channels": input_channels,
        "checkpoint": str(checkpoint_path),
        "checkpoint_selection": "best validation mIoU checkpoint",
        "device": str(device),
        "num_samples": len(dataset),
        "num_batches": len(batch_metrics),
        # Keep the canonical metrics field aligned with train_ablation.py validation:
        # all test pixels are pooled before calculating mIoU.
        "metrics": global_pixel_metrics,
        "global_pixel_metrics": global_pixel_metrics,
        "batch_average_metrics": batch_average_metrics,
        "aggregation": {
            "metrics": "global pixel metrics; primary paper aggregation",
            "global_pixel_metrics": "metrics from all test pixels pooled together",
            "batch_average_metrics": "mean of per-batch metrics; auxiliary compatibility report",
        },
    }

    for directory in output_dirs:
        write_json(directory / "test_metrics.json", evaluation)
        update_results_json(directory / "results.json", evaluation)

    print(json.dumps(evaluation, ensure_ascii=False, indent=2))
    print("Updated output directories:")
    for directory in output_dirs:
        print(directory)


if __name__ == "__main__":
    main()
