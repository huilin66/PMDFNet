"""Official train→val entry point for SKG-PMDFNet experiments.

This entry point intentionally never creates a test loader.  Test evaluation is
performed by ``eval_skg.py`` after model and hyperparameter choices are frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# Direct invocation fallback; launchers set this before starting Python.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from env_utils import get_data_root
from knowledge.input_normalization import ChannelNormalizer, _as_chw, compute_channel_normalizer
from knowledge.prototype_bank import load_prototype_artifact
from knowledge.prototype_prior import PrototypePrior
from knowledge.spectral_descriptor import load_descriptor_config
from losses import CrossEntropyDiceLoss, KnowledgeConsistencyLoss
from networks.skg_dual_segformer import SKGDualSegFormer, SingleStreamSegFormer
from reproducibility import (
    env_bool,
    runtime_metadata,
    seed_worker as shared_seed_worker,
    set_global_seed,
)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_channels(value: str) -> tuple[int, ...]:
    channels = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not channels:
        raise argparse.ArgumentTypeError("Channel list cannot be empty")
    return channels


@dataclass
class TrainConfig:
    data_root: str
    output_dir: str
    experiment_name: str
    stage: str
    seed: int
    train_split: str
    val_split: str
    test_split: str
    eval_test: bool
    encoder: str
    pretrain: bool
    model_family: str
    channels1: str
    channels2: str
    fusion: str
    knowledge_mode: str
    knowledge_stages: str
    descriptor_config: str
    prototype_path: str
    prototype_k: int
    prototype_distance: str
    prototype_temperature: float
    prototype_update: str
    prototype_reg_weight: float
    knowledge_consistency_weight: float
    knowledge_confidence: str
    knowledge_confidence_threshold: float
    normalization: str
    normalization_pixels_per_image: int
    input_size: int
    batch_size: int
    epochs: int
    early_stopping_patience: int
    min_delta: float
    lr: float
    weight_decay: float
    scheduler: str
    augmentation: str
    num_workers: int
    device: str
    mixed_precision: bool
    deterministic: bool
    ignore_index: int
    resume: str
    max_train_samples: int
    max_val_samples: int
    save_gates: bool


class OfficialSegmentationDataset(Dataset):
    """Labeled MMLSv2 dataset that never uses non-train statistics."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        normalizer: ChannelNormalizer,
        input_size: int,
        *,
        training: bool = False,
        augmentation: str = "basic",
        max_samples: int = 0,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.normalizer = normalizer
        self.input_size = int(input_size)
        self.training = training
        self.augmentation = augmentation.lower()
        images_dir = self.root_dir / split / "images"
        masks_dir = self.root_dir / split / "masks"
        self.samples: list[tuple[Path, Path]] = []
        for image_path in sorted(images_dir.glob("*.tif")):
            mask_path = masks_dir / image_path.name
            if not mask_path.exists():
                raise FileNotFoundError(f"{split} must be labeled: no mask for {image_path.name}")
            self.samples.append((image_path, mask_path))
        if not self.samples:
            raise FileNotFoundError(f"No TIFF samples found in {images_dir}")
        if max_samples > 0:
            self.samples = self.samples[:max_samples]

    @property
    def image_paths(self) -> list[Path]:
        return [image for image, _ in self.samples]

    def manifest_rows(self) -> list[dict[str, str]]:
        return [
            {
                "image": str(image.relative_to(self.root_dir)).replace("\\", "/"),
                "mask": str(mask.relative_to(self.root_dir)).replace("\\", "/"),
            }
            for image, mask in self.samples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.augmentation == "none":
            return image, mask
        if self.augmentation != "basic":
            raise ValueError("Official augmentation must be 'basic' or 'none'")
        if torch.rand(()) < 0.5:
            image, mask = image.flip(-1), mask.flip(-1)
        if torch.rand(()) < 0.5:
            image, mask = image.flip(-2), mask.flip(-2)
        if torch.rand(()) < 0.5:
            turns = int(torch.randint(1, 4, ()).item())
            image, mask = torch.rot90(image, turns, (-2, -1)), torch.rot90(mask, turns, (-2, -1))
        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]
        image = self.normalizer.apply_numpy(_as_chw(tifffile.imread(image_path)))
        mask = tifffile.imread(mask_path).astype(np.int64)
        image_t = torch.from_numpy(image).float()
        mask_t = torch.from_numpy(mask).long()
        image_t, mask_t = self._augment(image_t, mask_t)
        if image_t.shape[-2:] != (self.input_size, self.input_size):
            image_t = F.interpolate(
                image_t.unsqueeze(0), size=(self.input_size, self.input_size), mode="bilinear", align_corners=False
            ).squeeze(0)
            mask_t = F.interpolate(
                mask_t.unsqueeze(0).unsqueeze(0).float(), size=(self.input_size, self.input_size), mode="nearest"
            ).squeeze(0).squeeze(0).long()
        return image_t, mask_t


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Official train→val runner for SKG-PMDFNet.")
    parser.add_argument("--data-root", default=get_data_root())
    parser.add_argument("--output-dir", default="outputs_skg/baselines")
    parser.add_argument("--stage", default="baselines")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--eval-test", type=str2bool, default=False, help="Must remain false; use eval_skg.py for frozen test evaluation.")
    parser.add_argument("--encoder", default="tu-convnext_tiny")
    parser.add_argument("--pretrain", type=str2bool, default=True)
    parser.add_argument("--channels1", default="0,1,2,3")
    parser.add_argument("--channels2", default="4,5,6")
    parser.add_argument("--model-family", choices=["single", "dual"], default="dual")
    parser.add_argument("--fusion", choices=["none", "add", "cat", "att", "knowledge_gate"], default="cat")
    parser.add_argument("--knowledge-mode", choices=["none", "descriptor", "prototype", "full"], default="none")
    parser.add_argument("--knowledge-stages", default="3,4")
    parser.add_argument("--descriptor-config", default="configs/descriptors/default.json")
    parser.add_argument("--prototype-path", default="")
    parser.add_argument("--prototype-k", type=int, default=4)
    parser.add_argument("--prototype-distance", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--prototype-temperature", type=float, default=0.1)
    parser.add_argument("--prototype-update", choices=["fixed", "learnable", "residual"], default="fixed")
    parser.add_argument("--prototype-reg-weight", type=float, default=0.0)
    parser.add_argument("--knowledge-consistency-weight", type=float, default=0.0)
    parser.add_argument("--knowledge-confidence", choices=["maxprob", "margin", "entropy", "distance"], default="maxprob")
    parser.add_argument("--knowledge-confidence-threshold", type=float, default=0.7)
    parser.add_argument("--normalization", choices=["auto", "none", "train_zscore", "robust"], default="auto")
    parser.add_argument("--normalization-pixels-per-image", type=int, default=32768)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--early-stopping-patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--augmentation", choices=["basic", "none"], default="basic")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mixed-precision", type=str2bool, default=True)
    parser.add_argument("--deterministic", type=str2bool, default=True)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--resume", default="")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--save-gates", type=str2bool, default=False)
    return parser


def set_seed(seed: int, deterministic: bool) -> dict:
    return set_global_seed(
        seed,
        deterministic=deterministic,
        strict=env_bool("TRAIN_STRICT_DETERMINISM", False),
    )


def choose_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def worker_seed(_: int) -> None:
    return shared_seed_worker(_)


def compute_metrics(confusion: np.ndarray) -> dict[str, float]:
    confusion = confusion.astype(np.float64)
    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1e-8)
    precision = tp[1] / max(tp[1] + fp[1], 1e-8)
    recall = tp[1] / max(tp[1] + fn[1], 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "miou": float(iou.mean()),
        "iou_bg": float(iou[0]),
        "iou_fg": float(iou[1]),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def update_confusion(confusion: np.ndarray, prediction: torch.Tensor, target: torch.Tensor, ignore_index: int) -> None:
    valid = target != ignore_index
    target = target[valid].reshape(-1)
    prediction = prediction[valid].reshape(-1)
    encoded = target * 2 + prediction
    values = torch.bincount(encoded, minlength=4).reshape(2, 2).cpu().numpy()
    confusion += values


def evaluate(model, loader: DataLoader, criterion: CrossEntropyDiceLoss, config: TrainConfig, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    confusion = np.zeros((2, 2), dtype=np.int64)
    autocast_enabled = config.mixed_precision and device.type == "cuda"
    with torch.no_grad():
        for image, target in loader:
            image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                logits = model(image)
                loss = criterion(logits, target)
            total_loss += float(loss.item())
            update_confusion(confusion, logits.argmax(dim=1), target, config.ignore_index)
    metrics = compute_metrics(confusion)
    metrics["loss"] = total_loss / max(1, len(loader))
    return metrics


def save_gate_snapshot(model, loader: DataLoader, device: torch.device, output_path: Path) -> None:
    """Save one validation gate map plus spatially averaged gate contributions."""

    model.eval()
    with torch.no_grad():
        image, _ = next(iter(loader))
        _, aux = model(image.to(device, non_blocking=True), return_aux=True)
    arrays: dict[str, np.ndarray] = {}
    for stage, gates in aux["gates"].items():
        gates = gates.detach().float().cpu()
        arrays[f"{stage}_mean"] = gates.mean(dim=(0, 2, 3)).numpy()
        arrays[f"{stage}_sample0"] = gates[0].numpy()
    if arrays:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **arrays)


def write_csv_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def create_run_dir(config: TrainConfig) -> Path:
    run_dir = Path(config.output_dir) / config.experiment_name / str(config.seed)
    if run_dir.exists() and not config.resume:
        raise FileExistsError(f"Run directory already exists: {run_dir}. Use a new experiment name/seed or --resume.")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "visualizations").mkdir(exist_ok=True)
    return run_dir


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"skg.{run_dir.as_posix()}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "run.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    best_metrics: dict[str, float],
    config: TrainConfig,
    normalizer: ChannelNormalizer,
) -> None:
    torch.save(
        {
            "format": "skg_checkpoint_v1",
            "epoch": epoch,
            "best_metric": float(best_metric),
            "best_metric_name": "validation_mIoU",
            "best_epoch": int(best_epoch),
            "best_metrics": dict(best_metrics),
            "best_val_miou": float(best_metrics.get("miou", best_metric)),
            "best_val_iou_fg": float(best_metrics.get("iou_fg", -np.inf)),
            "config": asdict(config),
            "input_normalizer": normalizer.state_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer=None, scheduler=None) -> tuple[int, float, int, dict[str, float]]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    best_metrics = checkpoint.get("best_metrics", {})
    best_miou = checkpoint.get("best_val_miou", checkpoint.get("best_metric", -np.inf))
    return (
        int(checkpoint.get("epoch", -1)) + 1,
        float(best_miou),
        int(checkpoint.get("best_epoch", 0)),
        dict(best_metrics),
    )


def resolve_normalizer(config: TrainConfig, train_paths: list[Path], prototype_metadata: dict | None) -> ChannelNormalizer:
    if prototype_metadata:
        artifact_normalizer = ChannelNormalizer.from_state_dict(prototype_metadata["input_normalizer"])
        if config.normalization == "auto":
            return artifact_normalizer
        if config.normalization != artifact_normalizer.mode:
            raise ValueError(
                "Normalization does not match the prototype artifact. Rebuild prototypes or use --normalization auto."
            )
        return artifact_normalizer
    mode = "none" if config.normalization == "auto" else config.normalization
    return compute_channel_normalizer(
        train_paths,
        mode=mode,
        max_pixels_per_image=config.normalization_pixels_per_image,
        seed=config.seed,
    )


def validate_knowledge_config(config: TrainConfig, descriptor_config: dict, prototype_metadata: dict | None) -> None:
    if config.model_family == "single":
        if config.fusion != "none" or config.knowledge_mode != "none" or config.prototype_path:
            raise ValueError("Single-stream baselines require fusion=none, knowledge-mode=none, and no prototype artifact.")
        return
    knowledge_requested = config.knowledge_mode in {"prototype", "full"} or config.fusion == "knowledge_gate"
    if knowledge_requested and not config.prototype_path:
        raise ValueError("Prototype/full knowledge requires --prototype-path built from the train split.")
    if config.knowledge_consistency_weight > 0 and config.knowledge_mode != "full":
        raise ValueError("Knowledge consistency loss requires --knowledge-mode full.")
    if config.prototype_reg_weight > 0 and config.prototype_update != "residual":
        raise ValueError("Prototype regularization requires --prototype-update residual.")
    if prototype_metadata:
        if prototype_metadata.get("source_split") != "train":
            raise ValueError("Prototype artifact is not marked as train-derived.")
        if prototype_metadata.get("descriptor_config") != descriptor_config:
            raise ValueError("Descriptor configuration does not match the prototype artifact.")
        if int(prototype_metadata.get("prototype_k", -1)) != config.prototype_k:
            raise ValueError("--prototype-k does not match the prototype artifact.")
        if prototype_metadata.get("distance") != config.prototype_distance:
            raise ValueError("--prototype-distance does not match the prototype artifact.")


def build_model(config: TrainConfig, descriptor_config: dict, prototype_bank):
    if config.model_family == "single":
        return SingleStreamSegFormer(
            encoder_name=config.encoder,
            encoder_weights="imagenet" if config.pretrain else None,
            in_channels=7,
            num_classes=2,
        )
    return SKGDualSegFormer(
        encoder_name=config.encoder,
        encoder_weights="imagenet" if config.pretrain else None,
        channels1=parse_channels(config.channels1),
        channels2=parse_channels(config.channels2),
        fusion=config.fusion,
        knowledge_mode=config.knowledge_mode,
        descriptor_config=descriptor_config,
        prototype_bank=prototype_bank,
        prototype_distance=config.prototype_distance,
        prototype_temperature=config.prototype_temperature,
        knowledge_stages=config.knowledge_stages,
    )


def run(config: TrainConfig) -> Path:
    if config.eval_test:
        raise ValueError("train_skg.py never evaluates test. Run eval_skg.py after freezing a checkpoint.")
    if config.train_split != "train":
        raise ValueError("Official training requires --train-split train.")
    if config.val_split == config.test_split:
        raise ValueError("Validation and test splits must remain distinct.")
    if not config.data_root:
        raise ValueError("--data-root is required (or set MMLSV2_DATA_ROOT in .env).")

    descriptor_config = load_descriptor_config(config.descriptor_config)
    prototype_bank = None
    prototype_metadata = None
    if config.prototype_path:
        prototype_bank, prototype_metadata = load_prototype_artifact(config.prototype_path, config.prototype_update)
    validate_knowledge_config(config, descriptor_config, prototype_metadata)

    provisional_train = OfficialSegmentationDataset(
        config.data_root,
        config.train_split,
        ChannelNormalizer.identity(),
        config.input_size,
        training=False,
        max_samples=config.max_train_samples,
    )
    normalizer = resolve_normalizer(config, provisional_train.image_paths, prototype_metadata)
    train_dataset = OfficialSegmentationDataset(
        config.data_root,
        config.train_split,
        normalizer,
        config.input_size,
        training=True,
        augmentation=config.augmentation,
        max_samples=config.max_train_samples,
    )
    val_dataset = OfficialSegmentationDataset(
        config.data_root,
        config.val_split,
        normalizer,
        config.input_size,
        training=False,
        max_samples=config.max_val_samples,
    )

    run_dir = create_run_dir(config)
    logger = setup_logger(run_dir)
    reproducibility = set_seed(config.seed, config.deterministic)
    device = choose_device(config.device)
    logger.info(
        "Device: %s | train samples: %d | val samples: %d | "
        "strict determinism: %s",
        device,
        len(train_dataset),
        len(val_dataset),
        reproducibility["strict_determinism"],
    )
    logger.info("Protocol: %s → %s for selection; %s is not loaded.", config.train_split, config.val_split, config.test_split)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, ensure_ascii=False)
    with open(run_dir / "input_normalizer.json", "w", encoding="utf-8") as handle:
        json.dump(normalizer.state_dict(), handle, indent=2, ensure_ascii=False)
    with open(run_dir / "split_manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"train": train_dataset.manifest_rows(), "val": val_dataset.manifest_rows()}, handle, indent=2)
    with open(run_dir / "git_commit.txt", "w", encoding="utf-8") as handle:
        handle.write(git_value(["rev-parse", "HEAD"]) + "\n")
    with open(run_dir / "environment.txt", "w", encoding="utf-8") as handle:
        handle.write(f"python={sys.version}\nplatform={platform.platform()}\ntorch={torch.__version__}\n")
        handle.write(f"git_dirty={bool(git_value(['status', '--short']))}\n")
    with open(run_dir / "reproducibility.json", "w", encoding="utf-8") as handle:
        json.dump(reproducibility, handle, indent=2, ensure_ascii=False)
    with open(run_dir / "runtime.json", "w", encoding="utf-8") as handle:
        json.dump(runtime_metadata(), handle, indent=2, ensure_ascii=False)
    if prototype_metadata:
        with open(run_dir / "prototype_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(prototype_metadata, handle, indent=2, ensure_ascii=False)

    generator = torch.Generator().manual_seed(config.seed)
    loader_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": worker_seed,
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, **loader_kwargs)

    model = build_model(config, descriptor_config, prototype_bank).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info("Model parameters: %d", parameter_count)

    criterion = CrossEntropyDiceLoss(ignore_index=config.ignore_index)
    consistency = KnowledgeConsistencyLoss(config.knowledge_confidence_threshold)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs) if config.scheduler == "cosine" else None
    autocast_enabled = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=autocast_enabled)
    start_epoch, best_miou, best_epoch, best_metrics = (0, -np.inf, 0, {})
    if config.resume:
        start_epoch, best_miou, best_epoch, best_metrics = load_checkpoint(
            config.resume, model, optimizer, scheduler
        )
        logger.info("Resumed at epoch %d with best validation mIoU %.5f", start_epoch, best_miou)

    max_val_iou_fg = float(best_metrics.get("iou_fg", -np.inf))
    no_improvement = 0
    metrics_path = run_dir / "training_metrics.csv"
    for epoch in range(start_epoch, config.epochs):
        model.train()
        total_loss = total_seg = total_kc = total_reg = 0.0
        for image, target in train_loader:
            image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            needs_aux = config.knowledge_consistency_weight > 0
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                if needs_aux:
                    logits, aux = model(image, return_aux=True)
                else:
                    logits, aux = model(image), None
                segmentation_loss = criterion(logits, target)
                kc_loss = segmentation_loss.new_zeros(())
                if config.knowledge_consistency_weight > 0:
                    confidence = PrototypePrior.confidence(aux["confidence"], config.knowledge_confidence)
                    kc_loss = consistency(logits, aux["prior"], confidence, target != config.ignore_index)
                regularization = model.knowledge_regularization()
                loss = (
                    segmentation_loss
                    + config.knowledge_consistency_weight * kc_loss
                    + config.prototype_reg_weight * regularization
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())
            total_seg += float(segmentation_loss.item())
            total_kc += float(kc_loss.item())
            total_reg += float(regularization.item())

        val = evaluate(model, val_loader, criterion, config, device)
        if scheduler is not None:
            scheduler.step()
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train_loader),
            "train_seg_loss": total_seg / len(train_loader),
            "train_kc_loss": total_kc / len(train_loader),
            "train_proto_reg": total_reg / len(train_loader),
            "lr": optimizer.param_groups[0]["lr"],
            "val_loss": val["loss"],
            "val_miou": val["miou"],
            "val_iou_fg": val["iou_fg"],
            "val_f1": val["f1"],
        }
        improved = val["miou"] > best_miou + config.min_delta
        if improved:
            best_miou = val["miou"]
            best_metrics = dict(val)
            best_epoch = epoch + 1
            no_improvement = 0
            save_checkpoint(
                run_dir / "checkpoints" / "best.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_miou,
                best_epoch,
                best_metrics,
                config,
                normalizer,
            )
            if config.save_gates:
                save_gate_snapshot(model, val_loader, device, run_dir / "visualizations" / "best_val_gates.npz")
        else:
            no_improvement += 1
        max_val_iou_fg = max(max_val_iou_fg, float(val["iou_fg"]))
        save_checkpoint(
            run_dir / "checkpoints" / "last.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_miou,
            best_epoch,
            best_metrics,
            config,
            normalizer,
        )
        write_csv_row(metrics_path, row)
        logger.info(
            "Epoch %03d | train %.4f | val loss %.4f | mIoU %.4f | IoU_fg %.4f | F1 %.4f%s",
            epoch + 1, row["train_loss"], val["loss"], val["miou"], val["iou_fg"], val["f1"], " | best mIoU" if improved else "",
        )
        if config.early_stopping_patience > 0 and no_improvement >= config.early_stopping_patience:
            logger.info("Early stopped after %d validations without val mIoU improvement.", no_improvement)
            break

    with open(run_dir / "val_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "selection_metric": "validation_mIoU",
                "best_val": best_metrics,
                "best_val_iou_fg": best_metrics.get("iou_fg", ""),
                "best_val_miou": best_miou,
                "max_val_iou_fg": max_val_iou_fg,
            },
            handle,
            indent=2,
        )
    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "stage": config.stage,
        "experiment_id": config.experiment_name,
        "seed": config.seed,
        "model": "SingleStream-SegFormer" if config.model_family == "single" else "SKG-PMDFNet",
        "encoder": config.encoder,
        "fusion": config.fusion,
        "knowledge_descriptor": Path(config.descriptor_config).name if config.knowledge_mode != "none" else "none",
        "prototype_type": config.prototype_update if config.prototype_path else "none",
        "prototype_k": config.prototype_k if config.prototype_path else "",
        "lambda_kc": config.knowledge_consistency_weight,
        "params": parameter_count,
        "selection_metric": "validation_mIoU",
        "best_val_epoch": best_epoch,
        "best_val_iou_fg": best_metrics.get("iou_fg", ""),
        "best_val_miou": best_miou,
        "max_val_iou_fg": max_val_iou_fg,
        "final_val_iou_fg": val["iou_fg"],
        "test_iou_fg": "",
        "test_miou": "",
        "test_f1": "",
        "run_dir": str(run_dir),
        "git_commit": git_value(["rev-parse", "HEAD"]),
    }
    summary_name = "baseline_summary.csv" if config.stage == "baselines" else f"{config.stage}_summary.csv"
    write_csv_row(Path(config.output_dir) / summary_name, summary)
    logger.info("Completed. Best validation mIoU: %.5f. No test data was loaded.", best_miou)
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = TrainConfig(**vars(args))
    run(config)


if __name__ == "__main__":
    main()
