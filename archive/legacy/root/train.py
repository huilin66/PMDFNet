import logging
import json
import os
import shutil
from datetime import datetime

# This must be present before CUDA kernels are first used.  The launcher also
# sets it explicitly for each seed; this fallback covers direct invocation.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MarsSegDataset, MosaicCastDataset
from env_utils import get_data_root
from losses import UnetFormerLoss
from networks import get_model
from reproducibility import (
    env_bool,
    file_sha256,
    repository_metadata,
    runtime_metadata,
    seed_worker as shared_seed_worker,
    set_global_seed,
)

RANDOM_SEED = int(os.environ.get("TRAIN_SEED", "42"))
STRICT_DETERMINISM = env_bool("TRAIN_STRICT_DETERMINISM", False)


def set_seed(seed=42):
    return set_global_seed(seed, deterministic=True, strict=STRICT_DETERMINISM)


def seed_worker(worker_id):
    return shared_seed_worker(worker_id)


def setup_logger(save_dir):
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"train_log_{timestamp}.txt")
    logger_name = f"pmdfnet.train.{os.path.abspath(save_dir)}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # A single train_pipeline invocation should own exactly one file and one
    # console handler. This avoids duplicated lines when several experiments
    # are launched in the same Python process.
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(log_format)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, timestamp, log_file


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def finalize_experiment(staging_dir, final_dir):
    """Copy a completed run, including logs and checkpoints, to the dataset."""
    os.makedirs(os.path.dirname(final_dir), exist_ok=True)
    if os.path.exists(final_dir):
        raise FileExistsError(f"Final output directory already exists: {final_dir}")
    shutil.copytree(staging_dir, final_dir)


def calculate_metrics(preds, targets):
    """
    Calculate metrics for binary segmentation (Class 1: Landslide, Class 0: Background)
    preds: [B, H, W] (0 or 1)
    targets: [B, H, W] (0 or 1)
    """
    return calculate_metrics_from_confusion(calculate_confusion(preds, targets))


def calculate_confusion(preds, targets):
    """Return a global [target, prediction] confusion matrix for a batch."""

    preds = preds.reshape(-1).long()
    targets = targets.reshape(-1).long()
    encoded = targets * 2 + preds
    return torch.bincount(encoded, minlength=4).reshape(2, 2).to(torch.float64).cpu()


def calculate_metrics_from_confusion(confusion):
    """Calculate metrics from an accumulated pixel-level confusion matrix."""

    confusion = torch.as_tensor(confusion, dtype=torch.float64)
    tp = confusion[1, 1]
    fp = confusion[0, 1]
    fn = confusion[1, 0]
    tn = confusion[0, 0]

    # Precision, Recall, F1 for Foreground
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # IoU
    iou_fg = tp / (tp + fp + fn + 1e-8)
    iou_bg = tn / (tn + fp + fn + 1e-8)
    miou = (iou_fg + iou_bg) / 2

    return {
        "miou": miou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "iou_fg": iou_fg.item(),
        "iou_bg": iou_bg.item(),
    }


def train_pipeline(model_name, conduct_val=False):
    reproducibility = set_seed(RANDOM_SEED)

    if model_name.startswith(("dual_", "pmdfnet_")):
        BATCH_SIZE = 16
    elif model_name.startswith("ocrnet"):
        BATCH_SIZE = 8
    elif model_name.startswith("mask2former"):
        BATCH_SIZE = 8
    elif model_name.startswith("oneformer"):
        BATCH_SIZE = 8
    else:
        BATCH_SIZE = 32
    BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", BATCH_SIZE))
    LR = float(os.environ.get("TRAIN_LR", "1e-4"))
    WEIGHT_DECAY = float(os.environ.get("TRAIN_WEIGHT_DECAY", "5e-4"))
    EPOCHS = int(os.environ.get("TRAIN_EPOCHS", "100"))
    VAL_INTERVAL = int(os.environ.get("TRAIN_VAL_INTERVAL", "1"))
    NUM_WORKERS = int(os.environ.get("TRAIN_NUM_WORKERS", "4"))
    IN_CHANNELS = 7

    DATASET_ROOT = (
        os.environ.get("MARS_DATA_ROOT")
        or os.environ.get("DATA_ROOT")
        or get_data_root(
            "/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase_updateB2"
        )
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_ROOT = os.environ.get("TRAIN_OUTPUT_ROOT", "outputs")
    run_id = f"{model_name}_seed{RANDOM_SEED}_{timestamp}"
    EXPERIMENT_DIR = os.path.join(OUTPUT_ROOT, run_id)

    FINALIZE_RESULTS = os.environ.get("TRAIN_FINALIZE_RESULTS", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    FINAL_OUTPUT_ROOT = os.environ.get(
        "TRAIN_FINAL_OUTPUT_ROOT", os.path.join(DATASET_ROOT, "outputs")
    )
    FINAL_EXPERIMENT_DIR = os.path.join(FINAL_OUTPUT_ROOT, run_id)
    mapping_manifest = os.path.join(DATASET_ROOT, "mapping_manifest.json")

    CHECKPOINT_DIR = os.path.join(EXPERIMENT_DIR, "checkpoints")
    LOG_DIR = os.path.join(EXPERIMENT_DIR, "logs")
    TB_DIR = os.path.join(EXPERIMENT_DIR, "tensorboard")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TB_DIR, exist_ok=True)

    requested_device = os.environ.get("TRAIN_DEVICE", "cuda:0")
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=TB_DIR)
    logger, _, log_file = setup_logger(LOG_DIR)
    logger.info(f"Model: {model_name}")
    logger.info(f"Experiment started at {timestamp}")
    logger.info(f"Staging outputs: {EXPERIMENT_DIR}")
    if FINALIZE_RESULTS:
        logger.info(f"Final outputs after success: {FINAL_EXPERIMENT_DIR}")
    else:
        logger.info("Final dataset output copy is disabled")
    logger.info(f"Using device: {device}")
    logger.info(f"Dataset root: {DATASET_ROOT}")
    logger.info(
        f"Seed: {RANDOM_SEED} | strict determinism: {STRICT_DETERMINISM} | "
        f"workers: {NUM_WORKERS} | epochs: {EPOCHS} | batch size: {BATCH_SIZE}"
    )
    logger.info(
        "Checkpoint policy: last.pth is saved every epoch; best.pth is selected "
        "by validation mIoU"
    )

    run_config = {
        "run_id": run_id,
        "model_name": model_name,
        "seed": RANDOM_SEED,
        "dataset_root": DATASET_ROOT,
        "mapping_manifest": os.path.abspath(mapping_manifest)
        if os.path.isfile(mapping_manifest)
        else None,
        "mapping_manifest_sha256": file_sha256(mapping_manifest),
        "staging_output_dir": os.path.abspath(EXPERIMENT_DIR),
        "final_output_dir": os.path.abspath(FINAL_EXPERIMENT_DIR)
        if FINALIZE_RESULTS
        else None,
        "finalize_results": FINALIZE_RESULTS,
        "device": str(device),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "validation_enabled": bool(conduct_val),
        "validation_interval": VAL_INTERVAL,
        "best_checkpoint_criterion": "validation_mIoU",
        "validation_metric_aggregation": "global_pixel_confusion",
        "checkpoint_files": ["checkpoints/best.pth", "checkpoints/last.pth"],
        "log_file": os.path.abspath(log_file),
        "tensorboard_dir": os.path.abspath(TB_DIR),
        "reproducibility": reproducibility,
        "runtime": runtime_metadata(),
        "repository": repository_metadata(os.path.dirname(__file__)),
    }
    write_json(os.path.join(EXPERIMENT_DIR, "run_config.json"), run_config)

    model = get_model(model_name, in_channels=IN_CHANNELS, num_classes=2).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name} Model Parameters: {total_params / 1e6:.2f} M")

    train_dataset_raw = MarsSegDataset(root_dir=DATASET_ROOT, split="train")
    train_dataset = MosaicCastDataset(train_dataset_raw)
    val_dataset = MarsSegDataset(root_dir=DATASET_ROOT, split="val")

    gt = torch.Generator()
    gt.manual_seed(RANDOM_SEED)
    gv = torch.Generator()
    gv.manual_seed(RANDOM_SEED)

    if len(train_dataset) == 0:
        logger.warning("Dataset not found. Creating dummy data.")
        train_loader = []
        val_loader = []
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=gt,
        )
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                worker_init_fn=seed_worker,
                generator=gv,
            )
        else:
            val_loader = None
            logger.warning("Validation set is empty.")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = UnetFormerLoss()

    best_miou = float("-inf")
    best_epoch = None
    best_metrics = {}
    last_epoch = None
    last_metrics = {}
    logger.info("Start Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]", unit="batch"
        )
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)

            if isinstance(output, tuple):
                final_output, aux_output = output
                loss_main = criterion(final_output, target)
                loss_aux = criterion(aux_output, target)
                loss = loss_main + 0.4 * loss_aux
                output = final_output
            else:
                loss = criterion(output, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_val = loss.item()
            train_loss += loss_val

            pbar.set_postfix({"loss": f"{loss_val:.4f}"})
            writer.add_scalar(
                "Train/Batch_Loss", loss_val, epoch * len(train_loader) + batch_idx
            )

        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        logger.info(f"Epoch [{epoch + 1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f}")
        writer.add_scalar("Train/Epoch_Loss", avg_train_loss, epoch)
        last_epoch = epoch + 1
        last_metrics = {"train_loss": float(avg_train_loss)}

        # turn off validation in this task
        if conduct_val and val_loader and (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            val_loss = 0

            total_confusion = torch.zeros((2, 2), dtype=torch.float64)

            with torch.no_grad():
                pbar_val = tqdm(
                    val_loader,
                    desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]",
                    unit="batch",
                    leave=False,
                )
                for val_data, val_target in pbar_val:
                    val_data, val_target = val_data.to(device), val_target.to(device)

                    output = model(val_data)
                    if isinstance(output, tuple):
                        final_output, aux_output = output
                        if hasattr(criterion, "main_loss") and hasattr(
                            criterion, "aux_loss"
                        ):
                            loss_main = criterion.main_loss(final_output, val_target)
                            loss_aux = criterion.aux_loss(aux_output, val_target)
                        else:
                            loss_main = criterion(final_output, val_target)
                            loss_aux = criterion(aux_output, val_target)
                        loss = loss_main + 0.4 * loss_aux
                        output = final_output
                    else:
                        if hasattr(criterion, "main_loss"):
                            loss = criterion.main_loss(output, val_target)
                        else:
                            loss = criterion(output, val_target)

                    val_loss += loss.item()

                    preds = torch.argmax(output, dim=1)

                    total_confusion += calculate_confusion(preds, val_target)

            avg_val_loss = val_loss / len(val_loader)
            epoch_metrics = calculate_metrics_from_confusion(total_confusion)
            last_metrics = {
                "train_loss": float(avg_train_loss),
                "val_loss": float(avg_val_loss),
                **epoch_metrics,
            }
            logger.info(
                f"Epoch [{epoch + 1}/{EPOCHS}] Val Loss: {avg_val_loss:.4f} | "
                f"global mIoU: {epoch_metrics['miou']:.4f} | "
                f"F1: {epoch_metrics['f1']:.4f} | "
                f"IoU (FG): {epoch_metrics['iou_fg']:.4f} | "
                f"IoU (BG): {epoch_metrics['iou_bg']:.4f}"
            )

            writer.add_scalar("Val/Loss", avg_val_loss, epoch)
            writer.add_scalar("Val/mIoU", epoch_metrics["miou"], epoch)
            writer.add_scalar("Val/F1", epoch_metrics["f1"], epoch)
            writer.add_scalar("Val/Recall", epoch_metrics["recall"], epoch)
            writer.add_scalar("Val/Precision", epoch_metrics["precision"], epoch)
            writer.add_scalar("Val/IoU_FG", epoch_metrics["iou_fg"], epoch)
            writer.add_scalar("Val/IoU_BG", epoch_metrics["iou_bg"], epoch)
            if epoch_metrics["miou"] > best_miou:
                best_miou = epoch_metrics["miou"]
                best_epoch = epoch + 1
                best_metrics = dict(last_metrics)
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best.pth"))
                logger.info(f"New Best Model Saved! (mIoU: {best_miou:.4f})")
        scheduler.step()
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "last.pth"))
        writer.flush()

    logger.info("Training Completed.")
    if best_epoch is None:
        logger.info("No best.pth was written because validation was disabled or empty.")
    else:
        logger.info(
            f"Best checkpoint: epoch {best_epoch}, validation mIoU {best_miou:.4f}"
        )
    logger.info("Model checkpoints saved: best.pth and last.pth")

    results = {
        "status": "completed",
        "run_id": run_id,
        "model_name": model_name,
        "seed": RANDOM_SEED,
        "dataset_root": DATASET_ROOT,
        "staging_output_dir": os.path.abspath(EXPERIMENT_DIR),
        "final_output_dir": os.path.abspath(FINAL_EXPERIMENT_DIR)
        if FINALIZE_RESULTS
        else None,
        "log_file": os.path.abspath(log_file),
        "tensorboard_dir": os.path.abspath(TB_DIR),
        "checkpoint_files": ["checkpoints/best.pth", "checkpoints/last.pth"],
        "best_checkpoint_criterion": "validation_mIoU",
        "validation_metric_aggregation": "global_pixel_confusion",
        "best": {
            "epoch": best_epoch,
            "validation_miou": float(best_miou)
            if best_epoch is not None
            else None,
            "metrics": best_metrics,
        },
        "last": {
            "epoch": last_epoch,
            "metrics": last_metrics,
        },
    }

    # Close the event writer and flush the text log before copying the complete
    # run directory. The staging directory remains available for recovery and
    # debugging if finalization itself fails.
    writer.flush()
    writer.close()
    write_json(os.path.join(EXPERIMENT_DIR, "results.json"), results)
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)

    if FINALIZE_RESULTS:
        finalize_experiment(EXPERIMENT_DIR, FINAL_EXPERIMENT_DIR)

    return results


if __name__ == "__main__":
    model_names = [
        "m3lsnet",
        "ocrnet_hrnet_w48",
        # "unet_resnet50",
        "upernet_convnexttiny",
        "segformer_mitb2",
        "segformer_convnexttiny",
        "dual_segformer_convnexttiny_chv1_add",
        "dual_segformer_convnextsmall_chv1_add",
        "dual_segformer_convnextbase_chv1_add",
        "dual_segformer_convnextlarge_chv1_add",
    ]
    for model_name in model_names:
        train_pipeline(
            model_name,
            conduct_val=True,  # turn off validation in this task
        )
