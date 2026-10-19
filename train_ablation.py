import argparse
import csv
import json
import logging
import os
import random
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from env_utils import get_env_value


DEFAULTS = {
    "data_root": get_env_value("MMLSV2_DATA_ROOT", "MARS_DATA_ROOT", "DATA_ROOT"),
    "train_split": "train",
    "val_split": "val",
    "split_file": "",
    "fold": -1,
    "experiment_name": "ablation",
    "output_dir": "outputs_ablation",
    "summary_csv": "",
    "model_name": "segformer_convnexttiny",
    "arch": "segformer",
    "encoder": "tu-convnext_tiny",
    "pretrain": True,
    "channels1": "0,1,2,3",
    "channels2": "4,5,6",
    "fusion": "add",
    "in_channels": 7,
    "input_channels": "0,1,2,3,4,5,6",
    "num_classes": 2,
    "input_size": 128,
    "batch_size": 16,
    "max_train_samples": 0,
    "max_val_samples": 0,
    "epochs": 100,
    "lr": 1e-4,
    "weight_decay": 5e-4,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "loss": "unetformer",
    "ignore_index": 255,
    "augmentation": "mars",
    "aug_prob": 0.5,
    "mosaic_prob": 0.5,
    "num_workers": 4,
    "device": "auto",
    "seed": 42,
    "mixed_precision": False,
    "deterministic": True,
    "resume": "",
    "init_checkpoint": "",
    "val_interval": 1,
    "save_interval": 1,
    "primary_metric": "miou",
    "early_stopping_patience": 0,
    "min_delta": 0.0,
    "normalization": "auto",
    "dataset_type": "mmlsv2",
    "use_file_list_dataset": False,
    "save_auxiliary_checkpoints": False,
    "finalize_results": False,
    "final_output_dir": "",
    "dry_run": False,
    "skip_model_init": False,
}


# Experiment groups accepted by the training entry points. ``core``, ``ablations``
# and ``extended`` are the disjoint formal groups (19 x 3 = 57 runs);
# ``single_channel`` and ``modality_grouping`` are separate supplementary
# protocol groups and are not part of the formal 57-run count. Keep this tuple
# in sync with supported protocols.
EXPERIMENT_GROUPS = (
    "core",
    "ablations",
    "extended",
    "single_channel",
    "modality_grouping",
    "all",
)


def experiment_in_group(experiment, group):
    """Return True when ``experiment`` (a protocol entry) belongs to ``group``.

    Single source of truth for the trainer, the VRAM probe and the launcher
    scripts, so they can never disagree about which runs a group contains.
    """
    if group == "all":
        return True
    groups = experiment.get("group", "core")
    if isinstance(groups, str):
        groups = [groups]
    return group in groups


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_channels(value):
    if isinstance(value, (list, tuple)):
        channels = [int(v) for v in value]
    else:
        channels = [int(v.strip()) for v in str(value).split(",") if v.strip() != ""]
    invalid = [channel for channel in channels if channel < 0 or channel >= 7]
    if invalid:
        raise ValueError(f"Channel indices must be in [0, 6], got {invalid}")
    return channels


def build_parser(defaults=None):
    defaults = defaults or DEFAULTS
    parser = argparse.ArgumentParser(
        description="Reproducible training entry for PMDFNet ablations."
    )
    parser.add_argument("--config", default="", help="JSON config for one experiment.")
    parser.add_argument(
        "--batch-config",
        default="",
        help="JSON file with {'common': {...}, 'experiments': [{...}]} for ablations.",
    )
    parser.add_argument("--data-root", default=defaults["data_root"])
    parser.add_argument("--train-split", default=defaults["train_split"])
    parser.add_argument("--val-split", default=defaults["val_split"])
    parser.add_argument("--split-file", default=defaults["split_file"])
    parser.add_argument("--fold", type=int, default=defaults["fold"])
    parser.add_argument("--experiment-name", default=defaults["experiment_name"])
    parser.add_argument("--output-dir", default=defaults["output_dir"])
    parser.add_argument(
        "--summary-csv",
        default=defaults["summary_csv"],
        help="Append one row per completed run to this CSV for cross-experiment analysis.",
    )
    parser.add_argument("--model-name", default=defaults["model_name"])
    parser.add_argument("--arch", default=defaults["arch"])
    parser.add_argument("--encoder", default=defaults["encoder"])
    parser.add_argument("--pretrain", type=str2bool, default=defaults["pretrain"])
    parser.add_argument("--channels1", default=defaults["channels1"])
    parser.add_argument("--channels2", default=defaults["channels2"])
    parser.add_argument("--fusion", default=defaults["fusion"])
    parser.add_argument("--in-channels", type=int, default=defaults["in_channels"])
    parser.add_argument(
        "--input-channels",
        default=defaults["input_channels"],
        help="Comma-separated source channel indices selected from the 7-channel input.",
    )
    parser.add_argument("--num-classes", type=int, default=defaults["num_classes"])
    parser.add_argument("--input-size", type=int, default=defaults["input_size"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=defaults["max_train_samples"],
        help="Use only the first N training samples; 0 means all samples.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=defaults["max_val_samples"],
        help="Use only the first N validation samples; 0 means all samples.",
    )
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--optimizer", default=defaults["optimizer"])
    parser.add_argument("--scheduler", default=defaults["scheduler"])
    parser.add_argument("--loss", default=defaults["loss"])
    parser.add_argument("--ignore-index", type=int, default=defaults["ignore_index"])
    parser.add_argument("--augmentation", default=defaults["augmentation"])
    parser.add_argument("--aug-prob", type=float, default=defaults["aug_prob"])
    parser.add_argument("--mosaic-prob", type=float, default=defaults["mosaic_prob"])
    parser.add_argument("--num-workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--mixed-precision", type=str2bool, default=defaults["mixed_precision"])
    parser.add_argument("--deterministic", type=str2bool, default=defaults["deterministic"])
    parser.add_argument("--resume", default=defaults["resume"])
    parser.add_argument("--init-checkpoint", default=defaults["init_checkpoint"])
    parser.add_argument("--val-interval", type=int, default=defaults["val_interval"])
    parser.add_argument("--save-interval", type=int, default=defaults["save_interval"])
    parser.add_argument(
        "--primary-metric",
        default=defaults["primary_metric"],
        choices=["miou", "iou_fg", "f1", "val_loss"],
        help="Metric used for best.pth and optional early stopping.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=defaults["early_stopping_patience"],
        help="Stop after N validations without primary metric improvement; 0 disables it.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=defaults["min_delta"],
        help="Minimum primary metric improvement for checkpoint/early stopping.",
    )
    parser.add_argument(
        "--normalization",
        default=defaults["normalization"],
        choices=["auto", "train", "val", "test", "none"],
    )
    parser.add_argument(
        "--dataset-type",
        default=defaults["dataset_type"],
        choices=["mars_ls", "mmlsv2"],
        help="Dataset type: mars_ls (raw DN, apply mean/std norm) or mmlsv2 (pre-normalized [0,1], skip norm).",
    )
    parser.add_argument(
        "--use-file-list-dataset",
        type=str2bool,
        default=defaults["use_file_list_dataset"],
        help="Read split/images and split/masks with the configurable normalization/channel protocol.",
    )
    parser.add_argument(
        "--save-auxiliary-checkpoints",
        type=str2bool,
        default=defaults["save_auxiliary_checkpoints"],
        help="Also save best_<metric>.pth for non-primary metrics; otherwise only best.pth/last.pth are saved.",
    )
    parser.add_argument(
        "--finalize-results",
        type=str2bool,
        default=defaults["finalize_results"],
        help="Explicitly copy a completed run to final_output_dir or <data_root>/outputs.",
    )
    parser.add_argument(
        "--final-output-dir",
        default=defaults["final_output_dir"],
        help="Optional copy destination; empty means <data_root>/outputs when finalize-results is explicitly enabled.",
    )
    parser.add_argument("--dry-run", action="store_true", default=defaults["dry_run"])
    parser.add_argument(
        "--skip-model-init",
        action="store_true",
        default=defaults["skip_model_init"],
        help="Useful for config/data smoke tests on machines without model deps.",
    )
    return parser


@dataclass
class RunConfig:
    data_root: str
    train_split: str
    val_split: str
    split_file: str
    fold: int
    experiment_name: str
    output_dir: str
    summary_csv: str
    model_name: str
    arch: str
    encoder: str
    pretrain: bool
    channels1: str
    channels2: str
    fusion: str
    in_channels: int
    input_channels: str
    num_classes: int
    input_size: int
    batch_size: int
    max_train_samples: int
    max_val_samples: int
    epochs: int
    lr: float
    weight_decay: float
    optimizer: str
    scheduler: str
    loss: str
    ignore_index: int
    augmentation: str
    aug_prob: float
    mosaic_prob: float
    num_workers: int
    device: str
    seed: int
    mixed_precision: bool
    deterministic: bool
    resume: str
    init_checkpoint: str
    val_interval: int
    save_interval: int
    primary_metric: str
    early_stopping_patience: int
    min_delta: float
    normalization: str
    dataset_type: str
    use_file_list_dataset: bool
    save_auxiliary_checkpoints: bool
    finalize_results: bool
    final_output_dir: str
    dry_run: bool
    skip_model_init: bool


def namespace_to_config(ns):
    data = {k: getattr(ns, k) for k in DEFAULTS.keys()}
    return RunConfig(**data)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_config(*parts):
    merged = dict(DEFAULTS)
    for part in parts:
        if part:
            merged.update(part)
    return merged


def get_git_info():
    def run_git(args):
        try:
            return subprocess.check_output(
                ["git"] + args, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return ""

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(run_git(["status", "--short"])),
    }


def make_run_dir(cfg):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in cfg.experiment_name)
    run_dir = Path(cfg.output_dir) / f"{safe_name}_{timestamp}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logger(run_dir):
    logger = logging.getLogger(f"train_ablation.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(run_dir / "logs" / "train.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def save_run_metadata(cfg, run_dir):
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
    with open(run_dir / "command.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(sys.argv) + "\n")
    with open(run_dir / "git.json", "w", encoding="utf-8") as f:
        json.dump(get_git_info(), f, indent=2, ensure_ascii=False)


def finalize_run_results(cfg, run_dir):
    """Copy a completed run only when explicitly requested by the config/CLI."""
    if not cfg.finalize_results:
        return ""
    if not cfg.data_root:
        raise ValueError("finalize_results requires a non-empty data_root")

    output_root = Path(cfg.final_output_dir) if cfg.final_output_dir else Path(cfg.data_root) / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / Path(run_dir).name
    if target.exists():
        raise FileExistsError(f"Final output already exists: {target}")
    shutil.copytree(run_dir, target)
    return str(target)


def set_seed(seed, deterministic=True):
    from reproducibility import env_bool, set_global_seed

    return set_global_seed(
        seed,
        deterministic=deterministic,
        strict=env_bool("TRAIN_STRICT_DETERMINISM", False),
    )


def model_backend_metadata(cfg):
    """Record backend choices that can change a model's implementation.

    M3LSNet has two intentionally supported paths: the real mamba_ssm
    selective scan and the built-in gated-convolution simulator.  The path is
    runtime-dependent, so it belongs in the run artifact rather than only in
    the static protocol JSON.
    """
    model_name = str(getattr(cfg, "model_name", "") or "").lower()
    if not model_name.startswith("m3lsnet"):
        return {"model_backend": "standard_pytorch_or_smp"}
    try:
        import importlib

        m3lsnet_module = importlib.import_module("networks.M3LSNet")

        available = m3lsnet_module.selective_scan_fn is not None
        return {
            "model_backend": (
                "mamba_ssm_selective_scan" if available else "gated_conv_simulator"
            ),
            "m3lsnet_backend": (
                "mamba_ssm_selective_scan" if available else "gated_conv_simulator"
            ),
            "mamba_ssm_available": bool(available),
        }
    except Exception as exc:
        return {
            "model_backend": "backend_detection_failed",
            "m3lsnet_backend": "backend_detection_failed",
            "mamba_ssm_available": False,
            "backend_error": str(exc),
        }


def seed_worker(worker_id):
    from reproducibility import seed_worker as shared_seed_worker

    return shared_seed_worker(worker_id)


class FileListMarsDataset:
    def __init__(self, root_dir, rows, split, cfg, is_train):
        import numpy as np
        import torch
        import torch.nn.functional as F
        import tifffile

        from augmentations import MarsAugmentor
        from dataset import (
            MARS_MEAN_TEST,
            MARS_MEAN_TRAIN,
            MARS_MEAN_VAL,
            MARS_STD_TEST,
            MARS_STD_TRAIN,
            MARS_STD_VAL,
        )

        self.np = np
        self.torch = torch
        self.F = F
        self.tifffile = tifffile
        self.root_dir = Path(root_dir)
        self.rows = rows
        self.split = split
        self.cfg = cfg
        self.is_train = is_train
        self.input_size = cfg.input_size
        self.input_channels = parse_channels(getattr(cfg, "input_channels", "0,1,2,3,4,5,6"))
        if not self.input_channels:
            raise ValueError("input_channels must contain at least one source channel index")
        self.augmentor = (
            MarsAugmentor(prob=cfg.aug_prob, channel_indices=self.input_channels)
            if is_train and cfg.augmentation.lower() != "none"
            else None
        )

        norm_key = cfg.normalization if cfg.normalization != "auto" else split
        # mmlsv2 data is pre-normalized to [0,1], always skip normalization
        if getattr(cfg, "dataset_type", "mars_ls") == "mmlsv2":
            norm_key = "none"
        if norm_key == "train":
            mean, std = MARS_MEAN_TRAIN, MARS_STD_TRAIN
        elif norm_key == "val":
            mean, std = MARS_MEAN_VAL, MARS_STD_VAL
        elif norm_key == "test":
            mean, std = MARS_MEAN_TEST, MARS_STD_TEST
        else:
            mean, std = None, None
        self.mean = None if mean is None else np.array(mean, dtype=np.float32).reshape(7, 1, 1)
        self.std = None if std is None else np.array(std, dtype=np.float32).reshape(7, 1, 1)

    def __len__(self):
        return len(self.rows)

    def _resolve_path(self, value, split, subdir):
        if value:
            p = Path(value)
            if p.is_absolute():
                return p
            direct = self.root_dir / p
            if direct.exists():
                return direct
            if len(p.parts) == 1:
                return self.root_dir / split / subdir / p
            return direct
        return None

    def __getitem__(self, index):
        row = self.rows[index]
        image_value = row.get("image") or row.get("image_path") or row.get("filename") or row.get("file")
        mask_value = row.get("mask") or row.get("mask_path") or row.get("label") or row.get("label_path")
        split = row.get("split", self.split)

        image_path = self._resolve_path(image_value, split, "images")
        if mask_value:
            mask_path = self._resolve_path(mask_value, split, "masks")
        else:
            mask_path = self.root_dir / split / "masks" / Path(image_path).name

        image = self.tifffile.imread(str(image_path)).astype(self.np.float32)
        if image.ndim == 3 and image.shape[2] == 7:
            image = image.transpose(2, 0, 1)
        if image.ndim != 3 or image.shape[0] != 7:
            raise ValueError(
                f"Expected a 7-channel image at {image_path}, got shape {tuple(image.shape)}"
            )
        image[image < -100000] = 0.0
        if self.mean is not None:
            image = (image - self.mean) / (self.std + 1e-8)
        image = image[self.input_channels]

        mask = self.tifffile.imread(str(mask_path)).astype(self.np.float32)
        image_t = self.torch.from_numpy(image).float()
        mask_t = self.torch.from_numpy(mask).long()

        if self.augmentor is not None:
            image_t, mask_t = self.augmentor(image_t, mask_t)

        if image_t.shape[-1] != self.input_size or image_t.shape[-2] != self.input_size:
            image_t = self.F.interpolate(
                image_t.unsqueeze(0),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mask_t = self.F.interpolate(
                mask_t.unsqueeze(0).unsqueeze(0).float(),
                size=(self.input_size, self.input_size),
                mode="nearest",
            ).squeeze(0).squeeze(0).long()

        return image_t, mask_t


def read_split_rows(split_file):
    path = Path(split_file)
    if path.suffix.lower() == ".json":
        data = load_json(path)
        if isinstance(data, dict):
            data = data.get("samples", data.get("rows", []))
        return data
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".txt":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append({"image": line})
        return rows
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_labeled_folder_dataset(root_dir, split, cfg, is_train):
    images_dir = Path(root_dir) / split / "images"
    if not images_dir.exists():
        return FileListMarsDataset(root_dir, [], split, cfg, is_train)
    rows = [
        {"image": image_path.name, "split": split}
        for image_path in sorted(images_dir.glob("*.tif"))
    ]
    return FileListMarsDataset(root_dir, rows, split, cfg, is_train)


def build_datasets(cfg, logger):
    import torch
    import torch.nn.functional as F

    from dataset import MarsSegDataset, MMLSV2Dataset, MosaicCastDataset

    # Select dataset class based on dataset_type
    DatasetClass = MMLSV2Dataset if cfg.dataset_type == "mmlsv2" else MarsSegDataset
    logger.info("Dataset type: %s -> using %s", cfg.dataset_type, DatasetClass.__name__)

    if cfg.split_file:
        rows = read_split_rows(cfg.split_file)
        if cfg.fold >= 0 and rows and "fold" in rows[0]:
            train_rows = [r for r in rows if str(r.get("fold")) != str(cfg.fold)]
            val_rows = [r for r in rows if str(r.get("fold")) == str(cfg.fold)]
        else:
            train_rows = [r for r in rows if r.get("split", cfg.train_split) == cfg.train_split]
            val_rows = [r for r in rows if r.get("split", "") == cfg.val_split]
        train_dataset = FileListMarsDataset(cfg.data_root, train_rows, cfg.train_split, cfg, True)
        val_dataset = FileListMarsDataset(cfg.data_root, val_rows, cfg.val_split, cfg, False)
    elif cfg.use_file_list_dataset:
        train_dataset = build_labeled_folder_dataset(
            cfg.data_root, cfg.train_split, cfg, is_train=True
        )
        val_dataset = build_labeled_folder_dataset(
            cfg.data_root, cfg.val_split, cfg, is_train=False
        )
        logger.info(
            "Using file-list protocol with normalization=%s and input_channels=%s.",
            cfg.normalization,
            cfg.input_channels,
        )
    else:
        train_dataset = DatasetClass(
            root_dir=cfg.data_root, split=cfg.train_split, size=cfg.input_size
        )
        val_mask_dir = Path(cfg.data_root) / cfg.val_split / "masks"
        if cfg.val_split == "test" and val_mask_dir.exists():
            val_dataset = build_labeled_folder_dataset(
                cfg.data_root, cfg.val_split, cfg, is_train=False
            )
            logger.info("Using labeled %s split for validation.", cfg.val_split)
        else:
            val_dataset = DatasetClass(
                root_dir=cfg.data_root, split=cfg.val_split, size=cfg.input_size
            )
        if cfg.augmentation.lower() == "none" and hasattr(train_dataset, "augmentor"):
            train_dataset.augmentor = None

        if cfg.input_size != 128:
            train_dataset = ResizeDataset(train_dataset, cfg.input_size, F, torch)
            val_dataset = ResizeDataset(val_dataset, cfg.input_size, F, torch)

    if cfg.mosaic_prob > 0:
        train_dataset = MosaicCastDataset(train_dataset, prob=cfg.mosaic_prob, size=cfg.input_size)

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    return train_dataset, val_dataset


class ResizeDataset:
    def __init__(self, dataset, size, F, torch):
        self.dataset = dataset
        self.size = size
        self.F = F
        self.torch = torch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        if image.shape[-2:] == (self.size, self.size):
            return image, mask
        image = self.F.interpolate(
            image.unsqueeze(0),
            size=(self.size, self.size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mask = self.F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(self.size, self.size),
            mode="nearest",
        ).squeeze(0).squeeze(0).long()
        return image, mask


def build_model(cfg):
    if cfg.model_name and cfg.model_name.lower() not in {"auto", "custom"}:
        from networks import get_model

        # M3LSNet is a custom architecture that cannot be rebuilt through
        # ``plain_model``; get_model already forwards in_channels/num_classes.
        if cfg.model_name.lower().startswith("m3lsnet"):
            return get_model("m3lsnet", in_channels=cfg.in_channels, num_classes=cfg.num_classes)

        # Any other literal model_name is a legacy checkpoint identifier. It used
        # to fall through to ``plain_model`` and return None, which failed much
        # later with an unrelated error; fail loudly here instead.
        raise ValueError(
            "Unsupported literal model_name for paper configs: "
            f"{cfg.model_name!r}. Use one of the arch values handled by "
            "plain_model, 'm3lsnet', or 'custom' (default)."
        )

    import segmentation_models_pytorch as smp
    from networks.dual_model import UniversalDualWrapper

    arch = cfg.arch.lower()
    weights = "imagenet" if cfg.pretrain else None

    def plain_model(in_channels):
        if arch.endswith("segformer") or arch in {"segformer", "pmdfnet"}:
            return smp.Segformer(
                encoder_name=cfg.encoder,
                encoder_weights=weights,
                in_channels=in_channels,
                classes=cfg.num_classes,
            )
        if arch.endswith("upernet") or arch == "upernet":
            return smp.UPerNet(
                encoder_name=cfg.encoder,
                encoder_weights=weights,
                in_channels=in_channels,
                classes=cfg.num_classes,
            )
        # Must be matched before the ``endswith("unet")`` branch below, which
        # would otherwise also capture "unetplusplus".
        if arch in {"unetplusplus", "unet_plus_plus", "unetpp"}:
            return smp.UnetPlusPlus(
                encoder_name=cfg.encoder,
                encoder_weights=weights,
                in_channels=in_channels,
                classes=cfg.num_classes,
            )
        if arch == "unet" or arch.endswith("unet"):
            return smp.Unet(
                encoder_name=cfg.encoder,
                encoder_weights=weights,
                in_channels=in_channels,
                classes=cfg.num_classes,
            )
        if arch == "deeplabv3plus":
            return smp.DeepLabV3Plus(
                encoder_name=cfg.encoder,
                encoder_weights=weights,
                in_channels=in_channels,
                classes=cfg.num_classes,
            )
        if arch == "fpn":
            return smp.FPN(
                encoder_name=cfg.encoder,
                encoder_weights=weights,
                in_channels=in_channels,
                classes=cfg.num_classes,
            )
        if arch == "manet":
            return smp.MAnet(
                encoder_name=cfg.encoder,
                encoder_weights=weights,
                in_channels=in_channels,
                classes=cfg.num_classes,
            )
        if arch == "ocrnet":
            # networks.OCRNet.SMP_OCRNet returns ``(final, aux)`` logits while
            # training; forward_loss already applies the 0.4 aux-loss weighting.
            from networks.OCRNet import SMP_OCRNet

            return SMP_OCRNet(
                encoder_name=cfg.encoder,
                in_channels=in_channels,
                classes=cfg.num_classes,
            )
        raise ValueError(f"Unsupported arch: {cfg.arch}")

    # PMDFNet is the paper-facing name of the legacy dual-stream
    # implementation.  Keep ``dual_*`` as a backward-compatible alias for
    # historical configs and checkpoints.
    if arch in {"dual_segformer", "pmdfnet"} or arch.startswith(("dual_", "pmdfnet_")):
        ch1 = parse_channels(cfg.channels1)
        ch2 = parse_channels(cfg.channels2)
        main_model = plain_model(len(ch1))
        aux_model = plain_model(len(ch2))
        return UniversalDualWrapper(main_model, aux_model, ch1, ch2, fusion_type=cfg.fusion)

    return plain_model(cfg.in_channels)


def build_loss(cfg):
    import torch.nn as nn

    name = cfg.loss.lower()
    if name == "ce":
        from losses.utils_loss import DeterministicCrossEntropyLoss

        return DeterministicCrossEntropyLoss(ignore_index=cfg.ignore_index)
    if name == "combined":
        from losses import CombinedLoss

        return CombinedLoss(
            ce_weight=0.5,
            dice_weight=0.5,
            ignore_index=cfg.ignore_index,
        )
    if name == "dice":
        from losses.unetformer_loss import DiceLoss

        return DiceLoss(ignore_index=cfg.ignore_index)
    if name == "unetformer":
        from losses import UnetFormerLoss

        return UnetFormerLoss(ignore_index=cfg.ignore_index)
    raise ValueError(f"Unsupported loss: {cfg.loss}")


def build_optimizer(cfg, model):
    import torch

    name = cfg.optimizer.lower()
    params = model.parameters()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


def build_scheduler(cfg, optimizer):
    import torch

    name = cfg.scheduler.lower()
    if name in {"none", "off", ""}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.1)
    raise ValueError(f"Unsupported scheduler: {cfg.scheduler}")


def compute_metrics_from_confusion(conf):
    import numpy as np

    conf = conf.astype(np.float64)
    tp = np.diag(conf)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    tn = conf.sum() - (tp + fp + fn)
    iou = tp / np.maximum(tp + fp + fn, 1e-8)
    precision_fg = tp[1] / max(tp[1] + fp[1], 1e-8)
    recall_fg = tp[1] / max(tp[1] + fn[1], 1e-8)
    f1_fg = 2 * precision_fg * recall_fg / max(precision_fg + recall_fg, 1e-8)
    return {
        "miou": float(np.mean(iou)),
        "iou_bg": float(iou[0]),
        "iou_fg": float(iou[1]),
        "precision": float(precision_fg),
        "recall": float(recall_fg),
        "f1": float(f1_fg),
        "tn": float(tn[1]),
        "fp": float(fp[1]),
        "fn": float(fn[1]),
        "tp": float(tp[1]),
    }


def update_confusion(conf, preds, targets, num_classes, ignore_index):
    import torch

    preds = preds.view(-1)
    targets = targets.view(-1)
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]
    valid = (targets >= 0) & (targets < num_classes)
    preds = preds[valid]
    targets = targets[valid]
    encoded = targets * num_classes + preds
    bincount = torch.bincount(encoded, minlength=num_classes * num_classes)
    conf += bincount.view(num_classes, num_classes).cpu().numpy()


def forward_loss(model, criterion, data, target, mixed_precision, device_type):
    import torch

    with torch.autocast(device_type=device_type, enabled=mixed_precision):
        output = model(data)
        if isinstance(output, tuple):
            final_output, aux_output = output
            if hasattr(criterion, "main_loss") and hasattr(criterion, "aux_loss"):
                loss_main = criterion.main_loss(final_output, target)
                loss_aux = criterion.aux_loss(aux_output, target)
            else:
                loss_main = criterion(final_output, target)
                loss_aux = criterion(aux_output, target)
            loss = loss_main + 0.4 * loss_aux
            output = final_output
        else:
            if hasattr(criterion, "main_loss"):
                loss = criterion.main_loss(output, target)
            else:
                loss = criterion(output, target)
    return output, loss


def evaluate(model, loader, criterion, cfg, device, logger):
    import numpy as np
    import torch
    from tqdm import tqdm

    model.eval()
    total_loss = 0.0
    conf = np.zeros((cfg.num_classes, cfg.num_classes), dtype=np.int64)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Val", leave=False):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output, loss = forward_loss(
                model, criterion, data, target, cfg.mixed_precision, device_type
            )
            total_loss += float(loss.item())
            preds = torch.argmax(output, dim=1)
            update_confusion(conf, preds, target, cfg.num_classes, cfg.ignore_index)

    metrics = compute_metrics_from_confusion(conf)
    metrics["val_loss"] = total_loss / max(len(loader), 1)
    logger.info(
        "Val loss %.4f | mIoU %.4f | F1 %.4f | IoU_FG %.4f | IoU_BG %.4f",
        metrics["val_loss"],
        metrics["miou"],
        metrics["f1"],
        metrics["iou_fg"],
        metrics["iou_bg"],
    )
    return metrics


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, cfg):
    import torch

    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "config": asdict(cfg),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    import torch

    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return int(checkpoint.get("epoch", 0)) + 1, float(checkpoint.get("best_metric", 0.0))
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint.state_dict())
    return 0, 0.0


def append_metrics(path, row):
    exists = Path(path).exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


SUMMARY_FIELDS = [
    "completed_at",
    "status",
    "experiment_name",
    "run_dir",
    "data_root",
    "train_split",
    "val_split",
    "split_file",
    "fold",
    "seed",
    "model_name",
    "arch",
    "encoder",
    "pretrain",
    "channels1",
    "channels2",
    "fusion",
    "in_channels",
    "input_channels",
    "num_classes",
    "input_size",
    "batch_size",
    "max_train_samples",
    "max_val_samples",
    "epochs",
    "lr",
    "weight_decay",
    "optimizer",
    "scheduler",
    "loss",
    "augmentation",
    "aug_prob",
    "mosaic_prob",
    "primary_metric",
    "early_stopping_patience",
    "min_delta",
    "normalization",
    "use_file_list_dataset",
    "save_auxiliary_checkpoints",
    "finalize_results",
    "final_output_dir",
    "mixed_precision",
    "deterministic",
    "train_samples",
    "val_samples",
    "best_epoch",
    "best_miou",
    "best_iou_fg",
    "best_iou_bg",
    "best_f1",
    "best_val_loss",
    "best_iou_fg_epoch",
    "best_iou_fg_miou",
    "best_iou_fg_value",
    "best_iou_fg_f1",
    "best_f1_epoch",
    "best_f1_miou",
    "best_f1_iou_fg",
    "best_f1_value",
    "best_val_loss_epoch",
    "best_val_loss_miou",
    "best_val_loss_iou_fg",
    "best_val_loss_f1",
    "best_val_loss_value",
    "final_epoch",
    "final_miou",
    "final_iou_fg",
    "final_iou_bg",
    "final_f1",
    "final_val_loss",
    "final_train_loss",
    "best_checkpoint",
    "best_miou_checkpoint",
    "best_iou_fg_checkpoint",
    "best_f1_checkpoint",
    "best_val_loss_checkpoint",
    "last_checkpoint",
    "git_commit",
    "git_branch",
    "git_dirty",
]


def append_summary(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    normalized = {field: row.get(field, "") for field in SUMMARY_FIELDS}
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(normalized)


def make_summary_row(
    cfg,
    run_dir,
    status,
    train_samples,
    val_samples,
    best_row=None,
    final_row=None,
    metric_best_rows=None,
):
    git_info = get_git_info()
    best_row = best_row or {}
    final_row = final_row or {}
    metric_best_rows = metric_best_rows or {}
    best_iou_fg_row = metric_best_rows.get("iou_fg", {})
    best_f1_row = metric_best_rows.get("f1", {})
    best_val_loss_row = metric_best_rows.get("val_loss", {})
    return {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "experiment_name": cfg.experiment_name,
        "run_dir": str(run_dir),
        "data_root": cfg.data_root,
        "train_split": cfg.train_split,
        "val_split": cfg.val_split,
        "split_file": cfg.split_file,
        "fold": cfg.fold,
        "seed": cfg.seed,
        "model_name": cfg.model_name,
        "arch": cfg.arch,
        "encoder": cfg.encoder,
        "pretrain": cfg.pretrain,
        "channels1": cfg.channels1,
        "channels2": cfg.channels2,
        "fusion": cfg.fusion,
        "in_channels": cfg.in_channels,
        "input_channels": cfg.input_channels,
        "num_classes": cfg.num_classes,
        "input_size": cfg.input_size,
        "batch_size": cfg.batch_size,
        "max_train_samples": cfg.max_train_samples,
        "max_val_samples": cfg.max_val_samples,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "optimizer": cfg.optimizer,
        "scheduler": cfg.scheduler,
        "loss": cfg.loss,
        "augmentation": cfg.augmentation,
        "aug_prob": cfg.aug_prob,
        "mosaic_prob": cfg.mosaic_prob,
        "primary_metric": cfg.primary_metric,
        "early_stopping_patience": cfg.early_stopping_patience,
        "min_delta": cfg.min_delta,
        "normalization": cfg.normalization,
        "use_file_list_dataset": cfg.use_file_list_dataset,
        "save_auxiliary_checkpoints": cfg.save_auxiliary_checkpoints,
        "finalize_results": cfg.finalize_results,
        "final_output_dir": (
            cfg.final_output_dir
            if cfg.final_output_dir
            else str(Path(cfg.data_root) / "outputs")
            if cfg.finalize_results and cfg.data_root
            else ""
        ),
        "mixed_precision": cfg.mixed_precision,
        "deterministic": cfg.deterministic,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "best_epoch": best_row.get("epoch", ""),
        "best_miou": best_row.get("miou", best_row.get("best_miou", "")),
        "best_iou_fg": best_row.get("iou_fg", ""),
        "best_iou_bg": best_row.get("iou_bg", ""),
        "best_f1": best_row.get("f1", ""),
        "best_val_loss": best_row.get("val_loss", ""),
        "best_iou_fg_epoch": best_iou_fg_row.get("epoch", ""),
        "best_iou_fg_miou": best_iou_fg_row.get("miou", ""),
        "best_iou_fg_value": best_iou_fg_row.get("iou_fg", ""),
        "best_iou_fg_f1": best_iou_fg_row.get("f1", ""),
        "best_f1_epoch": best_f1_row.get("epoch", ""),
        "best_f1_miou": best_f1_row.get("miou", ""),
        "best_f1_iou_fg": best_f1_row.get("iou_fg", ""),
        "best_f1_value": best_f1_row.get("f1", ""),
        "best_val_loss_epoch": best_val_loss_row.get("epoch", ""),
        "best_val_loss_miou": best_val_loss_row.get("miou", ""),
        "best_val_loss_iou_fg": best_val_loss_row.get("iou_fg", ""),
        "best_val_loss_f1": best_val_loss_row.get("f1", ""),
        "best_val_loss_value": best_val_loss_row.get("val_loss", ""),
        "final_epoch": final_row.get("epoch", ""),
        "final_miou": final_row.get("miou", ""),
        "final_iou_fg": final_row.get("iou_fg", ""),
        "final_iou_bg": final_row.get("iou_bg", ""),
        "final_f1": final_row.get("f1", ""),
        "final_val_loss": final_row.get("val_loss", ""),
        "final_train_loss": final_row.get("train_loss", ""),
        "best_checkpoint": str(run_dir / "checkpoints" / "best.pth"),
        "best_miou_checkpoint": (
            str(run_dir / "checkpoints" / "best.pth")
            if cfg.primary_metric == "miou"
            else str(run_dir / "checkpoints" / "best_miou.pth")
            if cfg.save_auxiliary_checkpoints
            else ""
        ),
        "best_iou_fg_checkpoint": (
            str(run_dir / "checkpoints" / "best.pth")
            if cfg.primary_metric == "iou_fg"
            else str(run_dir / "checkpoints" / "best_iou_fg.pth")
            if cfg.save_auxiliary_checkpoints
            else ""
        ),
        "best_f1_checkpoint": (
            str(run_dir / "checkpoints" / "best.pth")
            if cfg.primary_metric == "f1"
            else str(run_dir / "checkpoints" / "best_f1.pth")
            if cfg.save_auxiliary_checkpoints
            else ""
        ),
        "best_val_loss_checkpoint": (
            str(run_dir / "checkpoints" / "best.pth")
            if cfg.primary_metric == "val_loss"
            else str(run_dir / "checkpoints" / "best_val_loss.pth")
            if cfg.save_auxiliary_checkpoints
            else ""
        ),
        "last_checkpoint": str(run_dir / "checkpoints" / "last.pth"),
        "git_commit": git_info["commit"],
        "git_branch": git_info["branch"],
        "git_dirty": git_info["dirty"],
    }


def metric_is_better(metric_name, score, best_score, min_delta):
    if metric_name == "val_loss":
        return score < best_score - min_delta
    return score > best_score + min_delta


def run_experiment(cfg):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm

    reproducibility = set_seed(cfg.seed, cfg.deterministic)
    run_dir = make_run_dir(cfg)
    save_run_metadata(cfg, run_dir)
    from reproducibility import runtime_metadata

    with open(run_dir / "reproducibility.json", "w", encoding="utf-8") as f:
        json.dump(reproducibility, f, indent=2, ensure_ascii=False)
    run_runtime = runtime_metadata()
    run_runtime.update(model_backend_metadata(cfg))
    with open(run_dir / "runtime.json", "w", encoding="utf-8") as f:
        json.dump(run_runtime, f, indent=2, ensure_ascii=False)
    logger = setup_logger(run_dir)
    logger.info("Run directory: %s", run_dir)

    train_dataset, val_dataset = build_datasets(cfg, logger)
    if cfg.max_train_samples > 0 and len(train_dataset) > cfg.max_train_samples:
        train_dataset = Subset(train_dataset, list(range(cfg.max_train_samples)))
        logger.info("Smoke/sample limit: train samples capped to %d", len(train_dataset))
    if cfg.max_val_samples > 0 and len(val_dataset) > cfg.max_val_samples:
        val_dataset = Subset(val_dataset, list(range(cfg.max_val_samples)))
        logger.info("Smoke/sample limit: val samples capped to %d", len(val_dataset))
    train_samples = len(train_dataset)
    val_samples = len(val_dataset)
    if cfg.dry_run:
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            logger.info("Dry run sample image shape: %s mask shape: %s", sample[0].shape, sample[1].shape)
        else:
            logger.warning("Dry run found empty train dataset.")
        if cfg.skip_model_init:
            logger.info("Skipped model initialization.")
            if cfg.summary_csv:
                append_summary(
                    cfg.summary_csv,
                    make_summary_row(cfg, run_dir, "dry_run", train_samples, val_samples),
                )
            return run_dir

    if cfg.skip_model_init:
        logger.info("Skipped model initialization.")
        if cfg.summary_csv:
            append_summary(
                cfg.summary_csv,
                make_summary_row(cfg, run_dir, "skipped_model_init", train_samples, val_samples),
            )
        return run_dir

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    logger.info("Using device: %s", device)

    model = build_model(cfg).to(device)
    backend_metadata = model_backend_metadata(cfg)
    if backend_metadata.get("m3lsnet_backend"):
        logger.info("M3LSNet backend: %s", backend_metadata["m3lsnet_backend"])
    criterion = build_loss(cfg).to(device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    if cfg.init_checkpoint:
        logger.info("Loading initial checkpoint: %s", cfg.init_checkpoint)
        load_checkpoint(cfg.init_checkpoint, model, map_location=device)

    start_epoch = 0
    tracked_metrics = ["miou", "iou_fg", "f1", "val_loss"]
    best_scores = {
        "miou": float("-inf"),
        "iou_fg": float("-inf"),
        "f1": float("-inf"),
        "val_loss": float("inf"),
    }
    metric_best_rows = {}
    if cfg.resume:
        logger.info("Resuming checkpoint: %s", cfg.resume)
        start_epoch, loaded_best_metric = load_checkpoint(
            cfg.resume, model, optimizer, scheduler, map_location=device
        )
        best_scores[cfg.primary_metric] = loaded_best_metric

    gt = torch.Generator()
    gt.manual_seed(cfg.seed)
    gv = torch.Generator()
    gv.manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=gt,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=gv,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision and device.type == "cuda")
    metrics_path = run_dir / "metrics.csv"
    best_row = {}
    final_row = {}
    epochs_without_improvement = 0

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs} Train")
        for data, target in pbar:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output, loss = forward_loss(
                model, criterion, data, target, cfg.mixed_precision, device_type
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item())
            pbar.set_postfix(loss=f"{float(loss.item()):.4f}")

        avg_train_loss = train_loss / max(len(train_loader), 1)
        logger.info("Epoch %d train loss %.4f", epoch + 1, avg_train_loss)

        val_metrics = {}
        if len(val_dataset) > 0 and (epoch + 1) % cfg.val_interval == 0:
            val_metrics = evaluate(model, val_loader, criterion, cfg, device, logger)

        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        row.update(val_metrics)

        primary_improved = False
        if val_metrics:
            for metric_name in tracked_metrics:
                score = float(val_metrics[metric_name])
                if metric_is_better(metric_name, score, best_scores[metric_name], cfg.min_delta):
                    best_scores[metric_name] = score
                    metric_best_rows[metric_name] = dict(row)
                    if cfg.save_auxiliary_checkpoints:
                        save_checkpoint(
                            run_dir / "checkpoints" / f"best_{metric_name}.pth",
                            model,
                            optimizer,
                            scheduler,
                            epoch,
                            score,
                            cfg,
                        )
                        logger.info("Saved best_%s checkpoint: %.4f", metric_name, score)
                    if metric_name == cfg.primary_metric:
                        primary_improved = True
                        save_checkpoint(
                            run_dir / "checkpoints" / "best.pth",
                            model,
                            optimizer,
                            scheduler,
                            epoch,
                            score,
                            cfg,
                        )
                        logger.info("Updated primary best checkpoint (%s): %.4f", cfg.primary_metric, score)

            if primary_improved:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        row["best_miou"] = "" if best_scores["miou"] == float("-inf") else best_scores["miou"]
        row["best_iou_fg_so_far"] = "" if best_scores["iou_fg"] == float("-inf") else best_scores["iou_fg"]
        row["best_f1_so_far"] = "" if best_scores["f1"] == float("-inf") else best_scores["f1"]
        row["best_val_loss_so_far"] = "" if best_scores["val_loss"] == float("inf") else best_scores["val_loss"]
        append_metrics(metrics_path, row)
        final_row = row
        best_row = metric_best_rows.get("miou", best_row)

        primary_best = best_scores[cfg.primary_metric]
        checkpoint_metric = 0.0 if primary_best in {float("-inf"), float("inf")} else primary_best
        if (epoch + 1) % cfg.save_interval == 0 or epoch + 1 == cfg.epochs:
            save_checkpoint(
                run_dir / "checkpoints" / "last.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                checkpoint_metric,
                cfg,
            )

        if (
            cfg.early_stopping_patience > 0
            and val_metrics
            and epochs_without_improvement >= cfg.early_stopping_patience
        ):
            logger.info(
                "Early stopping at epoch %d: no %s improvement for %d validations.",
                epoch + 1,
                cfg.primary_metric,
                epochs_without_improvement,
            )
            break

    best_miou = best_scores["miou"]
    if best_miou == float("-inf"):
        best_miou = 0.0
    logger.info("Training completed. Best mIoU: %.4f", best_miou)
    if cfg.summary_csv:
        append_summary(
            cfg.summary_csv,
            make_summary_row(
                cfg,
                run_dir,
                "completed",
                train_samples,
                val_samples,
                best_row,
                final_row,
                metric_best_rows,
            ),
        )
        logger.info("Appended run summary: %s", cfg.summary_csv)
    final_dir = finalize_run_results(cfg, run_dir)
    if final_dir:
        logger.info("Copied completed run to dataset outputs: %s", final_dir)
    return run_dir


def parse_single_config(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="")
    pre_parser.add_argument("--batch-config", default="")
    pre_args, _ = pre_parser.parse_known_args(argv)

    defaults = dict(DEFAULTS)
    if pre_args.config:
        defaults.update(load_json(pre_args.config))
    parser = build_parser(defaults)
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_single_config(argv)
    if args.batch_config:
        batch = load_json(args.batch_config)
        common = batch.get("common", {})
        experiments = batch.get("experiments", [])
        if not experiments:
            raise ValueError("batch-config must contain a non-empty 'experiments' list")
        run_dirs = []
        for exp in experiments:
            merged = merge_config(common, exp)
            cfg = RunConfig(**{k: merged[k] for k in DEFAULTS.keys()})
            run_dirs.append(run_experiment(cfg))
        print("Completed runs:")
        for run_dir in run_dirs:
            print(run_dir)
        return

    cfg = namespace_to_config(args)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
