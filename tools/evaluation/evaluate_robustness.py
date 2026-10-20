"""Evaluate frozen checkpoints under five controlled Test-set conditions.

The protocol is deliberately paired: every model sees the same sorted Test
image list and, for stochastic noise, the same deterministic noise realization
for each image/condition/level.  Results include per-image confusion counts so
that image-level bootstrap, paired tests, FP/FN summaries and qualitative
failure-case selection can be performed without rerunning inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
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
from tools.evaluation.evaluate_test_metrics import (  # noqa: E402
    confusion_counts,
    load_checkpoint,
    mean_metric_dict,
    metrics_from_counts,
    build_evaluation_model,
    sample_standard_deviation,
)
from tools.evaluation.measure_model_complexity import _as_first_tensor, _m3lsnet_backend  # noqa: E402
from tools.evaluation.paper_run_utils import (  # noqa: E402
    load_protocol,
    parse_id_list,
    parse_seed_list,
    resolve_formal_run,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate formal checkpoints on clean Test plus five controlled conditions."
    )
    parser.add_argument("--config", default="configs/mmlsv2_paper_protocol.json")
    parser.add_argument(
        "--robustness-config",
        default="configs/mmlsv2_robustness_protocol.json",
    )
    parser.add_argument("--run-root", default="outputs_experiments/pmdfnet_mmlsv2")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", ""))
    parser.add_argument("--ids", default="M4,L1", help="Paper/protocol ids; default compares M4 and L1/M5.")
    parser.add_argument("--seeds", default="41 42 43")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "4")))
    parser.add_argument(
        "--output-dir",
        default="outputs_experiments/robustness",
        help="Directory for summary JSON/CSV and per-image JSONL.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Record missing runs/errors and continue; default is fail-fast.",
    )
    return parser.parse_args(argv)


class PerturbedTestDataset(Dataset):
    """Public Test images with a deterministic condition applied in test space."""

    def __init__(
        self,
        root_dir: str | Path,
        condition: dict,
        condition_index: int,
        random_seed: int,
        input_channels: list[int] | None = None,
    ):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.images_dir = self.root_dir / "test" / "images"
        self.masks_dir = self.root_dir / "test" / "masks"
        if not self.images_dir.is_dir() or not self.masks_dir.is_dir():
            raise FileNotFoundError(
                f"Expected mapped public Test images/masks under {self.root_dir / 'test'}"
            )
        self.image_files = sorted(self.images_dir.glob("*.tif"))
        self.mask_by_stem = {path.stem: path for path in self.masks_dir.glob("*.tif")}
        missing = [path.name for path in self.image_files if path.stem not in self.mask_by_stem]
        if missing:
            raise FileNotFoundError(f"Missing masks for {len(missing)} Test images; example: {missing[0]}")
        if not self.image_files:
            raise RuntimeError(f"No Test images found in {self.images_dir}")
        self.condition = condition
        self.condition_index = int(condition_index)
        self.random_seed = int(random_seed)
        self.input_channels = list(range(7)) if input_channels is None else list(input_channels)
        invalid = [channel for channel in self.input_channels if channel < 0 or channel >= 7]
        if not self.input_channels or invalid:
            raise ValueError(f"Invalid input_channels: {self.input_channels}")
        self.mean = np.asarray(MARS_MEAN_TEST, dtype=np.float32).reshape(7, 1, 1)
        self.std = np.asarray(MARS_STD_TEST, dtype=np.float32).reshape(7, 1, 1)

    def __len__(self):
        return len(self.image_files)

    def _apply_condition(self, image: np.ndarray, index: int) -> np.ndarray:
        kind = self.condition.get("kind", "clean")
        if kind == "clean":
            return image
        channels = [int(channel) for channel in self.condition.get("channels", [])]
        value = float(self.condition.get("value", 0.0))
        if not channels:
            raise ValueError(f"Condition {self.condition.get('id')} has no channels")
        output = image.copy()
        if kind == "additive_offset":
            output[channels] += value
        elif kind == "gaussian_noise":
            # The condition/level/image combination is deterministic and shared
            # across all evaluated models, which makes comparisons paired.
            seed = (
                self.random_seed
                + self.condition_index * 1_000_003
                + int(self.condition.get("level_index", 0)) * 10_007
                + index
            )
            rng = np.random.default_rng(seed)
            noise = rng.normal(0.0, value, size=output[channels].shape).astype(np.float32)
            output[channels] += noise
        elif kind == "zero_channels":
            # value=1 means full removal; value=0 leaves the channel intact.
            output[channels] *= max(0.0, 1.0 - value)
        else:
            raise ValueError(f"Unsupported robustness condition kind: {kind}")
        return output

    def __getitem__(self, index: int):
        image_path = self.image_files[index]
        mask_path = self.mask_by_stem[image_path.stem]
        image = tifffile.imread(str(image_path)).astype(np.float32)
        if image.ndim == 3 and image.shape[-1] == 7:
            image = image.transpose(2, 0, 1)
        if image.ndim != 3 or image.shape[0] != 7:
            raise ValueError(f"Expected 7-channel image, got {image.shape}: {image_path}")
        image[image < -100000] = 0.0
        image = (image - self.mean) / (self.std + 1e-8)
        image = self._apply_condition(image, index)
        image = image[self.input_channels]

        mask = np.squeeze(tifffile.imread(str(mask_path)).astype(np.int64))
        if mask.ndim != 2 or tuple(mask.shape) != tuple(image.shape[-2:]):
            raise ValueError(f"Image/mask shape mismatch for {image_path.name}: {image.shape} / {mask.shape}")
        mask = (mask > 0).astype(np.int64)
        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(mask).long(),
            image_path.name,
        )


def condition_records(protocol: dict) -> list[dict]:
    records = [
        {
            "id": "clean",
            "kind": "clean",
            "channels": [],
            "units": "none",
            "level_name": "clean",
            "level_index": 0,
            "value": 0.0,
        }
    ]
    for condition_index, condition in enumerate(protocol.get("conditions", []), start=1):
        for level_index, level in enumerate(condition.get("levels", [])):
            item = dict(condition)
            item.update(
                {
                    "condition_index": condition_index,
                    "level_index": level_index,
                    "level_name": level["name"],
                    "value": float(level["value"]),
                }
            )
            records.append(item)
    return records


def evaluate_condition(model, dataset, device, batch_size: int, num_workers: int) -> tuple[dict, list[dict]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    total_tp = total_fp = total_fn = total_tn = 0
    batch_metrics = []
    per_image = []
    with torch.inference_mode():
        for images, targets, names in tqdm(loader, desc=dataset.condition["id"], unit="batch", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            predictions = torch.argmax(_as_first_tensor(model(images)), dim=1)
            batch_counts = confusion_counts(predictions, targets)
            total_tp += batch_counts[0]
            total_fp += batch_counts[1]
            total_fn += batch_counts[2]
            total_tn += batch_counts[3]
            batch_metrics.append(metrics_from_counts(*batch_counts))
            for index, name in enumerate(names):
                counts = confusion_counts(predictions[index], targets[index])
                metrics = metrics_from_counts(*counts)
                per_image.append(
                    {
                        "image": str(name),
                        "tp": counts[0],
                        "fp": counts[1],
                        "fn": counts[2],
                        "tn": counts[3],
                        "predicted_positive_pixels": counts[0] + counts[1],
                        "target_positive_pixels": counts[0] + counts[2],
                        "error_pixels": counts[1] + counts[2],
                        **metrics,
                    }
                )
    global_counts = (total_tp, total_fp, total_fn, total_tn)
    summary = {
        "num_samples": len(dataset),
        "num_batches": len(batch_metrics),
        "global_counts": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "tn": total_tn,
        },
        "metrics": metrics_from_counts(*global_counts),
        "batch_average_metrics": mean_metric_dict(batch_metrics),
        "aggregation": "global pixel metrics are primary; batch average is auxiliary",
    }
    return summary, per_image


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def across_seed_summary(summary_records: list[dict]) -> list[dict]:
    """Aggregate robustness mIoU and auxiliary metrics across training seeds.

    Each row is a fixed model/condition/level evaluated with three seeds.  The
    mean is descriptive and the reported across-seed SD is the sample SD.
    """
    grouped = defaultdict(list)
    for record in summary_records:
        if record.get("status") != "completed":
            continue
        key = (
            record["paper_id"],
            record["condition_id"],
            record["level_name"],
            record["level_value"],
        )
        grouped[key].append(record)

    rows = []
    for (paper_id, condition_id, level_name, level_value), records in sorted(grouped.items()):
        row = {
            "paper_id": paper_id,
            "condition_id": condition_id,
            "level_name": level_name,
            "level_value": level_value,
            "n_seeds": len(records),
            "seeds": ",".join(str(item["seed"]) for item in sorted(records, key=lambda item: item["seed"])),
        }
        metric_keys = sorted(records[0]["metrics"])
        for metric in metric_keys:
            values = [float(item["metrics"][metric]) for item in records]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = sample_standard_deviation(values)
        rows.append(row)
    return rows


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.data_root:
        raise ValueError("Provide --data-root or set DATA_ROOT/MMLSV2_DATA_ROOT")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers cannot be negative")

    paper_protocol_path = Path(args.config).expanduser().resolve()
    robustness_path = Path(args.robustness_config).expanduser().resolve()
    paper_protocol = load_protocol(paper_protocol_path)
    robustness_protocol = json.loads(robustness_path.read_text(encoding="utf-8"))
    ids = parse_id_list(args.ids, tuple(robustness_protocol.get("default_model_ids", ["M4", "L1"])))
    seeds = parse_seed_list(
        args.seeds,
        tuple(int(seed) for seed in robustness_protocol.get("default_seeds", [41, 42, 43])),
    )
    conditions = condition_records(robustness_protocol)
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device} but CUDA is unavailable")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = output_dir / "robustness_per_image.jsonl"
    summary_records = []
    with per_image_path.open("w", encoding="utf-8") as per_image_handle:
        for experiment_id in ids:
            for seed in seeds:
                run = resolve_formal_run(args.run_root, paper_protocol, experiment_id, seed)
                print(f"[{experiment_id}/s{seed}] loading {run['checkpoint_path']}", flush=True)
                try:
                    model_args = argparse.Namespace(run_config=run["config_path"], model_name="")
                    model, model_name, config_path, input_channels = build_evaluation_model(model_args)
                    model = model.to(device)
                    load_checkpoint(model, Path(run["checkpoint_path"]), device)
                    model.eval()
                    config_payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
                    model_metadata = {
                        "model_name": model_name,
                        "run_config": config_path,
                        "input_channels": input_channels,
                        "device": str(device),
                        **_m3lsnet_backend(config_payload.get("model_name", model_name)),
                    }
                    for condition in conditions:
                        dataset_condition = dict(condition)
                        dataset = PerturbedTestDataset(
                            args.data_root,
                            dataset_condition,
                            condition_index=int(condition.get("condition_index", 0)),
                            random_seed=int(robustness_protocol.get("random_seed", 0)),
                            input_channels=input_channels,
                        )
                        metrics, per_image = evaluate_condition(
                            model, dataset, device, args.batch_size, args.num_workers
                        )
                        per_image_key = {
                            "paper_id": experiment_id,
                            "stored_id": run["stored_id"],
                            "seed": seed,
                            "condition_id": condition["id"],
                            "level_name": condition["level_name"],
                            "level_value": condition["value"],
                            "rows": per_image,
                        }
                        per_image_handle.write(
                            json.dumps(per_image_key, ensure_ascii=False) + "\n"
                        )
                        summary_records.append(
                            {
                                "status": "completed",
                                "paper_id": experiment_id,
                                "stored_id": run["stored_id"],
                                "seed": seed,
                                "experiment_name": run["experiment_name"],
                                "run_dir": run["run_dir"],
                                "checkpoint": run["checkpoint_path"],
                                "condition_id": condition["id"],
                                "condition_kind": condition.get("kind", "clean"),
                                "channels": condition.get("channels", []),
                                "units": condition.get("units", ""),
                                "level_name": condition["level_name"],
                                "level_value": condition["value"],
                                **model_metadata,
                                **metrics,
                            }
                        )
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception as exc:
                    if not args.allow_missing:
                        raise
                    summary_records.append(
                        {
                            "status": "failed",
                            "paper_id": experiment_id,
                            "seed": seed,
                            "run_dir": run["run_dir"],
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    print(f"[{experiment_id}/s{seed}] failed: {exc}", file=sys.stderr, flush=True)

    payload = {
        "status": "completed" if summary_records and all(item["status"] == "completed" for item in summary_records) else "partial",
        "evaluation_type": "paired_robustness_test_metrics",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_protocol": str(paper_protocol_path),
        "robustness_protocol": str(robustness_path),
        "robustness_protocol_sha256": sha256(robustness_path),
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "ids": ids,
        "seeds": seeds,
        "conditions_including_clean": len(conditions),
        "per_image_metrics": str(per_image_path),
        "records": summary_records,
        "across_seed_summary": across_seed_summary(summary_records),
        "statistical_convention": {
            "across_seed": "sample standard deviation (ddof=1) over seed-level estimates",
            "per_image": "per-image metrics are retained for paired bootstrap and sign-flip analyses",
        },
        "material_passport": {
            "verification_status": "EXECUTED",
            "pairing": "same sorted Test image list and deterministic noise realization across models",
            "perturbation_space": "after fixed MARS test normalization, before input-channel selection",
            "clean_reference": "included for every model/seed",
        },
    }
    (output_dir / "robustness_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fieldnames = sorted({key for record in summary_records for key in record})
    with (output_dir / "robustness_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_records)
    across_rows = across_seed_summary(summary_records)
    across_fieldnames = sorted({key for record in across_rows for key in record})
    with (output_dir / "robustness_across_seed.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=across_fieldnames)
        writer.writeheader()
        writer.writerows(across_rows)
    failed = [item for item in summary_records if item["status"] != "completed"]
    print(
        f"Wrote {len(summary_records)} condition records and per-image metrics to {output_dir}; "
        f"incomplete={len(failed)}"
    )
    return 1 if failed and not args.allow_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

