"""Compute frozen-Test boundary, per-tile, area-stratified and uncertainty diagnostics.

This evaluator is intentionally separate from the canonical ``test_metrics.json``
writer.  It does not change checkpoint selection or the primary paper metrics;
it performs one deterministic inference pass over the public labeled Test split
and writes auxiliary diagnostics for already-frozen runs.

The reported diagnostics are:

* boundary precision/recall/F1/IoU with a 3x3 contour and configurable pixel
  tolerance (the paper protocol uses two pixels);
* per-tile foreground IoU/F1/error and boundary distributions;
* deterministic small/medium/large foreground-area strata, each containing
  approximately one third of the Test tiles;
* binary predictive entropy in nats, including entropy on misclassified pixels,
  and Pearson correlation with the per-tile error rate.

The primary pixel metrics remain pooled over all Test pixels.  Per-tile means
and area-stratified means are descriptive and are reported separately so they
cannot be confused with the canonical global metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.evaluate_test_metrics import (  # noqa: E402
    LabeledTestDataset,
    build_evaluation_model,
    load_checkpoint,
    metrics_from_counts,
)
from tools.evaluation.paper_run_utils import (  # noqa: E402
    load_protocol,
    parse_id_list,
    parse_seed_list,
    resolve_formal_run,
)


DEFAULT_DIAGNOSTIC_IDS = ("M4", "L1")
DEFAULT_DIAGNOSTIC_SEEDS = (41, 42, 43)
PER_TILE_METRICS = (
    "miou",
    "iou_fg",
    "f1",
    "boundary_f1",
    "boundary_iou",
    "error_rate",
    "mean_entropy",
    "mean_error_entropy",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen checkpoints on public Test and compute boundary, "
            "per-tile, foreground-area and uncertainty diagnostics."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-root", default="outputs_experiments/pmdfnet_mmlsv2")
    parser.add_argument("--config", default="configs/mmlsv2_paper_protocol.json")
    parser.add_argument("--ids", default="M4,L1", help="Comma-separated protocol IDs.")
    parser.add_argument(
        "--seeds",
        default="41 42 43",
        help="Whitespace- or comma-separated training seeds.",
    )
    parser.add_argument("--output-dir", default="outputs_experiments/test_diagnostics")
    parser.add_argument("--device", default=os.environ.get("TRAIN_DEVICE", "cuda:0"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--boundary-tolerance",
        type=int,
        default=2,
        help="Pixel tolerance for matching predicted and GT contours.",
    )
    parser.add_argument(
        "--area-groups",
        type=int,
        default=3,
        help="Number of deterministic foreground-area strata (default: 3).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write tabular/JSON outputs without rendering matplotlib figures.",
    )
    return parser.parse_args(argv)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_divide(numerator: float, denominator: float, empty_value: float = 0.0) -> float:
    if denominator == 0:
        return float(empty_value)
    return float(numerator / denominator)


def binary_boundary(mask: torch.Tensor) -> torch.Tensor:
    """Return a one-pixel contour using dilate(mask) - erode(mask).

    ``mask`` is a two-dimensional boolean tensor.  Pooling keeps this helper
    independent of OpenCV/scipy and exactly matches a 3x3 binary morphology
    operation for the interior pixels.
    """

    values = mask.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    dilated = F.max_pool2d(values, kernel_size=3, stride=1, padding=1) >= 0.5
    eroded = -F.max_pool2d(-values, kernel_size=3, stride=1, padding=1) >= 0.5
    return (dilated & ~eroded).squeeze(0).squeeze(0)


def binary_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius < 0:
        raise ValueError("dilation radius must be non-negative")
    if radius == 0:
        return mask.bool()
    values = mask.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel = 2 * radius + 1
    return (
        F.max_pool2d(values, kernel_size=kernel, stride=1, padding=radius) >= 0.5
    ).squeeze(0).squeeze(0)


def boundary_statistics(
    pred: torch.Tensor,
    target: torch.Tensor,
    tolerance: int,
) -> tuple[dict[str, float | int], dict[str, int]]:
    pred_boundary = binary_boundary(pred)
    target_boundary = binary_boundary(target)
    pred_count = int(pred_boundary.sum().item())
    target_count = int(target_boundary.sum().item())

    pred_matched = pred_boundary & binary_dilate(target_boundary, tolerance)
    target_matched = target_boundary & binary_dilate(pred_boundary, tolerance)
    matched_pred_count = int(pred_matched.sum().item())
    matched_target_count = int(target_matched.sum().item())

    pred_band = binary_dilate(pred_boundary, tolerance)
    target_band = binary_dilate(target_boundary, tolerance)
    band_intersection = int((pred_band & target_band).sum().item())
    band_union = int((pred_band | target_band).sum().item())

    precision = safe_divide(matched_pred_count, pred_count, empty_value=1.0 if target_count == 0 else 0.0)
    recall = safe_divide(matched_target_count, target_count, empty_value=1.0 if pred_count == 0 else 0.0)
    f1 = safe_divide(2.0 * precision * recall, precision + recall, empty_value=1.0 if pred_count == 0 and target_count == 0 else 0.0)
    iou = safe_divide(band_intersection, band_union, empty_value=1.0 if pred_count == 0 and target_count == 0 else 0.0)

    metrics = {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "boundary_iou": iou,
    }
    counts = {
        "boundary_pred_pixels": pred_count,
        "boundary_gt_pixels": target_count,
        "boundary_matched_pred_pixels": matched_pred_count,
        "boundary_matched_gt_pixels": matched_target_count,
        "boundary_band_intersection": band_intersection,
        "boundary_band_union": band_union,
    }
    return metrics, counts


def extract_logits(output: object) -> torch.Tensor:
    if isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("Model returned an empty sequence")
        return extract_logits(output[0])
    if isinstance(output, dict):
        for key in ("out", "logits", "masks", "pred"):
            if key in output:
                return extract_logits(output[key])
        for value in output.values():
            if torch.is_tensor(value):
                return value
        raise TypeError("Model output dictionary contains no tensor")
    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported model output type: {type(output)}")
    if output.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W] logits, got {tuple(output.shape)}")
    return output


def probabilities_and_predictions(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.shape[1] == 1:
        probability = torch.sigmoid(logits[:, 0])
        prediction = (probability >= 0.5).long()
    elif logits.shape[1] >= 2:
        probability = torch.softmax(logits, dim=1)[:, 1]
        prediction = torch.argmax(logits, dim=1).long()
    else:
        raise ValueError(f"Logits have no class channel: {tuple(logits.shape)}")
    return probability, prediction


def entropy_from_probability(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.clamp(1e-7, 1.0 - 1e-7)
    return -(
        probability * probability.log()
        + (1.0 - probability) * (1.0 - probability).log()
    )


def per_tile_record(
    image_name: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    probability: torch.Tensor,
    boundary_tolerance: int,
) -> tuple[dict, dict[str, int]]:
    prediction = prediction.bool()
    target = target.bool()
    tp = int((prediction & target).sum().item())
    fp = int((prediction & ~target).sum().item())
    fn = int((~prediction & target).sum().item())
    tn = int((~prediction & ~target).sum().item())
    metrics = metrics_from_counts(tp, fp, fn, tn)
    boundary_metrics, boundary_counts = boundary_statistics(
        prediction, target, boundary_tolerance
    )

    entropy = entropy_from_probability(probability)
    error = prediction != target
    error_count = int(error.sum().item())
    entropy_values = entropy.detach().float()
    error_entropy_values = entropy_values[error]
    mean_error_entropy = (
        float(error_entropy_values.mean().item()) if error_count else float("nan")
    )
    error_entropy_p95 = (
        float(torch.quantile(error_entropy_values, 0.95).item())
        if error_count
        else float("nan")
    )

    height, width = target.shape[-2:]
    pixel_count = int(target.numel())
    gt_foreground_pixels = int(target.sum().item())
    pred_foreground_pixels = int(prediction.sum().item())
    row = {
        "image": image_name,
        "height": int(height),
        "width": int(width),
        "pixel_count": pixel_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "error_pixels": error_count,
        "error_rate": safe_divide(error_count, pixel_count),
        "gt_foreground_pixels": gt_foreground_pixels,
        "pred_foreground_pixels": pred_foreground_pixels,
        "gt_foreground_ratio": safe_divide(gt_foreground_pixels, pixel_count),
        "pred_foreground_ratio": safe_divide(pred_foreground_pixels, pixel_count),
        "mean_entropy": float(entropy_values.mean().item()),
        "mean_error_entropy": mean_error_entropy,
        "error_entropy_p95": error_entropy_p95,
        "max_entropy": float(entropy_values.max().item()),
        **metrics,
        **boundary_metrics,
        **boundary_counts,
    }
    return row, boundary_counts


def assign_area_groups(rows: list[dict], group_count: int) -> dict[str, dict]:
    if group_count < 2:
        raise ValueError("area-groups must be at least 2")
    ordered = sorted(
        range(len(rows)),
        key=lambda index: (float(rows[index]["gt_foreground_ratio"]), rows[index]["image"]),
    )
    labels = ["small", "medium", "large"] if group_count == 3 else [f"q{index + 1}" for index in range(group_count)]
    chunks = np.array_split(np.asarray(ordered, dtype=np.int64), group_count)
    group_info = {}
    for group_index, chunk in enumerate(chunks):
        label = labels[group_index]
        ratios = []
        for row_index in chunk.tolist():
            rows[row_index]["area_group"] = label
            ratios.append(float(rows[row_index]["gt_foreground_ratio"]))
        group_info[label] = {
            "n_tiles": len(ratios),
            "min_gt_foreground_ratio": min(ratios) if ratios else None,
            "max_gt_foreground_ratio": max(ratios) if ratios else None,
            "mean_gt_foreground_ratio": float(np.mean(ratios)) if ratios else None,
            "selection": "rank tiles by GT foreground ratio, tie-break by image filename, then np.array_split",
        }
    return group_info


def mean_and_std(rows: list[dict], metric: str) -> dict[str, float | None]:
    values = np.asarray(
        [float(row[metric]) for row in rows if math.isfinite(float(row[metric]))],
        dtype=np.float64,
    )
    if values.size == 0:
        return {"mean": None, "std": None, "median": None, "q25": None, "q75": None}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
    }


def pooled_metrics(rows: list[dict]) -> dict:
    counts = {key: int(sum(int(row[key]) for row in rows)) for key in ("tp", "fp", "fn", "tn")}
    return {**counts, **metrics_from_counts(**counts)}


def pooled_boundary_metrics(rows: list[dict]) -> dict:
    counts = {
        key: int(sum(int(row[key]) for row in rows))
        for key in (
            "boundary_pred_pixels",
            "boundary_gt_pixels",
            "boundary_matched_pred_pixels",
            "boundary_matched_gt_pixels",
            "boundary_band_intersection",
            "boundary_band_union",
        )
    }
    precision = safe_divide(
        counts["boundary_matched_pred_pixels"],
        counts["boundary_pred_pixels"],
        empty_value=1.0 if counts["boundary_gt_pixels"] == 0 else 0.0,
    )
    recall = safe_divide(
        counts["boundary_matched_gt_pixels"],
        counts["boundary_gt_pixels"],
        empty_value=1.0 if counts["boundary_pred_pixels"] == 0 else 0.0,
    )
    return {
        **counts,
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": safe_divide(
            2.0 * precision * recall,
            precision + recall,
            empty_value=1.0
            if counts["boundary_pred_pixels"] == 0 and counts["boundary_gt_pixels"] == 0
            else 0.0,
        ),
        "boundary_iou": safe_divide(
            counts["boundary_band_intersection"],
            counts["boundary_band_union"],
            empty_value=1.0
            if counts["boundary_pred_pixels"] == 0 and counts["boundary_gt_pixels"] == 0
            else 0.0,
        ),
    }


def correlation(rows: list[dict], x_key: str, y_key: str) -> dict[str, float | int | None]:
    values = np.asarray(
        [
            [float(row[x_key]), float(row[y_key])]
            for row in rows
            if math.isfinite(float(row[x_key])) and math.isfinite(float(row[y_key]))
        ],
        dtype=np.float64,
    )
    if values.shape[0] < 2 or np.std(values[:, 0]) == 0 or np.std(values[:, 1]) == 0:
        coefficient = None
    else:
        coefficient = float(np.corrcoef(values[:, 0], values[:, 1])[0, 1])
    return {"pearson_r": coefficient, "n_tiles": int(values.shape[0])}


def summarize_area_groups(rows: list[dict], group_info: dict[str, dict]) -> list[dict]:
    output = []
    for label, info in group_info.items():
        group_rows = [row for row in rows if row.get("area_group") == label]
        record = {
            "area_group": label,
            "n_tiles": len(group_rows),
            "min_gt_foreground_ratio": info["min_gt_foreground_ratio"],
            "max_gt_foreground_ratio": info["max_gt_foreground_ratio"],
            "mean_gt_foreground_ratio": info["mean_gt_foreground_ratio"],
        }
        for metric in PER_TILE_METRICS:
            stats = mean_and_std(group_rows, metric)
            record[f"{metric}_mean"] = stats["mean"]
            record[f"{metric}_std"] = stats["std"]
        record.update({f"global_{key}": value for key, value in pooled_metrics(group_rows).items()})
        record.update(
            {f"global_{key}": value for key, value in pooled_boundary_metrics(group_rows).items()}
        )
        output.append(record)
    return output


def summarize_run(
    model_id: str,
    stored_id: str,
    seed: int,
    run: dict,
    rows: list[dict],
    group_info: dict[str, dict],
    boundary_tolerance: int,
) -> dict:
    per_tile = {}
    for metric in PER_TILE_METRICS:
        per_tile[metric] = mean_and_std(rows, metric)
    uncertainty = {
        "mean_error_entropy_vs_error_rate": correlation(
            rows, "mean_error_entropy", "error_rate"
        ),
        "mean_entropy_vs_error_rate": correlation(rows, "mean_entropy", "error_rate"),
    }
    return {
        "model_id": model_id,
        "stored_id": stored_id,
        "seed": int(seed),
        "run_dir": run["run_dir"],
        "checkpoint": run["checkpoint_path"],
        "n_tiles": len(rows),
        "boundary_tolerance_pixels": boundary_tolerance,
        "global_pixel_metrics": pooled_metrics(rows),
        "global_boundary_metrics": pooled_boundary_metrics(rows),
        "per_tile_distribution": per_tile,
        "area_group_definition": group_info,
        "area_stratified": summarize_area_groups(rows, group_info),
        "uncertainty": uncertainty,
        "aggregation": {
            "global_pixel_metrics": "TP/FP/FN/TN pooled over all Test pixels",
            "global_boundary_metrics": "contour counts pooled over all Test tiles",
            "per_tile_distribution": "descriptive distribution over Test tiles",
            "uncertainty": "binary predictive entropy in natural-log nats; error entropy is calculated only on misclassified pixels",
        },
    }


def across_seed_summaries(run_summaries: list[dict]) -> list[dict]:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for summary in run_summaries:
        by_model[summary["model_id"]].append(summary)
    rows = []
    for model_id, summaries in sorted(by_model.items()):
        record = {
            "model_id": model_id,
            "n_seeds": len(summaries),
            "seeds": ",".join(str(item["seed"]) for item in sorted(summaries, key=lambda item: item["seed"])),
        }
        for scope_key, metric_key in (
            ("global_pixel_metrics", "miou"),
            ("global_pixel_metrics", "iou_fg"),
            ("global_pixel_metrics", "f1"),
            ("global_boundary_metrics", "boundary_f1"),
            ("global_boundary_metrics", "boundary_iou"),
        ):
            values = [float(item[scope_key][metric_key]) for item in summaries]
            prefix = f"{scope_key}_{metric_key}"
            record[f"{prefix}_mean"] = float(np.mean(values))
            record[f"{prefix}_std"] = float(np.std(values, ddof=0))
        for metric in ("mean_entropy_vs_error_rate", "mean_error_entropy_vs_error_rate"):
            values = [
                item["uncertainty"][metric]["pearson_r"]
                for item in summaries
                if item["uncertainty"][metric]["pearson_r"] is not None
            ]
            record[f"uncertainty_{metric}_mean"] = float(np.mean(values)) if values else None
            record[f"uncertainty_{metric}_std"] = float(np.std(values, ddof=0)) if values else None
        rows.append(record)
    return rows


def render_plots(output_dir: Path, tile_rows: list[dict], area_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    data = {metric: [] for metric in ("miou", "iou_fg", "f1", "boundary_f1", "boundary_iou")}
    for key in sorted({(row["model_id"], row["seed"]) for row in tile_rows}):
        model_id, seed = key
        labels.append(f"{model_id}\ns{seed}")
        selected = [row for row in tile_rows if (row["model_id"], row["seed"]) == key]
        for metric in data:
            data[metric].append(
                [float(row[metric]) for row in selected if math.isfinite(float(row[metric]))]
            )
    figure, axes = plt.subplots(1, len(data), figsize=(18, 4.5), sharey=False)
    for axis, (metric, values) in zip(axes, data.items()):
        axis.boxplot(values, tick_labels=labels, showfliers=False)
        axis.set_title(metric)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=30)
    figure.suptitle("Per-tile Test distributions")
    figure.tight_layout()
    figure.savefig(plots_dir / "per_tile_distributions.png", dpi=220)
    plt.close(figure)

    present_area_labels = {row["area_group"] for row in area_rows}
    area_labels = [
        label for label in ("small", "medium", "large") if label in present_area_labels
    ]
    area_labels.extend(sorted(present_area_labels - set(area_labels)))
    run_labels = sorted({(row["model_id"], row["seed"]) for row in area_rows})
    x = np.arange(len(area_labels), dtype=np.float64)
    width = 0.8 / max(1, len(run_labels))
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for index, key in enumerate(run_labels):
        selected = {row["area_group"]: row for row in area_rows if (row["model_id"], row["seed"]) == key}
        means = [float(selected[label]["iou_fg_mean"]) for label in area_labels]
        stds = [float(selected[label]["iou_fg_std"]) for label in area_labels]
        offset = (index - (len(run_labels) - 1) / 2.0) * width
        axis.bar(x + offset, means, width, yerr=stds, capsize=3, label=f"{key[0]} s{key[1]}")
    axis.set_xticks(x, area_labels)
    axis.set_ylabel("Foreground IoU (mean ± tile std)")
    axis.set_title("Foreground-area stratification")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(plots_dir / "foreground_area_stratification.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5.5))
    for key in run_labels:
        selected = [row for row in tile_rows if (row["model_id"], row["seed"]) == key]
        x_values = np.asarray(
            [float(row["error_rate"]) for row in selected if math.isfinite(float(row["mean_error_entropy"]))],
            dtype=np.float64,
        )
        y_values = np.asarray(
            [float(row["mean_error_entropy"]) for row in selected if math.isfinite(float(row["mean_error_entropy"]))],
            dtype=np.float64,
        )
        axis.scatter(x_values, y_values, alpha=0.55, s=22, label=f"{key[0]} s{key[1]}")
    axis.set_xlabel("Per-tile error rate")
    axis.set_ylabel("Mean entropy on error pixels (nats)")
    axis.set_title("Uncertainty versus error rate")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(plots_dir / "uncertainty_error_scatter.png", dpi=220)
    plt.close(figure)


def evaluate_run(
    data_root: Path,
    run: dict,
    model_id: str,
    seed: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    boundary_tolerance: int,
    area_groups: int,
) -> tuple[dict, list[dict], list[dict]]:
    model_args = SimpleNamespace(run_config=run["config_path"], model_name="")
    model, _, _, input_channels = build_evaluation_model(model_args)
    dataset = LabeledTestDataset(str(data_root), input_channels=input_channels)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = model.to(device)
    load_checkpoint(model, Path(run["checkpoint_path"]), device)
    model.eval()

    rows = []
    with torch.inference_mode():
        for images, targets in tqdm(
            loader,
            desc=f"{model_id}/s{seed} Test diagnostics",
            leave=False,
        ):
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            probability, prediction = probabilities_and_predictions(
                extract_logits(model(images))
            )
            for index in range(images.shape[0]):
                image_name = dataset.image_files[len(rows)].name
                row, _ = per_tile_record(
                    image_name,
                    prediction[index].detach().cpu(),
                    targets[index].detach().cpu(),
                    probability[index].detach().cpu(),
                    boundary_tolerance,
                )
                row["model_id"] = model_id
                row["seed"] = int(seed)
                rows.append(row)

    group_info = assign_area_groups(rows, area_groups)
    area_rows = summarize_area_groups(rows, group_info)
    summary = summarize_run(
        model_id,
        run["stored_id"],
        seed,
        run,
        rows,
        group_info,
        boundary_tolerance,
    )
    return summary, rows, area_rows


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers cannot be negative")
    if args.boundary_tolerance < 0:
        raise ValueError("boundary-tolerance cannot be negative")

    data_root = Path(args.data_root).expanduser().resolve()
    protocol_path = Path(args.config).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    protocol = load_protocol(protocol_path)
    ids = parse_id_list(args.ids, DEFAULT_DIAGNOSTIC_IDS)
    seeds = parse_seed_list(args.seeds, DEFAULT_DIAGNOSTIC_SEEDS)
    requested_device = args.device
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_summaries = []
    tile_rows = []
    area_rows = []
    resolved_runs = []
    for model_id in ids:
        for seed in seeds:
            run = resolve_formal_run(run_root, protocol, model_id, seed)
            resolved_runs.append(run)
            summary, rows, grouped = evaluate_run(
                data_root,
                run,
                model_id,
                seed,
                device,
                args.batch_size,
                args.num_workers,
                args.boundary_tolerance,
                args.area_groups,
            )
            run_summaries.append(summary)
            tile_rows.extend(rows)
            for row in grouped:
                area_rows.append({"model_id": model_id, "seed": seed, **row})

    distribution_rows = []
    for key in sorted({(row["model_id"], row["seed"]) for row in tile_rows}):
        selected = [row for row in tile_rows if (row["model_id"], row["seed"]) == key]
        for metric in PER_TILE_METRICS:
            distribution_rows.append(
                {
                    "model_id": key[0],
                    "seed": key[1],
                    "metric": metric,
                    **mean_and_std(selected, metric),
                }
            )

    write_csv(output_dir / "per_tile.csv", tile_rows)
    write_csv(output_dir / "foreground_area_stratified.csv", area_rows)
    write_csv(output_dir / "per_tile_distributions.csv", distribution_rows)
    write_csv(
        output_dir / "diagnostics_summary.csv",
        [
            {
                "model_id": summary["model_id"],
                "stored_id": summary["stored_id"],
                "seed": summary["seed"],
                "n_tiles": summary["n_tiles"],
                "global_miou": summary["global_pixel_metrics"]["miou"],
                "global_iou_fg": summary["global_pixel_metrics"]["iou_fg"],
                "global_f1": summary["global_pixel_metrics"]["f1"],
                "global_boundary_f1": summary["global_boundary_metrics"]["boundary_f1"],
                "global_boundary_iou": summary["global_boundary_metrics"]["boundary_iou"],
                "mean_tile_miou": summary["per_tile_distribution"]["miou"]["mean"],
                "mean_tile_f1": summary["per_tile_distribution"]["f1"]["mean"],
                "mean_error_entropy": summary["per_tile_distribution"]["mean_error_entropy"]["mean"],
                "entropy_error_rate_pearson_r": summary["uncertainty"]["mean_error_entropy_vs_error_rate"]["pearson_r"],
                "entropy_error_rate_n_tiles": summary["uncertainty"]["mean_error_entropy_vs_error_rate"]["n_tiles"],
            }
            for summary in run_summaries
        ],
    )
    across_rows = across_seed_summaries(run_summaries)
    write_csv(output_dir / "across_seed_summary.csv", across_rows)

    if not args.no_plots:
        render_plots(output_dir, tile_rows, area_rows)

    manifest = {
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": "frozen public Test diagnostics",
        "data_root": str(data_root),
        "protocol": str(protocol_path),
        "protocol_version": protocol.get("protocol_version", ""),
        "run_root": str(run_root),
        "requested_ids": ids,
        "seeds": seeds,
        "resolved_runs": resolved_runs,
        "device_requested": requested_device,
        "device_used": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "test_tiles": len(tile_rows) // max(1, len(ids) * len(seeds)),
        "boundary": {
            "contour": "dilate_3x3(mask) minus erode_3x3(mask)",
            "tolerance_pixels": args.boundary_tolerance,
            "boundary_f1": "matched contour precision/recall F1; matching is symmetric within tolerance",
            "boundary_iou": "IoU of tolerance-dilated contour bands",
        },
        "foreground_area": {
            "groups": args.area_groups,
            "definition": "rank by GT foreground ratio, tie-break by filename, then split into equal-count groups",
        },
        "uncertainty": {
            "probability": "foreground softmax probability for two-class logits, or sigmoid for one-channel logits",
            "entropy": "binary predictive entropy in natural-log nats",
            "error_entropy": "mean entropy over misclassified pixels; tiles without errors are excluded from correlation",
            "correlation": "Pearson correlation between per-tile entropy and per-tile error rate",
        },
        "outputs": {
            "per_tile": str(output_dir / "per_tile.csv"),
            "area_stratified": str(output_dir / "foreground_area_stratified.csv"),
            "distributions": str(output_dir / "per_tile_distributions.csv"),
            "summary": str(output_dir / "diagnostics_summary.csv"),
            "across_seed_summary": str(output_dir / "across_seed_summary.csv"),
            "plots": str(output_dir / "plots") if not args.no_plots else None,
        },
        "material_passport": {
            "verification_status": "EXECUTED",
            "checkpoint_selection": "frozen formal run best.pth resolved from protocol summary.csv",
            "inference_rerun": True,
            "primary_metrics_unchanged": True,
        },
    }
    write_json(output_dir / "diagnostics_manifest.json", manifest)
    write_json(
        output_dir / "diagnostics_summary.json",
        {
            "manifest": str(output_dir / "diagnostics_manifest.json"),
            "run_summaries": run_summaries,
            "across_seed_summary": across_rows,
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
