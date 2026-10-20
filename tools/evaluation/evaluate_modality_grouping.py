"""Evaluate and statistically compare the frozen modality-grouping protocol.

The protocol has four semantic groups:

* G0: the original physically motivated grouping, reusing L1/M5;
* G1: the strongest already evaluated alternative, reusing B2 exactly;
* G2: an equal-capacity DEM/Grayscale swap;
* G3: an equal-capacity Slope/R swap.

Only G2 and G3 require new training.  This evaluator uses the same sorted
public Test image list for every group and seed, reports global-pixel metrics
separately from per-tile metrics, and performs the primary paired inference on
per-tile mIoU for G0 minus G1.  G0-vs-G2/G3 statistics are labelled
exploratory; they are not used to create a second confirmatory claim.
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
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.evaluate_test_diagnostics import (  # noqa: E402
    extract_logits,
    mean_and_std,
    per_tile_record,
    pooled_boundary_metrics,
    pooled_metrics,
    probabilities_and_predictions,
)
from tools.evaluation.evaluate_test_metrics import (  # noqa: E402
    LabeledTestDataset,
    build_evaluation_model,
    load_checkpoint,
    sample_standard_deviation,
)
from tools.evaluation.paper_run_utils import (  # noqa: E402
    load_protocol,
    parse_seed_list,
    resolve_formal_run,
)


GROUP_ORDER = ("G0", "G1", "G2", "G3")
PRIMARY_METRIC = "miou"
PER_TILE_REPORT_METRICS = ("miou", "f1", "boundary_f1")
SHARED_CONFIG_KEYS = (
    "arch",
    "encoder",
    "pretrain",
    "fusion",
    "in_channels",
    "input_channels",
    "num_classes",
    "input_size",
    "batch_size",
    "epochs",
    "lr",
    "weight_decay",
    "optimizer",
    "scheduler",
    "loss",
    "augmentation",
    "aug_prob",
    "mosaic_prob",
    "normalization",
    "dataset_type",
    "use_file_list_dataset",
    "deterministic",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate G0-G3 modality grouping checkpoints on public Test and "
            "compute paired per-tile mIoU statistics."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--main-run-root",
        default="outputs_experiments/pmdfnet_mmlsv2",
        help="Run root containing the frozen L1/M5 and B2 reference checkpoints.",
    )
    parser.add_argument(
        "--main-config",
        default="configs/mmlsv2_paper_protocol.json",
        help="Main protocol used to resolve G0=L1 and G1=B2.",
    )
    parser.add_argument(
        "--grouping-run-root",
        default="outputs_experiments/pmdfnet_mmlsv2_modality_grouping",
        help="Run root containing the new G2/G3 checkpoints.",
    )
    parser.add_argument(
        "--grouping-config",
        default="configs/mmlsv2_modality_grouping_protocol.json",
        help="Supplementary G2/G3 protocol.",
    )
    parser.add_argument("--groups", default="G0,G1,G2,G3")
    parser.add_argument("--seeds", default="41 42 43")
    parser.add_argument(
        "--output-dir",
        default="outputs_experiments/modality_grouping",
        help="Directory for per-tile, summary and paired-statistic outputs.",
    )
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--boundary-tolerance",
        type=int,
        default=2,
        help="Pixel tolerance used for the supplementary Boundary F1 report.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--permutation-resamples", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20261019)
    parser.add_argument(
        "--comparisons",
        default="G0:G1,G0:G2,G0:G3",
        help="Comma-separated left:right comparisons; differences are left minus right.",
    )
    parser.add_argument("--primary-comparison", default="G0:G1")
    return parser.parse_args(argv)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def parse_groups(value: str) -> list[str]:
    groups = []
    for item in value.split(","):
        item = item.strip()
        if item and item not in groups:
            groups.append(item)
    if not groups:
        raise ValueError("--groups must contain at least one group")
    unknown = [group for group in groups if group not in GROUP_ORDER]
    if unknown:
        raise ValueError(f"Unknown grouping IDs: {unknown}; expected {GROUP_ORDER}")
    return groups


def parse_comparisons(value: str) -> list[tuple[str, str]]:
    comparisons = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 2 or any(part not in GROUP_ORDER for part in parts):
            raise ValueError(
                f"Invalid comparison {item!r}; use GROUP:GROUP, e.g. G0:G1"
            )
        pair = (parts[0], parts[1])
        if pair[0] == pair[1]:
            raise ValueError(f"A comparison must contain two different groups: {item}")
        if pair not in comparisons:
            comparisons.append(pair)
    if not comparisons:
        raise ValueError("--comparisons must contain at least one comparison")
    return comparisons


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join(str(part) for part in (base_seed, *parts)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**63 - 1)


def normalize_channels(value: object) -> str:
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = str(value).split(",")
    return ",".join(str(int(item)) for item in values if str(item).strip())


def config_value_equal(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        try:
            return float(left) == float(right)
        except (TypeError, ValueError):
            return False
    return str(left) == str(right)


def load_run_config(run: dict) -> dict:
    path = Path(run["config_path"])
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_group_run(
    group_id: str,
    seed: int,
    main_protocol: dict,
    grouping_protocol: dict,
    main_run_root: Path,
    grouping_run_root: Path,
) -> dict:
    if group_id in grouping_protocol.get("reference_groups", {}):
        reference = grouping_protocol["reference_groups"][group_id]
        # The explicitly supplied main protocol is authoritative for this
        # evaluation; the reference path is retained as provenance metadata.
        source_protocol = main_protocol
        # The CLI run root is authoritative for the current evaluation.  The
        # protocol's source_run_root remains provenance metadata/defaults, but
        # must not prevent evaluating a relocated output tree.
        source_root = main_run_root
        source_id = str(reference["source_id"])
        run = resolve_formal_run(source_root, source_protocol, source_id, seed)
        config = load_run_config(run)
        expected_channels = {
            "channels1": reference.get("stored_channels1"),
            "channels2": reference.get("stored_channels2"),
        }
        for key, expected in expected_channels.items():
            if expected is not None and normalize_channels(config.get(key, "")) != normalize_channels(expected):
                raise ValueError(
                    f"{group_id} source {source_id} config mismatch for {key}: "
                    f"expected {expected}, got {config.get(key)}. "
                    "Do not silently substitute a branch-order-swapped checkpoint."
                )
        return {
            "group_id": group_id,
            "label": reference.get("label", group_id),
            "semantic_stream_1": reference.get("semantic_stream_1", ""),
            "semantic_stream_2": reference.get("semantic_stream_2", ""),
            "source_id": source_id,
            "source_protocol": str(resolve_repo_path(main_protocol["_path"])),
            "run": run,
            "config": config,
            "reuse_checkpoint": bool(reference.get("reuse_checkpoint", False)),
            "stream_order_note": reference.get("stream_order_note", ""),
        }

    experiment_ids = {str(item.get("id")) for item in grouping_protocol.get("experiments", [])}
    if group_id not in experiment_ids:
        raise ValueError(f"Grouping protocol does not define {group_id}")
    run = resolve_formal_run(grouping_run_root, grouping_protocol, group_id, seed)
    config = load_run_config(run)
    experiment = next(item for item in grouping_protocol["experiments"] if item.get("id") == group_id)
    return {
        "group_id": group_id,
        "label": experiment.get("notes", group_id),
        "semantic_stream_1": experiment.get("semantic_stream_1", ""),
        "semantic_stream_2": experiment.get("semantic_stream_2", ""),
        "source_id": group_id,
        "source_protocol": str(resolve_repo_path(grouping_protocol["_path"])),
        "run": run,
        "config": config,
        "reuse_checkpoint": False,
        "stream_order_note": "",
    }


def validate_shared_configs(runs_for_seed: dict[str, dict]) -> dict:
    """Check that grouping comparisons differ only in the intended channels."""
    baseline_group = "G0" if "G0" in runs_for_seed else next(iter(runs_for_seed))
    baseline = runs_for_seed[baseline_group]["config"]
    mismatches = []
    for group_id, resolved in runs_for_seed.items():
        config = resolved["config"]
        for key in SHARED_CONFIG_KEYS:
            if key in {"channels1", "channels2"}:
                continue
            if not config_value_equal(config.get(key), baseline.get(key)):
                mismatches.append(
                    {
                        "group_id": group_id,
                        "key": key,
                        "baseline_value": baseline.get(key),
                        "observed_value": config.get(key),
                    }
                )
    if mismatches:
        raise ValueError(
            "Modality grouping configs are not isomorphic outside channel grouping: "
            + json.dumps(mismatches, ensure_ascii=False)
        )
    return {
        "status": "passed",
        "baseline_group": baseline_group,
        "checked_keys": list(SHARED_CONFIG_KEYS),
        "scope": "same PMDFNet architecture, backbone, fusion, loss and training settings; channels1/channels2 intentionally vary",
    }


def evaluate_run(
    data_root: Path,
    resolved: dict,
    seed: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    boundary_tolerance: int,
) -> tuple[dict, list[dict]]:
    run = resolved["run"]
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
        offset = 0
        for images, targets in tqdm(
            loader,
            desc=f"{resolved['group_id']}/s{seed} grouping Test",
            unit="batch",
            leave=False,
        ):
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            probability, prediction = probabilities_and_predictions(
                extract_logits(model(images))
            )
            for index in range(images.shape[0]):
                image_name = dataset.image_files[offset + index].name
                row, _ = per_tile_record(
                    image_name,
                    prediction[index].detach().cpu(),
                    targets[index].detach().cpu(),
                    probability[index].detach().cpu(),
                    boundary_tolerance,
                )
                row.update(
                    {
                        "group_id": resolved["group_id"],
                        "source_id": resolved["source_id"],
                        "seed": int(seed),
                    }
                )
                rows.append(row)
            offset += int(images.shape[0])

    if offset != len(dataset) or len(rows) != len(dataset):
        raise RuntimeError(
            f"Test row count mismatch for {resolved['group_id']}/s{seed}: "
            f"dataset={len(dataset)}, rows={len(rows)}, offset={offset}"
        )

    summary = {
        "group_id": resolved["group_id"],
        "source_id": resolved["source_id"],
        "seed": int(seed),
        "run_dir": run["run_dir"],
        "checkpoint": run["checkpoint_path"],
        "n_tiles": len(rows),
        "global_pixel_metrics": pooled_metrics(rows),
        "global_boundary_metrics": pooled_boundary_metrics(rows),
        "per_tile_distribution": {
            metric: mean_and_std(rows, metric) for metric in PER_TILE_REPORT_METRICS
        },
        "aggregation": {
            "global_pixel_metrics": "TP/FP/FN/TN pooled over all Test pixels",
            "global_boundary_metrics": "contour counts pooled over all Test tiles",
            "per_tile_distribution": "descriptive distribution over Test tiles",
        },
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, rows


def bootstrap_mean_ci(
    differences: np.ndarray,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    indices = rng.integers(0, differences.size, size=(resamples, differences.size))
    means = differences[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_sign_flip_pvalue(
    differences: np.ndarray,
    resamples: int,
    rng: np.random.Generator,
) -> float:
    observed = abs(float(differences.mean()))
    signs = rng.integers(0, 2, size=(resamples, differences.size), dtype=np.int8)
    signs = signs * 2 - 1
    null_means = signs @ differences / differences.size
    exceedances = int(np.count_nonzero(np.abs(null_means) >= observed))
    return float((exceedances + 1) / (resamples + 1))


def paired_stats(
    tile_rows: list[dict],
    comparisons: list[tuple[str, str]],
    primary_comparison: tuple[str, str],
    seeds: list[int],
    bootstrap_resamples: int,
    permutation_resamples: int,
    random_seed: int,
) -> tuple[list[dict], list[dict]]:
    by_key = {
        (row["group_id"], int(row["seed"]), row["image"]): row
        for row in tile_rows
    }
    per_seed_rows = []
    for left_id, right_id in comparisons:
        role = "primary" if (left_id, right_id) == primary_comparison else "exploratory"
        for seed in seeds:
            left = {
                image: row
                for (group, row_seed, image), row in by_key.items()
                if group == left_id and row_seed == seed
            }
            right = {
                image: row
                for (group, row_seed, image), row in by_key.items()
                if group == right_id and row_seed == seed
            }
            if set(left) != set(right):
                raise ValueError(
                    f"Unpaired grouping rows for {left_id}:{right_id}/seed={seed}: "
                    f"left_only={sorted(set(left) - set(right))[:3]}, "
                    f"right_only={sorted(set(right) - set(left))[:3]}"
                )
            images = sorted(left)
            left_values = np.asarray([float(left[name][PRIMARY_METRIC]) for name in images])
            right_values = np.asarray([float(right[name][PRIMARY_METRIC]) for name in images])
            differences = left_values - right_values
            bootstrap_rng = np.random.default_rng(
                stable_seed(random_seed, "bootstrap", left_id, right_id, seed)
            )
            permutation_rng = np.random.default_rng(
                stable_seed(random_seed, "permutation", left_id, right_id, seed)
            )
            ci_low, ci_high = bootstrap_mean_ci(
                differences, bootstrap_resamples, bootstrap_rng
            )
            per_seed_rows.append(
                {
                    "scope": "per_seed",
                    "comparison_role": role,
                    "left_id": left_id,
                    "right_id": right_id,
                    "difference_definition": f"{left_id} minus {right_id}",
                    "seed": int(seed),
                    "n_pairs": int(differences.size),
                    "left_tile_miou_mean": float(left_values.mean()),
                    "right_tile_miou_mean": float(right_values.mean()),
                    "mean_tile_miou_difference": float(differences.mean()),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "paired_sign_flip_pvalue": paired_sign_flip_pvalue(
                        differences, permutation_resamples, permutation_rng
                    ),
                }
            )

    aggregate_rows = []
    by_comparison = defaultdict(list)
    for row in per_seed_rows:
        by_comparison[(row["left_id"], row["right_id"])].append(row)
    for (left_id, right_id), rows in sorted(by_comparison.items()):
        aggregate_rows.append(
            {
                "scope": "mean_across_seeds",
                "comparison_role": rows[0]["comparison_role"],
                "left_id": left_id,
                "right_id": right_id,
                "difference_definition": f"{left_id} minus {right_id}",
                "seed": "mean_across_seeds",
                "n_seeds": len(rows),
                "n_pairs_per_seed": rows[0]["n_pairs"],
                "left_tile_miou_mean": float(np.mean([row["left_tile_miou_mean"] for row in rows])),
                "right_tile_miou_mean": float(np.mean([row["right_tile_miou_mean"] for row in rows])),
                "mean_tile_miou_difference": float(
                    np.mean([row["mean_tile_miou_difference"] for row in rows])
                ),
                "note": "Descriptive mean of per-seed paired estimates; no seed-level CI or p-value.",
            }
        )
    return per_seed_rows, aggregate_rows


def flatten_summary(summary: dict) -> dict:
    row = {
        "group_id": summary["group_id"],
        "source_id": summary["source_id"],
        "seed": summary["seed"],
        "n_tiles": summary["n_tiles"],
        "global_miou": summary["global_pixel_metrics"]["miou"],
        "global_f1": summary["global_pixel_metrics"]["f1"],
        "global_boundary_f1": summary["global_boundary_metrics"]["boundary_f1"],
        "mean_tile_miou": summary["per_tile_distribution"]["miou"]["mean"],
        "mean_tile_f1": summary["per_tile_distribution"]["f1"]["mean"],
        "mean_tile_boundary_f1": summary["per_tile_distribution"]["boundary_f1"]["mean"],
        "run_dir": summary["run_dir"],
        "checkpoint": summary["checkpoint"],
    }
    return row


def across_seed_summary(run_summaries: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for summary in run_summaries:
        grouped[summary["group_id"]].append(summary)
    rows = []
    for group_id, summaries in sorted(grouped.items()):
        row = {"group_id": group_id, "n_seeds": len(summaries)}
        row["seeds"] = ",".join(str(item["seed"]) for item in sorted(summaries, key=lambda item: item["seed"]))
        for label, values in (
            (
                "global_miou",
                [item["global_pixel_metrics"]["miou"] for item in summaries],
            ),
            (
                "global_f1",
                [item["global_pixel_metrics"]["f1"] for item in summaries],
            ),
            (
                "global_boundary_f1",
                [item["global_boundary_metrics"]["boundary_f1"] for item in summaries],
            ),
            (
                "mean_tile_miou",
                [item["per_tile_distribution"]["miou"]["mean"] for item in summaries],
            ),
            (
                "mean_tile_f1",
                [item["per_tile_distribution"]["f1"]["mean"] for item in summaries],
            ),
            (
                "mean_tile_boundary_f1",
                [item["per_tile_distribution"]["boundary_f1"]["mean"] for item in summaries],
            ),
        ):
            row[f"{label}_mean"] = float(np.mean(values))
            row[f"{label}_std"] = sample_standard_deviation(values)
        rows.append(row)
    return rows


def main(argv=None) -> int:
    args = parse_args(argv)
    groups = parse_groups(args.groups)
    seeds = parse_seed_list(args.seeds, (41, 42, 43))
    comparisons = parse_comparisons(args.comparisons)
    primary_comparison_parts = parse_comparisons(args.primary_comparison)
    if len(primary_comparison_parts) != 1:
        raise ValueError("--primary-comparison must contain exactly one GROUP:GROUP pair")
    primary_comparison = primary_comparison_parts[0]
    required_groups = set(groups)
    for left_id, right_id in comparisons:
        required_groups.update((left_id, right_id))
    missing_groups = sorted(required_groups - set(groups))
    if missing_groups:
        raise ValueError(
            f"Comparisons require groups not included in --groups: {missing_groups}"
        )
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers cannot be negative")
    if args.boundary_tolerance < 0:
        raise ValueError("boundary-tolerance cannot be negative")
    if args.bootstrap_resamples <= 0 or args.permutation_resamples <= 0:
        raise ValueError("resample counts must be positive")

    data_root = resolve_repo_path(args.data_root)
    main_config_path = resolve_repo_path(args.main_config)
    grouping_config_path = resolve_repo_path(args.grouping_config)
    main_run_root = resolve_repo_path(args.main_run_root)
    grouping_run_root = resolve_repo_path(args.grouping_run_root)
    output_dir = resolve_repo_path(args.output_dir)
    main_protocol = load_protocol(main_config_path)
    grouping_protocol = load_protocol(grouping_config_path)
    grouping_protocol["_path"] = str(grouping_config_path)

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {requested_device}, but CUDA is unavailable")
    device = torch.device(requested_device)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_runs = []
    run_summaries = []
    tile_rows = []
    protocol_checks = []
    for seed in seeds:
        runs_for_seed = {}
        for group_id in groups:
            resolved = resolve_group_run(
                group_id,
                seed,
                main_protocol,
                grouping_protocol,
                main_run_root,
                grouping_run_root,
            )
            runs_for_seed[group_id] = resolved
            resolved_runs.append(
                {
                    "group_id": group_id,
                    "seed": seed,
                    "source_id": resolved["source_id"],
                    "source_protocol": resolved["source_protocol"],
                    "run": resolved["run"],
                    "semantic_stream_1": resolved["semantic_stream_1"],
                    "semantic_stream_2": resolved["semantic_stream_2"],
                    "stream_order_note": resolved["stream_order_note"],
                }
            )
        protocol_checks.append({"seed": seed, **validate_shared_configs(runs_for_seed)})
        for group_id in groups:
            summary, rows = evaluate_run(
                data_root,
                runs_for_seed[group_id],
                seed,
                device,
                args.batch_size,
                args.num_workers,
                args.boundary_tolerance,
            )
            run_summaries.append(summary)
            tile_rows.extend(rows)

    write_csv(output_dir / "grouping_per_tile.csv", tile_rows)
    write_csv(output_dir / "grouping_results.csv", [flatten_summary(item) for item in run_summaries])
    across_rows = across_seed_summary(run_summaries)
    write_csv(output_dir / "grouping_across_seed.csv", across_rows)

    paired_per_seed, paired_aggregate = paired_stats(
        tile_rows,
        comparisons,
        primary_comparison,
        seeds,
        args.bootstrap_resamples,
        args.permutation_resamples,
        args.random_seed,
    )
    write_csv(
        output_dir / "grouping_paired_stats.csv",
        paired_per_seed + paired_aggregate,
    )

    group_definitions = {}
    for item in resolved_runs:
        group_definitions.setdefault(
            item["group_id"],
            {
                "source_id": item["source_id"],
                "semantic_stream_1": item["semantic_stream_1"],
                "semantic_stream_2": item["semantic_stream_2"],
                "stream_order_note": item["stream_order_note"],
            },
        )
    payload = {
        "status": "completed",
        "evaluation_type": "modality_grouping_global_and_per_tile_test",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "main_protocol": str(main_config_path),
        "main_protocol_sha256": sha256(main_config_path),
        "grouping_protocol": str(grouping_config_path),
        "grouping_protocol_sha256": sha256(grouping_config_path),
        "main_run_root": str(main_run_root),
        "grouping_run_root": str(grouping_run_root),
        "groups": groups,
        "group_definitions": group_definitions,
        "seeds": seeds,
        "primary_metric": PRIMARY_METRIC,
        "metrics": {
            "global": "global-pixel mIoU/F1 and pooled contour Boundary F1",
            "per_tile": "per-Test-image mIoU/F1/Boundary F1 means and distributions",
            "paired_inference": "per-tile mIoU only",
        },
        "boundary_tolerance_pixels": args.boundary_tolerance,
        "per_seed_results": run_summaries,
        "across_seed_results": across_rows,
        "paired_stats": {
            "primary_comparison": f"{primary_comparison[0]} minus {primary_comparison[1]}",
            "comparisons": [f"{left} minus {right}" for left, right in comparisons],
            "per_seed": paired_per_seed,
            "mean_across_seeds": paired_aggregate,
            "bootstrap_resamples": args.bootstrap_resamples,
            "permutation_resamples": args.permutation_resamples,
            "random_seed": args.random_seed,
            "multiple_comparison_policy": "Only G0-G1 is primary; G0-G2 and G0-G3 are exploratory and descriptive.",
        },
        "artifacts": {
            "per_tile_csv": str(output_dir / "grouping_per_tile.csv"),
            "per_seed_csv": str(output_dir / "grouping_results.csv"),
            "across_seed_csv": str(output_dir / "grouping_across_seed.csv"),
            "paired_stats_csv": str(output_dir / "grouping_paired_stats.csv"),
        },
        "resolved_runs": resolved_runs,
        "protocol_checks": protocol_checks,
        "aggregation_notes": {
            "global_vs_tile": "Global-pixel metrics pool all Test pixels; paired statistics use one per-tile mIoU per image and are not the same estimand.",
            "seed_summary": "Mean and sample standard deviation (ddof=1) across the three training seeds; no seed is selected using Test performance.",
            "within_seed_tile_distribution": "Per-tile distribution standard deviations use population standard deviation (ddof=0) over the complete fixed public Test-tile set.",
            "inference": "G0-G1 is a targeted comparison with a previously evaluated alternative selected from existing results; it is not labelled a fully preregistered confirmatory test.",
        },
        "material_passport": {
            "verification_status": "EXECUTED",
            "pairing": "same sorted public Test image filename within each training seed",
            "checkpoint_selection": "best validation mIoU checkpoint resolved from each protocol summary",
            "inference_rerun": True,
        },
    }
    write_json(output_dir / "grouping_results.json", payload)
    print(
        json.dumps(
            {
                "status": "completed",
                "groups": groups,
                "seeds": seeds,
                "per_seed_rows": len(run_summaries),
                "test_tiles_per_run": len(tile_rows) // max(1, len(run_summaries)),
                "paired_stat_rows": len(paired_per_seed) + len(paired_aggregate),
                "primary_comparison": f"{primary_comparison[0]} minus {primary_comparison[1]}",
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
