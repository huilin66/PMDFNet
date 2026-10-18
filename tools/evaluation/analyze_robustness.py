"""Paired image-level statistics and deterministic failure-case selection.

The robustness evaluator stores one JSON object per model/seed/condition and
keeps the per-image confusion counts inside that object.  This script uses
those counts without rerunning inference.  The primary comparison is paired
within each seed: the same Test image is compared for the reference and
candidate model.  Bootstrap intervals are therefore intervals over images,
not over individual pixels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_METRICS = (
    "miou",
    "f1",
    "iou_fg",
    "precision",
    "recall",
    "fp",
    "fn",
    "error_pixels",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute paired image-level statistics from robustness JSONL output."
    )
    parser.add_argument(
        "--input",
        default="outputs_experiments/robustness/robustness_per_image.jsonl",
        help="robustness_per_image.jsonl produced by evaluate_robustness.py",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory; defaults to <input-dir>/analysis.",
    )
    parser.add_argument("--reference-id", default="M4")
    parser.add_argument("--candidate-id", default="L1")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--permutation-resamples", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20261018)
    parser.add_argument("--failure-seed", type=int, default=41)
    parser.add_argument("--failure-condition", default="clean")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args(argv)


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


def load_per_image_records(path: Path) -> tuple[dict, dict]:
    """Return metadata and rows indexed by model and pairing key."""
    rows_by_model: dict[str, dict[tuple[int, str, str, str], dict]] = defaultdict(dict)
    metadata: dict = {"models": set(), "seeds": set(), "conditions": set()}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            model_id = str(record["paper_id"])
            seed = int(record["seed"])
            condition_id = str(record["condition_id"])
            level_name = str(record["level_name"])
            level_value = float(record.get("level_value", 0.0))
            metadata["models"].add(model_id)
            metadata["seeds"].add(seed)
            metadata["conditions"].add((condition_id, level_name, level_value))
            for row in record.get("rows", []):
                image = str(row["image"])
                key = (seed, condition_id, level_name, image)
                if key in rows_by_model[model_id]:
                    raise ValueError(
                        f"Duplicate model/seed/condition/image at line {line_number}: "
                        f"{model_id}, {key}"
                    )
                normalized = dict(row)
                normalized["level_value"] = level_value
                rows_by_model[model_id][key] = normalized
    if not rows_by_model:
        raise ValueError(f"No per-image records found in {path}")
    return metadata, rows_by_model


def bootstrap_mean_ci(
    values: np.ndarray,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if resamples <= 0:
        raise ValueError("bootstrap-resamples must be positive")
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    bootstrap_means = values[indices].mean(axis=1)
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(low), float(high)


def paired_sign_flip_pvalue(
    differences: np.ndarray,
    resamples: int,
    rng: np.random.Generator,
) -> float:
    """Two-sided Monte Carlo paired sign-flip test for mean difference."""
    if differences.size == 0:
        return float("nan")
    if resamples <= 0:
        raise ValueError("permutation-resamples must be positive")
    observed = abs(float(differences.mean()))
    signs = rng.integers(0, 2, size=(resamples, differences.size), dtype=np.int8)
    signs = signs * 2 - 1
    null_means = signs @ differences / differences.size
    exceedances = int(np.count_nonzero(np.abs(null_means) >= observed))
    return float((exceedances + 1) / (resamples + 1))


def pair_rows(
    rows_by_model: dict,
    reference_id: str,
    candidate_id: str,
    condition_id: str,
    level_name: str,
    seed: int,
) -> list[tuple[str, dict, dict]]:
    reference = rows_by_model.get(reference_id, {})
    candidate = rows_by_model.get(candidate_id, {})
    reference_keys = {
        key for key in reference if key[:3] == (seed, condition_id, level_name)
    }
    candidate_keys = {
        key for key in candidate if key[:3] == (seed, condition_id, level_name)
    }
    if reference_keys != candidate_keys:
        missing_reference = sorted(candidate_keys - reference_keys)
        missing_candidate = sorted(reference_keys - candidate_keys)
        raise ValueError(
            f"Unpaired rows for {condition_id}/{level_name}/seed={seed}; "
            f"missing reference={missing_reference[:3]}, "
            f"missing candidate={missing_candidate[:3]}"
        )
    return [
        (key[3], reference[key], candidate[key])
        for key in sorted(reference_keys, key=lambda item: item[3])
    ]


def metric_record(
    condition_id: str,
    level_name: str,
    level_value: float,
    seed: int,
    metric: str,
    reference_values: np.ndarray,
    candidate_values: np.ndarray,
    bootstrap_resamples: int,
    permutation_resamples: int,
    random_seed: int,
) -> dict:
    differences = candidate_values - reference_values
    bootstrap_rng = np.random.default_rng(
        stable_seed(random_seed, "bootstrap", condition_id, level_name, seed, metric)
    )
    permutation_rng = np.random.default_rng(
        stable_seed(random_seed, "permutation", condition_id, level_name, seed, metric)
    )
    ci_low, ci_high = bootstrap_mean_ci(differences, bootstrap_resamples, bootstrap_rng)
    return {
        "scope": "per_seed",
        "condition_id": condition_id,
        "level_name": level_name,
        "level_value": float(level_value),
        "seed": int(seed),
        "metric": metric,
        "n_pairs": int(differences.size),
        "reference_mean": float(reference_values.mean()),
        "candidate_mean": float(candidate_values.mean()),
        "mean_difference_candidate_minus_reference": float(differences.mean()),
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "paired_sign_flip_pvalue": paired_sign_flip_pvalue(
            differences, permutation_resamples, permutation_rng
        ),
    }


def failure_case_row(
    image: str,
    seed: int,
    condition_id: str,
    level_name: str,
    reference_id: str,
    candidate_id: str,
    reference: dict,
    candidate: dict,
) -> dict:
    row = {
        "image": image,
        "seed": int(seed),
        "condition_id": condition_id,
        "level_name": level_name,
        "level_value": float(reference.get("level_value", 0.0)),
        "reference_id": reference_id,
        "candidate_id": candidate_id,
        "reference_miou": float(reference["miou"]),
        "candidate_miou": float(candidate["miou"]),
        "reference_f1": float(reference["f1"]),
        "candidate_f1": float(candidate["f1"]),
        "reference_fp": int(reference["fp"]),
        "candidate_fp": int(candidate["fp"]),
        "reference_fn": int(reference["fn"]),
        "candidate_fn": int(candidate["fn"]),
        "reference_error_pixels": int(reference["error_pixels"]),
        "candidate_error_pixels": int(candidate["error_pixels"]),
    }
    row["candidate_minus_reference_miou"] = row["candidate_miou"] - row["reference_miou"]
    row["candidate_minus_reference_f1"] = row["candidate_f1"] - row["reference_f1"]
    row["candidate_minus_reference_fp"] = row["candidate_fp"] - row["reference_fp"]
    row["candidate_minus_reference_fn"] = row["candidate_fn"] - row["reference_fn"]
    row["candidate_minus_reference_error_pixels"] = (
        row["candidate_error_pixels"] - row["reference_error_pixels"]
    )
    row["candidate_error_reduction_pixels"] = (
        row["reference_error_pixels"] - row["candidate_error_pixels"]
    )
    row["mean_error_pixels"] = (
        row["reference_error_pixels"] + row["candidate_error_pixels"]
    ) / 2.0
    return row


def select_failure_cases(rows: list[dict], top_k: int) -> dict[str, list[dict]]:
    if top_k <= 0:
        raise ValueError("top-k must be positive")
    candidate_better = sorted(
        (row for row in rows if row["candidate_error_reduction_pixels"] > 0),
        key=lambda row: (-row["candidate_error_reduction_pixels"], row["image"]),
    )[:top_k]
    reference_better = sorted(
        (row for row in rows if row["candidate_error_reduction_pixels"] < 0),
        key=lambda row: (row["candidate_error_reduction_pixels"], row["image"]),
    )[:top_k]
    both_hard = sorted(
        rows,
        key=lambda row: (-row["mean_error_pixels"], row["image"]),
    )[:top_k]
    largest_disagreement = sorted(
        rows,
        key=lambda row: (
            -abs(row["candidate_error_reduction_pixels"]),
            row["image"],
        ),
    )[:top_k]
    return {
        "candidate_better": candidate_better,
        "reference_better": reference_better,
        "both_hard": both_hard,
        "largest_disagreement": largest_disagreement,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")
    if args.bootstrap_resamples <= 0 or args.permutation_resamples <= 0:
        raise ValueError("resample counts must be positive")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_path.parent / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, rows_by_model = load_per_image_records(input_path)
    for model_id in (args.reference_id, args.candidate_id):
        if model_id not in rows_by_model:
            raise ValueError(
                f"Model {model_id!r} not found in {input_path}; "
                f"available={sorted(rows_by_model)}"
            )

    condition_keys = sorted(metadata["conditions"], key=lambda item: (item[0], item[1]))
    seeds = sorted(metadata["seeds"])
    stats_rows: list[dict] = []
    seed_means: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for condition_id, level_name, level_value in condition_keys:
        for seed in seeds:
            pairs = pair_rows(
                rows_by_model,
                args.reference_id,
                args.candidate_id,
                condition_id,
                level_name,
                seed,
            )
            if not pairs:
                raise ValueError(f"No paired images for {condition_id}/{level_name}/seed={seed}")
            for metric in DEFAULT_METRICS:
                reference_values = np.asarray(
                    [float(reference[metric]) for _, reference, _ in pairs], dtype=np.float64
                )
                candidate_values = np.asarray(
                    [float(candidate[metric]) for _, _, candidate in pairs], dtype=np.float64
                )
                record = metric_record(
                    condition_id,
                    level_name,
                    level_value,
                    seed,
                    metric,
                    reference_values,
                    candidate_values,
                    args.bootstrap_resamples,
                    args.permutation_resamples,
                    args.random_seed,
                )
                stats_rows.append(record)
                seed_means[(condition_id, level_name, metric)].append(record)

    aggregate_rows: list[dict] = []
    for key, records in sorted(seed_means.items()):
        condition_id, level_name, metric = key
        aggregate_rows.append(
            {
                "scope": "mean_across_seeds",
                "condition_id": condition_id,
                "level_name": level_name,
                "level_value": float(records[0]["level_value"]),
                "seed": "mean_across_seeds",
                "n_seeds": len(records),
                "n_pairs_per_seed": int(records[0]["n_pairs"]),
                "metric": metric,
                "reference_mean": float(np.mean([row["reference_mean"] for row in records])),
                "candidate_mean": float(np.mean([row["candidate_mean"] for row in records])),
                "mean_difference_candidate_minus_reference": float(
                    np.mean(
                        [row["mean_difference_candidate_minus_reference"] for row in records]
                    )
                ),
                "note": "descriptive mean of three per-seed paired estimates; no seed-level CI",
            }
        )
    all_stats_rows = stats_rows + aggregate_rows
    write_csv(output_dir / "robustness_paired_stats.csv", all_stats_rows)

    failure_pairs = pair_rows(
        rows_by_model,
        args.reference_id,
        args.candidate_id,
        args.failure_condition,
        "clean" if args.failure_condition == "clean" else "",
        args.failure_seed,
    )
    # The clean condition uses level_name=clean. For a non-clean condition, the
    # first matching level is selected deterministically by the protocol order.
    if not failure_pairs:
        matching_levels = sorted(
            {
                key[2]
                for key in rows_by_model[args.reference_id]
                if key[0] == args.failure_seed and key[1] == args.failure_condition
            }
        )
        if not matching_levels:
            raise ValueError(
                f"No failure-analysis condition {args.failure_condition!r} for seed {args.failure_seed}"
            )
        failure_level = matching_levels[0]
        failure_pairs = pair_rows(
            rows_by_model,
            args.reference_id,
            args.candidate_id,
            args.failure_condition,
            failure_level,
            args.failure_seed,
        )
    else:
        failure_level = "clean"
    failure_rows = [
        failure_case_row(
            image,
            args.failure_seed,
            args.failure_condition,
            failure_level,
            args.reference_id,
            args.candidate_id,
            reference,
            candidate,
        )
        for image, reference, candidate in failure_pairs
    ]
    selected = select_failure_cases(failure_rows, args.top_k)
    selected_images = []
    seen_images = set()
    for category in ("candidate_better", "reference_better", "both_hard", "largest_disagreement"):
        for row in selected[category]:
            if row["image"] not in seen_images:
                seen_images.add(row["image"])
                selected_images.append(row)
    failure_payload = {
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "reference_id": args.reference_id,
        "candidate_id": args.candidate_id,
        "condition_id": args.failure_condition,
        "level_name": failure_level,
        "seed": args.failure_seed,
        "top_k_per_category": args.top_k,
        "categories": selected,
        "selected_unique_cases": selected_images,
        "selection_rule": {
            "candidate_better": "largest positive reference_error_pixels - candidate_error_pixels",
            "reference_better": "largest negative reference_error_pixels - candidate_error_pixels",
            "both_hard": "largest mean of the two models' error_pixels",
            "largest_disagreement": "largest absolute error_pixels difference",
            "tie_break": "image filename ascending",
        },
    }
    (output_dir / "failure_cases.json").write_text(
        json.dumps(failure_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    failure_csv_rows = []
    for category, category_rows in selected.items():
        for row in category_rows:
            failure_csv_rows.append({"category": category, **row})
    write_csv(output_dir / "failure_cases.csv", failure_csv_rows)

    payload = {
        "status": "completed",
        "evaluation_type": "paired_image_level_bootstrap_and_sign_flip",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "reference_id": args.reference_id,
        "candidate_id": args.candidate_id,
        "metrics": list(DEFAULT_METRICS),
        "seeds": seeds,
        "conditions": [
            {"condition_id": key[0], "level_name": key[1], "level_value": key[2]}
            for key in condition_keys
        ],
        "bootstrap_resamples": args.bootstrap_resamples,
        "permutation_resamples": args.permutation_resamples,
        "random_seed": args.random_seed,
        "pairing": {
            "unit": "one Test image within one training seed",
            "key": "seed + condition_id + level_name + image filename",
            "difference": "candidate minus reference",
            "pixel_metrics": "computed from per-image TP/FP/FN/TN counts; pixels are not treated as independent samples",
        },
        "records": stats_rows,
        "aggregate_records": aggregate_rows,
        "failure_cases": str(output_dir / "failure_cases.json"),
        "material_passport": {
            "verification_status": "EXECUTED",
            "source": "robustness_per_image.jsonl",
            "inference_rerun": False,
            "bootstrap_ci": "percentile 95% interval over paired Test images within each seed",
            "paired_test": "two-sided Monte Carlo sign-flip randomization test on paired image differences",
        },
    }
    (output_dir / "robustness_paired_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "paired_stat_rows": len(stats_rows),
                "aggregate_stat_rows": len(aggregate_rows),
                "failure_case_categories": {
                    category: len(rows) for category, rows in selected.items()
                },
                "selected_unique_cases": len(selected_images),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
