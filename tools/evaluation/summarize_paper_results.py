"""Aggregate paper runs across seeds and, when available, public-test metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, stdev


RUN_PATTERN = re.compile(r"^(?P<experiment>.+)_s(?P<seed>\d+)$")
TRAIN_METRICS = (
    "best_miou",
    "best_iou_fg",
    "best_iou_bg",
    "best_f1",
    "best_val_loss",
    "final_miou",
    "final_iou_fg",
    "final_f1",
)
TEST_METRICS = ("miou", "iou_fg", "iou_bg", "f1", "precision", "recall")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        default=str(root / "outputs_experiments/pmdfnet_mmlsv2/summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "outputs_experiments/pmdfnet_mmlsv2/aggregate"),
    )
    return parser.parse_args()


def as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": None, "std": None}
    return {
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
    }


def resolve_run_dir(summary_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return summary_path.parent.parent.parent / path


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    with summary_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        match = RUN_PATTERN.match(row.get("experiment_name", ""))
        if not match:
            continue
        record = dict(row)
        record["seed"] = int(match.group("seed"))
        record["experiment"] = match.group("experiment")
        run_dir = resolve_run_dir(summary_path, row["run_dir"])
        test_path = run_dir / "test_metrics.json"
        if test_path.is_file():
            with test_path.open(encoding="utf-8") as f:
                record["test"] = json.load(f).get("metrics", {})
        else:
            record["test"] = {}
        grouped.setdefault(record["experiment"], []).append(record)

    aggregate_rows = []
    aggregate_json = {"summary_csv": str(summary_path), "experiments": {}}
    for experiment, records in sorted(grouped.items()):
        records.sort(key=lambda item: item["seed"])
        payload = {
            "experiment": experiment,
            "seeds": [record["seed"] for record in records],
            "train": {},
            "test": {},
            "per_seed": [],
        }
        flat = {"experiment": experiment, "seeds": ",".join(str(r["seed"]) for r in records)}
        for metric in TRAIN_METRICS:
            values = [as_float(record.get(metric, "")) for record in records]
            values = [value for value in values if value is not None]
            result = stats(values)
            payload["train"][metric] = result
            flat[f"train_{metric}_mean"] = result["mean"]
            flat[f"train_{metric}_std"] = result["std"]
        for metric in TEST_METRICS:
            values = [as_float(record.get("test", {}).get(metric)) for record in records]
            values = [value for value in values if value is not None]
            result = stats(values)
            payload["test"][metric] = result
            flat[f"test_{metric}_mean"] = result["mean"]
            flat[f"test_{metric}_std"] = result["std"]
        for record in records:
            payload["per_seed"].append(
                {
                    "seed": record["seed"],
                    "run_dir": record["run_dir"],
                    "best_miou": as_float(record.get("best_miou", "")),
                    "test": record.get("test", {}),
                }
            )
        aggregate_json["experiments"][experiment] = payload
        aggregate_rows.append(flat)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "paper_aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(aggregate_rows[0]) if aggregate_rows else ["experiment", "seeds"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    with (output_dir / "paper_aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate_json, f, ensure_ascii=False, indent=2)
    print(json.dumps(aggregate_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
