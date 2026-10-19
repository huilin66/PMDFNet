#!/usr/bin/env python3
"""Summarize training results from output directories into formatted tables.

Usage:
    python tools/analysis/summarize_results.py outputs_jstar/ablation outputs_jstar/comparison
    python tools/analysis/summarize_results.py outputs_experiments/chv3 --sort-by best_miou
    python tools/analysis/summarize_results.py outputs_jstar --recursive --output results_table.csv
"""

import argparse
import csv
import sys
from pathlib import Path


# Key columns to display in the summary table
DISPLAY_COLUMNS = [
    "experiment_name",
    "model_name",
    "dataset_type",
    "loss",
    "fusion",
    "augmentation",
    "mosaic_prob",
    "seed",
    "best_epoch",
    "best_miou",
    "best_iou_fg",
    "best_f1",
    "best_val_loss",
    "best_iou_fg_value",
    "best_iou_fg_epoch",
    "best_f1_value",
    "best_f1_epoch",
    "final_epoch",
    "final_miou",
    "final_iou_fg",
    "final_f1",
    "final_val_loss",
]

# Columns that should be formatted as floats
FLOAT_COLUMNS = {
    "best_miou", "best_iou_fg", "best_iou_bg", "best_f1", "best_val_loss",
    "best_iou_fg_value", "best_iou_fg_miou", "best_iou_fg_f1",
    "best_f1_value", "best_f1_miou", "best_f1_iou_fg",
    "best_val_loss_value", "best_val_loss_miou", "best_val_loss_iou_fg", "best_val_loss_f1",
    "final_miou", "final_iou_fg", "final_iou_bg", "final_f1", "final_val_loss",
    "final_train_loss", "mosaic_prob", "aug_prob", "lr", "weight_decay",
}


def find_summary_csvs(paths, recursive=False):
    """Find all summary CSV files in the given paths."""
    csv_files = []
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix == ".csv":
            csv_files.append(path)
        elif path.is_dir():
            if recursive:
                csv_files.extend(sorted(path.rglob("*summary*.csv")))
            else:
                csv_files.extend(sorted(path.glob("*summary*.csv")))
                # Also check one level deep
                for sub in sorted(path.iterdir()):
                    if sub.is_dir():
                        csv_files.extend(sorted(sub.glob("*summary*.csv")))
    # Deduplicate while preserving order
    seen = set()
    result = []
    for f in csv_files:
        resolved = f.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(f)
    return result


def read_summary_csv(csv_path):
    """Read a summary CSV and return list of row dicts."""
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows that didn't complete
            status = row.get("status", "")
            if status and status != "completed":
                continue
            row["_source_file"] = str(csv_path)
            rows.append(row)
    return rows


def safe_float(value, default=None):
    """Try to convert value to float."""
    if value is None or value == "" or value == "None":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def format_float(value, decimals=4):
    """Format a float value for display."""
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def sort_rows(rows, sort_by="best_miou", descending=True):
    """Sort rows by a given column."""
    def key_fn(row):
        val = safe_float(row.get(sort_by))
        if val is None:
            return float("-inf") if descending else float("inf")
        return val
    return sorted(rows, key=key_fn, reverse=descending)


def print_table(rows, columns=None, title=""):
    """Print a formatted table to stdout."""
    if not rows:
        print("No results found.")
        return

    # Determine which columns actually have data
    if columns is None:
        columns = DISPLAY_COLUMNS

    # Filter columns that exist in at least one row
    available_cols = []
    for col in columns:
        if any(row.get(col, "") != "" for row in rows):
            available_cols.append(col)

    if not available_cols:
        print("No displayable columns found.")
        return

    # Compute column widths
    col_widths = {}
    for col in available_cols:
        header_len = len(col)
        max_val_len = 0
        for row in rows:
            val = row.get(col, "")
            if col in FLOAT_COLUMNS:
                fval = safe_float(val)
                val_str = format_float(fval)
            else:
                val_str = str(val) if val else "-"
            max_val_len = max(max_val_len, len(val_str))
        col_widths[col] = max(header_len, max_val_len)

    # Print title
    if title:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")

    # Print header
    header = " | ".join(col.ljust(col_widths[col]) for col in available_cols)
    print(f"\n{header}")
    print("-+-".join("-" * col_widths[col] for col in available_cols))

    # Print rows
    for row in rows:
        cells = []
        for col in available_cols:
            val = row.get(col, "")
            if col in FLOAT_COLUMNS:
                fval = safe_float(val)
                val_str = format_float(fval)
            else:
                val_str = str(val) if val else "-"
            cells.append(val_str.ljust(col_widths[col]))
        print(" | ".join(cells))

    print(f"\nTotal: {len(rows)} experiments")


def export_csv(rows, output_path, columns=None):
    """Export rows to a CSV file."""
    if not rows:
        print("No data to export.")
        return

    if columns is None:
        # Use all keys from first row, excluding internal keys
        columns = [k for k in rows[0].keys() if not k.startswith("_")]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Exported {len(rows)} rows to: {output_path}")


def print_ranking(rows, metric="best_miou", top_n=10):
    """Print a ranking table by a specific metric."""
    valid_rows = [r for r in rows if safe_float(r.get(metric)) is not None]
    if not valid_rows:
        return

    ranked = sort_rows(valid_rows, sort_by=metric, descending=True)
    print(f"\n{'='*60}")
    print(f"  Top-{top_n} by {metric}")
    print(f"{'='*60}")
    print(f"{'Rank':>4} | {'Experiment':<45} | {metric:>12}")
    print(f"{'-'*4}-+-{'-'*45}-+-{'-'*12}")
    for i, row in enumerate(ranked[:top_n], 1):
        name = row.get("experiment_name", "?")[:45]
        val = safe_float(row.get(metric))
        print(f"{i:>4} | {name:<45} | {val:>12.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize training results from output directories."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Output directories or summary CSV files to scan.",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively search for summary CSVs.",
    )
    parser.add_argument(
        "--sort-by",
        default="best_miou",
        help="Column to sort results by (default: best_miou).",
    )
    parser.add_argument(
        "--descending",
        default=True,
        type=lambda x: x.lower() != "false",
        help="Sort descending (default: true).",
    )
    parser.add_argument(
        "--output", "-o",
        default="",
        help="Export results to a CSV file.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Show top-N ranking (default: 10).",
    )
    parser.add_argument(
        "--columns",
        default="",
        help="Comma-separated list of columns to display.",
    )
    parser.add_argument(
        "--all-columns",
        action="store_true",
        help="Display all available columns.",
    )
    args = parser.parse_args()

    # Find CSV files
    csv_files = find_summary_csvs(args.paths, recursive=args.recursive)
    if not csv_files:
        print(f"Error: No summary CSV files found in: {args.paths}")
        sys.exit(1)

    print(f"Found {len(csv_files)} summary file(s):")
    for f in csv_files:
        print(f"  - {f}")

    # Read all rows
    all_rows = []
    for csv_file in csv_files:
        rows = read_summary_csv(csv_file)
        all_rows.extend(rows)

    if not all_rows:
        print("No completed experiments found.")
        sys.exit(0)

    # Sort
    all_rows = sort_rows(all_rows, sort_by=args.sort_by, descending=args.descending)

    # Determine columns
    if args.all_columns:
        columns = [k for k in all_rows[0].keys() if not k.startswith("_")]
    elif args.columns:
        columns = [c.strip() for c in args.columns.split(",")]
    else:
        columns = None  # Use default DISPLAY_COLUMNS

    # Print table
    print_table(all_rows, columns=columns, title="Experiment Results Summary")

    # Print rankings
    print_ranking(all_rows, metric="best_miou", top_n=args.top)
    print_ranking(all_rows, metric="best_iou_fg", top_n=args.top)
    print_ranking(all_rows, metric="best_f1", top_n=args.top)

    # Check for foreground collapse
    collapsed = [r for r in all_rows if safe_float(r.get("final_iou_fg")) == 0.0]
    if collapsed:
        print(f"\n⚠️  Warning: {len(collapsed)}/{len(all_rows)} experiments have final IoU_fg = 0 (foreground collapse)")

    # Export
    if args.output:
        export_csv(all_rows, args.output)


if __name__ == "__main__":
    main()
