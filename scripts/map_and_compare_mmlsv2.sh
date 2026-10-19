#!/usr/bin/env bash
set -euo pipefail

# Map MMLSv2 to the Mars-LS representation and compare it with MARS_LSC_ROOT.
# The three roots are loaded from the repository .env through _env.sh:
#   MMLSV2_SRC_DATA_ROOT  source MMLSv2 root
#   MMLSV2_DATA_ROOT      mapped output root
#   MARS_LSC_ROOT         reference Mars-LS root
#
# Optional overrides:
#   PYTHON_BIN=python
#   SPLITS="train val test"
#   ATOL=1e-3
#   OVERWRITE=1
#   IGNORE_MASKS=1
#
# Compare only, without running the mapping step:
#   bash scripts/map_and_compare_mmlsv2.sh --compare-only

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

. "${SCRIPT_DIR}/_env.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLITS="${SPLITS:-train val test}"
ATOL="${ATOL:-1e-3}"
OVERWRITE="${OVERWRITE:-0}"
IGNORE_MASKS="${IGNORE_MASKS:-0}"
COMPARE_ONLY="${COMPARE_ONLY:-0}"

usage() {
  cat <<'EOF'
Usage: bash scripts/map_and_compare_mmlsv2.sh [options]

Options:
  --compare-only       Skip mapping and compare existing MMLSV2_DATA_ROOT only.
  --overwrite          Allow the mapping step to overwrite existing TIFF files.
  --ignore-masks       Compare images only.
  --atol VALUE         Pixel absolute tolerance; default is 1e-3.
  -h, --help           Show this help.

Environment overrides:
  SPLITS="train val test"  PYTHON_BIN=python  OVERWRITE=1  IGNORE_MASKS=1
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --compare-only)
      COMPARE_ONLY=1
      ;;
    --overwrite)
      OVERWRITE=1
      ;;
    --ignore-masks)
      IGNORE_MASKS=1
      ;;
    --atol)
      if [ "$#" -lt 2 ]; then
        echo "--atol requires a value" >&2
        exit 2
      fi
      ATOL="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

# OpenCV emits warnings for GeoTIFF metadata tags that it does not interpret.
# They do not affect the pixel data used by this mapping workflow.
export OPENCV_LOG_LEVEL=ERROR

: "${MMLSV2_SRC_DATA_ROOT:?MMLSV2_SRC_DATA_ROOT is not set in .env}"
: "${MMLSV2_DATA_ROOT:?MMLSV2_DATA_ROOT is not set in .env}"
: "${MARS_LSC_ROOT:?MARS_LSC_ROOT is not set in .env}"

if [ ! -d "${MMLSV2_SRC_DATA_ROOT}" ]; then
  echo "Source dataset does not exist: ${MMLSV2_SRC_DATA_ROOT}" >&2
  exit 1
fi
if [ ! -d "${MARS_LSC_ROOT}" ]; then
  echo "Reference dataset does not exist: ${MARS_LSC_ROOT}" >&2
  exit 1
fi
if [ "${MMLSV2_SRC_DATA_ROOT}" = "${MMLSV2_DATA_ROOT}" ]; then
  echo "MMLSV2_DATA_ROOT must differ from MMLSV2_SRC_DATA_ROOT" >&2
  exit 1
fi
if [ "${MARS_LSC_ROOT}" = "${MMLSV2_DATA_ROOT}" ]; then
  echo "MMLSV2_DATA_ROOT must differ from MARS_LSC_ROOT" >&2
  exit 1
fi

read -r -a SPLIT_LIST <<< "${SPLITS}"
if [ "${#SPLIT_LIST[@]}" -eq 0 ]; then
  echo "SPLITS must contain at least one split" >&2
  exit 1
fi

MAP_ARGS=(
  --source-root "${MMLSV2_SRC_DATA_ROOT}"
  --output-root "${MMLSV2_DATA_ROOT}"
  --reference-root "${MARS_LSC_ROOT}"
  --splits "${SPLIT_LIST[@]}"
  --test-masks
)
if [ "${OVERWRITE}" = "1" ] || [ "${OVERWRITE}" = "true" ] || [ "${OVERWRITE}" = "TRUE" ]; then
  MAP_ARGS+=(--overwrite)
fi

echo "[map] source=${MMLSV2_SRC_DATA_ROOT}"
echo "[map] output=${MMLSV2_DATA_ROOT}"
echo "[map] reference=${MARS_LSC_ROOT}"
echo "[map] splits=${SPLITS} overwrite=${OVERWRITE} compare_only=${COMPARE_ONLY}"

if [ "${COMPARE_ONLY}" = "1" ] || [ "${COMPARE_ONLY}" = "true" ] || [ "${COMPARE_ONLY}" = "TRUE" ]; then
  if [ ! -d "${MMLSV2_DATA_ROOT}" ]; then
    echo "Mapped dataset does not exist: ${MMLSV2_DATA_ROOT}" >&2
    exit 1
  fi
  echo "[compare] skipping mapping step"
else
  "${PYTHON_BIN}" tools/data/map_mmlsv2_to_mars.py "${MAP_ARGS[@]}"
fi

# The mapping tool above performs the conversion.  The following small Python
# block compares the materialized output with the reference TIFF files.
"${PYTHON_BIN}" - "${MMLSV2_DATA_ROOT}" "${MARS_LSC_ROOT}" "${ATOL}" "${IGNORE_MASKS}" "${SPLIT_LIST[@]}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm


mapped_root = Path(sys.argv[1]).resolve()
reference_root = Path(sys.argv[2]).resolve()
atol = float(sys.argv[3])
ignore_masks = sys.argv[4].strip().lower() in {"1", "true", "yes", "on"}
splits = sys.argv[5:]
BAND_NAMES = ["Thermal", "Slope", "DEM", "Grayscale", "R", "G", "B"]

if atol < 0:
    raise SystemExit("ATOL must be non-negative")


def tif_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    result = {}
    for pattern in ("*.tif", "*.tiff"):
        for path in directory.glob(pattern):
            result[path.name] = path
    return result


def as_bands(array):
    """Return seven bands in HxW form for either HxWx7 or 7xHxW TIFFs."""

    array = np.asarray(array)
    if array.ndim != 3:
        return None
    if array.shape[-1] == len(BAND_NAMES):
        return [array[..., index] for index in range(len(BAND_NAMES))]
    if array.shape[0] == len(BAND_NAMES):
        return [array[index, ...] for index in range(len(BAND_NAMES))]
    return None


def new_band_stats():
    return [
        {
            "index": index,
            "name": name,
            "compared_files": 0,
            "shape_mismatch_files": 0,
            "total_values": 0,
            "exact_values": 0,
            "within_atol_values": 0,
            "absolute_error_sum": 0.0,
            "max_absolute_error": None,
        }
        for index, name in enumerate(BAND_NAMES)
    ]


def update_band_stats(stats, left, right, tolerance):
    left_bands = as_bands(left)
    right_bands = as_bands(right)
    if left_bands is None or right_bands is None:
        return

    for index, (left_band, right_band) in enumerate(zip(left_bands, right_bands)):
        item = stats[index]
        item["compared_files"] += 1
        if left_band.shape != right_band.shape:
            item["shape_mismatch_files"] += 1
            continue

        diff = np.abs(left_band.astype(np.float64) - right_band.astype(np.float64))
        item["total_values"] += int(diff.size)
        item["exact_values"] += int(np.count_nonzero(diff == 0))
        item["within_atol_values"] += int(np.count_nonzero(diff <= tolerance))
        item["absolute_error_sum"] += float(diff.sum())
        max_error = float(np.max(diff)) if diff.size else 0.0
        if item["max_absolute_error"] is None:
            item["max_absolute_error"] = max_error
        else:
            item["max_absolute_error"] = max(item["max_absolute_error"], max_error)


def finalize_band_stats(stats):
    finalized = []
    for item in stats:
        total = item["total_values"]
        exact_fraction = item["exact_values"] / total if total else 0.0
        within_fraction = item["within_atol_values"] / total if total else 0.0
        finalized.append(
            {
                "index": item["index"],
                "name": item["name"],
                "compared_files": item["compared_files"],
                "shape_mismatch_files": item["shape_mismatch_files"],
                "total_values": total,
                "exact_values": item["exact_values"],
                "within_atol_values": item["within_atol_values"],
                "exact_fraction": exact_fraction,
                "within_atol_fraction": within_fraction,
                "mean_absolute_error": (
                    item["absolute_error_sum"] / total if total else None
                ),
                "max_absolute_error": item["max_absolute_error"],
                "exact_match": bool(
                    total > 0
                    and item["exact_values"] == total
                    and item["shape_mismatch_files"] == 0
                ),
                "within_atol_match": bool(
                    total > 0
                    and item["within_atol_values"] == total
                    and item["shape_mismatch_files"] == 0
                ),
            }
        )
    return finalized


def compare_file_set(mapped_dir: Path, reference_dir: Path, tolerance: float):
    mapped = tif_files(mapped_dir)
    reference = tif_files(reference_dir)
    mapped_names = set(mapped)
    reference_names = set(reference)
    common_names = sorted(mapped_names & reference_names)
    exact_count = 0
    close_count = 0
    shape_mismatch = 0
    mean_errors = []
    max_errors = []
    mismatches = []
    band_stats = new_band_stats()
    total_values = 0
    within_atol_values = 0

    for name in tqdm(
        common_names,
        desc=f"Compare {mapped_dir.parent.name}/{mapped_dir.name}",
        unit="file",
    ):
        left = np.asarray(tifffile.imread(str(mapped[name])))
        right = np.asarray(tifffile.imread(str(reference[name])))
        update_band_stats(band_stats, left, right, tolerance)
        if left.shape != right.shape:
            shape_mismatch += 1
            mismatches.append({
                "name": name,
                "shape_left": list(left.shape),
                "shape_right": list(right.shape),
            })
            continue

        diff = np.abs(left.astype(np.float64) - right.astype(np.float64))
        total_values += int(diff.size)
        within_atol_values += int(np.count_nonzero(diff <= tolerance))
        exact = bool(np.all(diff == 0))
        close = bool(np.all(diff <= tolerance))
        exact_count += int(exact)
        close_count += int(close)
        mean_errors.append(float(np.mean(diff)))
        max_errors.append(float(np.max(diff)))
        if not close:
            mismatches.append({
                "name": name,
                "shape": list(left.shape),
                "dtype_left": str(left.dtype),
                "dtype_right": str(right.dtype),
                "exact_fraction": float(np.mean(diff == 0)),
                "within_atol_fraction": float(np.mean(diff <= tolerance)),
                "mean_absolute_error": float(np.mean(diff)),
                "max_absolute_error": float(np.max(diff)),
            })

    report = {
        "mapped_count": len(mapped_names),
        "reference_count": len(reference_names),
        "common_count": len(common_names),
        "missing_in_mapped": sorted(reference_names - mapped_names),
        "extra_in_mapped": sorted(mapped_names - reference_names),
        "compared_count": len(common_names),
        "exact_match_count": exact_count,
        "within_atol_count": close_count,
        "shape_mismatch_count": shape_mismatch,
        "exact_match": (
            mapped_names == reference_names
            and exact_count == len(common_names)
            and shape_mismatch == 0
        ),
        "within_atol_match": (
            mapped_names == reference_names
            and close_count == len(common_names)
            and shape_mismatch == 0
        ),
        "mean_absolute_error_per_file": (
            float(np.mean(mean_errors)) if mean_errors else None
        ),
        "max_absolute_error": max(max_errors) if max_errors else None,
        "total_values": total_values,
        "within_atol_values": within_atol_values,
        "pixel_match_percentage": (
            within_atol_values / total_values * 100.0
            if total_values
            else None
        ),
        "mismatches": mismatches,
        "per_band": finalize_band_stats(band_stats),
    }
    return report


def aggregate_band_stats(split_reports):
    aggregate = new_band_stats()
    for split_report in split_reports.values():
        for item in split_report["images"].get("per_band", []):
            target = aggregate[item["index"]]
            target["compared_files"] += item["compared_files"]
            target["shape_mismatch_files"] += item["shape_mismatch_files"]
            target["total_values"] += item["total_values"]
            target["exact_values"] += item["exact_values"]
            target["within_atol_values"] += item["within_atol_values"]
            if item["mean_absolute_error"] is not None:
                target["absolute_error_sum"] += (
                    item["mean_absolute_error"] * item["total_values"]
                )
            if item["max_absolute_error"] is not None:
                if target["max_absolute_error"] is None:
                    target["max_absolute_error"] = item["max_absolute_error"]
                else:
                    target["max_absolute_error"] = max(
                        target["max_absolute_error"], item["max_absolute_error"]
                    )
    return finalize_band_stats(aggregate)


def print_match_table(split_reports, tolerance, ignore_masks):
    print(f"=== Pixel match percentage (absolute error <= {tolerance:g}) ===")
    print(
        f"{'Split':<8}"
        + "".join(f"{name:>12}" for name in BAND_NAMES)
        + f"{'Mask':>12}"
    )
    print("-" * (8 + 12 * (len(BAND_NAMES) + 1)))

    for split, split_report in split_reports.items():
        percentages = []
        for item in split_report["images"].get("per_band", []):
            total = item["total_values"]
            percentage = (
                item["within_atol_values"] / total * 100.0
                if total
                else None
            )
            percentages.append("N/A" if percentage is None else f"{percentage:.4f}%")

        mask_percentage = None
        if not ignore_masks and split_report["masks"].get("status") == "checked":
            mask_percentage = split_report["masks"]["files"].get(
                "pixel_match_percentage"
            )
        percentages.append(
            "N/A"
            if mask_percentage is None
            else f"{mask_percentage:.4f}%"
        )
        print(
            f"{split:<8}"
            + "".join(f"{percentage:>12}" for percentage in percentages)
        )


split_reports = {}
for split in splits:
    image_report = compare_file_set(
        mapped_root / split / "images",
        reference_root / split / "images",
        atol,
    )
    split_report = {
        "images": image_report,
        "images_match": bool(image_report["within_atol_match"]),
    }

    if ignore_masks:
        split_report["masks"] = {"status": "ignored"}
        split_report["masks_match"] = None
    else:
        mapped_masks = tif_files(mapped_root / split / "masks")
        reference_masks = tif_files(reference_root / split / "masks")
        if not mapped_masks and not reference_masks:
            split_report["masks"] = {"status": "not_available"}
            split_report["masks_match"] = None
        elif not reference_masks:
            # Competition test labels may be hidden/unavailable while MMLSv2
            # provides public masks.  Do not treat those extra public labels
            # as an image mapping failure.
            split_report["masks"] = {
                "status": "reference_not_available",
                "mapped_count": len(mapped_masks),
                "reference_count": 0,
            }
            split_report["masks_match"] = None
        else:
            mask_report = compare_file_set(
                mapped_root / split / "masks",
                reference_root / split / "masks",
                atol,
            )
            split_report["masks"] = {
                "status": "checked",
                "files": mask_report,
            }
            split_report["masks_match"] = bool(mask_report["within_atol_match"])

    split_report["complete_match"] = bool(
        split_report["images_match"]
        and split_report["masks_match"] is not False
    )
    split_reports[split] = split_report

image_complete = all(item["images_match"] for item in split_reports.values())
checked_masks = [
    item["masks_match"]
    for item in split_reports.values()
    if item["masks_match"] is not None
]
masks_complete = all(checked_masks) if checked_masks else None
complete = bool(image_complete and masks_complete is not False)
per_band = aggregate_band_stats(split_reports)

report = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "mapped_root": str(mapped_root),
    "reference_root": str(reference_root),
    "splits": splits,
    "atol": atol,
    "ignore_masks": ignore_masks,
    "split_reports": split_reports,
    "per_band": per_band,
    "images_complete_match": image_complete,
    "masks_complete_match": masks_complete,
    "complete_match": complete,
}
report_path = mapped_root / "mapping_match_report.json"
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)

print_match_table(split_reports, atol, ignore_masks)
print(f"Report: {report_path}")

raise SystemExit(0 if complete else 2)
PY
