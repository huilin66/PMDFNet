"""Map the public MMLSv2 release to the local Mars-LS representation.

The two local releases contain the same patch names, but use different channel
orders and numeric encodings.  This script keeps the source dataset untouched
and writes a new, competition-style copy:

    MMLSv2: [R, G, B, DEM, Slope, Thermal, Grayscale], values in [0, 1]
    Mars-LS: [Thermal, Slope, DEM, Grayscale, R, G, B], raw-like DN values

The affine conversion constants were calibrated from the paired train/val
images available in the workspace.  The slope channel has a small number of
border pixels where the public release stores zero while the local updateB2
copy contains a non-zero value; the verifier reports this explicitly.

Examples (PowerShell):

    python tools/data/map_mmlsv2_to_mars.py `
      --source-root Z:\\huilin\\bdd\\cp_data\\mmlsv2 `
      --output-root Z:\\huilin\\bdd\\cp_data\\mmlsv2_mapped_mars_ls `
      --reference-root Z:\\huilin\\bdd\\cp_data\\mars_seg\\Mars_LSc_2025_dataset_1st_phase_updateB2

Use ``--verify-only`` to run the comparison without writing output files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

# OpenCV does not interpret several GeoTIFF metadata tags used by the source
# files.  Keep those non-fatal TIFF warnings quiet unless the caller chose a
# different OpenCV log level explicitly.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np
import tifffile

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - requirements.txt includes tqdm
    def tqdm(iterable, **_kwargs):
        return iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env_utils import get_env_value


# Output order is the historical competition order used by dataset.py.
# For every output channel, this gives the input MMLSv2 channel index.
MMLS_TO_MARS_CHANNELS = (5, 4, 3, 6, 0, 1, 2)

# raw_value = normalized_value * scale + offset
MMLS_TO_MARS_SCALE = np.array(
    [254.0, 62.8026087, 12822.0, 254.0, 189.0, 213.0, 237.0],
    dtype=np.float64,
)
MMLS_TO_MARS_OFFSET = np.array(
    [0.0, 0.0, -5834.0, 0.0, 18.0, 18.0, 18.0],
    dtype=np.float64,
)


def _tif_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.tif") if p.is_file())


def read_image(path: Path) -> np.ndarray:
    image = tifffile.imread(str(path)).astype(np.float32, copy=False)
    if image.ndim == 3 and image.shape[-1] == 7:
        return image
    if image.ndim == 3 and image.shape[0] == 7:
        return np.moveaxis(image, 0, -1)
    raise ValueError(f"Expected a seven-channel TIFF, got {path}: {image.shape}")


def read_mask(path: Path) -> np.ndarray:
    # MMLSv2 masks are LZW-compressed.  OpenCV can read them without requiring
    # imagecodecs, while the output is written uncompressed for portability.
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Unable to read mask: {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.uint8, copy=False)


def map_image(image: np.ndarray) -> np.ndarray:
    ordered = image[..., list(MMLS_TO_MARS_CHANNELS)].astype(np.float64)
    mapped = ordered * MMLS_TO_MARS_SCALE.reshape(1, 1, 7)
    mapped += MMLS_TO_MARS_OFFSET.reshape(1, 1, 7)
    return mapped.astype(np.float32)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # No compression keeps the generated files readable by tifffile without
    # the optional imagecodecs package and reproduces the local updateB2 layout.
    tifffile.imwrite(str(path), image.astype(np.float32, copy=False), compression=None)


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), mask.astype(np.uint8, copy=False), compression=None)


def _update_error(
    state: dict[str, np.ndarray | int], mapped: np.ndarray, reference: np.ndarray
) -> None:
    error = np.abs(mapped.astype(np.float64) - reference.astype(np.float64))
    state["images"] = int(state["images"]) + 1
    state["pixels"] = int(state["pixels"]) + error.shape[0] * error.shape[1]
    state["sum_abs"] = np.asarray(state["sum_abs"]) + error.sum(axis=(0, 1))
    state["max_abs"] = np.maximum(np.asarray(state["max_abs"]), error.max(axis=(0, 1)))
    state["within_1e-3"] = np.asarray(state["within_1e-3"]) + (
        error <= 1e-3
    ).sum(axis=(0, 1))


def _new_error_state() -> dict[str, np.ndarray | int]:
    return {
        "images": 0,
        "pixels": 0,
        "sum_abs": np.zeros(7, dtype=np.float64),
        "max_abs": np.zeros(7, dtype=np.float64),
        "within_1e-3": np.zeros(7, dtype=np.int64),
    }


def _finalize_error_state(state: dict[str, np.ndarray | int]) -> dict[str, object]:
    pixels = max(int(state["pixels"]), 1)
    return {
        "images": int(state["images"]),
        "mean_abs_by_channel": (
            np.asarray(state["sum_abs"]) / pixels
        ).round(8).tolist(),
        "max_abs_by_channel": np.asarray(state["max_abs"]).round(6).tolist(),
        "fraction_within_1e-3_by_channel": (
            np.asarray(state["within_1e-3"]) / pixels
        ).round(8).tolist(),
    }


def _mask_summary(source: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    diff = source != reference
    return {
        "pixels": int(source.size),
        "different_pixels": int(diff.sum()),
        "exact_fraction": float((~diff).mean()),
        "max_abs_difference": int(np.abs(source.astype(np.int16) - reference.astype(np.int16)).max()),
    }


def map_dataset(
    source_root: Path,
    output_root: Path,
    reference_root: Path | None,
    splits: Iterable[str],
    verify_only: bool,
    include_test_masks: bool,
    overwrite: bool,
) -> dict[str, object]:
    report: dict[str, object] = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "reference_root": str(reference_root) if reference_root else None,
        "input_channel_order": ["R", "G", "B", "DEM", "Slope", "Thermal", "Grayscale"],
        "output_channel_order": ["Thermal", "Slope", "DEM", "Grayscale", "R", "G", "B"],
        "mmls_to_mars_channels": list(MMLS_TO_MARS_CHANNELS),
        "scale": MMLS_TO_MARS_SCALE.tolist(),
        "offset": MMLS_TO_MARS_OFFSET.tolist(),
        "splits": {},
    }

    for split in splits:
        source_images = source_root / split / "images"
        source_masks = source_root / split / "masks"
        image_paths = _tif_files(source_images)
        if not image_paths:
            raise FileNotFoundError(f"No TIFF images found in {source_images}")

        target_image_dir = output_root / split / "images"
        target_mask_dir = output_root / split / "masks"
        reference_image_dir = reference_root / split / "images" if reference_root else None
        reference_mask_dir = reference_root / split / "masks" if reference_root else None

        error_state = _new_error_state()
        mask_diffs = 0
        mask_compared = 0
        image_count = 0

        for source_image_path in tqdm(
            image_paths,
            desc=f"Map {split}",
            unit="image",
        ):
            name = source_image_path.name
            mapped = map_image(read_image(source_image_path))

            if reference_image_dir is not None:
                reference_path = reference_image_dir / name
                if reference_path.exists():
                    _update_error(error_state, mapped, read_image(reference_path))

            if not verify_only:
                target_path = target_image_dir / name
                if target_path.exists() and not overwrite:
                    raise FileExistsError(
                        f"Target exists; use --overwrite or choose another output root: {target_path}"
                    )
                write_image(target_path, mapped)

            source_mask_path = source_masks / name
            should_copy_mask = split != "test" or include_test_masks
            if should_copy_mask and source_mask_path.exists():
                mask = read_mask(source_mask_path)
                if reference_mask_dir is not None:
                    reference_mask_path = reference_mask_dir / name
                    if reference_mask_path.exists():
                        reference_mask = read_mask(reference_mask_path)
                        summary = _mask_summary(mask, reference_mask)
                        mask_compared += 1
                        mask_diffs += int(summary["different_pixels"])
                if not verify_only:
                    target_mask_path = target_mask_dir / name
                    if target_mask_path.exists() and not overwrite:
                        raise FileExistsError(
                            f"Target exists; use --overwrite or choose another output root: {target_mask_path}"
                        )
                    write_mask(target_mask_path, mask)

            image_count += 1

        split_report: dict[str, object] = {"images": image_count}
        if int(error_state["images"]) > 0:
            split_report["image_verification"] = _finalize_error_state(error_state)
        if mask_compared:
            split_report["mask_verification"] = {
                "images": mask_compared,
                "different_pixels_total": mask_diffs,
            }
        report["splits"][split] = split_report  # type: ignore[index]

    if not verify_only:
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "mapping_manifest.json"
        manifest_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        report["manifest"] = str(manifest_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default=get_env_value(
            "MMLSV2_DATA_ROOT", "MARS_DATA_ROOT", "DATA_ROOT",
            default=r"Z:\huilin\bdd\cp_data\mmlsv2",
        ),
        help="MMLSv2 root containing train/val/test.",
    )
    parser.add_argument(
        "--output-root",
        default=r"Z:\huilin\bdd\cp_data\mmlsv2_mapped_mars_ls",
        help="New directory for the mapped dataset.",
    )
    parser.add_argument(
        "--reference-root",
        default=r"Z:\huilin\bdd\cp_data\mars_seg\Mars_LSc_2025_dataset_1st_phase_updateB2",
        help="Optional local competition copy used for pixel-level verification.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the mapping without creating output files.",
    )
    parser.add_argument(
        "--test-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include public MMLSv2 test masks in the output (default: yes).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files in an existing output directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = map_dataset(
        source_root=Path(args.source_root),
        output_root=Path(args.output_root),
        reference_root=Path(args.reference_root) if args.reference_root else None,
        splits=args.splits,
        verify_only=args.verify_only,
        include_test_masks=args.test_masks,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
