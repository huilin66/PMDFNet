"""Compose a prediction comparison sheet from frozen interpretability outputs.

The figure is intentionally inference-free.  It reuses the RGB preview, the
Slope preview, and the binary prediction saved by ``cam_analysis.py`` for the
same performance-conditioned Test samples.  This keeps the instance-analysis
figure aligned with the CAM sample-selection manifest without changing model
predictions or selecting cases a second time.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REQUIRED_FILES = ("rgb.png", "slope.png", "gt.png", "pred.png")

FONT_CANDIDATES = (
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
)

HEADER_FONT_CANDIDATES = (
    "C:/Windows/Fonts/timesbi.ttf",
    "C:/Windows/Fonts/georgiaz.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSerif-BoldItalic.ttf",
)


METHOD_LABELS = {
    "M1": "U-Net",
    "M2": "DeepLabV3+",
    "M3": "SegFormer",
    "M4": "Single-stream ConvNeXt",
    "M5": "PMDFNet-Small",
    "L1": "PMDFNet-Small",
    "M6": "UPerNet",
    "M7": "FPN",
    "M8": "MANet",
    "M9": "U-Net++",
    "M10": "OCRNet",
    "M11": "M3LSNet",
}


def method_label(model_id: str) -> str:
    """Return the paper-facing method name while retaining IDs in metadata."""

    normalized = model_id.strip().upper()
    return METHOD_LABELS.get(normalized, model_id.strip())


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose RGB/Slope/model-prediction grids from frozen CAM outputs."
    )
    parser.add_argument(
        "--model-dir",
        action="append",
        required=True,
        help="Repeated MODEL_ID=CAM_OUTPUT_DIR entries, e.g. --model-dir M4=... --model-dir L1=...",
    )
    parser.add_argument(
        "--sample-ids",
        default="",
        help="Comma-separated image filenames; default is the common manifest intersection.",
    )
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument(
        "--output",
        default="outputs_experiments/interpretability/prediction_comparison_rgb_slope.png",
    )
    parser.add_argument("--scale", type=int, default=2)
    return parser.parse_args(argv)


def parse_model_dirs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--model-dir must be MODEL_ID=PATH, got {value!r}")
        model_id, raw_path = value.split("=", 1)
        model_id = model_id.strip()
        raw_path = raw_path.strip()
        if not model_id or not raw_path:
            raise ValueError(f"Invalid --model-dir: {value!r}")
        if model_id in result:
            raise ValueError(f"Duplicate model ID: {model_id}")
        result[model_id] = Path(raw_path).expanduser().resolve()
    if len(result) < 2:
        raise ValueError("At least two model directories are required for a comparison")
    return result


def load_manifest(root: Path) -> dict:
    path = root / "cam_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing CAM manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sample_map(root: Path, manifest: dict) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for record in manifest.get("samples", []):
        image_name = str(record["image"])
        sample_dir = Path(record["sample_dir"])
        if not sample_dir.is_absolute():
            sample_dir = root / sample_dir
        if not sample_dir.is_dir():
            # The manifest may have been generated on the remote server.  The
            # rank/name fallback makes copied output directories portable.
            rank = int(record.get("rank", 0))
            candidates = sorted(root.glob(f"samples/rank_{rank:03d}_*"))
            if candidates:
                sample_dir = candidates[0]
        if not sample_dir.is_dir():
            raise FileNotFoundError(f"Missing sample directory for {image_name}: {sample_dir}")
        missing = [name for name in REQUIRED_FILES if not (sample_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {missing} in {sample_dir}")
        result[image_name] = sample_dir
    return result


def common_images(maps: dict[str, dict[str, Path]]) -> list[str]:
    iterator = iter(maps.values())
    common = set(next(iterator))
    for mapping in iterator:
        common &= set(mapping)
    return sorted(common)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a readable bold font on both the A100 Linux host and Windows."""

    for raw_path in FONT_CANDIDATES:
        path = Path(raw_path)
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def load_header_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for raw_path in HEADER_FONT_CANDIDATES:
        path = Path(raw_path)
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return load_font(size)


def prepare_image(path: Path, scale: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    return image


def compose_grid(rows: list[list[Image.Image]], labels: list[str], scale: int) -> Image.Image:
    """Compose a single-header, white-gutter grid matching the reference style."""

    if not rows or not labels:
        raise ValueError("Cannot compose an empty comparison grid")
    if any(len(row) != len(labels) for row in rows):
        raise ValueError("Every grid row must have one image per header label")

    gutter = max(4, 2 * scale)
    border = max(2, scale)
    column_widths = [max(row[index].width for row in rows) for index in range(len(labels))]
    row_heights = [max(image.height for image in row) for row in rows]
    header_font = load_header_font(max(24, 18 * scale))
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
    text_heights = [
        probe.textbbox((0, 0), label, font=header_font)[3]
        - probe.textbbox((0, 0), label, font=header_font)[1]
        for label in labels
    ]
    header_height = max(46, max(text_heights) + 2 * max(8, 4 * scale))
    grid_width = sum(column_widths) + gutter * (len(labels) - 1)
    grid_height = sum(row_heights) + gutter * (len(rows) - 1)
    canvas = Image.new(
        "RGB",
        (2 * border + grid_width, border + header_height + gutter + grid_height + border),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    x = border
    for index, label in enumerate(labels):
        bbox = draw.textbbox((0, 0), label, font=header_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = x + (column_widths[index] - text_width) / 2 - bbox[0]
        text_y = border + (header_height - text_height) / 2 - bbox[1]
        draw.text((text_x, text_y), label, fill="black", font=header_font)
        x += column_widths[index] + gutter

    y = border + header_height + gutter
    for row, row_height in zip(rows, row_heights):
        x = border
        for index, image in enumerate(row):
            y_offset = (row_height - image.height) // 2
            x_offset = (column_widths[index] - image.width) // 2
            canvas.paste(image, (x + x_offset, y + y_offset))
            x += column_widths[index] + gutter
        y += row_height + gutter
    return canvas


def main(argv=None) -> int:
    args = parse_args(argv)
    model_dirs = parse_model_dirs(args.model_dir)
    manifests = {model_id: load_manifest(root) for model_id, root in model_dirs.items()}
    maps = {model_id: sample_map(root, manifests[model_id]) for model_id, root in model_dirs.items()}

    images = common_images(maps)
    if args.sample_ids.strip():
        requested = [item.strip() for item in args.sample_ids.split(",") if item.strip()]
        missing = [item for item in requested if item not in images]
        if missing:
            raise FileNotFoundError(f"Requested samples are not common to all models: {missing}")
        images = requested
    elif args.max_samples > 0:
        images = images[: args.max_samples]
    if not images:
        raise RuntimeError("No common prediction samples found")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_ids = list(model_dirs)
    rows: list[list[Image.Image]] = []
    for image_name in images:
        reference_dir = maps[model_ids[0]][image_name]
        columns = [
            prepare_image(reference_dir / "rgb.png", args.scale),
            prepare_image(reference_dir / "slope.png", args.scale),
        ]
        for model_id in model_ids:
            columns.append(
                prepare_image(
                    maps[model_id][image_name] / "pred.png",
                    args.scale,
                )
            )
        columns.append(prepare_image(reference_dir / "gt.png", args.scale))
        rows.append(columns)

    display_columns = ["RGB", "Slope"] + [method_label(model_id) for model_id in model_ids] + ["GT"]
    sheet = compose_grid(rows, display_columns, args.scale)
    sheet.save(output_path)

    manifest = {
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis": "prediction_comparison_figure",
        "models": model_ids,
        "method_labels": {model_id: method_label(model_id) for model_id in model_ids},
        "samples": images,
        "source_manifests": {
            model_id: str(model_dirs[model_id] / "cam_manifest.json") for model_id in model_ids
        },
        "output": str(output_path),
        "columns": ["RGB", "Slope"]
        + [f"{method_label(model_id)} Prediction" for model_id in model_ids]
        + ["GT"],
        "display_columns": display_columns,
        "layout": {
            "style": "single header row with white gutters",
            "repeat_labels_per_row": False,
            "header_explains_panels": "Method names label prediction panels; internal model IDs remain in the manifest.",
        },
        "legend": {
            "slope": "per-tile 2--98 percentile-scaled raw MMLSv2 Slope band (channel index 1)",
            "gt": "ground-truth binary segmentation mask; white=foreground and black=background",
            "prediction": "binary segmentation prediction; white=foreground and black=background",
            "selection": "same performance-conditioned PMDFNet advantage samples as the source CAM manifests",
        },
    }
    save_json(output_path.with_name(output_path.stem + "_manifest.json"), manifest)
    print(json.dumps({"status": "completed", "samples": len(images), "output": str(output_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
