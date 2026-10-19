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


def labeled_image(path: Path, label: str, scale: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    label_height = max(24, 13 * scale * (str(label).count("\n") + 1))
    canvas = Image.new("RGB", (image.width, image.height + label_height), "white")
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), label, fill="black", font=ImageFont.load_default())
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
    rows: list[Image.Image] = []
    for image_name in images:
        reference_dir = maps[model_ids[0]][image_name]
        columns = [
            labeled_image(reference_dir / "rgb.png", "RGB", args.scale),
            labeled_image(reference_dir / "slope.png", "Slope", args.scale),
            labeled_image(reference_dir / "gt.png", "GT", args.scale),
        ]
        for model_id in model_ids:
            columns.append(
                labeled_image(
                    maps[model_id][image_name] / "pred.png",
                    f"{model_id} Prediction",
                    args.scale,
                )
            )
        row_width = sum(item.width for item in columns)
        row_height = max(item.height for item in columns)
        row_image = Image.new("RGB", (row_width, row_height), "white")
        x = 0
        for item in columns:
            row_image.paste(item, (x, 0))
            x += item.width
        rows.append(row_image)

    title_height = 30
    sheet_width = max(row.width for row in rows)
    sheet_height = title_height + sum(row.height for row in rows)
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    title = f"Prediction comparison | RGB | Slope | GT | {', '.join(model_ids)} Prediction | samples: {len(images)}"
    draw.text((4, 8), title, fill="black", font=ImageFont.load_default())
    y = title_height
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(output_path)

    manifest = {
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis": "prediction_comparison_figure",
        "models": model_ids,
        "samples": images,
        "source_manifests": {
            model_id: str(model_dirs[model_id] / "cam_manifest.json") for model_id in model_ids
        },
        "output": str(output_path),
        "columns": ["RGB", "Slope", "GT"] + [f"{model_id} Prediction" for model_id in model_ids],
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
