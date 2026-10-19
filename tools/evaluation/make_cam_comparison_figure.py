"""Compose publication-style CAM comparison sheets.

The script is intentionally inference-free.  It consumes the sample folders
written by ``cam_analysis.py`` and therefore cannot silently change model
predictions, sample selection or CAM settings while making a figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REQUIRED_FILES = ("rgb.png", "slope.png", "gt.png", "cam_overlay.png")

FONT_CANDIDATES = (
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose CAM comparison grids from frozen CAM outputs.")
    parser.add_argument(
        "--model-dir",
        action="append",
        required=True,
        help="Repeated MODEL_ID=CAM_OUTPUT_DIR entries, e.g. --model-dir M4=... --model-dir L1=...",
    )
    parser.add_argument("--sample-ids", default="", help="Comma-separated image filenames; default is the common manifest intersection.")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument(
        "--output",
        default="outputs_experiments/interpretability/cam/cam_comparison_rgb_slope.png",
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
        if not model_id or not raw_path.strip():
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
    result = {}
    for record in manifest.get("samples", []):
        image_name = str(record["image"])
        sample_dir = Path(record["sample_dir"])
        if not sample_dir.is_absolute():
            sample_dir = root / sample_dir
        if not sample_dir.is_dir():
            # Older or copied manifests may contain a path rooted at the
            # original machine; fall back to rank/name under this output root.
            rank = int(record.get("rank", 0))
            candidate = next(root.glob(f"samples/rank_{rank:03d}_*"), None)
            if candidate is not None:
                sample_dir = candidate
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


def labeled_image(path: Path, label: str, scale: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    font = load_font(max(28, 24 * scale))
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
    bbox = probe.multiline_textbbox((0, 0), str(label), font=font, spacing=2 * scale)
    text_height = bbox[3] - bbox[1]
    padding = max(10, 5 * scale)
    label_height = max(48, text_height + 2 * padding)
    canvas = Image.new("RGB", (image.width, image.height + label_height), "white")
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text((padding, padding // 2), label, fill="black", font=font, spacing=2 * scale)
    return canvas


def cam_target_label(sample_dir: Path, manifest: dict) -> str:
    """Return a readable per-sample CAM class label for the comparison header."""

    metadata_path = sample_dir / "metadata.json"
    metadata = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    role = metadata.get("cam_role")
    class_id = metadata.get("cam_class")
    if role:
        return str(role).capitalize()
    if class_id is None:
        class_id = manifest.get("cam_class")
    if class_id is None or class_id == "per_sample":
        return "CAM"
    try:
        class_id = int(class_id)
    except (TypeError, ValueError):
        return f"class {class_id}"
    if class_id == int(manifest.get("foreground_class_id", 1)):
        return "Foreground"
    if class_id == int(manifest.get("background_class_id", 0)):
        return "Background"
    return f"class {class_id}"


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
        raise RuntimeError("No common CAM samples found")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_ids = list(model_dirs)
    rows = []
    target_classes = {model_id: {} for model_id in model_ids}
    for image_name in images:
        columns = [
            labeled_image(maps[model_ids[0]][image_name] / "rgb.png", "RGB", args.scale),
            labeled_image(maps[model_ids[0]][image_name] / "slope.png", "Slope", args.scale),
            labeled_image(maps[model_ids[0]][image_name] / "gt.png", "GT", args.scale),
        ]
        for model_id in model_ids:
            sample_dir = maps[model_id][image_name]
            target_label = cam_target_label(sample_dir, manifests[model_id])
            target_classes[model_id][image_name] = target_label
            columns.append(labeled_image(sample_dir / "cam_overlay.png", f"{model_id}\nCAM", args.scale))
        row_width = sum(item.width for item in columns)
        row_height = max(item.height for item in columns)
        row_image = Image.new("RGB", (row_width, row_height), "white")
        x = 0
        for item in columns:
            row_image.paste(item, (x, 0))
            x += item.width
        rows.append(row_image)

    title_font = load_font(max(28, 20 * args.scale))
    title_probe = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
    title = f"CAM comparison | RGB | Slope | GT | {', '.join(model_ids)} CAM | samples: {len(images)}"
    title_bbox = title_probe.textbbox((0, 0), title, font=title_font)
    title_height = max(64, title_bbox[3] - title_bbox[1] + 2 * max(10, 5 * args.scale))
    sheet_width = max(row.width for row in rows)
    sheet_height = title_height + sum(row.height for row in rows)
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((max(10, 5 * args.scale), max(8, 4 * args.scale)), title, fill="black", font=title_font)
    y = title_height
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(output_path)

    manifest = {
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis": "cam_comparison_figure",
        "models": model_ids,
        "samples": images,
        "source_manifests": {model_id: str(model_dirs[model_id] / "cam_manifest.json") for model_id in model_ids},
        "target_classes": target_classes,
        "output": str(output_path),
        "columns": ["RGB", "Slope", "GT"] + [f"{model_id} CAM" for model_id in model_ids],
        "legend": {
            "slope": "per-tile 2--98 percentile-scaled MMLSv2 Slope band",
            "gt": "ground-truth binary segmentation mask; white=foreground and black=background",
            "cam_overlay": "RGB preview blended with the model-specific Grad-CAM heatmap",
        },
    }
    save_json(output_path.with_name(output_path.stem + "_manifest.json"), manifest)
    print(json.dumps({"status": "completed", "samples": len(images), "output": str(output_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
