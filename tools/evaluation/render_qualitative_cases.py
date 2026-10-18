"""Render deterministic Test qualitative panels for selected failure cases."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.evaluate_robustness import PerturbedTestDataset  # noqa: E402
from tools.evaluation.evaluate_test_metrics import (  # noqa: E402
    build_evaluation_model,
    load_checkpoint,
)
from tools.evaluation.measure_model_complexity import _as_first_tensor  # noqa: E402
from tools.evaluation.paper_run_utils import (  # noqa: E402
    load_protocol,
    resolve_formal_run,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render RGB/GT/prediction/FP-FN panels for deterministic failure cases."
    )
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-root", default="outputs_experiments/pmdfnet_mmlsv2")
    parser.add_argument("--config", default="configs/mmlsv2_paper_protocol.json")
    parser.add_argument("--reference-id", default="M4")
    parser.add_argument("--candidate-id", default="L1")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--output-dir", default="outputs_experiments/robustness/qualitative")
    parser.add_argument("--max-cases", type=int, default=0)
    return parser.parse_args(argv)


def to_chw(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] == 7:
        image = image.transpose(2, 0, 1)
    if image.ndim != 3 or image.shape[0] != 7:
        raise ValueError(f"Expected 7-channel TIFF, got {image.shape}")
    return image.astype(np.float32, copy=False)


def rgb_preview(image: np.ndarray) -> np.ndarray:
    channels = to_chw(image)[[4, 5, 6]].copy()
    output = np.zeros_like(channels, dtype=np.float32)
    for index, channel in enumerate(channels):
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            continue
        low, high = np.percentile(finite, [2.0, 98.0])
        if high <= low:
            high = low + 1.0
        output[index] = np.clip((channel - low) / (high - low), 0.0, 1.0)
    return (output.transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)


def binary_preview(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask) > 0
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    output[mask] = (255, 255, 255)
    return output


def fp_fn_preview(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction) > 0
    target = np.asarray(target) > 0
    false_positive = prediction & ~target
    false_negative = ~prediction & target
    true_positive = prediction & target
    output = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    output[true_positive] = (80, 80, 80)
    output[false_positive] = (230, 40, 40)
    output[false_negative] = (40, 80, 230)
    return output


def panel(array: np.ndarray, label: str, scale: int = 2) -> Image.Image:
    image = Image.fromarray(array, mode="RGB")
    image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    label_height = 24
    canvas = Image.new("RGB", (image.width, image.height + label_height), "white")
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 5), label, fill="black", font=ImageFont.load_default())
    return canvas


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)


def load_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("selected_unique_cases", [])
    if not cases:
        raise ValueError(f"No selected_unique_cases in {path}")
    return cases


def main(argv=None) -> int:
    args = parse_args(argv)
    cases_path = Path(args.cases_json).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(cases_path)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    paper_protocol = load_protocol(Path(args.config).expanduser().resolve())
    seed = int(cases[0]["seed"])
    if any(int(case["seed"]) != seed for case in cases):
        raise ValueError("Selected qualitative cases must use one seed")
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device} but CUDA is unavailable")

    model_ids = [args.reference_id, args.candidate_id]
    models = {}
    datasets = {}
    model_metadata = {}
    for model_id in model_ids:
        run = resolve_formal_run(args.run_root, paper_protocol, model_id, seed)
        model_args = argparse.Namespace(run_config=run["config_path"], model_name="")
        model, resolved_name, config_path, input_channels = build_evaluation_model(model_args)
        model = model.to(device)
        load_checkpoint(model, Path(run["checkpoint_path"]), device)
        model.eval()
        condition = {
            "id": "clean",
            "kind": "clean",
            "channels": [],
            "level_name": "clean",
            "level_index": 0,
            "value": 0.0,
        }
        dataset = PerturbedTestDataset(
            data_root,
            condition,
            condition_index=0,
            random_seed=0,
            input_channels=input_channels,
        )
        models[model_id] = model
        datasets[model_id] = dataset
        model_metadata[model_id] = {
            "stored_id": run["stored_id"],
            "checkpoint": run["checkpoint_path"],
            "run_config": config_path,
            "input_channels": input_channels,
            "experiment_name": resolved_name,
        }

    index_by_name = {path.name: index for index, path in enumerate(datasets[args.reference_id].image_files)}
    rendered = []
    with torch.inference_mode():
        for case in cases:
            image_name = str(case["image"])
            if image_name not in index_by_name:
                raise FileNotFoundError(f"Selected image not found in mapped Test set: {image_name}")
            index = index_by_name[image_name]
            reference_dataset = datasets[args.reference_id]
            source = tifffile.imread(str(reference_dataset.image_files[index]))
            target = reference_dataset[index][1].numpy()
            predictions = {}
            for model_id in model_ids:
                image_tensor, _, returned_name = datasets[model_id][index]
                if returned_name != image_name:
                    raise ValueError(
                        f"Dataset ordering mismatch for {model_id}: {returned_name} vs {image_name}"
                    )
                output = _as_first_tensor(models[model_id](image_tensor.unsqueeze(0).to(device)))
                predictions[model_id] = torch.argmax(output, dim=1)[0].cpu().numpy()

            reference_prediction = predictions[args.reference_id]
            candidate_prediction = predictions[args.candidate_id]
            panels = [
                panel(rgb_preview(source), "RGB"),
                panel(binary_preview(target), "GT"),
                panel(binary_preview(reference_prediction), args.reference_id),
                panel(binary_preview(candidate_prediction), args.candidate_id),
                panel(fp_fn_preview(reference_prediction, target), f"{args.reference_id} FP/FN"),
                panel(fp_fn_preview(candidate_prediction, target), f"{args.candidate_id} FP/FN"),
            ]
            sheet_width = sum(item.width for item in panels)
            sheet_height = max(item.height for item in panels) + 38
            sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
            draw = ImageDraw.Draw(sheet)
            title = (
                f"{image_name} | seed={seed} | "
                f"{args.reference_id} mIoU={case['reference_miou']:.4f}, "
                f"{args.candidate_id} mIoU={case['candidate_miou']:.4f} | "
                f"error delta={case['candidate_minus_reference_error_pixels']:+d}"
            )
            draw.text((4, 8), title, fill="black", font=ImageFont.load_default())
            x = 0
            for item in panels:
                sheet.paste(item, (x, 34))
                x += item.width
            output_path = output_dir / f"{safe_name(image_name.rsplit('.', 1)[0])}.png"
            sheet.save(output_path)
            rendered.append(
                {
                    "image": image_name,
                    "output": str(output_path),
                    "reference_id": args.reference_id,
                    "candidate_id": args.candidate_id,
                    "reference_miou": case["reference_miou"],
                    "candidate_miou": case["candidate_miou"],
                    "reference_fp": case["reference_fp"],
                    "candidate_fp": case["candidate_fp"],
                    "reference_fn": case["reference_fn"],
                    "candidate_fn": case["candidate_fn"],
                    "reference_error_pixels": case["reference_error_pixels"],
                    "candidate_error_pixels": case["candidate_error_pixels"],
                }
            )

    manifest = {
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_json": str(cases_path),
        "data_root": str(data_root),
        "seed": seed,
        "condition": "clean",
        "device": str(device),
        "models": model_metadata,
        "legend": {
            "FP/FN panel": "red= false positive, blue= false negative, gray= true positive",
            "RGB": "raw mapped Test TIFF channels R/G/B with per-image 2-98 percentile display scaling",
        },
        "rendered_cases": rendered,
    }
    (output_dir / "qualitative_cases.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "rendered_cases": len(rendered),
                "output_dir": str(output_dir),
                "manifest": str(output_dir / "qualitative_cases.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
