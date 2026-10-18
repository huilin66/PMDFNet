"""Grad-CAM analysis for frozen PMDFNet checkpoints.

This is the PMDFNet/PyTorch counterpart of the MRSN CAM analysis workflow.
It produces one reproducible directory per selected image containing RGB, GT,
prediction, FP/FN and Grad-CAM views.  Model construction and checkpoint
loading are delegated to the formal evaluation code, so CAM is an analysis of
the trained paper model rather than a second model implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.evaluate_test_metrics import (  # noqa: E402
    build_evaluation_model,
    load_checkpoint,
)
from tools.evaluation.interpretability_common import (  # noqa: E402
    MappedLabeledDataset,
    SpatialHook,
    as_first_tensor,
    binary_metrics,
    binary_preview,
    find_spatial_layer,
    fp_fn_preview,
    rgb_preview,
    safe_name,
    save_json,
)
from tools.evaluation.paper_run_utils import (  # noqa: E402
    load_protocol,
    resolve_formal_run,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic Grad-CAM visualizations for a frozen PMDFNet run."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-config", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--run-root", default="outputs_experiments/pmdfnet_mmlsv2")
    parser.add_argument("--config", default="configs/mmlsv2_paper_protocol.json")
    parser.add_argument("--experiment-id", default="L1", help="Formal protocol/paper ID when run-config is omitted.")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--cam-layer", default="auto", help="Named module or auto decoder-feature selection.")
    parser.add_argument(
        "--cam-target",
        choices=("pred_mask", "gt_mask", "full_image", "error_region"),
        default="gt_mask",
        help="Spatial target used to pool the selected class logit.",
    )
    parser.add_argument("--class-id", type=int, default=1)
    parser.add_argument(
        "--selection",
        choices=("fixed", "all", "lowest_miou", "highest_miou"),
        default="fixed",
        help="fixed is the paper-safe selection rule; ranking modes are exploratory.",
    )
    parser.add_argument("--selection-seed", type=int, default=20261018)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--ids", default="", help="Comma-separated image filenames; overrides selection.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--resize",
        type=int,
        default=0,
        help="Optional square resize before inference; default follows the paper path (Test keeps source size, Val uses run input_size).",
    )
    return parser.parse_args(argv)


def resolve_run(args: argparse.Namespace) -> dict:
    if bool(args.run_config) != bool(args.checkpoint):
        raise ValueError("--run-config and --checkpoint must be supplied together")
    if args.run_config:
        config_path = Path(args.run_config).expanduser().resolve()
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        if not config_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing run config or checkpoint: {config_path}, {checkpoint_path}")
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        return {
            "paper_id": args.experiment_id,
            "seed": args.seed,
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "experiment_name": payload.get("experiment_name", config_path.parent.name),
        }
    protocol = load_protocol(Path(args.config).expanduser().resolve())
    return resolve_formal_run(args.run_root, protocol, args.experiment_id, args.seed)


def load_model(run: dict, device: torch.device):
    model_args = argparse.Namespace(run_config=run["config_path"], model_name="")
    model, resolved_name, config_path, input_channels = build_evaluation_model(model_args)
    model = model.to(device)
    load_checkpoint(model, Path(run["checkpoint_path"]), device)
    model.eval()
    config_payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return model, resolved_name, config_payload, input_channels


def config_resize(config_payload: dict, split: str, requested: int) -> int | None:
    if requested > 0:
        return requested
    if split == "val":
        return int(config_payload.get("input_size", 128))
    return None


def config_normalization(config_payload: dict, split: str) -> str:
    # Public Test evaluation is intentionally pinned to MARS Test statistics,
    # matching evaluate_test_metrics.py.  Train/Val follow the run config.
    if str(config_payload.get("dataset_type", "mars_ls")).lower() == "mmlsv2":
        return "none"
    if split == "test":
        return "test"
    return str(config_payload.get("normalization", "train"))


def choose_indices(
    dataset: MappedLabeledDataset,
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[int], dict]:
    name_to_index = {path.name: index for index, path in enumerate(dataset.image_files)}
    if args.ids.strip():
        names = [item.strip() for item in args.ids.split(",") if item.strip()]
        missing = [name for name in names if name not in name_to_index]
        if missing:
            raise FileNotFoundError(f"Requested image IDs are missing from {dataset.split}: {missing}")
        return [name_to_index[name] for name in names], {
            "rule": "explicit_image_ids",
            "seed": args.selection_seed,
            "top_k": len(names),
        }

    count = len(dataset)
    top_k = count if args.selection == "all" or args.top_k <= 0 else min(args.top_k, count)
    if args.selection == "fixed":
        rng = np.random.default_rng(args.selection_seed)
        indices = sorted(rng.choice(count, size=top_k, replace=False).tolist())
        return indices, {
            "rule": "fixed_seeded_sample_ids",
            "seed": args.selection_seed,
            "top_k": top_k,
            "model_independent": True,
        }

    scores: list[tuple[float, int]] = []
    with torch.inference_mode():
        for index in range(count):
            image, target, _ = dataset[index]
            logits = as_first_tensor(model(image.unsqueeze(0).to(device)))
            prediction = torch.argmax(logits, dim=1)[0].cpu().numpy()
            metric = binary_metrics(prediction, target.numpy())
            scores.append((float(metric["miou"]), index))
    reverse = args.selection == "highest_miou"
    scores.sort(key=lambda item: (item[0], item[1]), reverse=reverse)
    indices = [index for _, index in scores[:top_k]]
    return indices, {
        "rule": args.selection,
        "seed": args.selection_seed,
        "top_k": top_k,
        "model_independent": False,
        "warning": "ranking uses model predictions and is exploratory, not a frozen paper selection rule",
    }


def target_mask(
    mode: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    class_id: int,
) -> tuple[torch.Tensor, bool]:
    if mode == "pred_mask":
        mask = prediction == class_id
    elif mode == "gt_mask":
        mask = target == class_id
    elif mode == "error_region":
        mask = (prediction != target) & ((prediction == class_id) | (target == class_id))
    else:
        mask = torch.ones_like(target, dtype=torch.bool)
    fallback = not bool(mask.any().item())
    if fallback:
        mask = torch.ones_like(target, dtype=torch.bool)
    return mask, fallback


def normalize_cam(cam: torch.Tensor) -> np.ndarray:
    cam = cam.detach().float().cpu().numpy()
    cam = np.maximum(cam, 0.0)
    low, high = np.percentile(cam, [1.0, 99.0]) if np.isfinite(cam).any() else (0.0, 1.0)
    if high <= low:
        low = float(np.nanmin(cam)) if np.isfinite(cam).any() else 0.0
        high = low + 1.0
    return np.clip((cam - low) / (high - low), 0.0, 1.0).astype(np.float32)


def grad_cam(
    model: torch.nn.Module,
    hook: SpatialHook,
    image: torch.Tensor,
    class_id: int,
    mask: torch.Tensor,
) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    hook.clear()
    logits = as_first_tensor(model(image.unsqueeze(0)))
    if logits.ndim != 4 or class_id < 0 or class_id >= logits.shape[1]:
        raise ValueError(f"Unexpected logits shape/class: {tuple(logits.shape)}, class={class_id}")
    target_mask_resized = torch.nn.functional.interpolate(
        mask.float().unsqueeze(0).unsqueeze(0),
        size=tuple(logits.shape[-2:]),
        mode="nearest",
    ).squeeze(0).squeeze(0)
    score = (logits[0, class_id] * target_mask_resized).sum() / (target_mask_resized.sum() + 1e-6)
    score.backward()
    if hook.activation is None or hook.gradient is None:
        raise RuntimeError("CAM hook did not capture activation and gradient")
    activation = hook.activation[0]
    gradient = hook.gradient[0]
    weights = gradient.mean(dim=(-2, -1), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=0, keepdim=True))
    cam = torch.nn.functional.interpolate(
        cam.unsqueeze(0), size=tuple(image.shape[-2:]), mode="bilinear", align_corners=False
    ).squeeze(0).squeeze(0)
    return normalize_cam(cam)


def overlay_cam(rgb: np.ndarray, cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import matplotlib.cm as cm

    heat = (cm.get_cmap("jet")(cam)[..., :3] * 255.0 + 0.5).astype(np.uint8)
    overlay = (0.55 * rgb.astype(np.float32) + 0.45 * heat.astype(np.float32)).clip(0, 255).astype(np.uint8)
    gray = (cam * 255.0 + 0.5).astype(np.uint8)
    return overlay, gray


def save_sample(
    sample_dir: Path,
    source: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    cam: np.ndarray,
    cam_mask: np.ndarray,
    metadata: dict,
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    output_size = (int(prediction.shape[1]), int(prediction.shape[0]))
    rgb = rgb_preview(source, output_size=output_size)
    overlay, gray = overlay_cam(rgb, cam)
    Image.fromarray(rgb, mode="RGB").save(sample_dir / "rgb.png")
    Image.fromarray(binary_preview(target), mode="RGB").save(sample_dir / "gt.png")
    Image.fromarray(binary_preview(prediction), mode="RGB").save(sample_dir / "pred.png")
    Image.fromarray(fp_fn_preview(prediction, target), mode="RGB").save(sample_dir / "fp_fn.png")
    Image.fromarray(binary_preview(cam_mask), mode="RGB").save(sample_dir / "cam_target.png")
    Image.fromarray(overlay, mode="RGB").save(sample_dir / "cam_overlay.png")
    Image.fromarray(gray, mode="L").save(sample_dir / "cam_gray.png")
    save_json(sample_dir / "metadata.json", metadata)


def main(argv=None) -> int:
    args = parse_args(argv)
    run = resolve_run(args)
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device} but CUDA is unavailable")

    model, resolved_name, config_payload, input_channels = load_model(run, device)
    resize_to = config_resize(config_payload, args.split, args.resize)
    normalization = config_normalization(config_payload, args.split)
    dataset = MappedLabeledDataset(
        args.data_root,
        args.split,
        input_channels,
        normalization=normalization,
        resize_to=resize_to,
    )
    if args.class_id < 0 or args.class_id >= int(config_payload.get("num_classes", 2)):
        raise ValueError(f"class-id {args.class_id} is outside the model class range")

    indices, selection = choose_indices(dataset, model, device, args)
    if not indices:
        raise RuntimeError("No images selected")
    probe = dataset[indices[0]][0].unsqueeze(0).to(device)
    layer_name, layer = find_spatial_layer(model, probe, args.cam_layer)
    hook = SpatialHook(layer, with_gradient=True)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path("outputs_experiments/interpretability/cam") / f"{args.experiment_id}_s{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        for rank, index in enumerate(indices, start=1):
            image, target, image_name = dataset[index]
            image_device = image.unsqueeze(0).to(device)
            with torch.enable_grad():
                model.zero_grad(set_to_none=True)
                logits = as_first_tensor(model(image_device))
                prediction_tensor = torch.argmax(logits, dim=1)[0]
                mask_tensor, fallback = target_mask(
                    args.cam_target,
                    prediction_tensor,
                    target.to(device),
                    args.class_id,
                )
                cam = grad_cam(model, hook, image_device[0], args.class_id, mask_tensor)
            prediction = prediction_tensor.detach().cpu().numpy().astype(np.uint8)
            target_np = target.numpy().astype(np.uint8)
            cam_mask = mask_tensor.detach().cpu().numpy().astype(np.uint8)
            metrics = binary_metrics(prediction, target_np)
            sample_dir = output_dir / "samples" / f"rank_{rank:03d}_{safe_name(image_name.rsplit('.', 1)[0])}"
            metadata = {
                "rank": rank,
                "index": index,
                "image": image_name,
                "split": args.split,
                "model_id": args.experiment_id,
                "experiment_name": run.get("experiment_name", resolved_name),
                "seed": args.seed,
                "run_config": run["config_path"],
                "checkpoint": run["checkpoint_path"],
                "input_channels": input_channels,
                "normalization": normalization,
                "resize_to": resize_to,
                "cam_layer": layer_name,
                "cam_class": args.class_id,
                "cam_target": args.cam_target,
                "cam_target_fallback_to_full_image": fallback,
                "selection": selection,
                "metrics": metrics,
            }
            save_sample(sample_dir, dataset.source(index), target_np, prediction, cam, cam_mask, metadata)
            records.append({"rank": rank, "index": index, "image": image_name, "sample_dir": str(sample_dir), **metrics})
            print(f"[{rank}/{len(indices)}] {image_name} mIoU={metrics['miou']:.4f}", flush=True)
    finally:
        hook.close()

    manifest = {
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis": "grad_cam",
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "split": args.split,
        "model_id": args.experiment_id,
        "resolved_model_name": resolved_name,
        "seed": args.seed,
        "run_config": run["config_path"],
        "checkpoint": run["checkpoint_path"],
        "device": str(device),
        "input_channels": input_channels,
        "normalization": normalization,
        "resize_to": resize_to,
        "cam_layer": layer_name,
        "cam_class": args.class_id,
        "cam_target": args.cam_target,
        "selection": selection,
        "legend": {
            "cam_overlay": "RGB preview blended with a jet Grad-CAM heatmap; red/yellow indicates larger positive contribution",
            "fp_fn": "red=false positive, blue=false negative, gray=true positive",
            "cam_target": "spatial region pooled from the selected class logit; empty targets fall back to the full image",
        },
        "samples": records,
    }
    save_json(output_dir / "cam_manifest.json", manifest)
    with (output_dir / "cam_meta.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for record in records for key in record})
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"status": "completed", "samples": len(records), "output_dir": str(output_dir), "layer": layer_name}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
