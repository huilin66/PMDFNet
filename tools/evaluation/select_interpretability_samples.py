"""Select fixed image IDs where PMDFNet outperforms comparison models.

This selector is intentionally separate from CAM/t-SNE inference.  It evaluates
the frozen checkpoints on the same labeled split, computes one binary mIoU per
image and model, and writes a deterministic image list.  CAM and t-SNE then
consume that same list so every model is visualized on identical images.

The selection is a performance-conditioned qualitative analysis.  It must not
be presented as an unbiased estimate of overall Test performance.
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
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.evaluate_test_metrics import (  # noqa: E402
    build_evaluation_model,
    load_checkpoint,
)
from tools.evaluation.interpretability_common import (  # noqa: E402
    MappedLabeledDataset,
    as_first_tensor,
    binary_metrics,
)
from tools.evaluation.paper_run_utils import (  # noqa: E402
    load_protocol,
    parse_id_list,
    resolve_formal_run,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select deterministic images where target PMDFNet models outperform reference models."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--target-model-ids", default="L1")
    parser.add_argument("--reference-model-ids", default="M1,M2,M3,M4")
    parser.add_argument("--run-root", default="outputs_experiments/pmdfnet_mmlsv2")
    parser.add_argument("--config", default="configs/mmlsv2_paper_protocol.json")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--metric", choices=("miou", "iou_fg", "f1"), default="miou")
    parser.add_argument("--reference-reduce", choices=("max", "mean"), default="max")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--min-advantage", type=float, default=1e-6)
    parser.add_argument("--min-target-score", type=float, default=0.0)
    parser.add_argument("--selection-seed", type=int, default=20261018)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def parse_ids(value: str) -> list[str]:
    return parse_id_list(value, ())


def normalization_for(config_payload: dict, split: str) -> str:
    if str(config_payload.get("dataset_type", "mars_ls")).lower() == "mmlsv2":
        return "none"
    return "test" if split == "test" else str(config_payload.get("normalization", "train"))


def resize_for(config_payload: dict, split: str) -> int | None:
    return int(config_payload.get("input_size", 128)) if split == "val" else None


def evaluate_model(
    model_id: str,
    args: argparse.Namespace,
    protocol: dict,
    device: torch.device,
) -> tuple[dict[str, dict], dict]:
    run = resolve_formal_run(args.run_root, protocol, model_id, args.seed)
    model_args = argparse.Namespace(run_config=run["config_path"], model_name="")
    model, resolved_name, config_path, input_channels = build_evaluation_model(model_args)
    model = model.to(device)
    load_checkpoint(model, Path(run["checkpoint_path"]), device)
    model.eval()
    config_payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    dataset = MappedLabeledDataset(
        args.data_root,
        args.split,
        input_channels,
        normalization=normalization_for(config_payload, args.split),
        resize_to=resize_for(config_payload, args.split),
    )

    try:
        from tqdm import tqdm

        iterator = tqdm(range(len(dataset)), desc=f"{model_id} {args.split}", unit="image")
    except ImportError:
        iterator = range(len(dataset))

    scores: dict[str, dict] = {}
    with torch.inference_mode():
        for index in iterator:
            image, target, image_name = dataset[index]
            logits = as_first_tensor(model(image.unsqueeze(0).to(device)))
            if tuple(logits.shape[-2:]) != tuple(target.shape[-2:]):
                logits = F.interpolate(logits, size=tuple(target.shape[-2:]), mode="bilinear", align_corners=False)
            prediction = torch.argmax(logits, dim=1)[0].detach().cpu().numpy()
            scores[image_name] = {
                "index": int(index),
                **binary_metrics(prediction, target.numpy()),
            }

    metadata = {
        "model_id": model_id,
        "resolved_model_name": resolved_name,
        "run": run,
        "input_channels": input_channels,
        "normalization": normalization_for(config_payload, args.split),
        "resize_to": resize_for(config_payload, args.split),
        "num_images": len(scores),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores, metadata


def reduce_reference(values: list[float], mode: str) -> float:
    if mode == "max":
        return float(max(values))
    return float(np.mean(values))


def write_csv(path: Path, rows: list[dict], model_ids: list[str], metric: str) -> None:
    fields = [
        "rank",
        "image",
        "index",
        "target_score",
        "reference_score",
        "advantage",
        "eligible",
    ]
    fields.extend(f"{model_id}_{metric}" for model_id in model_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    args = parse_args(argv)
    target_ids = parse_ids(args.target_model_ids)
    reference_ids = parse_ids(args.reference_model_ids)
    if not target_ids:
        raise ValueError("--target-model-ids must contain at least one model")
    if not reference_ids:
        raise ValueError("--reference-model-ids must contain at least one model")
    overlap = sorted(set(target_ids) & set(reference_ids))
    if overlap:
        raise ValueError(f"Target and reference model IDs overlap: {overlap}")
    if args.top_k == 0:
        raise ValueError("--top-k cannot be zero")

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device} but CUDA is unavailable")
    protocol = load_protocol(Path(args.config).expanduser().resolve())
    all_model_ids = target_ids + reference_ids
    per_model: dict[str, dict[str, dict]] = {}
    model_metadata = {}
    for model_id in all_model_ids:
        per_model[model_id], model_metadata[model_id] = evaluate_model(model_id, args, protocol, device)

    common_images = set(per_model[target_ids[0]])
    for model_id in all_model_ids[1:]:
        common_images &= set(per_model[model_id])
    if not common_images:
        raise RuntimeError("No common image IDs were evaluated across target/reference models")

    rows = []
    for image_name in sorted(common_images):
        target_values = [float(per_model[model_id][image_name][args.metric]) for model_id in target_ids]
        reference_values = [float(per_model[model_id][image_name][args.metric]) for model_id in reference_ids]
        target_score = float(np.mean(target_values))
        reference_score = reduce_reference(reference_values, args.reference_reduce)
        advantage = target_score - reference_score
        rows.append(
            {
                "image": image_name,
                "index": int(per_model[target_ids[0]][image_name]["index"]),
                "target_score": target_score,
                "reference_score": reference_score,
                "advantage": advantage,
                "eligible": bool(advantage >= args.min_advantage and target_score >= args.min_target_score),
                **{
                    f"{model_id}_{args.metric}": float(per_model[model_id][image_name][args.metric])
                    for model_id in all_model_ids
                },
            }
        )

    eligible = [row for row in rows if row["eligible"]]
    eligible.sort(key=lambda row: (-row["advantage"], -row["target_score"], row["image"]))
    selected = eligible if args.top_k < 0 else eligible[: args.top_k]
    if not selected:
        raise RuntimeError(
            "No images satisfy the PMDFNet advantage rule; lower --min-advantage or inspect selection_scores.csv"
        )
    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_ids.txt").write_text(
        "".join(f"{row['image']}\n" for row in selected),
        encoding="utf-8",
    )
    write_csv(output_dir / "selection_scores.csv", rows, all_model_ids, args.metric)
    write_csv(output_dir / "selected_samples.csv", selected, all_model_ids, args.metric)
    manifest = {
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis": "pmdfnet_advantage_sample_selection",
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "split": args.split,
        "seed": args.seed,
        "device": str(device),
        "metric": args.metric,
        "target_model_ids": target_ids,
        "reference_model_ids": reference_ids,
        "reference_reduce": args.reference_reduce,
        "selection_rule": "mean(target_models) - reference_reduce(reference_models) >= min_advantage and target_score >= min_target_score",
        "min_advantage": args.min_advantage,
        "min_target_score": args.min_target_score,
        "selection_seed": args.selection_seed,
        "common_image_count": len(common_images),
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "selected_images": [row["image"] for row in selected],
        "models": model_metadata,
        "outputs": {
            "selected_ids": str(output_dir / "selected_ids.txt"),
            "selection_scores": str(output_dir / "selection_scores.csv"),
            "selected_samples": str(output_dir / "selected_samples.csv"),
        },
        "warning": "Performance-conditioned qualitative selection; do not report it as an unbiased Test sample estimate.",
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "common_images": len(common_images),
                "eligible_images": len(eligible),
                "selected_images": len(selected),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
