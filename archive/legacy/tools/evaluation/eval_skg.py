"""Frozen-checkpoint evaluator for SKG-PMDFNet.

No optimization, checkpoint selection, or early stopping is performed here.
Use it exactly once per fully frozen test configuration in official experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from knowledge.input_normalization import ChannelNormalizer
from knowledge.prototype_bank import load_prototype_artifact
from knowledge.spectral_descriptor import load_descriptor_config
from losses import CrossEntropyDiceLoss
from train_skg import OfficialSegmentationDataset, TrainConfig, build_model, choose_device, evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one frozen SKG checkpoint on a labeled split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=0, help="0 reuses the training batch size.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--corruption", default="none", choices=["none"])
    parser.add_argument("--severity", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError("Not an skg_checkpoint_v1 checkpoint")
    config = TrainConfig(**checkpoint["config"])
    data_root = args.data_root or config.data_root
    if not data_root:
        raise ValueError("--data-root is required when it was not stored in the checkpoint")
    if args.corruption != "none" or args.severity != 0:
        raise ValueError("Controlled corruptions are not implemented in this first evaluator; use --corruption none --severity 0.")

    prototype_bank = None
    if config.prototype_path:
        prototype_bank, _ = load_prototype_artifact(config.prototype_path, config.prototype_update)
    descriptor_config = load_descriptor_config(config.descriptor_config)
    # Evaluation must not request pretrained weights; all parameters come from
    # the selected frozen checkpoint.
    config.pretrain = False
    model = build_model(config, descriptor_config, prototype_bank)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = choose_device(args.device)
    model.to(device)

    normalizer = ChannelNormalizer.from_state_dict(checkpoint.get("input_normalizer"))
    dataset = OfficialSegmentationDataset(
        data_root,
        args.split,
        normalizer,
        config.input_size,
        training=False,
        max_samples=0,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    criterion = CrossEntropyDiceLoss(ignore_index=config.ignore_index)
    metrics = evaluate(model, loader, criterion, config, device)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "corruption": args.corruption,
        "severity": args.severity,
        "samples": len(dataset),
        "metrics": metrics,
    }
    output_path = Path(args.output_json) if args.output_json else Path(args.checkpoint).parent.parent / f"{args.split}_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
