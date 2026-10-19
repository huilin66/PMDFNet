"""Build class spectral prototypes exclusively from one labeled training split."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile
import torch

from .input_normalization import ChannelNormalizer, _as_chw, compute_channel_normalizer
from .prototype_bank import save_prototype_artifact
from .spectral_descriptor import SpectralDescriptor


@dataclass(frozen=True)
class PrototypeBuildConfig:
    root_dir: str
    split: str = "train"
    prototype_k: int = 4
    distance: str = "cosine"
    normalization: str = "none"
    max_pixels_per_image: int = 2048
    normalization_pixels_per_image: int = 32768
    seed: int = 42
    max_samples: int = 0


def list_labeled_samples(root_dir: str | Path, split: str) -> list[tuple[Path, Path]]:
    root_dir = Path(root_dir)
    images_dir = root_dir / split / "images"
    masks_dir = root_dir / split / "masks"
    samples = []
    for image_path in sorted(images_dir.glob("*.tif")):
        mask_path = masks_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path}: {mask_path}")
        samples.append((image_path, mask_path))
    if not samples:
        raise FileNotFoundError(f"No labeled TIFF samples found under {images_dir}")
    return samples


def _sample_class_pixels(
    samples: Iterable[tuple[Path, Path]],
    descriptor: SpectralDescriptor,
    normalizer: ChannelNormalizer,
    max_pixels_per_image: int,
    seed: int,
) -> tuple[list[np.ndarray], list[int]]:
    rng = np.random.default_rng(seed)
    grouped: list[list[np.ndarray]] = [[], []]
    per_class_counts = [0, 0]
    descriptor.eval()
    with torch.no_grad():
        for image_path, mask_path in samples:
            image = normalizer.apply_numpy(_as_chw(tifffile.imread(image_path)))
            mask = tifffile.imread(mask_path).astype(np.int64)
            z = descriptor(torch.from_numpy(image).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
            z = z.cpu().numpy().reshape(-1, descriptor.output_channels)
            mask = mask.reshape(-1)
            for label in (0, 1):
                indices = np.flatnonzero(mask == label)
                if not len(indices):
                    continue
                if len(indices) > max_pixels_per_image:
                    indices = rng.choice(indices, size=max_pixels_per_image, replace=False)
                grouped[label].append(z[indices])
                per_class_counts[label] += len(indices)
    if not grouped[0] or not grouped[1]:
        raise ValueError("Training split must contain both background and landslide pixels")
    return [np.concatenate(parts, axis=0) for parts in grouped], per_class_counts


def _fit_kmeans(features: np.ndarray, k: int, distance: str, seed: int) -> np.ndarray:
    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.preprocessing import normalize
    except ImportError as error:
        raise ImportError("Prototype construction requires scikit-learn; install scikit-learn first.") from error
    if len(features) < k:
        raise ValueError(f"K={k} exceeds available class samples ({len(features)})")
    if distance == "cosine":
        features = normalize(features, norm="l2", axis=1)
    elif distance != "euclidean":
        raise ValueError("prototype distance must be cosine or euclidean")
    model = MiniBatchKMeans(
        n_clusters=k,
        batch_size=min(4096, len(features)),
        n_init=10,
        random_state=seed,
        reassignment_ratio=0.0,
    )
    model.fit(features)
    centers = model.cluster_centers_.astype(np.float32)
    if distance == "cosine":
        centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
    return centers


def build_prototype_artifact(
    config: PrototypeBuildConfig,
    descriptor_config: dict,
    output_path: str | Path,
) -> dict:
    """Create and save a prototype artifact using only ``config.split`` data."""

    if config.split != "train":
        raise ValueError("Official prototypes must be built from split='train' only")
    samples = list_labeled_samples(config.root_dir, config.split)
    if config.max_samples > 0:
        samples = samples[: config.max_samples]
    normalizer = compute_channel_normalizer(
        [image for image, _ in samples],
        mode=config.normalization,
        max_pixels_per_image=config.normalization_pixels_per_image,
        seed=config.seed,
    )
    descriptor = SpectralDescriptor(descriptor_config)
    grouped, counts = _sample_class_pixels(
        samples,
        descriptor,
        normalizer,
        config.max_pixels_per_image,
        config.seed,
    )
    prototypes = np.stack(
        [_fit_kmeans(grouped[label], config.prototype_k, config.distance, config.seed + label) for label in (0, 1)],
        axis=0,
    )
    metadata = {
        "source_split": "train",
        "root_dir": str(Path(config.root_dir).resolve()),
        "sample_count": len(samples),
        "sampled_pixel_count": {"background": counts[0], "landslide": counts[1]},
        "prototype_k": config.prototype_k,
        "distance": config.distance,
        "descriptor_config": descriptor.config_dict(),
        "descriptor_dim": descriptor.output_channels,
        "input_normalizer": normalizer.state_dict(),
        "seed": config.seed,
    }
    return save_prototype_artifact(output_path, prototypes, metadata)


def write_metadata_copy(metadata: dict, output_path: str | Path) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def main(argv: list[str] | None = None) -> None:
    import argparse

    from env_utils import get_data_root
    from .spectral_descriptor import load_descriptor_config

    parser = argparse.ArgumentParser(description="Build train-only MMLSv2 spectral prototypes.")
    parser.add_argument("--data-root", default=get_data_root())
    parser.add_argument("--split", default="train")
    parser.add_argument("--descriptor-config", default="configs/descriptors/default.json")
    parser.add_argument("--prototype-k", type=int, default=4)
    parser.add_argument("--distance", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--normalization", choices=["none", "train_zscore", "robust"], default="none")
    parser.add_argument("--max-pixels-per-image", type=int, default=2048)
    parser.add_argument("--normalization-pixels-per-image", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="Use first N train images; intended for smoke tests only.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if not args.data_root:
        parser.error("--data-root is required (or set MMLSV2_DATA_ROOT in .env)")
    config = PrototypeBuildConfig(
        root_dir=args.data_root,
        split=args.split,
        prototype_k=args.prototype_k,
        distance=args.distance,
        normalization=args.normalization,
        max_pixels_per_image=args.max_pixels_per_image,
        normalization_pixels_per_image=args.normalization_pixels_per_image,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    metadata = build_prototype_artifact(config, load_descriptor_config(args.descriptor_config), args.output)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
