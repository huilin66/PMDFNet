"""Train-split-only input normalization utilities.

MMLSv2 is already scaled to [0, 1], so ``none`` is the default.  The other
modes are deliberately computed from an explicit list of *training* images;
callers must never pass validation or test paths to these functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile


VALID_NORMALIZATION_MODES = {"none", "train_zscore", "robust"}


def _as_chw(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3-D image, got shape {image.shape}")
    if image.shape[-1] == 7:
        image = image.transpose(2, 0, 1)
    if image.shape[0] != 7:
        raise ValueError(f"Expected seven channels, got shape {image.shape}")
    image = image.copy()
    image[image < -100000] = 0.0
    return image


@dataclass(frozen=True)
class ChannelNormalizer:
    """Serializable per-channel normalizer computed on the train split."""

    mode: str = "none"
    center: tuple[float, ...] = ()
    scale: tuple[float, ...] = ()
    source_split: str = "train"

    def __post_init__(self):
        if self.mode not in VALID_NORMALIZATION_MODES:
            raise ValueError(f"Unsupported normalization mode: {self.mode}")
        if self.mode != "none" and (len(self.center) != 7 or len(self.scale) != 7):
            raise ValueError("Non-identity normalizers require seven centers and scales")

    @classmethod
    def identity(cls) -> "ChannelNormalizer":
        return cls()

    def apply_numpy(self, image: np.ndarray) -> np.ndarray:
        image = _as_chw(image)
        if self.mode == "none":
            return image
        center = np.asarray(self.center, dtype=np.float32)[:, None, None]
        scale = np.asarray(self.scale, dtype=np.float32)[:, None, None]
        return (image - center) / np.maximum(scale, 1e-6)

    def state_dict(self) -> dict:
        return {
            "mode": self.mode,
            "center": list(self.center),
            "scale": list(self.scale),
            "source_split": self.source_split,
        }

    @classmethod
    def from_state_dict(cls, state: dict | None) -> "ChannelNormalizer":
        if not state:
            return cls.identity()
        return cls(
            mode=state.get("mode", "none"),
            center=tuple(float(v) for v in state.get("center", [])),
            scale=tuple(float(v) for v in state.get("scale", [])),
            source_split=state.get("source_split", "train"),
        )


def _sample_pixels(image: np.ndarray, max_pixels: int, rng: np.random.Generator) -> np.ndarray:
    pixels = image.reshape(7, -1)
    if pixels.shape[1] <= max_pixels:
        return pixels
    indices = rng.choice(pixels.shape[1], size=max_pixels, replace=False)
    return pixels[:, indices]


def compute_channel_normalizer(
    image_paths: Iterable[str | Path],
    mode: str = "none",
    *,
    max_pixels_per_image: int = 32768,
    seed: int = 42,
) -> ChannelNormalizer:
    """Compute normalization statistics solely from the supplied image paths."""

    mode = mode.lower()
    if mode not in VALID_NORMALIZATION_MODES:
        raise ValueError(f"normalization must be one of {sorted(VALID_NORMALIZATION_MODES)}")
    if mode == "none":
        return ChannelNormalizer.identity()

    paths = [Path(path) for path in image_paths]
    if not paths:
        raise ValueError("Cannot compute train-only normalization from an empty image list")

    rng = np.random.default_rng(seed)
    if mode == "train_zscore":
        total = np.zeros(7, dtype=np.float64)
        total_sq = np.zeros(7, dtype=np.float64)
        count = 0
        for path in paths:
            pixels = _sample_pixels(_as_chw(tifffile.imread(path)), max_pixels_per_image, rng)
            total += pixels.sum(axis=1, dtype=np.float64)
            total_sq += np.square(pixels, dtype=np.float64).sum(axis=1, dtype=np.float64)
            count += pixels.shape[1]
        if count == 0:
            raise ValueError("No pixels available for train-only normalization")
        center = total / count
        variance = np.maximum(total_sq / count - np.square(center), 1e-12)
        scale = np.sqrt(variance)
    else:
        samples = []
        for path in paths:
            samples.append(_sample_pixels(_as_chw(tifffile.imread(path)), max_pixels_per_image, rng))
        pixels = np.concatenate(samples, axis=1)
        center = np.median(pixels, axis=1)
        q1, q3 = np.percentile(pixels, [25, 75], axis=1)
        scale = np.maximum(q3 - q1, 1e-6)

    return ChannelNormalizer(
        mode=mode,
        center=tuple(float(v) for v in center),
        scale=tuple(float(v) for v in scale),
        source_split="train",
    )
