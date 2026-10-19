"""Shared data, model and visualization helpers for PMDFNet interpretation tools.

The interpretation scripts deliberately reuse the formal-run model builder and
checkpoint loader used by the numerical evaluation scripts.  This keeps the
input channels, normalization, architecture and checkpoint semantics aligned
with the paper protocol.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from dataset import (
    MARS_MEAN_TEST,
    MARS_MEAN_TRAIN,
    MARS_MEAN_VAL,
    MARS_STD_TEST,
    MARS_STD_TRAIN,
    MARS_STD_VAL,
)


def as_first_tensor(output: Any) -> torch.Tensor:
    """Extract the segmentation logits from common PMDFNet output formats."""

    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("Model returned an empty tuple/list")
        return as_first_tensor(output[0])
    if isinstance(output, dict):
        for key in ("out", "logits", "main", "prediction"):
            if key in output:
                return as_first_tensor(output[key])
        if not output:
            raise ValueError("Model returned an empty dictionary")
        return as_first_tensor(next(iter(output.values())))
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected tensor-like model output, got {type(output)}")
    return output


def as_feature_tensor(output: Any) -> torch.Tensor:
    """Extract a spatial feature tensor from a module hook output."""

    if isinstance(output, (tuple, list)):
        for item in output:
            try:
                tensor = as_feature_tensor(item)
                if tensor.ndim >= 3:
                    return tensor
            except (TypeError, ValueError):
                continue
        raise ValueError("Hook output did not contain a feature tensor")
    if isinstance(output, dict):
        for item in output.values():
            try:
                tensor = as_feature_tensor(item)
                if tensor.ndim >= 3:
                    return tensor
            except (TypeError, ValueError):
                continue
        raise ValueError("Hook output dictionary did not contain a feature tensor")
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected tensor hook output, got {type(output)}")
    if output.ndim not in (3, 4):
        raise ValueError(f"Expected a 3D/4D spatial feature tensor, got {tuple(output.shape)}")
    if output.ndim == 3:
        # Some transformer decoders expose [B, N, C].  If N is a square, turn
        # it into [B, C, H, W] so CAM and prototype extraction remain spatial.
        batch, tokens, channels = output.shape
        side = int(math.sqrt(tokens))
        if side * side != tokens:
            raise ValueError(f"Cannot reshape non-square token feature {tuple(output.shape)}")
        return output.transpose(1, 2).reshape(batch, channels, side, side)
    return output


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)


def to_chw(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] == 7:
        image = image.transpose(2, 0, 1)
    if image.ndim != 3 or image.shape[0] != 7:
        raise ValueError(f"Expected a 7-channel TIFF, got {image.shape}")
    return image.astype(np.float32, copy=False)


def _normalization_arrays(name: str | None, split: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    key = (name or "auto").lower()
    if key == "auto":
        key = split.lower()
    if key in {"none", "null", "off"}:
        return None, None
    if key == "train":
        return np.asarray(MARS_MEAN_TRAIN, dtype=np.float32), np.asarray(MARS_STD_TRAIN, dtype=np.float32)
    if key == "val":
        return np.asarray(MARS_MEAN_VAL, dtype=np.float32), np.asarray(MARS_STD_VAL, dtype=np.float32)
    if key == "test":
        return np.asarray(MARS_MEAN_TEST, dtype=np.float32), np.asarray(MARS_STD_TEST, dtype=np.float32)
    raise ValueError(f"Unsupported interpretation normalization: {name!r}")


class MappedLabeledDataset(Dataset):
    """Read a labeled mapped-MMLSv2 split with the paper's preprocessing.

    ``test`` deliberately uses the same MARS Test statistics as
    ``evaluate_test_metrics.py``.  Train/Val default to the normalization
    stored in the selected run config, normally the Train statistics.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        input_channels: Iterable[int],
        normalization: str | None = "auto",
        resize_to: int | None = None,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.split = str(split)
        self.input_channels = [int(value) for value in input_channels]
        if not self.input_channels:
            raise ValueError("input_channels must contain at least one channel")
        invalid = [value for value in self.input_channels if value < 0 or value >= 7]
        if invalid:
            raise ValueError(f"Channel indices must be in [0, 6], got {invalid}")
        self.resize_to = int(resize_to) if resize_to else None
        self.images_dir = self.root_dir / self.split / "images"
        self.masks_dir = self.root_dir / self.split / "masks"
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Missing split image directory: {self.images_dir}")
        if not self.masks_dir.is_dir():
            raise FileNotFoundError(f"Missing split mask directory: {self.masks_dir}")
        self.image_files = sorted(self.images_dir.glob("*.tif"))
        self.mask_by_stem = {path.stem: path for path in self.masks_dir.glob("*.tif")}
        missing = [path.name for path in self.image_files if path.stem not in self.mask_by_stem]
        if missing:
            raise FileNotFoundError(
                f"Missing masks for {len(missing)} {self.split} images; examples: {missing[:5]}"
            )
        if not self.image_files:
            raise RuntimeError(f"No TIFF images found in {self.images_dir}")

        mean, std = _normalization_arrays(normalization, self.split)
        self.mean = None if mean is None else mean.reshape(7, 1, 1)
        self.std = None if std is None else std.reshape(7, 1, 1)

    def __len__(self) -> int:
        return len(self.image_files)

    def source(self, index: int) -> np.ndarray:
        return to_chw(tifffile.imread(str(self.image_files[index])))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_path = self.image_files[index]
        image = self.source(index).copy()
        image[image < -100000] = 0.0
        if self.mean is not None:
            image = (image - self.mean) / (self.std + 1e-8)
        image = image[self.input_channels]

        mask = np.asarray(tifffile.imread(str(self.mask_by_stem[image_path.stem])))
        mask = np.squeeze(mask)
        if mask.ndim != 2:
            raise ValueError(f"Expected a 2D mask, got {mask.shape}: {image_path}")
        mask = (mask > 0).astype(np.int64)
        image_t = torch.from_numpy(image).float()
        mask_t = torch.from_numpy(mask).long()
        if self.resize_to and tuple(image_t.shape[-2:]) != (self.resize_to, self.resize_to):
            image_t = F.interpolate(
                image_t.unsqueeze(0),
                size=(self.resize_to, self.resize_to),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mask_t = F.interpolate(
                mask_t.unsqueeze(0).unsqueeze(0).float(),
                size=(self.resize_to, self.resize_to),
                mode="nearest",
            ).squeeze(0).squeeze(0).long()
        return image_t, mask_t, image_path.name


def rgb_preview(image: np.ndarray, output_size: tuple[int, int] | None = None) -> np.ndarray:
    """Create a per-image percentile-scaled RGB preview from R/G/B channels."""

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
    image_uint8 = (output.transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
    if output_size and tuple(image_uint8.shape[:2][::-1]) != tuple(output_size):
        image_uint8 = np.asarray(
            Image.fromarray(image_uint8, mode="RGB").resize(output_size, Image.Resampling.BILINEAR)
        )
    return image_uint8


def single_band_preview(
    image: np.ndarray,
    channel_index: int,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Create a grayscale preview for one raw input band.

    The preview is only for visualization.  It uses per-tile 2--98 percentile
    scaling so that the physical slope channel remains visible even when its
    raw dynamic range differs substantially between tiles.
    """

    channels = to_chw(image)
    if channel_index < 0 or channel_index >= channels.shape[0]:
        raise ValueError(f"Channel index {channel_index} is outside {channels.shape[0]} channels")
    band = channels[channel_index].astype(np.float32, copy=False)
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        scaled = np.zeros_like(band, dtype=np.float32)
    else:
        low, high = np.percentile(finite, [2.0, 98.0])
        if high <= low:
            high = low + 1.0
        scaled = np.clip((band - low) / (high - low), 0.0, 1.0)
        scaled[~np.isfinite(scaled)] = 0.0
    band_uint8 = (scaled * 255.0 + 0.5).astype(np.uint8)
    image_uint8 = np.repeat(band_uint8[..., None], 3, axis=2)
    if output_size and tuple(image_uint8.shape[:2][::-1]) != tuple(output_size):
        image_uint8 = np.asarray(
            Image.fromarray(image_uint8, mode="RGB").resize(output_size, Image.Resampling.BILINEAR)
        )
    return image_uint8


def slope_preview(image: np.ndarray, output_size: tuple[int, int] | None = None) -> np.ndarray:
    """Create the standardized grayscale preview of the MMLSv2 Slope band."""

    return single_band_preview(image, channel_index=1, output_size=output_size)


def binary_preview(mask: np.ndarray, output_size: tuple[int, int] | None = None) -> np.ndarray:
    mask = np.asarray(mask) > 0
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[mask] = (255, 255, 255)
    if output_size and tuple(image.shape[:2][::-1]) != tuple(output_size):
        image = np.asarray(
            Image.fromarray(image, mode="RGB").resize(output_size, Image.Resampling.NEAREST)
        )
    return image


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


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    prediction = np.asarray(prediction) > 0
    target = np.asarray(target) > 0
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())
    eps = 1e-8
    iou_fg = tp / (tp + fp + fn + eps)
    iou_bg = tn / (tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "miou": float((iou_fg + iou_bg) / 2.0),
        "iou_fg": float(iou_fg),
        "f1": float(f1),
        "error_pixels": int(fp + fn),
    }


def module_by_name(model: nn.Module, name: str) -> nn.Module:
    normalized = name.strip().removeprefix("module.")
    modules = dict(model.named_modules())
    if normalized in modules:
        return modules[normalized]
    matches = [(key, value) for key, value in modules.items() if key.endswith(normalized)]
    if len(matches) == 1:
        return matches[0][1]
    available = ", ".join(list(modules)[:30])
    raise KeyError(f"Layer {name!r} not found or ambiguous. Available examples: {available}")


def _probe_module_outputs(model: nn.Module, probe: torch.Tensor) -> dict[str, torch.Tensor]:
    outputs: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            try:
                tensor = as_feature_tensor(output)
                if tensor.ndim == 4:
                    outputs[name] = tensor
            except (TypeError, ValueError):
                pass

        return hook

    for name, module in model.named_modules():
        if not name:
            continue
        handles.append(module.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        as_first_tensor(model(probe))
    for handle in handles:
        handle.remove()
    return outputs


def find_spatial_layer(model: nn.Module, probe: torch.Tensor, requested: str = "auto") -> tuple[str, nn.Module]:
    """Resolve a hook layer; auto prefers the final decoder feature map."""

    if requested and requested.lower() != "auto":
        return requested, module_by_name(model, requested)
    outputs = _probe_module_outputs(model, probe)
    if not outputs:
        raise RuntimeError("Could not find a 4D spatial layer for interpretation")

    names = list(outputs)
    preferred: list[str] = []
    for name in names:
        lowered = name.lower()
        if "segmentation_head" in lowered or lowered.endswith(".head"):
            continue
        if any(token in lowered for token in ("decoder", "decode_head", "decode")):
            preferred.append(name)
    if preferred:
        selected = preferred[-1]
    else:
        conv_names = [
            name for name in names
            if isinstance(module_by_name(model, name), (nn.Conv2d, nn.ConvTranspose2d))
        ]
        selected = conv_names[-1] if conv_names else names[-1]
    return selected, module_by_name(model, selected)


class SpatialHook:
    """Capture the latest spatial activation and, optionally, its gradient."""

    def __init__(self, module: nn.Module, with_gradient: bool) -> None:
        self.activation: torch.Tensor | None = None
        self.gradient: torch.Tensor | None = None
        self.with_gradient = with_gradient
        self.handle = module.register_forward_hook(self._forward)
        self.backward_handle = (
            module.register_full_backward_hook(self._backward) if with_gradient else None
        )

    def _forward(self, _module, _inputs, output) -> None:
        self.activation = as_feature_tensor(output)

    def _backward(self, _module, _grad_inputs, grad_outputs) -> None:
        if grad_outputs and grad_outputs[0] is not None:
            self.gradient = as_feature_tensor(grad_outputs[0])

    def clear(self) -> None:
        self.activation = None
        self.gradient = None

    def close(self) -> None:
        self.handle.remove()
        if self.backward_handle is not None:
            self.backward_handle.remove()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        mask.float().unsqueeze(0).unsqueeze(0), size=size, mode="nearest"
    ).squeeze(0).squeeze(0).bool()


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
