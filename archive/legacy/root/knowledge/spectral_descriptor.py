"""Explicit cross-band spectral descriptors.

The default descriptor uses generic cross-band relations only.  It intentionally
does not assign unverified physical names to MMLSv2 channels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


DEFAULT_DESCRIPTOR_CONFIG = {
    "components": ["raw", "adjacent_difference", "normalized_difference", "group_statistics", "cross_group_nd"],
    "eps": 1e-6,
}
VALID_COMPONENTS = {
    "raw",
    "adjacent_difference",
    "normalized_difference",
    "group_statistics",
    "cross_group_nd",
}


def load_descriptor_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a JSON descriptor configuration, or return the documented default."""

    if not path:
        return dict(DEFAULT_DESCRIPTOR_CONFIG)
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    merged = dict(DEFAULT_DESCRIPTOR_CONFIG)
    merged.update(config)
    return merged


class SpectralDescriptor(nn.Module):
    """Build differentiable pixel-wise spectral relations from seven channels."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        config = {**DEFAULT_DESCRIPTOR_CONFIG, **(config or {})}
        self.components = tuple(config["components"])
        unknown = set(self.components) - VALID_COMPONENTS
        if unknown:
            raise ValueError(f"Unknown descriptor components: {sorted(unknown)}")
        if not self.components:
            raise ValueError("Descriptor needs at least one component")
        self.eps = float(config["eps"])

    @property
    def output_channels(self) -> int:
        channels = 0
        if "raw" in self.components:
            channels += 7
        if "adjacent_difference" in self.components:
            channels += 6
        if "normalized_difference" in self.components:
            channels += 6
        if "group_statistics" in self.components:
            # VN mean/std, SWIR mean/std, and their mean gap.
            channels += 5
        if "cross_group_nd" in self.components:
            channels += 1
        return channels

    def config_dict(self) -> dict[str, Any]:
        return {"components": list(self.components), "eps": self.eps}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.size(1) != 7:
            raise ValueError(f"Expected [B, 7, H, W] input, got {tuple(x.shape)}")
        outputs: list[torch.Tensor] = []
        if "raw" in self.components:
            outputs.append(x)

        adjacent_difference = x[:, 1:] - x[:, :-1]
        if "adjacent_difference" in self.components:
            outputs.append(adjacent_difference)
        if "normalized_difference" in self.components:
            denominator = x[:, 1:] + x[:, :-1]
            outputs.append(adjacent_difference / (denominator.abs() + self.eps))

        vn = x[:, :4]
        swir = x[:, 4:]
        vn_mean = vn.mean(dim=1, keepdim=True)
        swir_mean = swir.mean(dim=1, keepdim=True)
        if "group_statistics" in self.components:
            vn_std = vn.std(dim=1, keepdim=True, unbiased=False)
            swir_std = swir.std(dim=1, keepdim=True, unbiased=False)
            outputs.append(torch.cat([vn_mean, vn_std, swir_mean, swir_std, vn_mean - swir_mean], dim=1))
        if "cross_group_nd" in self.components:
            outputs.append((vn_mean - swir_mean) / (vn_mean.abs() + swir_mean.abs() + self.eps))

        return torch.cat(outputs, dim=1)
