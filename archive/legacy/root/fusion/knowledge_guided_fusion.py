"""Feature fusion blocks used by SKG-PMDFNet."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _group_norm(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class OrdinaryAttentionFusion(nn.Module):
    """Knowledge-free two-stream attention baseline with softmax branch weights."""

    def __init__(self, vn_channels: int, swir_channels: int, out_channels: int):
        super().__init__()
        self.vn_projection = nn.Conv2d(vn_channels, out_channels, kernel_size=1, bias=False)
        self.swir_projection = nn.Conv2d(swir_channels, out_channels, kernel_size=1, bias=False)
        hidden = max(out_channels // 2, 16)
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, hidden, kernel_size=1, bias=False),
            _group_norm(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 2, kernel_size=1),
        )

    def forward(self, vn: torch.Tensor, swir: torch.Tensor) -> torch.Tensor:
        vn = self.vn_projection(vn)
        swir = self.swir_projection(swir)
        weights = F.softmax(self.gate(torch.cat([vn, swir], dim=1)), dim=1)
        return weights[:, 0:1] * vn + weights[:, 1:2] * swir


class KnowledgeGuidedFusion(nn.Module):
    """Fuse VN, SWIR, and interaction features using descriptor and prior evidence.

    The three output gate channels are softmax-normalized, so they represent the
    relative spatial contribution of VN, SWIR, and cross-stream interaction.
    """

    def __init__(
        self,
        vn_channels: int,
        swir_channels: int,
        descriptor_channels: int,
        out_channels: int,
        prior_channels: int = 2,
    ):
        super().__init__()
        self.vn_projection = nn.Conv2d(vn_channels, out_channels, kernel_size=1, bias=False)
        self.swir_projection = nn.Conv2d(swir_channels, out_channels, kernel_size=1, bias=False)
        self.interaction = nn.Sequential(
            nn.Conv2d(vn_channels + swir_channels, out_channels, kernel_size=1, bias=False),
            _group_norm(out_channels),
            nn.GELU(),
        )
        knowledge_channels = max(out_channels // 4, 16)
        self.knowledge_projection = nn.Sequential(
            nn.Conv2d(descriptor_channels + prior_channels, knowledge_channels, kernel_size=1, bias=False),
            _group_norm(knowledge_channels),
            nn.GELU(),
        )
        hidden = max(out_channels // 2, 16)
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels * 3 + knowledge_channels, hidden, kernel_size=1, bias=False),
            _group_norm(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 3, kernel_size=1),
        )

    def forward(
        self,
        vn: torch.Tensor,
        swir: torch.Tensor,
        descriptor: torch.Tensor,
        prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_size = vn.shape[-2:]
        if swir.shape[-2:] != spatial_size:
            swir = F.interpolate(swir, size=spatial_size, mode="bilinear", align_corners=False)
        descriptor = F.interpolate(descriptor, size=spatial_size, mode="bilinear", align_corners=False)
        prior = F.interpolate(prior, size=spatial_size, mode="bilinear", align_corners=False)

        vn_projected = self.vn_projection(vn)
        swir_projected = self.swir_projection(swir)
        interaction = self.interaction(torch.cat([vn, swir], dim=1))
        knowledge = self.knowledge_projection(torch.cat([descriptor, prior], dim=1))
        gates = F.softmax(
            self.gate(torch.cat([vn_projected, swir_projected, interaction, knowledge], dim=1)),
            dim=1,
        )
        fused = (
            gates[:, 0:1] * vn_projected
            + gates[:, 1:2] * swir_projected
            + gates[:, 2:3] * interaction
        )
        return fused, gates
