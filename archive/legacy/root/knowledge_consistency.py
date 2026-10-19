"""Confidence-controlled agreement between prototype priors and segmentation logits."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class KnowledgeConsistencyLoss(nn.Module):
    """Apply KL(Q || P_pred) only where the train-derived prior is confident."""

    def __init__(self, confidence_threshold: float = 0.7):
        super().__init__()
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("knowledge confidence threshold must be in [0, 1]")
        self.confidence_threshold = float(confidence_threshold)

    def forward(
        self,
        logits: torch.Tensor,
        prior: torch.Tensor,
        confidence: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.shape != prior.shape:
            prior = F.interpolate(prior, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        if confidence.shape[-2:] != logits.shape[-2:]:
            confidence = F.interpolate(confidence.unsqueeze(1), size=logits.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        target = prior.detach().clamp_min(1e-8)
        per_pixel = F.kl_div(F.log_softmax(logits, dim=1), target, reduction="none").sum(dim=1)
        weights = confidence.detach().clamp(0.0, 1.0)
        weights = weights * (weights >= self.confidence_threshold).to(weights.dtype)
        if valid_mask is not None:
            weights = weights * valid_mask.to(weights.dtype)
        denominator = weights.sum()
        if denominator.detach().item() == 0:
            return per_pixel.sum() * 0.0
        return (per_pixel * weights).sum() / denominator
