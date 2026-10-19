"""Stable binary segmentation loss used as the fixed SKG training objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .utils_loss import deterministic_cross_entropy


class CrossEntropyDiceLoss(nn.Module):
    """Equal-weight CE + foreground Dice, with optional ignored labels."""

    def __init__(self, ignore_index: int = 255, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = deterministic_cross_entropy(logits, target, ignore_index=self.ignore_index)
        valid = target != self.ignore_index
        target_fg = (target == 1).to(logits.dtype) * valid.to(logits.dtype)
        probability_fg = F.softmax(logits, dim=1)[:, 1] * valid.to(logits.dtype)
        intersection = (probability_fg * target_fg).sum()
        denominator = probability_fg.sum() + target_fg.sum()
        dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
        return self.ce_weight * ce + self.dice_weight * (1.0 - dice)
