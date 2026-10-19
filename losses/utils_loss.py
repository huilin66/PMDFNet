import torch
import torch.nn as nn
import torch.nn.functional as F


def deterministic_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross entropy without CUDA's nondeterministic ``nll_loss2d`` kernel.

    ``torch.nn.functional.cross_entropy`` dispatches to ``nll_loss2d`` for
    dense segmentation targets.  On the CUDA/PyTorch versions used by the
    paper experiments that kernel is rejected when strict deterministic
    algorithms are enabled.  The equivalent log-softmax + gather expression
    is deterministic for the supported tensor shapes and keeps the same
    ignore-index and reduction semantics needed here.
    """

    if logits.ndim < 2:
        raise ValueError("logits must have a class dimension")
    if target.shape != (logits.shape[0], *logits.shape[2:]):
        raise ValueError(
            f"target shape {tuple(target.shape)} is incompatible with logits "
            f"shape {tuple(logits.shape)}"
        )
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"unsupported reduction: {reduction}")

    target = target.to(dtype=torch.long)
    valid = target != ignore_index
    safe_target = target.masked_fill(~valid, 0)
    log_probs = F.log_softmax(logits, dim=1)
    loss = -log_probs.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    loss = loss.masked_fill(~valid, 0.0)

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.sum() / valid.sum().clamp_min(1).to(dtype=loss.dtype)


class DeterministicCrossEntropyLoss(nn.Module):
    """Drop-in CE module for dense binary/multiclass segmentation."""

    def __init__(self, ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return deterministic_cross_entropy(
            logits,
            target,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: [B, C, H, W] (raw logits)
        # target: [B, H, W] (long)
        
        # Apply Softmax to get probabilities
        probs = torch.softmax(pred, dim=1)
        
        # Get Foreground probabilities (Class 1)
        # Assuming binary segmentation: Class 0=BG, Class 1=FG
        pred_fg = probs[:, 1]
        
        # Create one-hot like mask for target or just use where target==1
        target_fg = (target == 1).float()
        
        # Flatten for calculation
        pred_flat = pred_fg.contiguous().view(-1)
        target_flat = target_fg.contiguous().view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5, ignore_index=-100):
        super(CombinedLoss, self).__init__()
        self.ce = DeterministicCrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss()
        self.w_ce = ce_weight
        self.w_dice = dice_weight
        
    def forward(self, pred, target):
        loss_ce = self.ce(pred, target)
        loss_dice = self.dice(pred, target)
        return self.w_ce * loss_ce + self.w_dice * loss_dice
