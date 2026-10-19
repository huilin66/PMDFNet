"""Differentiable prototype-to-class prior maps and confidence estimates."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .prototype_bank import PrototypeBank


class PrototypePrior(nn.Module):
    """Convert a descriptor map to a two-class prototype posterior map."""

    def __init__(
        self,
        bank: PrototypeBank,
        distance: str = "cosine",
        temperature: float = 0.1,
    ):
        super().__init__()
        if distance not in {"cosine", "euclidean"}:
            raise ValueError("prototype distance must be 'cosine' or 'euclidean'")
        if temperature <= 0:
            raise ValueError("prototype temperature must be positive")
        self.bank = bank
        self.distance = distance
        self.temperature = float(temperature)

    def forward(self, descriptor: torch.Tensor) -> dict[str, torch.Tensor]:
        if descriptor.ndim != 4:
            raise ValueError("Descriptor must have shape [B, D, H, W]")
        prototypes = self.bank.effective_prototypes()
        if descriptor.size(1) != prototypes.size(-1):
            raise ValueError(
                f"Descriptor has {descriptor.size(1)} channels; prototypes expect {prototypes.size(-1)}"
            )

        pixels = descriptor.permute(0, 2, 3, 1).reshape(-1, descriptor.size(1))
        prototypes = prototypes.reshape(-1, prototypes.size(-1))
        if self.distance == "cosine":
            similarity = F.normalize(pixels, dim=-1) @ F.normalize(prototypes, dim=-1).t()
            prototype_scores = similarity
            nearest_distance = 1.0 - similarity.max(dim=1).values
        else:
            distances = torch.cdist(pixels.unsqueeze(0), prototypes.unsqueeze(0)).squeeze(0)
            prototype_scores = -distances
            nearest_distance = distances.min(dim=1).values

        class_scores = prototype_scores.view(-1, 2, self.bank.num_prototypes).max(dim=-1).values
        posterior = F.softmax(class_scores / self.temperature, dim=-1)
        batch, _, height, width = descriptor.shape
        posterior = posterior.view(batch, height, width, 2).permute(0, 3, 1, 2).contiguous()
        scores = class_scores.view(batch, height, width, 2).permute(0, 3, 1, 2).contiguous()
        nearest_distance = nearest_distance.view(batch, height, width)

        top2 = posterior.topk(k=2, dim=1).values
        entropy = -(posterior.clamp_min(1e-8) * posterior.clamp_min(1e-8).log()).sum(dim=1)
        maxprob = top2[:, 0]
        return {
            "posterior": posterior,
            "scores": scores,
            "maxprob": maxprob,
            "margin": top2[:, 0] - top2[:, 1],
            "entropy": entropy,
            "distance": nearest_distance,
        }

    @staticmethod
    def confidence(prior: dict[str, torch.Tensor], mode: str = "maxprob") -> torch.Tensor:
        mode = mode.lower()
        if mode == "maxprob":
            return prior["maxprob"]
        if mode == "margin":
            return prior["margin"]
        if mode == "entropy":
            return 1.0 - prior["entropy"] / torch.log(torch.tensor(2.0, device=prior["entropy"].device))
        if mode == "distance":
            return torch.exp(-prior["distance"])
        raise ValueError("knowledge confidence must be maxprob, margin, entropy, or distance")
