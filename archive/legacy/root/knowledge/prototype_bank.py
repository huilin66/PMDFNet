"""Train-derived class spectral prototype storage and artifact I/O."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class PrototypeBank(nn.Module):
    """Class-conditioned prototype bank with fixed, learnable, or residual updates."""

    VALID_UPDATE_MODES = {"fixed", "learnable", "residual"}

    def __init__(self, prototypes: torch.Tensor | np.ndarray, update_mode: str = "fixed"):
        super().__init__()
        update_mode = update_mode.lower()
        if update_mode not in self.VALID_UPDATE_MODES:
            raise ValueError(f"prototype update must be one of {sorted(self.VALID_UPDATE_MODES)}")
        prototypes = torch.as_tensor(prototypes, dtype=torch.float32)
        if prototypes.ndim != 3 or prototypes.size(0) != 2:
            raise ValueError("Prototype tensor must have shape [2, K, descriptor_dim]")
        self.update_mode = update_mode
        if update_mode == "learnable":
            self.prototypes = nn.Parameter(prototypes.clone())
        else:
            self.register_buffer("base_prototypes", prototypes.clone())
            if update_mode == "residual":
                self.delta = nn.Parameter(torch.zeros_like(prototypes))

    @property
    def num_prototypes(self) -> int:
        return int(self.effective_prototypes().size(1))

    @property
    def descriptor_dim(self) -> int:
        return int(self.effective_prototypes().size(2))

    def effective_prototypes(self) -> torch.Tensor:
        if self.update_mode == "learnable":
            return self.prototypes
        if self.update_mode == "residual":
            return self.base_prototypes + self.delta
        return self.base_prototypes

    def regularization_loss(self) -> torch.Tensor:
        if self.update_mode != "residual":
            return self.effective_prototypes().new_zeros(())
        return self.delta.square().mean()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_prototype_artifact(
    path: str | Path,
    prototypes: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Persist the array and JSON metadata without pickled objects."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata)
    metadata["artifact_format"] = "skg_prototypes_v1"
    np.savez_compressed(
        path,
        prototypes=np.asarray(prototypes, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    metadata["artifact_path"] = str(path)
    metadata["sha256"] = sha256_file(path)
    with open(path.with_suffix(".metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return metadata


def load_prototype_artifact(path: str | Path, update_mode: str = "fixed") -> tuple[PrototypeBank, dict[str, Any]]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        prototypes = data["prototypes"]
        metadata = json.loads(str(data["metadata_json"].item()))
    metadata["artifact_path"] = str(path)
    metadata["sha256"] = sha256_file(path)
    return PrototypeBank(prototypes, update_mode=update_mode), metadata
