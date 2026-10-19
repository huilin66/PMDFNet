"""SKG-PMDFNet: explicit spectral knowledge-guided two-stream segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from knowledge.prototype_bank import PrototypeBank, load_prototype_artifact
from knowledge.prototype_prior import PrototypePrior
from knowledge.spectral_descriptor import SpectralDescriptor
from .fusion import KnowledgeGuidedFusion, OrdinaryAttentionFusion


def _parse_channels(channels: str | Iterable[int]) -> list[int]:
    if isinstance(channels, str):
        values = [int(value.strip()) for value in channels.split(",") if value.strip()]
    else:
        values = [int(value) for value in channels]
    if not values:
        raise ValueError("Each stream must include at least one input channel")
    if min(values) < 0 or max(values) > 6:
        raise ValueError("MMLSv2 channel indices must be in [0, 6]")
    return values


def _parse_stages(stages: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(stages, str):
        stages = [int(value.strip()) for value in stages.split(",") if value.strip()]
    stages = tuple(sorted(set(int(value) for value in stages)))
    if not stages or min(stages) < 1:
        raise ValueError("knowledge stages must be non-empty encoder stages such as '3,4'")
    return stages


class SKGDualSegFormer(nn.Module):
    """ConvNeXt-Tiny/SegFormer dual stream model with optional explicit knowledge.

    ``forward(x)`` returns logits for normal segmentation workflows.  Passing
    ``return_aux=True`` additionally returns the train-derived prior, confidence,
    and per-stage gates needed for knowledge-consistency training and analysis.
    """

    VALID_FUSIONS = {"add", "cat", "att", "knowledge_gate"}
    VALID_KNOWLEDGE_MODES = {"none", "descriptor", "prototype", "full"}

    def __init__(
        self,
        *,
        encoder_name: str = "tu-convnext_tiny",
        encoder_weights: str | None = "imagenet",
        channels1: str | Iterable[int] = (0, 1, 2, 3),
        channels2: str | Iterable[int] = (4, 5, 6),
        num_classes: int = 2,
        fusion: str = "cat",
        knowledge_mode: str = "none",
        descriptor_config: dict | None = None,
        prototype_bank: PrototypeBank | None = None,
        prototype_distance: str = "cosine",
        prototype_temperature: float = 0.1,
        knowledge_stages: str | Iterable[int] = (3, 4),
    ):
        super().__init__()
        fusion = fusion.lower()
        knowledge_mode = knowledge_mode.lower()
        if fusion not in self.VALID_FUSIONS:
            raise ValueError(f"fusion must be one of {sorted(self.VALID_FUSIONS)}")
        if knowledge_mode not in self.VALID_KNOWLEDGE_MODES:
            raise ValueError(f"knowledge mode must be one of {sorted(self.VALID_KNOWLEDGE_MODES)}")
        if num_classes != 2:
            raise ValueError("The initial SKG implementation is defined for binary segmentation")
        if set(_parse_channels(channels1)) & set(_parse_channels(channels2)):
            raise ValueError("VIS/NIR and SWIR input channel groups must not overlap")
        if fusion == "knowledge_gate" and knowledge_mode != "full":
            raise ValueError("knowledge_gate fusion requires --knowledge-mode full")
        if knowledge_mode in {"prototype", "full"} and prototype_bank is None:
            raise ValueError("prototype/full knowledge mode requires a train-derived prototype bank")

        try:
            import segmentation_models_pytorch as smp
        except ImportError as error:
            raise ImportError("SKG-PMDFNet requires segmentation_models_pytorch.") from error

        self.channels1 = _parse_channels(channels1)
        self.channels2 = _parse_channels(channels2)
        self.fusion = fusion
        self.knowledge_mode = knowledge_mode
        self.knowledge_stages = _parse_stages(knowledge_stages)
        self.num_classes = num_classes
        self.main_model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=len(self.channels1),
            classes=num_classes,
        )
        self.aux_model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=len(self.channels2),
            classes=num_classes,
        )
        main_channels = list(self.main_model.encoder.out_channels)
        aux_channels = list(self.aux_model.encoder.out_channels)
        if len(main_channels) != len(aux_channels):
            raise ValueError("The two encoder feature pyramids do not align")
        if max(self.knowledge_stages) >= len(main_channels):
            raise ValueError(
                f"knowledge stage {max(self.knowledge_stages)} is unavailable; encoder has stages 1..{len(main_channels) - 1}"
            )
        unavailable_knowledge_stages = [
            stage
            for stage in self.knowledge_stages
            if main_channels[stage] <= 0 or aux_channels[stage] <= 0
        ]
        if unavailable_knowledge_stages:
            raise ValueError(
                "knowledge stages include zero-channel encoder placeholders: "
                f"{unavailable_knowledge_stages}. Choose actual feature stages instead."
            )

        self.descriptor = SpectralDescriptor(descriptor_config)
        self.prototype_prior: PrototypePrior | None = None
        if prototype_bank is not None:
            if prototype_bank.descriptor_dim != self.descriptor.output_channels:
                raise ValueError(
                    "Prototype descriptor dimension does not match the active spectral descriptor "
                    f"({prototype_bank.descriptor_dim} != {self.descriptor.output_channels})"
                )
            self.prototype_prior = PrototypePrior(
                prototype_bank,
                distance=prototype_distance,
                temperature=prototype_temperature,
            )

        self.concat_fusions = nn.ModuleDict()
        self.attention_fusions = nn.ModuleDict()
        self.knowledge_fusions = nn.ModuleDict()
        self.descriptor_injections = nn.ModuleDict()
        self.prior_injections = nn.ModuleDict()
        for stage in range(1, len(main_channels)):
            key = str(stage)
            out_channels = main_channels[stage]
            # timm/SMP may insert an empty feature-map placeholder (0 channels)
            # into the feature list. It must pass through untouched: creating a
            # Conv2d(0, 0, ...) here produced the B4 runtime error.
            if out_channels <= 0 or aux_channels[stage] <= 0:
                continue
            if fusion == "cat":
                self.concat_fusions[key] = nn.Conv2d(
                    main_channels[stage] + aux_channels[stage], out_channels, kernel_size=1, bias=False
                )
            elif fusion == "att":
                self.attention_fusions[key] = OrdinaryAttentionFusion(
                    main_channels[stage], aux_channels[stage], out_channels
                )
            elif fusion == "knowledge_gate" and stage in self.knowledge_stages:
                self.knowledge_fusions[key] = KnowledgeGuidedFusion(
                    main_channels[stage],
                    aux_channels[stage],
                    self.descriptor.output_channels,
                    out_channels,
                    prior_channels=num_classes,
                )
            elif fusion == "knowledge_gate":
                self.concat_fusions[key] = nn.Conv2d(
                    main_channels[stage] + aux_channels[stage], out_channels, kernel_size=1, bias=False
                )

            # Descriptor/prior direct injection is an explicit A1/A2 ablation.
            # The knowledge-gated model receives these signals inside its gate,
            # rather than receiving an additional unaccounted-for residual path.
            if (
                fusion != "knowledge_gate"
                and stage in self.knowledge_stages
                and knowledge_mode in {"descriptor", "full"}
            ):
                self.descriptor_injections[key] = nn.Conv2d(
                    self.descriptor.output_channels, out_channels, kernel_size=1, bias=False
                )
            if (
                fusion != "knowledge_gate"
                and stage in self.knowledge_stages
                and knowledge_mode in {"prototype", "full"}
            ):
                self.prior_injections[key] = nn.Conv2d(num_classes, out_channels, kernel_size=1, bias=False)

        self._last_gates: dict[str, torch.Tensor] = {}

    @classmethod
    def from_prototype_path(cls, prototype_path: str | Path, *, prototype_update: str = "fixed", **kwargs):
        bank, metadata = load_prototype_artifact(prototype_path, update_mode=prototype_update)
        model = cls(prototype_bank=bank, **kwargs)
        model.prototype_metadata = metadata
        return model

    def _fixed_fusion(self, stage: int, vn: torch.Tensor, swir: torch.Tensor) -> torch.Tensor:
        key = str(stage)
        if swir.shape[-2:] != vn.shape[-2:]:
            swir = F.interpolate(swir, size=vn.shape[-2:], mode="bilinear", align_corners=False)
        if self.fusion == "add":
            if vn.shape[1] != swir.shape[1]:
                raise ValueError("add fusion needs matching encoder feature channels")
            return vn + swir
        if self.fusion == "cat" or self.fusion == "knowledge_gate":
            return self.concat_fusions[key](torch.cat([vn, swir], dim=1))
        return self.attention_fusions[key](vn, swir)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        if x.ndim != 4 or x.size(1) != 7:
            raise ValueError(f"Expected [B, 7, H, W] input, got {tuple(x.shape)}")
        vn_features = self.main_model.encoder(x[:, self.channels1])
        swir_features = self.aux_model.encoder(x[:, self.channels2])

        descriptor = None
        prior_result = None
        if self.knowledge_mode != "none" or self.fusion == "knowledge_gate":
            descriptor = self.descriptor(x)
        if self.prototype_prior is not None:
            prior_result = self.prototype_prior(descriptor)

        fused_features = [vn_features[0]]
        gates: dict[str, torch.Tensor] = {}
        for stage, (vn, swir) in enumerate(zip(vn_features[1:], swir_features[1:]), start=1):
            key = str(stage)
            if vn.size(1) == 0 or swir.size(1) == 0:
                fused_features.append(vn)
                continue
            if self.fusion == "knowledge_gate" and stage in self.knowledge_stages:
                fused, gate = self.knowledge_fusions[key](vn, swir, descriptor, prior_result["posterior"])
                gates[f"stage{stage}"] = gate
            else:
                fused = self._fixed_fusion(stage, vn, swir)

            if key in self.descriptor_injections:
                descriptor_stage = F.interpolate(descriptor, size=fused.shape[-2:], mode="bilinear", align_corners=False)
                fused = fused + self.descriptor_injections[key](descriptor_stage)
            if key in self.prior_injections:
                prior_stage = F.interpolate(prior_result["posterior"], size=fused.shape[-2:], mode="bilinear", align_corners=False)
                fused = fused + self.prior_injections[key](prior_stage)
            fused_features.append(fused)

        decoder_output = self.main_model.decoder(fused_features)
        logits = self.main_model.segmentation_head(decoder_output)
        self._last_gates = gates
        if not return_aux:
            return logits
        aux = {
            "descriptor": descriptor,
            "prior": None if prior_result is None else prior_result["posterior"],
            "confidence": None if prior_result is None else prior_result,
            "gates": gates,
        }
        return logits, aux

    def knowledge_regularization(self) -> torch.Tensor:
        if self.prototype_prior is None:
            return next(self.parameters()).new_zeros(())
        return self.prototype_prior.bank.regularization_loss()

    def latest_gates(self, detach: bool = True) -> dict[str, torch.Tensor]:
        if not detach:
            return dict(self._last_gates)
        return {name: value.detach() for name, value in self._last_gates.items()}


class SingleStreamSegFormer(nn.Module):
    """Seven-channel single-stream reference model for fair official baselines."""

    def __init__(
        self,
        *,
        encoder_name: str = "tu-convnext_tiny",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 7,
        num_classes: int = 2,
    ):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as error:
            raise ImportError("SingleStreamSegFormer requires segmentation_models_pytorch.") from error
        self.model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        logits = self.model(x)
        if return_aux:
            return logits, {"descriptor": None, "prior": None, "confidence": None, "gates": {}}
        return logits

    def knowledge_regularization(self) -> torch.Tensor:
        return next(self.parameters()).new_zeros(())
