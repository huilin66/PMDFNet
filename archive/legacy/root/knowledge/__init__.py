"""Explicit, train-derived spectral knowledge components for SKG-PMDFNet."""

from .input_normalization import ChannelNormalizer, compute_channel_normalizer
from .prototype_bank import PrototypeBank, load_prototype_artifact
from .prototype_prior import PrototypePrior
from .spectral_descriptor import SpectralDescriptor, load_descriptor_config

__all__ = [
    "ChannelNormalizer",
    "PrototypeBank",
    "PrototypePrior",
    "SpectralDescriptor",
    "compute_channel_normalizer",
    "load_descriptor_config",
    "load_prototype_artifact",
]
