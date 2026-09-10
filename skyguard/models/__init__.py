"""
Neural Autoencoder Architectures and Baselines for SkyGuard AI.
"""

from skyguard.models.temporal_channel import TemporalSequenceAE
from skyguard.models.physics_channel import PhysicsInformedAE
from skyguard.models.dual_channel import DualChannelSkyGuardModel
from skyguard.models.baselines import (
    ThresholdQCBaseline,
    StatisticalIsolationForestBaseline,
    StandardMultivariateAE,
)

__all__ = [
    "TemporalSequenceAE",
    "PhysicsInformedAE",
    "DualChannelSkyGuardModel",
    "ThresholdQCBaseline",
    "StatisticalIsolationForestBaseline",
    "StandardMultivariateAE",
]
