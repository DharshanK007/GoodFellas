"""
Thermodynamic Atmospheric Physics Layer and Differentiable Losses for SkyGuard AI.
"""

from skyguard.physics.thermodynamics import (
    saturation_vapor_pressure,
    actual_vapor_pressure,
    virtual_temperature,
    atmospheric_refractive_index,
    compute_thermodynamic_state,
)
from skyguard.physics.residuals import PhysicsResidualCalculator
from skyguard.physics.loss import PhysicsConsistencyLoss

__all__ = [
    "saturation_vapor_pressure",
    "actual_vapor_pressure",
    "virtual_temperature",
    "atmospheric_refractive_index",
    "compute_thermodynamic_state",
    "PhysicsResidualCalculator",
    "PhysicsConsistencyLoss",
]
