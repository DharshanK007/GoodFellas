"""
Dynamic Advective Coupling and Verification Mechanism (DACM).
"""

from skyguard.dacm.geometry import haversine_distance_km, station_direction_vector
from skyguard.dacm.wind_vector import (
    meteorological_wind_to_vector,
    wind_vector_to_speed_direction,
    circular_wind_stability,
)
from skyguard.dacm.alignment import compute_wind_alignment
from skyguard.dacm.coupling import (
    DACMCouplingEngine,
    StationCouplingResult,
)
from skyguard.dacm.propagation import (
    AdvectivePropagationEstimator,
    PropagationWindow,
)
from skyguard.dacm.verification import DownstreamVerifier

__all__ = [
    "haversine_distance_km",
    "station_direction_vector",
    "meteorological_wind_to_vector",
    "wind_vector_to_speed_direction",
    "circular_wind_stability",
    "compute_wind_alignment",
    "DACMCouplingEngine",
    "StationCouplingResult",
    "AdvectivePropagationEstimator",
    "PropagationWindow",
    "DownstreamVerifier",
]
