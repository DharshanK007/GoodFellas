"""
Wind Vector Transformations and Circular Statistics for Atmospheric Flow.
"""

import math
from typing import List, Tuple, Optional
import numpy as np


def meteorological_wind_to_vector(
    wind_speed: float, wind_direction_deg: float
) -> np.ndarray:
    """
    Converts meteorological wind (speed in m/s, direction in degrees from which wind blows)
    into the atmospheric transport vector [u (East), v (North)] in m/s.
    
    Example:
    Wind from West (270°) transports air toward East (+u direction):
    u = -8 * sin(270°) = +8 m/s, v = -8 * cos(270°) = 0 m/s
    """
    rad = math.radians(wind_direction_deg)
    u = -wind_speed * math.sin(rad)
    v = -wind_speed * math.cos(rad)
    return np.array([u, v], dtype=np.float32)


def wind_vector_to_speed_direction(u: float, v: float) -> Tuple[float, float]:
    """
    Converts transport vector [u, v] back to meteorological speed (m/s)
    and meteorological direction (degrees from which wind blew).
    """
    speed = math.sqrt(u ** 2 + v ** 2)
    # Direction toward which air moves: atan2(u, v)
    # Meteorological direction (from which air came): + 180°
    rad_met = math.atan2(-u, -v)
    deg_met = (math.degrees(rad_met) + 360.0) % 360.0
    return speed, deg_met


def circular_wind_stability(
    recent_directions_deg: List[float], recent_speeds: Optional[List[float]] = None
) -> float:
    """
    Calculates circular wind stability score S_w in [0, 1] using directional resultant length.
    S_w ≈ 1.0 -> highly steady laminar flow
    S_w ≈ 0.0 -> highly turbulent / shifting / calm wind
    """
    if not recent_directions_deg:
        return 0.0

    if len(recent_directions_deg) == 1:
        return 1.0

    rads = np.radians(recent_directions_deg)
    if recent_speeds is not None and len(recent_speeds) == len(recent_directions_deg):
        weights = np.array(recent_speeds, dtype=np.float32)
        total_weight = np.sum(weights)
        if total_weight < 1e-3:
            return 0.0  # Calm wind
        mean_sin = np.sum(weights * np.sin(rads)) / total_weight
        mean_cos = np.sum(weights * np.cos(rads)) / total_weight
    else:
        mean_sin = np.mean(np.sin(rads))
        mean_cos = np.mean(np.cos(rads))

    # Mean resultant length R in [0, 1]
    R = float(np.sqrt(mean_sin ** 2 + mean_cos ** 2))
    return float(np.clip(R, 0.0, 1.0))
