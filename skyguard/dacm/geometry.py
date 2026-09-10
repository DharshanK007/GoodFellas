"""
Geospatial Calculations & Direction Vectors for Station Networks.
"""

import math
from typing import Tuple
import numpy as np


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes Great-Circle distance between two points on Earth in kilometers.
    """
    R = 6371.0088  # Mean Earth radius in km

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def station_direction_vector(
    lat_a: float, lon_a: float, lat_b: float, lon_b: float
) -> Tuple[np.ndarray, float]:
    """
    Calculates the 2D unit vector r_AB pointing from Station A toward Station B
    in local tangent plane coordinates (East, North), and the initial azimuth bearing in degrees.
    """
    phi1, phi2 = math.radians(lat_a), math.radians(lat_b)
    dlambda = math.radians(lon_b - lon_a)

    # Tangent plane displacement (East = x, North = y)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)

    bearing_rad = math.atan2(y, x)
    bearing_deg = (math.degrees(bearing_rad) + 360.0) % 360.0

    # East-North vector components:
    # 0° (North) -> [0, 1], 90° (East) -> [1, 0], 180° (South) -> [0, -1], 270° (West) -> [-1, 0]
    r_east = math.sin(bearing_rad)
    r_north = math.cos(bearing_rad)
    r_vec = np.array([r_east, r_north], dtype=np.float32)

    # Normalize unit vector
    norm = np.linalg.norm(r_vec)
    if norm > 1e-6:
        r_vec = r_vec / norm

    return r_vec, bearing_deg
