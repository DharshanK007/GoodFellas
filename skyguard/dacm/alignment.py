"""
Directional Alignment Calculations between Atmospheric Transport and Station Baseline.
"""

from typing import Tuple
import numpy as np
from skyguard.config import DACM_CONFIG


def compute_wind_alignment(
    v_a: np.ndarray,
    r_ab: np.ndarray,
    wind_speed_a: float,
    calm_threshold: float = DACM_CONFIG.CALM_WIND_THRESHOLD,
) -> Tuple[float, bool]:
    """
    Computes directional alignment a_AB in [0, 1] between atmospheric flow at A
    and the unit direction vector r_AB pointing from A to B.
    
    Returns:
    (alignment_score, is_calm)
    """
    if wind_speed_a < 1e-3:
        return 0.0, True

    norm_v = np.linalg.norm(v_a)
    if norm_v < 1e-3:
        return 0.0, True

    # Normalized dot product
    raw_cos = float(np.dot(v_a, r_ab) / norm_v)
    alignment = max(0.0, raw_cos)

    # Calm wind attenuation
    is_calm = wind_speed_a < calm_threshold
    if is_calm:
        calm_factor = max(0.0, wind_speed_a / calm_threshold)
        alignment *= calm_factor

    return float(np.clip(alignment, 0.0, 1.0)), is_calm
