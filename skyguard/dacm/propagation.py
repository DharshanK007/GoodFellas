"""
Advective Propagation Timing and Uncertainty Estimator.
"""

from dataclasses import dataclass
from typing import Optional
from skyguard.config import DACM_CONFIG


@dataclass
class PropagationWindow:
    """Estimated advective arrival timing window from source to downstream station."""
    tau_minutes: float
    min_arrival_minutes: float
    max_arrival_minutes: float
    v_parallel_ms: float
    uncertainty_minutes: float
    is_valid: bool = True
    reason: str = "VALID"


class AdvectivePropagationEstimator:
    """
    Estimates first-order advective travel time tau_AB and uncertainty window
    based on distance, wind speed, alignment, and stability.
    """

    def __init__(
        self,
        uncertainty_frac: float = DACM_CONFIG.PROPAGATION_UNCERTAINTY_FRAC,
        min_uncertainty_minutes: float = DACM_CONFIG.MIN_UNCERTAINTY_MINUTES,
    ):
        self.uncertainty_frac = uncertainty_frac
        self.min_uncertainty_minutes = min_uncertainty_minutes

    def estimate_travel_time(
        self,
        distance_km: float,
        wind_speed_ms: float,
        directional_alignment: float,
        wind_stability: float = 0.9,
    ) -> PropagationWindow:
        """
        Calculates expected arrival delay tau_AB and [tau - Delta_tau, tau + Delta_tau].
        """
        v_parallel_ms = wind_speed_ms * directional_alignment

        # If parallel velocity is near zero or negative, no direct advective link
        if v_parallel_ms < 0.2:
            return PropagationWindow(
                tau_minutes=0.0,
                min_arrival_minutes=0.0,
                max_arrival_minutes=0.0,
                v_parallel_ms=v_parallel_ms,
                uncertainty_minutes=0.0,
                is_valid=False,
                reason="INSUFFICIENT_PARALLEL_VELOCITY",
            )

        # Convert v_parallel to km/h: v (m/s) * 3.6
        v_parallel_kmh = v_parallel_ms * 3.6
        tau_hours = distance_km / v_parallel_kmh
        tau_minutes = tau_hours * 60.0

        # Uncertainty scales with instability (1 - S_w) and baseline fraction
        stability_penalty = 1.0 + (1.0 - wind_stability) * 1.5
        delta_tau = max(
            self.min_uncertainty_minutes,
            tau_minutes * self.uncertainty_frac * stability_penalty,
        )

        min_arr = max(1.0, tau_minutes - delta_tau)
        max_arr = tau_minutes + delta_tau

        return PropagationWindow(
            tau_minutes=tau_minutes,
            min_arrival_minutes=min_arr,
            max_arrival_minutes=max_arr,
            v_parallel_ms=v_parallel_ms,
            uncertainty_minutes=delta_tau,
            is_valid=True,
            reason="VALID_PROPAGATION_PATHWAY",
        )
