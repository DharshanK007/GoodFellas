"""
Regional Expectation Calculator using DACM & Station Reliability.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from skyguard.data.schema import AWSObservation
from skyguard.dacm.coupling import StationCouplingResult


@dataclass
class RegionalExpectationResult:
    """Regional expectation derived from surrounding network stations."""
    expected_temperature: Optional[float]
    expected_pressure: Optional[float]
    expected_humidity: Optional[float]
    deviation_temperature: float
    deviation_pressure: float
    deviation_humidity: float
    composite_regional_mismatch: float
    active_weights_sum: float
    is_available: bool
    contributing_stations: List[str]


class RegionalExpectationCalculator:
    """
    Computes regional expected values for [T, P, RH] by weighting candidate stations
    with dynamic DACM coupling and station reliability scores.
    """

    # Scale factors for normalizing deviations
    SCALE_DEV_T = 4.0    # °C
    SCALE_DEV_P = 3.0    # hPa
    SCALE_DEV_RH = 15.0  # %

    @classmethod
    def compute_regional_expectation(
        cls,
        target_obs: AWSObservation,
        current_network_observations: List[AWSObservation],
        couplings: List[StationCouplingResult],
    ) -> RegionalExpectationResult:
        """
        Calculates weighted regional expectation:
        E_region = Sum(W_effective_AB * Obs_B) / Sum(W_effective_AB)
        """
        stn_to_obs = {obs.station_id: obs for obs in current_network_observations}

        total_weight = 0.0
        weighted_t = 0.0
        weighted_p = 0.0
        weighted_rh = 0.0
        contributors = []

        for coup in couplings:
            if coup.target_station_id not in stn_to_obs:
                continue
            cand_obs = stn_to_obs[coup.target_station_id]
            weight = coup.effective_weight

            if weight > 0.05:
                total_weight += weight
                weighted_t += weight * cand_obs.temperature
                weighted_p += weight * cand_obs.pressure
                weighted_rh += weight * cand_obs.humidity
                contributors.append(coup.target_station_id)

        if total_weight < 0.05:
            # Regional expectation is unavailable due to low network connectivity / calm winds
            return RegionalExpectationResult(
                expected_temperature=None,
                expected_pressure=None,
                expected_humidity=None,
                deviation_temperature=0.0,
                deviation_pressure=0.0,
                deviation_humidity=0.0,
                composite_regional_mismatch=0.0,
                active_weights_sum=total_weight,
                is_available=False,
                contributing_stations=[],
            )

        exp_t = weighted_t / total_weight
        exp_p = weighted_p / total_weight
        exp_rh = weighted_rh / total_weight

        dev_t = abs(target_obs.temperature - exp_t)
        dev_p = abs(target_obs.pressure - exp_p)
        dev_rh = abs(target_obs.humidity - exp_rh)

        norm_dev_t = dev_t / cls.SCALE_DEV_T
        norm_dev_p = dev_p / cls.SCALE_DEV_P
        norm_dev_rh = dev_rh / cls.SCALE_DEV_RH

        composite_mismatch = float(np.clip((norm_dev_t + norm_dev_p + norm_dev_rh) / 3.0, 0.0, 1.0))

        return RegionalExpectationResult(
            expected_temperature=exp_t,
            expected_pressure=exp_p,
            expected_humidity=exp_rh,
            deviation_temperature=dev_t,
            deviation_pressure=dev_p,
            deviation_humidity=dev_rh,
            composite_regional_mismatch=composite_mismatch,
            active_weights_sum=total_weight,
            is_available=True,
            contributing_stations=contributors,
        )
