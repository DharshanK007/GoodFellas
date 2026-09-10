"""
Longitudinal Sensor Health & Degradation Tracker.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
import numpy as np
from skyguard.anomaly.fusion import FusionDecision, AnomalyClassification


class HealthState(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL_FAILURE = "CRITICAL_FAILURE"


@dataclass
class StationHealthReport:
    """Health summary for a single station."""
    station_id: str
    overall_health: HealthState
    reliability_score: float             # R_i in [0, 1]
    recent_fault_rate: float            # In range [0, 1]
    variable_health: Dict[str, HealthState]
    total_observations: int
    fault_counts: int


class SensorHealthTracker:
    """
    Tracks anomaly frequencies over rolling time windows to model
    sensor wear, calibration degradation, and dynamic station reliability.
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        # station_id -> deque of recent classification boolean flags (True if SENSOR_FAULT)
        self.station_fault_history: Dict[str, deque] = {}
        # station_id -> variable -> deque of recent variable error magnitudes
        self.variable_error_history: Dict[str, Dict[str, deque]] = {}
        self.total_counts: Dict[str, int] = {}
        self.total_faults: Dict[str, int] = {}

    def update(self, decision: FusionDecision) -> StationHealthReport:
        """
        Updates internal longitudinal tracking state with latest fusion decision.
        """
        stn = decision.station_id
        if stn not in self.station_fault_history:
            self.station_fault_history[stn] = deque(maxlen=self.window_size)
            self.variable_error_history[stn] = {
                "temperature": deque(maxlen=self.window_size),
                "pressure": deque(maxlen=self.window_size),
                "humidity": deque(maxlen=self.window_size),
            }
            self.total_counts[stn] = 0
            self.total_faults[stn] = 0

        is_fault = decision.classification == AnomalyClassification.STATION_SENSOR_FAULT
        self.station_fault_history[stn].append(is_fault)
        self.total_counts[stn] += 1
        if is_fault:
            self.total_faults[stn] += 1

        for var, err in decision.variable_attributions.items():
            if var in self.variable_error_history[stn]:
                self.variable_error_history[stn][var].append(err)

        # Calculate metrics
        history = list(self.station_fault_history[stn])
        recent_fault_rate = float(sum(history) / max(1, len(history)))

        # Variable-specific health
        var_health = {}
        worst_state = HealthState.HEALTHY

        for var, err_deque in self.variable_error_history[stn].items():
            err_list = list(err_deque)
            avg_err = float(np.mean(err_list)) if err_list else 0.0
            
            if avg_err > 2.5:
                state = HealthState.CRITICAL_FAILURE
            elif avg_err > 1.2 or (is_fault and err_list and err_list[-1] > 1.5):
                state = HealthState.DEGRADED
            else:
                state = HealthState.HEALTHY
            
            var_health[var] = state
            if state == HealthState.CRITICAL_FAILURE:
                worst_state = HealthState.CRITICAL_FAILURE
            elif state == HealthState.DEGRADED and worst_state != HealthState.CRITICAL_FAILURE:
                worst_state = HealthState.DEGRADED

        # If fault rate is high overall, ensure at least degraded
        if recent_fault_rate > 0.40:
            overall_health = HealthState.CRITICAL_FAILURE
        elif recent_fault_rate > 0.20 or worst_state != HealthState.HEALTHY:
            overall_health = HealthState.DEGRADED
        else:
            overall_health = HealthState.HEALTHY

        # Dynamic Reliability Score R_i
        if overall_health == HealthState.HEALTHY:
            reliability = float(np.clip(1.0 - (recent_fault_rate * 0.3), 0.90, 1.0))
        elif overall_health == HealthState.DEGRADED:
            reliability = float(np.clip(0.85 - (recent_fault_rate * 0.5), 0.50, 0.85))
        else:
            reliability = float(np.clip(0.50 - (recent_fault_rate * 0.5), 0.10, 0.50))

        return StationHealthReport(
            station_id=stn,
            overall_health=overall_health,
            reliability_score=reliability,
            recent_fault_rate=recent_fault_rate,
            variable_health=var_health,
            total_observations=self.total_counts[stn],
            fault_counts=self.total_faults[stn],
        )
