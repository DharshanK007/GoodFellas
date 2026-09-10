"""
Specific Sensor Fault Type Categorization.
"""

from enum import Enum
from typing import Dict, List, Optional
import numpy as np
from skyguard.data.schema import AWSObservation, QualityFlag


class SpecificFaultType(str, Enum):
    SUDDEN_SPIKE = "SUDDEN_SPIKE"
    FROZEN_SENSOR = "FROZEN_SENSOR"
    GRADUAL_DRIFT = "GRADUAL_DRIFT"
    CONSTANT_OFFSET = "CONSTANT_OFFSET"
    HIGH_NOISE = "HIGH_NOISE"
    COMMUNICATION_DROPOUT = "COMMUNICATION_DROPOUT"
    MULTIVARIATE_INCONSISTENCY = "MULTIVARIATE_INCONSISTENCY"
    UNSPECIFIED_ANOMALY = "UNSPECIFIED_ANOMALY"


class FaultTypeClassifier:
    """
    Identifies the underlying hardware/data failure mechanism
    from short-term time series dynamics and thermodynamic residuals.
    """

    @classmethod
    def classify_fault(
        cls,
        current_obs: AWSObservation,
        history: List[AWSObservation],
        variable_attributions: Dict[str, float],
        physics_residuals: Dict[str, float],
    ) -> SpecificFaultType:
        """
        Determines the most probable fault archetype.
        """
        if not history or len(history) < 3:
            return SpecificFaultType.UNSPECIFIED_ANOMALY

        # Identify failing variable
        failing_var = max(variable_attributions.items(), key=lambda x: x[1])[0]

        recent_vals = [getattr(obs, failing_var) for obs in history[-10:]] + [getattr(current_obs, failing_var)]
        recent_arr = np.array(recent_vals, dtype=np.float32)

        # 1. Check for Frozen Sensor (Zero variance across >= 4 timesteps)
        if len(recent_arr) >= 4 and np.std(recent_arr[-5:]) < 1e-4:
            return SpecificFaultType.FROZEN_SENSOR

        # 2. Check for Sudden Spike (Isolated 1-step extreme change that returns)
        step_diffs = np.abs(np.diff(recent_arr))
        if len(step_diffs) >= 2:
            latest_diff = step_diffs[-1]
            prev_diffs_mean = np.mean(step_diffs[:-1])
            if latest_diff > max(3.0, prev_diffs_mean * 4.0):
                return SpecificFaultType.SUDDEN_SPIKE

        # 3. Check for Gradual Drift (Monotonic drift trend)
        if len(recent_arr) >= 8:
            steps = np.arange(len(recent_arr))
            poly = np.polyfit(steps, recent_arr, 1)
            slope = poly[0]
            corr = np.corrcoef(steps, recent_arr)[0, 1]
            if abs(corr) > 0.85 and abs(slope) > 0.05:
                return SpecificFaultType.GRADUAL_DRIFT

        # 4. Check for High Noise (Standard deviation significantly higher than baseline)
        if len(recent_arr) >= 6:
            var_diffs = np.var(np.diff(recent_arr))
            if var_diffs > 4.0:
                return SpecificFaultType.HIGH_NOISE

        # 5. Check for Severe Multivariate Thermodynamic Inconsistency
        comp_phys = physics_residuals.get("composite_physics_inconsistency", 0.0)
        if comp_phys > 1.2:
            return SpecificFaultType.MULTIVARIATE_INCONSISTENCY

        return SpecificFaultType.UNSPECIFIED_ANOMALY
