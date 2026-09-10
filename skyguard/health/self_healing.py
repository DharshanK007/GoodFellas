"""
Non-Destructive Data-Level Self-Healing Imputation.
"""

from datetime import datetime, timezone
from typing import Dict, Optional
from skyguard.data.schema import AWSObservation, CorrectedObservation
from skyguard.anomaly.fusion import FusionDecision, AnomalyClassification


class SelfHealingImputer:
    """
    Provides data-level correction/imputation when a sensor fault is detected.
    The raw sensor observation is NEVER altered; imputed values are stored
    in a dedicated CorrectedObservation structure alongside full audit metadata.
    """

    @classmethod
    def impute_faulty_observation(
        cls,
        raw_obs: AWSObservation,
        decision: FusionDecision,
        expected_values: Dict[str, float],
        regional_expected_values: Optional[Dict[str, Optional[float]]] = None,
    ) -> CorrectedObservation:
        """
        Creates a CorrectedObservation container preserving raw data and adding imputed estimates.
        """
        if decision.classification != AnomalyClassification.STATION_SENSOR_FAULT:
            # For normal observations or genuine weather events, no imputation is needed
            return CorrectedObservation(
                raw_observation=raw_obs,
                imputed_temperature=None,
                imputed_pressure=None,
                imputed_humidity=None,
                imputation_reason="Observation verified as genuine or nominal; no correction needed.",
                imputation_confidence=1.0,
                processed_at=datetime.now(timezone.utc),
            )

        # Impute primarily from Dual-Channel physical model with optional regional blending
        imputed_t = expected_values.get("temperature", raw_obs.temperature)
        imputed_p = expected_values.get("pressure", raw_obs.pressure)
        imputed_rh = expected_values.get("humidity", raw_obs.humidity)

        if regional_expected_values:
            reg_t = regional_expected_values.get("temperature")
            reg_p = regional_expected_values.get("pressure")
            reg_rh = regional_expected_values.get("humidity")

            if reg_t is not None:
                imputed_t = 0.7 * imputed_t + 0.3 * reg_t
            if reg_p is not None:
                imputed_p = 0.7 * imputed_p + 0.3 * reg_p
            if reg_rh is not None:
                imputed_rh = 0.7 * imputed_rh + 0.3 * reg_rh

        # Ensure humidity bounds
        imputed_rh = max(0.0, min(100.0, float(imputed_rh)))

        failing_var = max(decision.variable_attributions.items(), key=lambda x: x[1])[0]
        reason = f"Imputed {failing_var} following confirmed station sensor fault (anomaly score={decision.local_anomaly_score:.2f})."

        return CorrectedObservation(
            raw_observation=raw_obs,
            imputed_temperature=round(float(imputed_t), 2),
            imputed_pressure=round(float(imputed_p), 2),
            imputed_humidity=round(float(imputed_rh), 2),
            imputation_reason=reason,
            imputation_confidence=round(decision.confidence, 4),
            processed_at=datetime.now(timezone.utc),
        )
