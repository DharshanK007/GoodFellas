"""
Transparent Explainable Audit Trail Generator for SkyGuard AI.
"""

from typing import Dict, List, Any
from skyguard.anomaly.fusion import FusionDecision, AnomalyClassification
from skyguard.diagnosis.fault_classifier import SpecificFaultType
from skyguard.data.schema import AWSObservation


class ExplanationGenerator:
    """
    Constructs model-derived, evidence-backed narrative explanation reports.
    Strictly avoids fabricating explanations.
    """

    @classmethod
    def generate_full_report(
        cls,
        obs: AWSObservation,
        decision: FusionDecision,
        fault_type: SpecificFaultType,
        physics_residuals: Dict[str, float],
        expected_values: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Builds a structured explainability audit record.
        """
        # Calculate variable contribution percentages
        total_var_err = sum(decision.variable_attributions.values()) + 1e-6
        var_percentages = {
            var: round((err / total_var_err) * 100.0, 1)
            for var, err in decision.variable_attributions.items()
        }

        # Level 1: Temporal Narrative
        temporal_narrative = (
            f"Temporal Sequence AE yielded an anomaly score of {decision.temporal_score:.2f}. "
            f"Expected T={expected_values.get('temperature', 0.0):.1f}°C, "
            f"P={expected_values.get('pressure', 0.0):.1f}hPa, "
            f"RH={expected_values.get('humidity', 0.0):.1f}%."
        )

        # Level 2: Physics Narrative
        tv_err = physics_residuals.get("r_virtual_temp_k", 0.0)
        e_err = physics_residuals.get("r_vapor_pressure_hpa", 0.0)
        n_err = physics_residuals.get("r_refractive_index", 0.0)
        physics_narrative = (
            f"Thermodynamic constraints evaluated: Virtual Temperature residual = {tv_err:.2f} K, "
            f"Vapor Pressure residual = {e_err:.2f} hPa, "
            f"Refractive Index residual = {n_err:.2e}. "
            f"Composite physics inconsistency = {decision.physics_score:.2f}."
        )

        # Level 3: DACM & Network Narrative
        if decision.downstream_confirmations:
            confirmations_text = [
                f"- {c.target_station_id}: {c.explanation}"
                for c in decision.downstream_confirmations[:3]
            ]
            dacm_narrative = (
                f"DACM identified dynamic advective connectivity (weight sum={decision.dacm_connectivity:.2f}). "
                f"Propagation evidence score = {decision.propagation_evidence:.2f}.\n"
                + "\n".join(confirmations_text)
            )
        else:
            dacm_narrative = (
                f"DACM network coupling = {decision.dacm_connectivity:.2f}. "
                "No active downstream partners found due to calm winds, crosswind geometry, or network boundary."
            )

        # Recommendation
        if decision.classification == AnomalyClassification.GENUINE_METEOROLOGICAL_EVENT:
            action = "Dispatch regional weather advisory; observation reflects true atmospheric front."
        elif decision.classification == AnomalyClassification.STATION_SENSOR_FAULT:
            failing_var = max(decision.variable_attributions.items(), key=lambda x: x[1])[0]
            action = f"Flag {failing_var} sensor at {decision.station_id} for physical inspection/recalibration ({fault_type.value})."
        elif decision.classification == AnomalyClassification.UNCERTAIN_INSUFFICIENT_EVIDENCE:
            action = "Hold for human meteorological review; insufficient dynamic network verification."
        else:
            action = "Nominal operation; no maintenance required."

        return {
            "station_id": decision.station_id,
            "timestamp": decision.timestamp.isoformat(),
            "classification": decision.classification.value,
            "fault_archetype": fault_type.value,
            "severity": decision.severity.value,
            "confidence": decision.confidence,
            "variable_contributions_pct": var_percentages,
            "reasoning_levels": {
                "level_1_temporal": temporal_narrative,
                "level_2_thermodynamics": physics_narrative,
                "level_3_dacm_network": dacm_narrative,
            },
            "summary": decision.summary_explanation,
            "recommended_action": action,
        }
