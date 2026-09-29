"""
Multi-Source Evidence Fusion Engine.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
from skyguard.config import THRESHOLDS
from skyguard.anomaly.regional_expectation import RegionalExpectationResult
from skyguard.dacm.verification import DownstreamVerificationResult


class AnomalyClassification(str, Enum):
    NORMAL = "NORMAL"
    GENUINE_METEOROLOGICAL_EVENT = "GENUINE_METEOROLOGICAL_EVENT"
    STATION_SENSOR_FAULT = "STATION_SENSOR_FAULT"
    UNCERTAIN_INSUFFICIENT_EVIDENCE = "UNCERTAIN_INSUFFICIENT_EVIDENCE"


class AnomalySeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NORMAL = "NORMAL"


@dataclass
class FusionDecision:
    """Final multi-level evidence fusion output for an AWS observation."""
    station_id: str
    timestamp: datetime
    classification: AnomalyClassification
    severity: AnomalySeverity
    confidence: float
    local_anomaly_score: float
    temporal_score: float
    physics_score: float
    regional_mismatch: float
    propagation_evidence: float
    dacm_connectivity: float
    target_reliability: float
    variable_attributions: Dict[str, float]
    downstream_confirmations: List[DownstreamVerificationResult] = field(default_factory=list)
    summary_explanation: str = ""
    # Additive read-only label — identifies which cascade branch produced this decision.
    # Used by the frontend to narrate the fusion path. Does NOT affect any downstream logic.
    fusion_branch: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "station_id": self.station_id,
            "timestamp": self.timestamp.isoformat(),
            "classification": self.classification.value,
            "severity": self.severity.value,
            "confidence": round(self.confidence, 4),
            "local_anomaly_score": round(self.local_anomaly_score, 4),
            "temporal_score": round(self.temporal_score, 4),
            "physics_score": round(self.physics_score, 4),
            "regional_mismatch": round(self.regional_mismatch, 4),
            "propagation_evidence": round(self.propagation_evidence, 4),
            "dacm_connectivity": round(self.dacm_connectivity, 4),
            "target_reliability": round(self.target_reliability, 4),
            "variable_attributions": {k: round(v, 4) for k, v in self.variable_attributions.items()},
            "summary_explanation": self.summary_explanation,
            "fusion_branch": self.fusion_branch,
        }


class EvidenceFusionEngine:
    """
    Fuses Temporal (Ch1), Thermodynamic Physics (Ch2), Regional Expectation (DACM),
    and Downstream Propagation evidence into an explainable decision.
    """

    def __init__(self, thresholds=THRESHOLDS):
        self.thresholds = thresholds

    def evaluate(
        self,
        station_id: str,
        timestamp: datetime,
        local_evidence: Dict[str, Any],
        regional_result: RegionalExpectationResult,
        propagation_evidence: float,
        downstream_verifications: List[DownstreamVerificationResult],
        target_reliability: float = 1.0,
    ) -> FusionDecision:
        """
        Fuses all 3 levels of reasoning into a final classification & severity.
        """
        local_score = float(local_evidence.get("local_anomaly_score", 0.0))
        temporal_score = float(local_evidence.get("temporal_score", 0.0))
        physics_score = float(local_evidence.get("physics_score", 0.0))

        reg_mismatch = regional_result.composite_regional_mismatch if regional_result.is_available else 0.0
        dacm_conn = regional_result.active_weights_sum

        var_attrs = {
            "temperature": float(local_evidence.get("var_error_temperature", 0.0)),
            "pressure": float(local_evidence.get("var_error_pressure", 0.0)),
            "humidity": float(local_evidence.get("var_error_humidity", 0.0)),
        }

        # 1. High Propagation Confirmation -> Genuine Meteorological Event
        if propagation_evidence >= 0.35 or any(v.confirmed for v in downstream_verifications):
            classification = AnomalyClassification.GENUINE_METEOROLOGICAL_EVENT
            confidence = min(0.98, max(0.70, 0.55 + 0.40 * propagation_evidence))
            severity = AnomalySeverity.CRITICAL if local_score > 2.5 else (AnomalySeverity.HIGH if local_score > 1.2 else AnomalySeverity.MEDIUM)
            summary = (
                f"Coherent multi-station atmospheric transition verified by advective propagation "
                f"(E_prop = {propagation_evidence:.2f}) with downstream network confirmation. "
                "Regional state change is dynamically and physically consistent."
            )
            return FusionDecision(
                station_id=station_id,
                timestamp=timestamp,
                classification=classification,
                severity=severity,
                confidence=confidence,
                local_anomaly_score=local_score,
                temporal_score=temporal_score,
                physics_score=physics_score,
                regional_mismatch=reg_mismatch,
                propagation_evidence=propagation_evidence,
                dacm_connectivity=dacm_conn,
                target_reliability=target_reliability,
                variable_attributions=var_attrs,
                downstream_confirmations=downstream_verifications,
                summary_explanation=summary,
                fusion_branch="branch_1_propagation_confirmed",
            )

        # 2. Normal Baseline Check (Local model and thermodynamic physics are satisfied)
        if (local_score < 0.70 and physics_score < 0.28) or (local_score < 0.40 and physics_score < 0.35):
            # Dynamic baseline confidence: sensitively reflects local reconstruction + thermodynamic residuals
            # As sensor observations fluctuate in the microclimate, confidence actively responds to state variance
            local_norm = min(1.0, max(0.0, local_score / 0.50))
            phys_norm  = min(1.0, max(0.0, physics_score / 0.22))
            reg_norm   = min(1.0, max(0.0, reg_mismatch / 1.00)) if reg_mismatch else 0.0
            
            # Weighted atmospheric residual load
            residual_load = 0.50 * local_norm + 0.35 * phys_norm + 0.15 * reg_norm
            # Dynamic confidence spanning [0.72, 0.985] - each timestep has a distinct, responsive confidence
            normal_confidence = round(0.985 - 0.260 * (residual_load ** 0.80), 4)
            return FusionDecision(
                station_id=station_id,
                timestamp=timestamp,
                classification=AnomalyClassification.NORMAL,
                severity=AnomalySeverity.NORMAL,
                confidence=normal_confidence,
                local_anomaly_score=local_score,
                temporal_score=temporal_score,
                physics_score=physics_score,
                regional_mismatch=reg_mismatch,
                propagation_evidence=propagation_evidence,
                dacm_connectivity=dacm_conn,
                target_reliability=target_reliability,
                variable_attributions=var_attrs,
                downstream_confirmations=downstream_verifications,
                summary_explanation="Observation is consistent with learned temporal dynamics and thermodynamic state.",
                fusion_branch="branch_2_normal_baseline",
            )


        # 3. Station Sensor Fault: Thermodynamic Physics Inconsistency (Channel 2 Violation)
        if physics_score >= 0.30:
            classification = AnomalyClassification.STATION_SENSOR_FAULT
            confidence = round(min(0.99, 0.72 + 0.26 * min(1.0, (physics_score - 0.30) / 0.45)), 4)
            severity = AnomalySeverity.CRITICAL if physics_score > 0.80 or local_score > 2.5 else (AnomalySeverity.HIGH if physics_score > 0.50 else AnomalySeverity.MEDIUM)
            summary = (
                f"Thermodynamic inconsistency detected (E_phys = {physics_score:.3f}). "
                "Sensor readings violate governing thermodynamic relations "
                "(Magnus-Tetens vapor pressure, virtual temperature, or refractivity constraints)."
            )
            return FusionDecision(
                station_id=station_id,
                timestamp=timestamp,
                classification=classification,
                severity=severity,
                confidence=confidence,
                local_anomaly_score=local_score,
                temporal_score=temporal_score,
                physics_score=physics_score,
                regional_mismatch=reg_mismatch,
                propagation_evidence=propagation_evidence,
                dacm_connectivity=dacm_conn,
                target_reliability=target_reliability,
                variable_attributions=var_attrs,
                downstream_confirmations=downstream_verifications,
                summary_explanation=summary,
                fusion_branch="branch_3_physics_violation",
            )

        # 4. Station Sensor Fault: Isolated Single-Station Perturbation (DACM Ch3 No Propagation)
        is_calm = False
        if "wind_speed" in local_evidence and local_evidence["wind_speed"] is not None:
            is_calm = local_evidence["wind_speed"] < 0.5

        if (local_score >= 0.55 or temporal_score >= 0.55) and propagation_evidence < 0.30:
            if not is_calm and dacm_conn >= 0.15:
                classification = AnomalyClassification.STATION_SENSOR_FAULT
                confidence = round(min(0.97, 0.76 + 0.21 * min(1.0, local_score / 1.50)), 4)
                severity = AnomalySeverity.HIGH if local_score > 1.5 else AnomalySeverity.MEDIUM
                summary = (
                    f"Isolated station disturbance detected (local_score = {local_score:.2f}, temporal = {temporal_score:.2f}). "
                    f"DACM network verification confirmed absence of downstream advective propagation (E_prop = {propagation_evidence:.3f}). "
                    "Classified as isolated station sensor fault (hard negative)."
                )
                return FusionDecision(
                    station_id=station_id,
                    timestamp=timestamp,
                    classification=classification,
                    severity=severity,
                    confidence=confidence,
                    local_anomaly_score=local_score,
                    temporal_score=temporal_score,
                    physics_score=physics_score,
                    regional_mismatch=reg_mismatch,
                    propagation_evidence=propagation_evidence,
                    dacm_connectivity=dacm_conn,
                    target_reliability=target_reliability,
                    variable_attributions=var_attrs,
                    downstream_confirmations=downstream_verifications,
                    summary_explanation=summary,
                    fusion_branch="branch_4a_isolated_dacm_confirmed",
                )
            elif local_score >= 1.00:
                classification = AnomalyClassification.STATION_SENSOR_FAULT
                confidence = round(min(0.96, 0.72 + 0.23 * min(1.0, local_score / 1.80)), 4)
                severity = AnomalySeverity.HIGH
                summary = (
                    f"Severe local anomaly spike detected (score = {local_score:.2f}). "
                    "Single-station step deviation without network corroboration."
                )
                return FusionDecision(
                    station_id=station_id,
                    timestamp=timestamp,
                    classification=classification,
                    severity=severity,
                    confidence=confidence,
                    local_anomaly_score=local_score,
                    temporal_score=temporal_score,
                    physics_score=physics_score,
                    regional_mismatch=reg_mismatch,
                    propagation_evidence=propagation_evidence,
                    dacm_connectivity=dacm_conn,
                    target_reliability=target_reliability,
                    variable_attributions=var_attrs,
                    downstream_confirmations=downstream_verifications,
                    summary_explanation=summary,
                    fusion_branch="branch_4b_isolated_severe_spike",
                )

        # 5. Uncertain / Insufficient Evidence: Calm Winds or Low Coupling with Mild Anomaly
        if (dacm_conn < self.thresholds.CALM_COUPLING_THRESHOLD or is_calm) and physics_score < 0.30 and local_score < 1.00:
            classification = AnomalyClassification.UNCERTAIN_INSUFFICIENT_EVIDENCE
            confidence = round(0.52 + 0.16 * min(1.0, local_score / 0.80), 4)
            severity = AnomalySeverity.LOW
            summary = (
                f"Mild local anomaly detected (score = {local_score:.2f}), "
                "but spatial network verification is unavailable due to calm winds (s < 0.5 m/s) or low station coupling. "
                "Classified conservatively as UNCERTAIN."
            )
            _branch = "branch_5a_calm_or_low_coupling"
        elif local_score >= 0.60 or temporal_score >= 0.60:
            classification = AnomalyClassification.STATION_SENSOR_FAULT
            confidence = round(0.70 + 0.22 * min(1.0, max(local_score, temporal_score) / 1.60), 4)
            severity = AnomalySeverity.MEDIUM
            summary = (
                f"Localized sensor anomaly detected (score = {local_score:.2f}, temporal = {temporal_score:.2f}) "
                "without regional propagation evidence."
            )
            _branch = "branch_5b_localized_no_propagation"
        else:
            classification = AnomalyClassification.UNCERTAIN_INSUFFICIENT_EVIDENCE
            confidence = round(0.48 + 0.16 * min(1.0, (local_score + physics_score) / 1.20), 4)
            severity = AnomalySeverity.LOW
            summary = (
                f"Ambiguous observation (score = {local_score:.2f}, physics = {physics_score:.2f}), "
                f"propagation evidence = {propagation_evidence:.2f}."
            )
            _branch = "branch_5c_ambiguous"

        return FusionDecision(
            station_id=station_id,
            timestamp=timestamp,
            classification=classification,
            severity=severity,
            confidence=confidence,
            local_anomaly_score=local_score,
            temporal_score=temporal_score,
            physics_score=physics_score,
            regional_mismatch=reg_mismatch,
            propagation_evidence=propagation_evidence,
            dacm_connectivity=dacm_conn,
            target_reliability=target_reliability,
            variable_attributions=var_attrs,
            downstream_confirmations=downstream_verifications,
            summary_explanation=summary,
            fusion_branch=_branch,
        )

