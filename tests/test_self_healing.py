"""
Unit Tests for Data-Level Self-Healing and Raw Data Immutability.
"""

import unittest
from datetime import datetime
from skyguard.data.schema import AWSObservation, QualityFlag
from skyguard.health.self_healing import SelfHealingImputer
from skyguard.anomaly.fusion import FusionDecision, AnomalyClassification, AnomalySeverity


class TestSelfHealing(unittest.TestCase):

    def test_raw_data_preservation_and_imputation(self):
        raw_obs = AWSObservation(
            station_id="AWS-01",
            timestamp=datetime(2026, 8, 30, 10, 0, 0),
            latitude=12.95,
            longitude=77.50,
            temperature=55.0,  # Extreme faulty spike
            pressure=1012.0,
            humidity=60.0,
        )

        decision_fault = FusionDecision(
            station_id="AWS-01",
            timestamp=raw_obs.timestamp,
            classification=AnomalyClassification.STATION_SENSOR_FAULT,
            severity=AnomalySeverity.CRITICAL,
            confidence=0.92,
            local_anomaly_score=2.8,
            temporal_score=2.5,
            physics_score=3.1,
            regional_mismatch=0.8,
            propagation_evidence=0.05,
            dacm_connectivity=0.75,
            target_reliability=0.9,
            variable_attributions={"temperature": 2.5, "pressure": 0.1, "humidity": 0.2},
            summary_explanation="Sensor fault confirmed.",
        )

        expected_values = {"temperature": 24.5, "pressure": 1012.1, "humidity": 60.2}
        regional_expected = {"temperature": 24.3, "pressure": 1011.9, "humidity": 61.0}

        corrected = SelfHealingImputer.impute_faulty_observation(
            raw_obs=raw_obs,
            decision=decision_fault,
            expected_values=expected_values,
            regional_expected_values=regional_expected,
        )

        # 1. Verify Raw data is STRICTLY preserved without modification
        self.assertEqual(corrected.raw_observation.temperature, 55.0)
        self.assertEqual(corrected.raw_observation.pressure, 1012.0)
        self.assertEqual(corrected.raw_observation.humidity, 60.0)

        # 2. Verify Imputed value is reasonable and populated
        self.assertIsNotNone(corrected.imputed_temperature)
        self.assertTrue(24.0 <= corrected.imputed_temperature <= 25.0)
        self.assertIn("temperature", corrected.imputation_reason.lower())
        self.assertEqual(corrected.imputation_confidence, 0.92)

    def test_no_imputation_on_genuine_met_event(self):
        raw_obs = AWSObservation(
            station_id="AWS-01",
            timestamp=datetime(2026, 8, 30, 10, 0, 0),
            latitude=12.95,
            longitude=77.50,
            temperature=18.0,  # Cold front drop
            pressure=1016.0,
            humidity=85.0,
        )

        decision_event = FusionDecision(
            station_id="AWS-01",
            timestamp=raw_obs.timestamp,
            classification=AnomalyClassification.GENUINE_METEOROLOGICAL_EVENT,
            severity=AnomalySeverity.HIGH,
            confidence=0.95,
            local_anomaly_score=1.8,
            temporal_score=1.6,
            physics_score=0.2,
            regional_mismatch=0.2,
            propagation_evidence=0.85,
            dacm_connectivity=0.80,
            target_reliability=1.0,
            variable_attributions={"temperature": 1.2, "pressure": 0.8, "humidity": 0.9},
            summary_explanation="Genuine cold front confirmed by downstream stations.",
        )

        corrected = SelfHealingImputer.impute_faulty_observation(
            raw_obs=raw_obs,
            decision=decision_event,
            expected_values={"temperature": 24.0, "pressure": 1012.0, "humidity": 65.0},
        )

        # Imputed value must be None because genuine events should NOT be overwritten or altered
        self.assertIsNone(corrected.imputed_temperature)
        self.assertIsNone(corrected.imputed_pressure)
        self.assertIsNone(corrected.imputed_humidity)


if __name__ == "__main__":
    unittest.main()
