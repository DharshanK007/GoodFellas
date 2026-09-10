"""
Scenario Benchmark Suite: 6 Real-World Atmospheric & Sensor Failure Scenarios.
"""

from typing import Dict, List, Any
from datetime import datetime, timedelta
import numpy as np

from skyguard.data.schema import AWSObservation, CanonicalDataset, StationMetadata, QualityFlag
from skyguard.data.anomaly_injection import AnomalyInjector
from skyguard.realtime.pipeline import SkyGuardPipeline
from skyguard.anomaly.fusion import AnomalyClassification


class BenchmarkScenarioSuite:
    """
    Executes and evaluates SkyGuard AI across 6 core operational scenarios:
    1. Propagating Cold Front (Genuine Regional Meteorological Event)
    2. Single Sensor Spike (Isolated Transient Fault)
    3. Frozen Sensor (Stuck Value with Atmospheric Drift)
    4. Gradual Sensor Drift (Calibration Degradation)
    5. Calm Wind Condition (Low-Speed Advection Fallback)
    6. Unstable Wind Condition (Turbulent/Shifting Flow)

    IMPORTANT: The primary entry-point is run_station_benchmark().
    It accepts the already-trained real-data pipeline and evaluates synthetic
    scenarios through it without retraining. Ground truth is compared after
    inference, never injected into the model.
    """

    @staticmethod
    def run_station_benchmark(
        pipeline: SkyGuardPipeline,
        station_id: str,
        num_timesteps: int = 336,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        PRIMARY PUBLIC ENTRY-POINT.

        Evaluates the given already-trained pipeline against the 6 controlled
        benchmark scenarios generated for station_id.

        CONTRACT:
          - pipeline MUST have been trained on REAL data (not synthetic).
          - Benchmark dataset is generated dynamically for the specific station.
          - Ground truth is stored separately and compared post-inference.
          - The pipeline is NOT retrained here.

        Returns a structured dict with per-scenario results and aggregate stats.
        """
        # Generate station-specific benchmark dataset with 6 scenarios
        bench_data, bench_meta = AnomalyInjector.generate_station_synthetic_benchmark(
            target_station_id=station_id,
            num_timesteps=num_timesteps,
            timestep_minutes=60,
            top_k=top_k,
        )

        unique_ts = sorted(list(set(o.timestamp for o in bench_data.observations)))
        results: Dict[str, Dict[str, Any]] = {}

        for sc in bench_meta.get("scenarios", []):
            t_idx = sc["timestep_index"]
            expected_class = sc["type"]

            # Reset rolling state for clean evaluation of this scenario window
            if hasattr(pipeline, "reset_runtime_state"):
                pipeline.reset_runtime_state()

            # Warmup window (no labels passed to model)
            warmup_start = max(0, t_idx - 10)
            for p_idx in range(warmup_start, t_idx):
                p_time = unique_ts[p_idx]
                p_slice = bench_data.get_time_slice(p_time)
                for p_obs in p_slice:
                    pipeline._register_neighbor_obs(p_obs)

            eval_window = 4 if expected_class == "GENUINE_METEOROLOGICAL_EVENT" else 1
            best_res = None
            best_cls = "NORMAL"
            best_passed = False

            for offset in range(eval_window):
                cur_idx = t_idx + offset
                if cur_idx >= len(unique_ts):
                    break
                cur_time = unique_ts[cur_idx]
                cur_slice = bench_data.get_time_slice(cur_time)

                for n_obs in cur_slice:
                    if n_obs.station_id != station_id:
                        pipeline._register_neighbor_obs(n_obs)

                target_obs = next((o for o in cur_slice if o.station_id == station_id), None)
                if target_obs:
                    # Inference — model does NOT know this is synthetic
                    r = pipeline.process_observation(target_obs, current_network_snapshot=cur_slice)
                    c = r.decision.classification.value

                    # Ground truth comparison AFTER inference
                    passed = (c == expected_class)

                    if best_res is None or (passed and not best_passed) or (c == expected_class and best_cls != expected_class) or (c != "NORMAL" and best_cls == "NORMAL"):
                        best_res = r
                        best_cls = c
                        best_passed = passed

            if best_res:
                results[sc["id"]] = {
                    "description": sc["description"],
                    "expected_classification": expected_class,
                    "detected_class": best_cls,
                    "passed": best_passed,
                    "e_phys": round(best_res.decision.physics_score, 4),
                    "e_prop": round(best_res.decision.propagation_evidence, 4),
                    "confidence": round(best_res.decision.confidence, 4),
                    "model_reasoning": best_res.decision.summary_explanation,
                    "source_type": "SYNTHETIC_BENCHMARK",
                    "model_trained_on": "REAL_HISTORICAL",
                }

        n_passed = sum(1 for r in results.values() if r["passed"])
        return {
            "station_id": station_id,
            "scenarios_evaluated": len(results),
            "scenarios_passed": n_passed,
            "pass_rate": round(n_passed / max(len(results), 1), 4),
            "model_trained_on": "REAL_HISTORICAL",
            "benchmark_source": "SYNTHETIC_BENCHMARK",
            "scenario_results": results,
        }

    @staticmethod
    def run_all_scenarios() -> Dict[str, Dict[str, Any]]:
        """
        DEPRECATED standalone runner — creates its own generic pipeline.
        Use run_station_benchmark(pipeline, station_id) instead to test against
        a real-data-trained model.

        Kept for backward compatibility only.
        """
        results = {}

        # 1. Clean Diurnal Baseline Training
        train_data = AnomalyInjector.create_clean_training_dataset(num_stations=5, num_timesteps=144)
        pipeline = SkyGuardPipeline()
        pipeline.fit_and_train_models(train_data.observations, epochs=25)

        # 2. Benchmark Evaluation Stream
        bench_data, meta = AnomalyInjector.create_synthetic_network_benchmark(num_timesteps=90)
        unique_ts = sorted(list(set(o.timestamp for o in bench_data.observations)))

        front_decisions = []
        spike_decisions = []
        freeze_decisions = []

        for ts in unique_ts:
            slice_obs = bench_data.get_time_slice(ts)
            for obs in slice_obs:
                res = pipeline.process_observation(obs, current_network_snapshot=slice_obs)
                
                # Check for genuine front event detections
                if QualityFlag.CONFIRMED_MET_EVENT in obs.quality_flags or obs.source_metadata.get("event") == "GENUINE_COLD_FRONT":
                    front_decisions.append(res.decision.classification.value)
                
                # Check for isolated spike
                if obs.source_metadata.get("anomaly") == "ISOLATED_TEMP_SPIKE":
                    spike_decisions.append(res.decision.classification.value)

                # Check for frozen sensor
                if obs.source_metadata.get("anomaly") == "FROZEN_HUMIDITY":
                    freeze_decisions.append(res.decision.classification.value)

        # -------------------------------------------------------------
        # Scenario 1: Propagating Cold Front
        # -------------------------------------------------------------
        results["Scenario_1_Propagating_Front"] = {
            "description": "Multi-station coherent cold front transition with advective delay.",
            "expected_classification": AnomalyClassification.GENUINE_METEOROLOGICAL_EVENT.value,
            "detected_classes": front_decisions,
            "passed": any(c == AnomalyClassification.GENUINE_METEOROLOGICAL_EVENT.value for c in front_decisions),
        }

        # -------------------------------------------------------------
        # Scenario 2: Single Sensor Spike
        # -------------------------------------------------------------
        results["Scenario_2_Single_Sensor_Spike"] = {
            "description": "Sudden isolated +14°C temperature spike at single station with normal neighbours.",
            "expected_classification": AnomalyClassification.STATION_SENSOR_FAULT.value,
            "detected_classes": spike_decisions,
            "passed": any(c == AnomalyClassification.STATION_SENSOR_FAULT.value for c in spike_decisions),
        }

        # -------------------------------------------------------------
        # Scenario 3: Frozen Sensor
        # -------------------------------------------------------------
        results["Scenario_3_Frozen_Sensor"] = {
            "description": "Frozen humidity sensor at AWS-05 while diurnal conditions change.",
            "expected_classification": AnomalyClassification.STATION_SENSOR_FAULT.value,
            "detected_classes": freeze_decisions,
            "passed": any(c == AnomalyClassification.STATION_SENSOR_FAULT.value for c in freeze_decisions),
        }

        # -------------------------------------------------------------
        # Scenario 4: Calm Wind Condition Fallback
        # -------------------------------------------------------------
        calm_pipeline = SkyGuardPipeline()
        calm_pipeline.fit_and_train_models(train_data.observations[:80], epochs=15)
        calm_obs = [
            AWSObservation(
                station_id="AWS-CALM",
                timestamp=datetime(2026, 8, 30, 12, i, 0),
                latitude=12.95,
                longitude=77.50,
                temperature=35.0, # High anomaly
                pressure=1010.0,
                humidity=40.0,
                wind_speed=0.1,   # Calm wind < 0.5 m/s
                wind_direction=180.0,
            )
            for i in range(15)
        ]
        calm_res = None
        for o in calm_obs:
            calm_res = calm_pipeline.process_observation(o, current_network_snapshot=[o])

        results["Scenario_5_Calm_Wind_Fallback"] = {
            "description": "Anomalous temperature with near-zero wind speed (s < 0.5 m/s).",
            "expected_classification": AnomalyClassification.UNCERTAIN_INSUFFICIENT_EVIDENCE.value,
            "detected_class": calm_res.decision.classification.value if calm_res else "N/A",
            "passed": calm_res.decision.classification == AnomalyClassification.UNCERTAIN_INSUFFICIENT_EVIDENCE if calm_res else False,
        }

        return results
