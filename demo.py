"""
SkyGuard AI Demonstration Runner & CLI Benchmark Viewer.
"""

import sys
import time
import argparse
from datetime import datetime

from skyguard.data.anomaly_injection import AnomalyInjector
from skyguard.realtime.pipeline import SkyGuardPipeline
from skyguard.evaluation.scenario_tests import BenchmarkScenarioSuite
from skyguard.evaluation.ablation_channel2 import Channel2AblationRunner
from skyguard.evaluation.ablation_dacm import DACMAblationRunner


def print_banner():
    print("=" * 80)
    print("  SKYGUARD AI: PHYSICS-INFORMED REAL-TIME ANOMALY & SENSOR HEALTH PLATFORM  ")
    print("  Dual-Channel Autoencoder + Dynamic Advective Coupling Mechanism (DACM)    ")
    print("=" * 80)


def run_cli_demo():
    print_banner()
    print("\n[Phase 1] Generating 5-Station AWS Network with Propagating Cold Front & Faults...")
    dataset, meta = AnomalyInjector.create_synthetic_network_benchmark(num_stations=5, num_timesteps=80)
    print(f"  > Initialized {len(dataset.stations)} AWS stations across East-West advective baseline.")
    print(f"  > Total ingested meteorological timesteps: {meta['num_timesteps']}")
    print(f"  > Embedded Genuine Front: AWS-01 (t=40) -> AWS-02 (t=47) -> AWS-04 (t=56)")
    print(f"  > Embedded Faults: AWS-03 Temp Spike (t=30), AWS-05 Frozen Humidity (t=60..80)")

    print("\n[Phase 2] Training Dual-Channel Models (Channel 1 Temporal + Channel 2 Physics)...")
    clean_train_data = AnomalyInjector.create_clean_training_dataset(num_stations=5, num_timesteps=144)
    pipeline = SkyGuardPipeline()
    train_obs = clean_train_data.observations
    start_train = time.time()
    loss_history = pipeline.fit_and_train_models(train_obs, epochs=30)
    train_dur = time.time() - start_train
    print(f"  > Training completed in {train_dur:.2f}s.")
    print(f"  > Final Channel 1 MSE Loss: {loss_history['loss_ch1'][-1]:.4f}")
    print(f"  > Final Channel 2 Physics Loss: {loss_history['loss_ch2'][-1]:.4f}")

    print("\n[Phase 3] Running Real-Time Streaming Simulation through DACM & Evidence Fusion...")
    unique_ts = sorted(list(set(o.timestamp for o in dataset.observations)))

    event_counts = {"GENUINE_METEOROLOGICAL_EVENT": 0, "STATION_SENSOR_FAULT": 0, "UNCERTAIN": 0, "NORMAL": 0}
    imputations = 0

    for idx, ts in enumerate(unique_ts):
        slice_obs = dataset.get_time_slice(ts)
        for obs in slice_obs:
            res = pipeline.process_observation(obs, current_network_snapshot=slice_obs)
            cls_name = res.decision.classification.value
            if "GENUINE" in cls_name:
                event_counts["GENUINE_METEOROLOGICAL_EVENT"] += 1
            elif "FAULT" in cls_name:
                event_counts["STATION_SENSOR_FAULT"] += 1
                if res.corrected_observation.imputed_temperature is not None:
                    imputations += 1
            elif "UNCERTAIN" in cls_name:
                event_counts["UNCERTAIN"] += 1
            else:
                event_counts["NORMAL"] += 1

            # Print highlights for key events
            if idx in (30, 40, 47, 56, 65) and obs.station_id in ("AWS-01", "AWS-02", "AWS-03", "AWS-04", "AWS-05"):
                if res.decision.classification.value != "NORMAL":
                    print(f"  [t={idx:02d} | {obs.station_id}] -> {res.decision.classification.value} (Conf: {res.decision.confidence:.2f})")
                    print(f"    Reasoning: {res.decision.summary_explanation}")
                    if res.fault_type.value != "UNSPECIFIED_ANOMALY":
                        print(f"    Diagnosed Fault: {res.fault_type.value}")

    print("\n[Phase 4] Execution Summary:")
    print(f"  > Genuine Meteorological Front Detections: {event_counts['GENUINE_METEOROLOGICAL_EVENT']}")
    print(f"  > Isolated Sensor Fault Detections:       {event_counts['STATION_SENSOR_FAULT']}")
    print(f"  > Non-Destructive Imputations Generated:  {imputations}")

    print("\n[Phase 5] Running Operational Scenario Suite...")
    scenarios = BenchmarkScenarioSuite.run_all_scenarios()
    for name, s in scenarios.items():
        status = "PASSED" if s["passed"] else "FAILED"
        print(f"  - {name}: [{status}] - {s['description']}")

    print("\n[Phase 6] Running DACM Dynamic Coupling Ablation Benchmark...")
    dacm_abl = DACMAblationRunner.run_dacm_ablation(dataset, meta)
    print("\n  ==========================================================================================")
    print("  Model Architecture                                   | Precision | Recall | F1-Score | FAR  ")
    print("  ==========================================================================================")
    for model_name, metrics in dacm_abl.items():
        p = metrics["precision"] * 100
        r = metrics["recall"] * 100
        f1 = metrics["f1"] * 100
        far = metrics["false_alarm_rate"] * 100
        print(f"  {model_name:<50} | {p:>8.1f}% | {r:>5.1f}% | {f1:>7.1f}% | {far:>4.1f}%")
    print("  ==========================================================================================\n")


def start_server(port: int = 8000):
    import uvicorn
    print_banner()
    print(f"\n[Dashboard Server] Starting SkyGuard AI interactive web dashboard on http://127.0.0.1:{port} ...")
    uvicorn.run("skyguard.dashboard.app:app", host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SkyGuard AI Platform")
    parser.add_argument("--server", action="store_true", help="Launch the web dashboard server")
    parser.add_argument("--port", type=int, default=8000, help="Port for dashboard server")
    args = parser.parse_args()

    if args.server:
        start_server(port=args.port)
    else:
        run_cli_demo()
        print("To launch the interactive Glassmorphic Web Dashboard, run:")
        print("  python demo.py --server")
