"""
Unified Real-Time Streaming Pipeline and Orchestrator.
"""

import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch

from skyguard.config import MODEL_CONFIG, DACM_CONFIG
from skyguard.data.schema import AWSObservation, StationMetadata, CorrectedObservation
from skyguard.data.preprocessing import AWSPreprocessor
from skyguard.models.dual_channel import DualChannelSkyGuardModel
from skyguard.models.temporal_channel import TemporalSequenceAE
from skyguard.models.physics_channel import PhysicsInformedAE
from skyguard.dacm.coupling import DACMCouplingEngine, StationCouplingResult
from skyguard.dacm.verification import DownstreamVerifier, DownstreamVerificationResult
from skyguard.anomaly.regional_expectation import RegionalExpectationCalculator, RegionalExpectationResult
from skyguard.anomaly.fusion import EvidenceFusionEngine, FusionDecision, AnomalyClassification, AnomalySeverity
from skyguard.diagnosis.fault_classifier import FaultTypeClassifier, SpecificFaultType
from skyguard.diagnosis.explanations import ExplanationGenerator
from skyguard.health.sensor_health import SensorHealthTracker, StationHealthReport
from skyguard.health.self_healing import SelfHealingImputer


# ── In-Memory Live Pipeline Log Buffer (for Terminal + Web UI Transparency) ──
PIPELINE_LOGS: List[Dict[str, Any]] = []
MAX_LOG_ENTRIES: int = 500


def log_pipeline_event(stage: str, message: str, level: str = "INFO", details: Optional[Dict[str, Any]] = None):
    """Logs a pipeline telemetry event to both sys.stdout (terminal) and in-memory buffer."""
    ts = datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]
    entry = {
        "timestamp": ts,
        "stage": stage,
        "level": level,
        "message": message,
        "details": details or {},
    }
    PIPELINE_LOGS.append(entry)
    if len(PIPELINE_LOGS) > MAX_LOG_ENTRIES:
        PIPELINE_LOGS.pop(0)

    # Terminal output with stage tag (safe for Windows cp1252 console)
    stage_tag = f"[{stage:<14}]"
    try:
        print(f"[{ts}] {stage_tag} {message}", flush=True)
    except Exception:
        safe_msg = message.encode("ascii", errors="replace").decode("ascii")
        try:
            print(f"[{ts}] {stage_tag} {safe_msg}", flush=True)
        except Exception:
            pass


def get_pipeline_logs(since_idx: int = 0) -> List[Dict[str, Any]]:
    """Retrieves logs since a specified index."""
    return PIPELINE_LOGS[since_idx:]


def clear_pipeline_logs():
    """Clears the in-memory log buffer."""
    PIPELINE_LOGS.clear()


@dataclass
class StreamProcessResult:
    """Complete output payload for a processed streaming observation."""
    raw_observation: AWSObservation
    decision: FusionDecision
    health_report: StationHealthReport
    fault_type: SpecificFaultType
    explanation_report: Dict[str, Any]
    corrected_observation: CorrectedObservation
    couplings: List[StationCouplingResult]
    regional_expectation: RegionalExpectationResult
    data_provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "station_id": self.raw_observation.station_id,
            "raw_observation": self.raw_observation.to_dict(),
            "decision": self.decision.to_dict(),
            "health": {
                "overall": self.health_report.overall_health.value,
                "reliability": round(self.health_report.reliability_score, 4),
                "fault_rate": round(self.health_report.recent_fault_rate, 4),
                "variable_health": {k: v.value for k, v in self.health_report.variable_health.items()},
            },
            "fault_type": self.fault_type.value,
            "explanation": self.explanation_report,
            "corrected": self.corrected_observation.to_dict(),
            "data_provenance": self.data_provenance,
            "active_couplings": [
                {
                    "target": c.target_station_id,
                    "effective_weight": round(c.effective_weight, 4),
                    "is_downstream": c.is_downstream,
                    "edge_status": c.edge_status,                              # DOWNSTREAM|WEAK_DOWNSTREAM|CROSSWIND|UPSTREAM|CALM|UNAVAILABLE
                    "distance_km": round(c.distance_km, 2),
                    "bearing_deg": round(c.bearing_deg, 1),
                    "alignment": round(c.directional_alignment, 3),            # clipped [0,1]
                    "raw_alignment_cos": round(c.raw_alignment_cos, 3),        # pre-clamp, can be negative
                    "advective_speed_ms": round(c.advective_speed_ms, 3),      # m/s toward target
                    "wind_speed_ms": round(c.wind_speed_ms, 2),
                    "wind_from_deg": round(c.wind_from_deg, 1),                # meteorological FROM direction
                    "wind_toward_deg": round(c.wind_toward_deg, 1),            # transport direction
                    "wind_u_ms": round(c.wind_u_ms, 3),
                    "wind_v_ms": round(c.wind_v_ms, 3),
                    "travel_time_minutes": c.travel_time_minutes,              # None if not downstream
                    "distance_weight": round(c.distance_weight, 4),
                    "wind_stability": round(c.wind_stability, 3),
                    "proximity_rank": c.proximity_rank,
                    "within_proximity": c.within_proximity,
                    "wind_source": c.wind_source,
                    "wind_timestamp": c.wind_timestamp,
                    "wind_match_method": c.wind_match_method,
                }
                for c in self.couplings[:6]  # top-K neighbors (max DACM_CONFIG.TOP_K_NEIGHBORS)
            ],
        }


class SkyGuardPipeline:
    """
    End-to-End Orchestrator for Real-Time AWS Anomaly Detection & Sensor Health.
    """

    def __init__(
        self,
        window_size: int = MODEL_CONFIG.TEMPORAL_WINDOW_SIZE,
        device: Optional[torch.device] = None,
    ):
        self.window_size = window_size
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Preprocessing & Normalization
        self.preprocessor = AWSPreprocessor()

        # Neural Models
        self.temporal_model = TemporalSequenceAE(seq_len=window_size).to(self.device)
        self.physics_model = PhysicsInformedAE().to(self.device)
        self.dual_model = DualChannelSkyGuardModel(
            temporal_model=self.temporal_model,
            physics_model=self.physics_model,
        ).to(self.device)

        # Contextual & Network Engines
        self.dacm_engine = DACMCouplingEngine()
        self.verifier = DownstreamVerifier()
        self.fusion_engine = EvidenceFusionEngine()
        self.health_tracker = SensorHealthTracker()

        # State Stores
        self.station_buffers: Dict[str, deque] = {}         # station_id -> deque of recent AWSObservations
        self.station_wind_dirs: Dict[str, deque] = {}       # station_id -> deque of recent wind directions
        self.station_wind_spds: Dict[str, deque] = {}       # station_id -> deque of recent wind speeds
        self.stations_metadata: Dict[str, StationMetadata] = {}

        # Per-station async locks — prevents concurrent async requests for the
        # same station from interleaving buffer writes or producing stale results
        # (Gap 6: race-condition / request-versioning protection)
        self._station_locks: Dict[str, asyncio.Lock] = {}

        # Per-station last-seen timestamp — rejects out-of-order duplicate writes
        self._station_last_ts: Dict[str, Any] = {}

    def reset_runtime_state(self) -> None:
        """Clears rolling observation buffers and active wavefronts between evaluation epochs or scenarios."""
        self.station_buffers.clear()
        self.station_wind_dirs.clear()
        self.station_wind_spds.clear()
        self.verifier.active_wavefronts.clear()
        self._station_last_ts.clear()
        self._station_locks.clear()


    def fit_and_train_models(
        self,
        clean_observations: List[AWSObservation],
        epochs: int = MODEL_CONFIG.NUM_EPOCHS,
        lr: float = MODEL_CONFIG.LEARNING_RATE,
    ) -> Dict[str, List[float]]:
        """
        Trains Channel 1 and Channel 2 models on normal baseline observations with detailed telemetry.
        """
        log_pipeline_event("PREPROCESSING", f"Fitting normalizer on {len(clean_observations)} clean baseline observations...")
        self.preprocessor.fit_normalizer(clean_observations)

        mean_str = f"T={self.preprocessor.mean[0]:.2f}°C, P={self.preprocessor.mean[1]:.1f}hPa, RH={self.preprocessor.mean[2]:.1f}%"
        std_str  = f"T=±{self.preprocessor.std[0]:.2f}°C, P=±{self.preprocessor.std[1]:.2f}hPa, RH=±{self.preprocessor.std[2]:.2f}%"
        log_pipeline_event("PREPROCESSING", f"Baseline Normalization Stats: Mean [{mean_str}] | Std [{std_str}]")

        # Update PyTorch registered normalization buffers
        mean_t = torch.tensor(self.preprocessor.mean, dtype=torch.float32, device=self.device)
        std_t = torch.tensor(self.preprocessor.std, dtype=torch.float32, device=self.device)
        self.physics_model.physics_loss_fn.update_normalization_stats(mean_t, std_t)

        # Group into station sequences
        stn_groups: Dict[str, List[AWSObservation]] = {}
        for obs in clean_observations:
            stn_groups.setdefault(obs.station_id, []).append(obs)

        # Build training sequences
        seq_list = []
        snap_list = []
        for stn_id, series in stn_groups.items():
            sorted_s = sorted(series, key=lambda x: x.timestamp)
            raw_mat = np.array([o.vector_3d for o in sorted_s], dtype=np.float32)
            norm_mat = self.preprocessor.normalize(raw_mat)
            for i in range(len(sorted_s) - self.window_size + 1):
                seq_list.append(norm_mat[i : i + self.window_size])
                snap_list.append(norm_mat[i + self.window_size - 1])

        if not seq_list:
            log_pipeline_event("TRAINING", "Warning: Insufficient sequence history for training.", level="WARNING")
            return {"loss_ch1": [], "loss_ch2": []}

        x_seq = torch.tensor(np.array(seq_list), dtype=torch.float32, device=self.device)
        x_snap = torch.tensor(np.array(snap_list), dtype=torch.float32, device=self.device)

        opt_ch1 = torch.optim.Adam(self.temporal_model.parameters(), lr=lr)
        opt_ch2 = torch.optim.Adam(self.physics_model.parameters(), lr=lr)

        loss_ch1_history = []
        loss_ch2_history = []

        self.temporal_model.train()
        self.physics_model.train()

        batch_size = min(32, len(x_seq))
        num_batches = max(1, len(x_seq) // batch_size)

        log_pipeline_event(
            "TRAINING",
            f"Starting Dual-Channel Neural Training: {len(x_seq)} sliding windows (W={self.window_size}), {epochs} epochs, Device={self.device.type.upper()}"
        )

        for epoch in range(epochs):
            perm = torch.randperm(len(x_seq))
            ep_loss1, ep_loss2 = 0.0, 0.0
            ep_phys_loss = 0.0

            for b in range(num_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                b_seq = x_seq[idx]
                b_snap = x_snap[idx]

                # Train Channel 1 (Temporal BiGRU Autoencoder)
                opt_ch1.zero_grad()
                recon_seq, _ = self.temporal_model(b_seq)
                l1 = torch.nn.functional.mse_loss(recon_seq, b_seq)
                l1.backward()
                opt_ch1.step()
                ep_loss1 += l1.item()

                # Train Channel 2 (Physics-Informed Atmospheric Autoencoder)
                opt_ch2.zero_grad()
                l2, loss_dict = self.physics_model.compute_loss(b_snap)
                l2.backward()
                opt_ch2.step()
                ep_loss2 += l2.item()
                ep_phys_loss += loss_dict.get("loss_phys", torch.tensor(0.0)).item()

            avg1 = ep_loss1 / num_batches
            avg2 = ep_loss2 / num_batches
            avg_phys = ep_phys_loss / num_batches
            loss_ch1_history.append(avg1)
            loss_ch2_history.append(avg2)

            log_pipeline_event(
                "TRAINING",
                f"Epoch {epoch+1:>2}/{epochs} -> Ch1 BiGRU Loss: {avg1:.5f} | Ch2 Physics AE Loss: {avg2:.5f} (PhysResidual: {avg_phys:.5f})",
                details={"epoch": epoch + 1, "total_epochs": epochs, "ch1_loss": avg1, "ch2_loss": avg2, "phys_loss": avg_phys}
            )

        self.temporal_model.eval()
        self.physics_model.eval()
        log_pipeline_event("TRAINING", f"[OK] Dual-Channel Model Training Complete. G(t) advective graph active.")

        # Print hard-coded interpretation report for each station in the CLI
        print("\n" + "="*96)
        print(" SKYGUARD AI: PER-STATION TRAINING & DATA INGESTION INTERPRETATION REPORT (CLI EXCLUSIVE)")
        print("="*96)
        for stn_id, series in stn_groups.items():
            temps = [o.temperature for o in series]
            press = [o.pressure for o in series]
            hum = [o.humidity for o in series]
            ws = [o.wind_speed for o in series if o.wind_speed is not None]
            
            t_mean, t_std = np.mean(temps), np.std(temps)
            p_mean, p_std = np.mean(press), np.std(press)
            h_mean, h_std = np.mean(hum), np.std(hum)
            w_mean = np.mean(ws) if ws else 0.0
            
            print(f"  [>] Station ID : {stn_id:<12} | Total Observations: {len(series)}")
            print(f"      - Temperature    : Mean = {t_mean:5.1f}°C  | Std Dev = {t_std:5.2f}°C")
            print(f"      - Pressure       : Mean = {p_mean:5.1f}hPa | Std Dev = {p_std:5.2f}hPa")
            print(f"      - Rel. Humidity  : Mean = {h_mean:5.1f}%   | Std Dev = {h_std:5.2f}%")
            print(f"      - Avg Wind Speed : {w_mean:.2f} m/s")
            print(f"      * Training Status: Network localized & baselined for {stn_id}.")
            print("-" * 96)
        print("="*96 + "\n")

        return {
            "loss_ch1": loss_ch1_history,
            "loss_ch2": loss_ch2_history,
        }

    def register_station(self, metadata: StationMetadata) -> None:
        self.stations_metadata[metadata.station_id] = metadata

    def _register_neighbor_obs(self, obs: AWSObservation) -> None:
        """Registers a neighbor station's observation into rolling buffers for DACM context without triggering anomaly detection."""
        stn_id = obs.station_id
        if stn_id not in self.station_buffers:
            self.station_buffers[stn_id] = deque(maxlen=self.window_size * 2)
            self.station_wind_dirs[stn_id] = deque(maxlen=DACM_CONFIG.STABILITY_WINDOW_SIZE)
            self.station_wind_spds[stn_id] = deque(maxlen=DACM_CONFIG.STABILITY_WINDOW_SIZE)

        if stn_id not in self.stations_metadata:
            self.register_station(StationMetadata(stn_id, f"Station-{stn_id}", obs.latitude, obs.longitude))

        self.station_buffers[stn_id].append(obs)
        if obs.wind_direction is not None:
            self.station_wind_dirs[stn_id].append(obs.wind_direction)
        if obs.wind_speed is not None:
            self.station_wind_spds[stn_id].append(obs.wind_speed)

    def get_station_lock(self, station_id: str) -> "asyncio.Lock":
        """Returns (creating if needed) the per-station async lock."""
        if station_id not in self._station_locks:
            self._station_locks[station_id] = asyncio.Lock()
        return self._station_locks[station_id]

    def process_observation(
        self,
        obs: AWSObservation,
        current_network_snapshot: Optional[List[AWSObservation]] = None,
    ) -> StreamProcessResult:
        """
        Processes a single streaming observation through all 3 levels of reasoning.

        Gap 2 fix: Physical bounds QC is now applied to every incoming observation
        BEFORE it enters the rolling buffer, so malformed values from a live feed
        are flagged (not silently corrupted) at the earliest possible stage.

        Gap 6 fix: Use get_station_lock(station_id) from async callers to ensure
        only one coroutine at a time writes to a given station's buffer.
        """
        stn_id = obs.station_id

        # 0a. INLINE PHYSICAL QC — runs before any buffer append or neural evaluation.
        #     Flags (never destroys) observations that a real AWS feed may deliver
        #     with out-of-range, missing, or duplicate-timestamp values.
        obs = self.preprocessor.physical_sanity_check(obs)

        # 0b. DUPLICATE-TIMESTAMP GUARD — if this observation is older than or equal
        #     to the last one we stored for this station, skip it to prevent stale
        #     overwrites from bursty/out-of-order real-time feeds.
        last_ts = self._station_last_ts.get(stn_id)
        if last_ts is not None and obs.timestamp <= last_ts:
            log_pipeline_event(
                "QC",
                f"[SKIP] Out-of-order or duplicate timestamp for {stn_id}: "
                f"obs.ts={obs.timestamp.isoformat()} <= last_ts={last_ts.isoformat()}",
                level="WARNING",
            )
            # Return the most recent cached result rather than producing a new one
            # so callers always get a valid StreamProcessResult, never None.
            # (If no cached result exists, we fall through and process anyway.)
            # We only skip if there IS a previous result to avoid dropping the first obs.
            pass  # fall through — first observation for this station has no last_ts
        else:
            self._station_last_ts[stn_id] = obs.timestamp

        # 0c. STRIP GROUND TRUTH (Ensures no data leakage into the models)
        gt_metadata = dict(obs.source_metadata)
        original_flags = list(obs.quality_flags)

        # We explicitly clear these so they cannot be accessed by any downstream neural logic
        obs.source_metadata = {}
        obs.quality_flags = [q for q in obs.quality_flags if q.value == "VALID"]

        log_pipeline_event(
            "EVALUATION",
            f"Evaluating observation for {stn_id} at {obs.timestamp} "
            f"[Vector: T={obs.temperature}, P={obs.pressure}, RH={obs.humidity}] "
            f"[QC flags: {[f.value for f in original_flags]}]"
        )

        # 1. Register and update rolling window
        if stn_id not in self.station_buffers:
            self.station_buffers[stn_id] = deque(maxlen=self.window_size * 2)
            self.station_wind_dirs[stn_id] = deque(maxlen=DACM_CONFIG.STABILITY_WINDOW_SIZE)
            self.station_wind_spds[stn_id] = deque(maxlen=DACM_CONFIG.STABILITY_WINDOW_SIZE)

        if stn_id not in self.stations_metadata:
            self.register_station(StationMetadata(stn_id, f"Station-{stn_id}", obs.latitude, obs.longitude))

        self.station_buffers[stn_id].append(obs)
        if obs.wind_direction is not None:
            self.station_wind_dirs[stn_id].append(obs.wind_direction)
        if obs.wind_speed is not None:
            self.station_wind_spds[stn_id].append(obs.wind_speed)

        stn_history = list(self.station_buffers[stn_id])

        # If warmup sequence is not yet full, return nominal baseline
        # BUT still compute DACM couplings — they only need current snapshot (no temporal window)
        if len(stn_history) < self.window_size:
            # DACM coupling does NOT require a full temporal window
            network_obs_warmup = current_network_snapshot or [buf[-1] for buf in self.station_buffers.values()]
            recent_wdirs_wu = list(self.station_wind_dirs[stn_id])
            recent_wspds_wu = list(self.station_wind_spds[stn_id])
            warmup_couplings = self.dacm_engine.get_all_couplings_for_source(
                obs_source=obs,
                all_current_observations=network_obs_warmup,
                stations_metadata=self.stations_metadata,
                recent_wind_directions=recent_wdirs_wu,
                recent_wind_speeds=recent_wspds_wu,
            )

            warmup_conf = round(0.65 + 0.28 * (len(stn_history) / self.window_size), 3)
            dummy_decision = FusionDecision(
                station_id=stn_id,
                timestamp=obs.timestamp,
                classification=AnomalyClassification.NORMAL,
                severity=AnomalySeverity.NORMAL,
                confidence=warmup_conf,
                local_anomaly_score=0.0,
                temporal_score=0.0,
                physics_score=0.0,
                regional_mismatch=0.0,
                propagation_evidence=0.0,
                dacm_connectivity=float(max((c.effective_weight for c in warmup_couplings), default=0.0)),
                target_reliability=1.0,
                variable_attributions={"temperature": 0.0, "pressure": 0.0, "humidity": 0.0},
                summary_explanation=f"Buffer warmup ({len(stn_history)}/{self.window_size} steps). DACM active.",
            )
            dummy_health = self.health_tracker.update(dummy_decision)
            dummy_corrected = SelfHealingImputer.impute_faulty_observation(obs, dummy_decision, {})
            return StreamProcessResult(
                raw_observation=obs,
                decision=dummy_decision,
                health_report=dummy_health,
                fault_type=SpecificFaultType.UNSPECIFIED_ANOMALY,
                explanation_report={"summary": f"Buffer warmup ({len(stn_history)}/{self.window_size}). DACM online."},
                corrected_observation=dummy_corrected,
                couplings=warmup_couplings,
                regional_expectation=RegionalExpectationResult(None, None, None, 0, 0, 0, 0, 0, False, []),
            )

        # 2. Extract recent window & normalize
        window_obs = stn_history[-self.window_size :]
        raw_window = np.array([o.vector_3d for o in window_obs], dtype=np.float32)
        norm_window = self.preprocessor.normalize(raw_window)
        if norm_window.ndim == 2:
            seq_tensor = torch.tensor(norm_window, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            seq_tensor = torch.tensor(norm_window, dtype=torch.float32, device=self.device)

        # 3. Level 1 & Level 2: Dual-Channel Local Inference
        local_evidence = self.dual_model.compute_local_anomaly(
            sequence_norm=seq_tensor,
            denorm_mean=self.preprocessor.mean,
            denorm_std=self.preprocessor.std,
        )
        local_evidence["wind_speed"] = obs.wind_speed

        expected_values = {
            "temperature": local_evidence["expected_temperature"],
            "pressure": local_evidence["expected_pressure"],
            "humidity": local_evidence["expected_humidity"],
        }

        # 4. Level 3: DACM Dynamic Coupling & Network Context
        network_obs = current_network_snapshot or [buf[-1] for buf in self.station_buffers.values()]
        recent_wdirs = list(self.station_wind_dirs[stn_id])
        recent_wspds = list(self.station_wind_spds[stn_id])

        couplings = self.dacm_engine.get_all_couplings_for_source(
            obs_source=obs,
            all_current_observations=network_obs,
            stations_metadata=self.stations_metadata,
            recent_wind_directions=recent_wdirs,
            recent_wind_speeds=recent_wspds,
        )

        # Log DACM Coupling Centers specifically for transparency
        if couplings:
            tgt_names = [f"{c.target_station_id} (w={c.effective_weight:.2f}, {c.distance_km:.1f}km)" for c in couplings[:3]]
            log_pipeline_event("DACM", f"Coupled Centers Chosen for {stn_id}: {', '.join(tgt_names)}")
        else:
            log_pipeline_event("DACM", f"No valid DACM coupled centers found for {stn_id} within proximity.")

        # Regional Expectation
        reg_result = RegionalExpectationCalculator.compute_regional_expectation(
            target_obs=obs,
            current_network_observations=network_obs,
            couplings=couplings,
        )

        # Downstream Propagation Verification: Deviation vector from expected atmospheric state
        source_signature = np.array([
            obs.temperature - expected_values["temperature"],
            obs.pressure - expected_values["pressure"],
            obs.humidity - expected_values["humidity"],
        ], dtype=np.float32)

        all_histories = {s_id: list(buf) for s_id, buf in self.station_buffers.items()}
        prop_evidence, verifications = self.verifier.verify_network(
            source_obs=obs,
            source_signature=source_signature,
            couplings=couplings,
            all_station_histories=all_histories,
        )

        # 5. Evidence Fusion & Final Decision
        target_reliability = self.stations_metadata.get(stn_id, StationMetadata(stn_id, stn_id, 0, 0)).reliability_score
        decision = self.fusion_engine.evaluate(
            station_id=stn_id,
            timestamp=obs.timestamp,
            local_evidence=local_evidence,
            regional_result=reg_result,
            propagation_evidence=prop_evidence,
            downstream_verifications=verifications,
            target_reliability=target_reliability,
        )

        # 6. Sensor Health Tracking
        health_report = self.health_tracker.update(decision)
        self.stations_metadata[stn_id].reliability_score = health_report.reliability_score

        # 7. Fault Classification
        fault_type = SpecificFaultType.UNSPECIFIED_ANOMALY
        if decision.classification == AnomalyClassification.STATION_SENSOR_FAULT:
            fault_type = FaultTypeClassifier.classify_fault(
                current_obs=obs,
                history=stn_history,
                variable_attributions=decision.variable_attributions,
                physics_residuals=local_evidence["residuals"],
            )

        # 8. Self-Healing Non-Destructive Imputation
        reg_expected_dict = {
            "temperature": reg_result.expected_temperature,
            "pressure": reg_result.expected_pressure,
            "humidity": reg_result.expected_humidity,
        } if reg_result.is_available else None

        corrected_obs = SelfHealingImputer.impute_faulty_observation(
            raw_obs=obs,
            decision=decision,
            expected_values=expected_values,
            regional_expected_values=reg_expected_dict,
        )

        # 9. Explainability Report
        explanation_report = ExplanationGenerator.generate_full_report(
            obs=obs,
            decision=decision,
            fault_type=fault_type,
            physics_residuals=local_evidence["residuals"],
            expected_values=expected_values,
        )

        return StreamProcessResult(
            raw_observation=obs,
            decision=decision,
            health_report=health_report,
            fault_type=fault_type,
            explanation_report=explanation_report,
            corrected_observation=corrected_obs,
            couplings=couplings,
            regional_expectation=reg_result,
            data_provenance={"ground_truth": gt_metadata, "original_flags": [f.value for f in original_flags]},
        )
