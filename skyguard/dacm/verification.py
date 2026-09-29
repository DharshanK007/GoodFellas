"""
Downstream & Upstream Propagation Verification and Wavefront Tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from skyguard.dacm.coupling import StationCouplingResult
from skyguard.dacm.propagation import AdvectivePropagationEstimator, PropagationWindow
from skyguard.data.schema import AWSObservation


@dataclass
class DownstreamVerificationResult:
    """Downstream/Upstream verification evidence for a single candidate."""
    source_station_id: str
    target_station_id: str
    dacm_weight: float
    propagation_window: PropagationWindow
    observed_lag_minutes: Optional[float]
    timing_consistency: float
    signature_similarity: float
    propagation_evidence: float
    confirmed: bool
    explanation: str


@dataclass
class AdvectiveWavefront:
    """Tracked atmospheric transition wavefront in the station network."""
    origin_station_id: str
    timestamp: datetime
    signature: np.ndarray
    wind_speed: float
    wind_direction: float
    confirmed_stations: List[str] = field(default_factory=list)
    is_active: bool = True


SIG_SCALE = np.array([2.0, 1.0, 15.0], dtype=np.float32)


class DownstreamVerifier:
    """
    Verifies whether an atmospheric transition signature propagates coherently
    across the station network by combining:
    1. Upstream backward verification (did an upstream station trigger this arrival?)
    2. Forward active wavefront tracking (did downstream stations subsequently confirm?)
    """

    def __init__(self, propagation_estimator: Optional[AdvectivePropagationEstimator] = None):
        self.estimator = propagation_estimator or AdvectivePropagationEstimator()
        self.active_wavefronts: List[AdvectiveWavefront] = []

    @staticmethod
    def calculate_event_signature(
        current_obs: AWSObservation, reference_obs: AWSObservation
    ) -> np.ndarray:
        """
        Calculates multivariate transition vector E = [Delta_T, Delta_P, Delta_RH].
        """
        dt = current_obs.temperature - reference_obs.temperature
        dp = current_obs.pressure - reference_obs.pressure
        drh = current_obs.humidity - reference_obs.humidity
        return np.array([dt, dp, drh], dtype=np.float32)

    @classmethod
    def compute_signature_similarity(cls, sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """
        Computes directional, sign, and magnitude similarity between scaled event signatures.
        """
        s_a = sig_a / SIG_SCALE
        s_b = sig_b / SIG_SCALE
        norm_a = float(np.linalg.norm(s_a))
        norm_b = float(np.linalg.norm(s_b))

        # Atmospheric event transitions must be substantial on both stations
        if norm_a < 0.60 or norm_b < 0.60:
            return 0.0

        # Physical multi-variable coupling: if event has significant pressure surge, downstream must corroborate pressure
        if abs(sig_a[1]) >= 0.70 and abs(sig_b[1]) < 0.45:
            return 0.0
        if abs(sig_b[1]) >= 0.70 and abs(sig_a[1]) < 0.45:
            return 0.0

        # Pressure sign consistency
        if abs(sig_a[1]) >= 0.50 and abs(sig_b[1]) >= 0.50:
            if (sig_a[1] > 0 and sig_b[1] < 0) or (sig_a[1] < 0 and sig_b[1] > 0):
                return 0.0

        # Temperature sign consistency for significant thermal shifts
        if abs(sig_a[0]) >= 1.5 and abs(sig_b[0]) >= 1.5:
            if (sig_a[0] > 0 and sig_b[0] < 0) or (sig_a[0] < 0 and sig_b[0] > 0):
                return 0.0

        cos_sim = float(np.dot(s_a, s_b) / (norm_a * norm_b))
        if cos_sim < 0.50:
            return 0.0

        # Relative magnitude ratio: require consistent physical event scale
        ratio = min(norm_a, norm_b) / max(norm_a, norm_b)
        if ratio < 0.30:
            return 0.0

        similarity = cos_sim * ratio
        return float(np.clip(similarity, 0.0, 1.0))

    @classmethod
    def compute_timing_consistency(
        cls, observed_lag_min: float, expected_tau_min: float, uncertainty_min: float
    ) -> float:
        diff = abs(observed_lag_min - expected_tau_min)
        sigma = max(15.0, uncertainty_min)
        score = float(np.exp(-diff / sigma))
        return float(np.clip(score, 0.0, 1.0))

    def register_wavefront(self, obs: AWSObservation, signature: np.ndarray) -> None:
        dt, dp = abs(signature[0]), abs(signature[1])
        is_substantial = (
            (dp >= 0.80 and signature[0] <= 0.2)  # Frontal pressure surge with cooling or capped heating
            or dp >= 1.20
            or (dt >= 4.0 and dp >= 0.60)
        )

        # Prune old wavefronts (> 3 hours old)
        cutoff = obs.timestamp - timedelta(hours=3)
        self.active_wavefronts = [w for w in self.active_wavefronts if w.timestamp >= cutoff]

        if is_substantial:
            existing = next(
                (w for w in self.active_wavefronts if w.origin_station_id == obs.station_id and abs((obs.timestamp - w.timestamp).total_seconds()) <= 3600),
                None,
            )
            if not existing:
                wf = AdvectiveWavefront(
                    origin_station_id=obs.station_id,
                    timestamp=obs.timestamp,
                    signature=signature,
                    wind_speed=obs.wind_speed or 0.0,
                    wind_direction=obs.wind_direction or 0.0,
                    confirmed_stations=[obs.station_id],
                    is_active=True,
                )
                self.active_wavefronts.append(wf)

    def verify_upstream_origin(
        self,
        current_obs: AWSObservation,
        current_sig: np.ndarray,
        all_station_histories: Dict[str, List[AWSObservation]],
    ) -> Tuple[float, Optional[str]]:
        """
        Checks whether current station's anomaly was caused by an upstream event.
        """
        best_evidence = 0.0
        best_origin = None

        for wf in self.active_wavefronts:
            if wf.origin_station_id == current_obs.station_id:
                continue

            origin_hist = all_station_histories.get(wf.origin_station_id, [])
            if not origin_hist:
                continue
            origin_obs = origin_hist[-1]

            from skyguard.dacm.geometry import haversine_distance_km, station_direction_vector
            from skyguard.dacm.wind_vector import meteorological_wind_to_vector
            from skyguard.dacm.alignment import compute_wind_alignment

            dist_km = haversine_distance_km(
                origin_obs.latitude, origin_obs.longitude, current_obs.latitude, current_obs.longitude
            )
            r_vec, _ = station_direction_vector(
                origin_obs.latitude, origin_obs.longitude, current_obs.latitude, current_obs.longitude
            )
            v_orig = meteorological_wind_to_vector(wf.wind_speed, wf.wind_direction)
            align, is_calm = compute_wind_alignment(v_orig, r_vec, wf.wind_speed)

            # Require directional downstream corridor alignment >= 0.40
            if align >= 0.40 and not is_calm:
                prop_win = self.estimator.estimate_travel_time(
                    dist_km, wf.wind_speed, align, wind_stability=0.9
                )
                if prop_win.is_valid:
                    lag_min = (current_obs.timestamp - wf.timestamp).total_seconds() / 60.0
                    if (prop_win.min_arrival_minutes - 15.0) <= lag_min <= (prop_win.max_arrival_minutes + 30.0):
                        t_cons = self.compute_timing_consistency(lag_min, prop_win.tau_minutes, prop_win.uncertainty_minutes)
                        sig_sim = self.compute_signature_similarity(wf.signature, current_sig)
                        evidence = align * t_cons * sig_sim
                        if evidence > best_evidence:
                            best_evidence = evidence
                            best_origin = wf.origin_station_id
                            if evidence >= 0.35 and current_obs.station_id not in wf.confirmed_stations:
                                wf.confirmed_stations.append(current_obs.station_id)

        return float(np.clip(best_evidence, 0.0, 1.0)), best_origin

    def verify_network(
        self,
        source_obs: AWSObservation,
        source_signature: np.ndarray,
        couplings: List[StationCouplingResult],
        all_station_histories: Dict[str, List[AWSObservation]],
    ) -> Tuple[float, List[DownstreamVerificationResult]]:
        """
        Evaluates propagation evidence by checking both upstream origins and downstream wavefront confirmations.
        """
        # Register new potential wavefront
        effective_sig = source_signature
        source_hist = all_station_histories.get(source_obs.station_id, [])
        if len(source_hist) >= 2:
            prev_obs = next((o for o in reversed(source_hist) if o.timestamp < source_obs.timestamp), source_hist[0])
            phys_sig = self.calculate_event_signature(source_obs, prev_obs)
            if len(source_hist) >= 3:
                pre_2 = next((o for o in reversed(source_hist) if (source_obs.timestamp - o.timestamp).total_seconds() >= 7200), prev_obs)
                sig_2 = self.calculate_event_signature(source_obs, pre_2)
                if np.linalg.norm(sig_2 / SIG_SCALE) > np.linalg.norm(phys_sig / SIG_SCALE):
                    phys_sig = sig_2
            if np.linalg.norm(phys_sig / SIG_SCALE) >= 1.5:
                effective_sig = phys_sig

        self.register_wavefront(source_obs, effective_sig)

        # 1. Check if this is an arrival of an upstream wavefront
        upstream_evidence, upstream_origin = self.verify_upstream_origin(
            source_obs, effective_sig, all_station_histories
        )

        # 2. Check downstream coupled partners in all_station_histories
        downstream_couplings = [c for c in couplings if c.is_downstream and c.effective_weight >= 0.08]
        downstream_evidence = 0.0

        # Only evaluate wavefronts originating from this station within recent 2 hours
        relevant_wfs = [
            wf for wf in self.active_wavefronts
            if wf.origin_station_id == source_obs.station_id
            and 0.0 <= (source_obs.timestamp - wf.timestamp).total_seconds() <= 7200
        ]

        for wf in relevant_wfs:
            for c in downstream_couplings:
                target_hist = all_station_histories.get(c.target_station_id, [])
                if len(target_hist) >= 2:
                    prop_win = self.estimator.estimate_travel_time(
                        distance_km=c.distance_km,
                        wind_speed_ms=wf.wind_speed or source_obs.wind_speed or 0.0,
                        directional_alignment=c.directional_alignment,
                        wind_stability=0.9,
                    )
                    if prop_win.is_valid:
                        # Check transitions in downstream history
                        for i in range(1, len(target_hist)):
                            n_obs = target_hist[i]
                            if n_obs.timestamp >= wf.timestamp:
                                n_prev = target_hist[i - 1]
                                n_sig = np.array([
                                    n_obs.temperature - n_prev.temperature,
                                    n_obs.pressure - n_prev.pressure,
                                    n_obs.humidity - n_prev.humidity,
                                ], dtype=np.float32)
                                sig_sim = self.compute_signature_similarity(wf.signature, n_sig)
                                lag_min = max(0.0, (n_obs.timestamp - wf.timestamp).total_seconds() / 60.0)
                                t_cons = self.compute_timing_consistency(lag_min, prop_win.tau_minutes, max(60.0, prop_win.uncertainty_minutes))
                                match_score = c.directional_alignment * sig_sim * t_cons
                                if match_score >= 0.15:
                                    if c.target_station_id not in wf.confirmed_stations:
                                        wf.confirmed_stations.append(c.target_station_id)

            downstream_stns = [s for s in wf.confirmed_stations if s != wf.origin_station_id]
            if len(downstream_stns) >= 2:
                cnt = len(downstream_stns)
                wf_evidence = min(0.95, 0.65 + 0.15 * cnt)
                downstream_evidence = max(downstream_evidence, wf_evidence)
            elif len(downstream_stns) == 1:
                downstream_evidence = max(downstream_evidence, 0.20)

        composite_prop_score = max(upstream_evidence, downstream_evidence)

        # Build verification report entries
        results = []
        for c in downstream_couplings:
            prop_win = self.estimator.estimate_travel_time(
                distance_km=c.distance_km,
                wind_speed_ms=source_obs.wind_speed or 0.0,
                directional_alignment=c.directional_alignment,
                wind_stability=c.wind_stability,
            )
            is_confirmed = composite_prop_score >= 0.35
            exp = (
                f"Dynamic coupling active toward {c.target_station_id} (weight={c.effective_weight:.2f}, "
                f"expected travel time={prop_win.tau_minutes:.1f} min)."
            )
            if upstream_origin:
                exp = f"Confirmed advection arrival originating from upstream station {upstream_origin}."
            elif is_confirmed:
                exp = f"Confirmed advection propagation to downstream station {c.target_station_id}."

            results.append(
                DownstreamVerificationResult(
                    source_station_id=c.source_station_id,
                    target_station_id=c.target_station_id,
                    dacm_weight=c.effective_weight,
                    propagation_window=prop_win,
                    observed_lag_minutes=None,
                    timing_consistency=composite_prop_score,
                    signature_similarity=composite_prop_score,
                    propagation_evidence=composite_prop_score,
                    confirmed=is_confirmed,
                    explanation=exp,
                )
            )

        return composite_prop_score, results

