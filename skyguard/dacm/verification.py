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
        Computes directional, sign, and magnitude similarity between event signatures.
        """
        norm_a = float(np.linalg.norm(sig_a))
        norm_b = float(np.linalg.norm(sig_b))

        # Atmospheric event transitions must be substantial on both stations (avoids diurnal noise matching)
        if norm_a < 3.0 or norm_b < 3.0:
            return 0.0

        # Sign consistency: if significant T or P changes have opposite signs, reject match
        if abs(sig_a[0]) >= 1.5 and abs(sig_b[0]) >= 1.5:
            if (sig_a[0] > 0 and sig_b[0] < 0) or (sig_a[0] < 0 and sig_b[0] > 0):
                return 0.0

        if abs(sig_a[1]) >= 1.0 and abs(sig_b[1]) >= 1.0:
            if (sig_a[1] > 0 and sig_b[1] < 0) or (sig_a[1] < 0 and sig_b[1] > 0):
                return 0.0

        cos_sim = float(np.dot(sig_a, sig_b) / (norm_a * norm_b))
        if cos_sim < 0.50:
            return 0.0

        # Relative magnitude ratio: require consistent physical event scale
        ratio = min(norm_a, norm_b) / max(norm_a, norm_b)
        if ratio < 0.40:
            return 0.0

        similarity = cos_sim * ratio
        return float(np.clip(similarity, 0.0, 1.0))

    @classmethod
    def compute_timing_consistency(
        cls, observed_lag_min: float, expected_tau_min: float, uncertainty_min: float
    ) -> float:
        diff = abs(observed_lag_min - expected_tau_min)
        sigma = max(3.0, uncertainty_min)
        score = float(np.exp(-diff / sigma))
        return float(np.clip(score, 0.0, 1.0))

    def register_wavefront(self, obs: AWSObservation, signature: np.ndarray) -> None:
        """Registers a newly detected atmospheric transition as an active candidate wavefront."""
        dt, dp, drh = abs(signature[0]), abs(signature[1]), abs(signature[2])
        is_substantial = (dt >= 1.0 and (dp >= 0.6 or drh >= 4.0)) or (dp >= 1.0 and drh >= 6.0)

        if is_substantial:
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
            # Prune old wavefronts (> 3 hours old)
            cutoff = obs.timestamp - timedelta(hours=3)
            self.active_wavefronts = [w for w in self.active_wavefronts if w.timestamp >= cutoff]

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

            # Require directional downstream corridor alignment >= 0.50
            if align >= 0.50 and not is_calm:
                prop_win = self.estimator.estimate_travel_time(
                    dist_km, wf.wind_speed, align, wind_stability=0.9
                )
                if prop_win.is_valid:
                    lag_min = (current_obs.timestamp - wf.timestamp).total_seconds() / 60.0
                    if (prop_win.min_arrival_minutes - 10.0) <= lag_min <= (prop_win.max_arrival_minutes + 15.0):
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
        self.register_wavefront(source_obs, source_signature)

        # 1. Check if this is an arrival of an upstream wavefront
        upstream_evidence, upstream_origin = self.verify_upstream_origin(
            source_obs, source_signature, all_station_histories
        )

        # 2. Check downstream coupled partners in all_station_histories directly
        downstream_couplings = [c for c in couplings if c.is_downstream and c.effective_weight >= 0.08]
        downstream_confirmed_count = 0
        best_downstream_evidence = 0.0

        # Evaluate against active origin wavefronts
        for wf in self.active_wavefronts:
            if wf.origin_station_id == source_obs.station_id:
                for c in downstream_couplings:
                    target_hist = all_station_histories.get(c.target_station_id, [])
                    if len(target_hist) >= 2:
                        n_pre = next((o for o in reversed(target_hist) if o.timestamp <= wf.timestamp), target_hist[0])
                        for n_obs in target_hist:
                            if n_obs.timestamp >= wf.timestamp:
                                n_sig = np.array([
                                    n_obs.temperature - n_pre.temperature,
                                    n_obs.pressure - n_pre.pressure,
                                    n_obs.humidity - n_pre.humidity,
                                ], dtype=np.float32)
                                sig_sim = self.compute_signature_similarity(wf.signature, n_sig)
                                prop_win = self.estimator.estimate_travel_time(
                                    distance_km=c.distance_km,
                                    wind_speed_ms=wf.wind_speed or source_obs.wind_speed or 0.0,
                                    directional_alignment=c.directional_alignment,
                                    wind_stability=0.9,
                                )
                                if sig_sim >= 0.35 and prop_win.is_valid:
                                    lag_min = max(0.0, (n_obs.timestamp - wf.timestamp).total_seconds() / 60.0)
                                    t_cons = self.compute_timing_consistency(lag_min, prop_win.tau_minutes, prop_win.uncertainty_minutes)
                                    match_score = c.directional_alignment * sig_sim * max(0.65, t_cons)
                                    if match_score >= 0.20:
                                        if c.target_station_id not in wf.confirmed_stations:
                                            wf.confirmed_stations.append(c.target_station_id)
                                        best_downstream_evidence = max(best_downstream_evidence, match_score)

            if len(wf.confirmed_stations) > 1:
                cnt = len(wf.confirmed_stations) - 1
                downstream_confirmed_count = max(downstream_confirmed_count, cnt)
                best_downstream_evidence = max(best_downstream_evidence, min(0.95, 0.55 + 0.25 * cnt))

        downstream_evidence = 0.0
        if downstream_confirmed_count > 0:
            downstream_evidence = max(best_downstream_evidence, min(0.95, 0.55 + 0.20 * downstream_confirmed_count))

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
