"""
Dynamic Advective Coupling Engine (DACM) — Proximity-First Two-Stage Architecture.

Stage 1 — Geographic Proximity Gate (pure distance, wind-agnostic):
    Only stations within PROXIMITY_RADIUS_KM are eligible as DACM candidates.
    Sorted by distance, capped at TOP_K_NEIGHBORS.
    Per master spec S62: wind direction alone must NOT couple distant stations.

Stage 2 — Wind-Enhanced Coupling (within the proximity pool):
    Among the proximity-selected neighbors, compute directional alignment,
    wind stability, historical relationship, and the full DACM coupling weight.
    W_DACM = W_phys * S_w * C_ij * R_j
    where W_phys already encodes both distance decay and wind advection boost.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from skyguard.config import DACM_CONFIG
from skyguard.data.schema import AWSObservation, StationMetadata
from skyguard.dacm.geometry import haversine_distance_km, station_direction_vector
from skyguard.dacm.wind_vector import meteorological_wind_to_vector, circular_wind_stability
from skyguard.dacm.alignment import compute_wind_alignment
from skyguard.dacm.propagation import AdvectivePropagationEstimator


@dataclass
class StationCouplingResult:
    """Dynamic coupling outcome between station A (source) and B (candidate/downstream)."""
    source_station_id: str
    target_station_id: str
    distance_km: float
    bearing_deg: float
    directional_alignment: float       # clipped to [0, 1] (alignment.py output)
    # Purely distance-based weight (before wind enhancement): exp(-d/gamma)
    distance_weight: float
    # Wind-enhanced physical coupling weight: exp(-d / [gamma*(1+alpha*s*a)])
    physical_coupling: float
    wind_stability: float
    historical_relationship: float
    target_reliability: float
    raw_dacm_weight: float   # physical_coupling * wind_stability * historical_relationship
    effective_weight: float  # raw_dacm_weight * target_reliability
    is_downstream: bool      # True iff edge_status in (DOWNSTREAM, WEAK_DOWNSTREAM)
    within_proximity: bool   # True iff distance_km <= PROXIMITY_RADIUS_KM
    proximity_rank: int      # 1 = closest neighbor in the proximity pool

    # ── New scientific DACM fields (added in v2) ──────────────────────────
    raw_alignment_cos: float = 0.0      # dot(v_a, r_ab)/|v_a| BEFORE max(0,...) clamp
                                        # negative values indicate upstream relationship
    advective_speed_ms: float = 0.0     # wind_speed_ms * clipped_alignment (parallel component)
    wind_speed_ms: float = 0.0          # scalar wind speed at source (m/s)
    wind_u_ms: float = 0.0              # eastward transport component at source (m/s)
    wind_v_ms: float = 0.0              # northward transport component at source (m/s)
    wind_from_deg: float = 0.0          # meteorological FROM direction (degrees)
    wind_toward_deg: float = 0.0        # atmospheric transport direction = (from + 180) % 360
    travel_time_minutes: Optional[float] = None  # None if not DOWNSTREAM/WEAK_DOWNSTREAM
    edge_status: str = "UNAVAILABLE"    # DOWNSTREAM|WEAK_DOWNSTREAM|CROSSWIND|UPSTREAM|CALM|UNAVAILABLE
    wind_source: str = "STATION_OBSERVATION"     # provenance tag
    wind_timestamp: str = ""            # ISO timestamp of wind obs used
    wind_match_method: str = "station_observation_exact"  # how wind was resolved


# Edge status thresholds (separate from is_downstream threshold)
_EDGE_DOWNSTREAM_THRESHOLD   = 0.50   # raw_cos >= this -> DOWNSTREAM
_EDGE_WEAK_DOWN_THRESHOLD    = 0.20   # 0.20 <= raw_cos < 0.50 -> WEAK_DOWNSTREAM
_EDGE_CROSSWIND_LOWER        = -0.20  # -0.20 < raw_cos < 0.20 -> CROSSWIND
                                      # raw_cos <= -0.20 -> UPSTREAM

_propagation_estimator = AdvectivePropagationEstimator()


def _classify_edge_status(
    raw_cos: float,
    wind_speed_ms: float,
    wind_available: bool,
) -> str:
    """Classify DACM edge status from pre-clamp alignment cosine and wind speed."""
    if not wind_available or wind_speed_ms < 1e-3:
        return "UNAVAILABLE"
    if wind_speed_ms < DACM_CONFIG.CALM_WIND_THRESHOLD:
        return "CALM"
    if raw_cos >= _EDGE_DOWNSTREAM_THRESHOLD:
        return "DOWNSTREAM"
    if raw_cos >= _EDGE_WEAK_DOWN_THRESHOLD:
        return "WEAK_DOWNSTREAM"
    if raw_cos > _EDGE_CROSSWIND_LOWER:
        return "CROSSWIND"
    return "UPSTREAM"



@dataclass
class ProximityNeighbor:
    """Lightweight Stage-1 proximity-selected neighbor record."""
    obs: AWSObservation
    metadata: StationMetadata
    distance_km: float
    rank: int  # 1-indexed, 1 = closest


class DACMCouplingEngine:
    """
    Computes dynamic, directional, wind-aware station relationships
    using a Proximity-First Two-Stage architecture.

    Stage 1 - Geographic proximity gate (wind-agnostic):
        select_proximity_neighbors() returns only nearby stations
        within PROXIMITY_RADIUS_KM, sorted by distance, top-K.

    Stage 2 - Wind-enhanced coupling (within proximity pool):
        compute_station_coupling() applies the full DACM formulation
        (alignment, wind stability, historical C_ij) only for Stage-1 candidates.
    """

    def __init__(
        self,
        gamma: float = DACM_CONFIG.SPATIAL_DECAY_GAMMA,
        alpha: float = DACM_CONFIG.WIND_ADVECTION_ALPHA,
        proximity_radius_km: float = DACM_CONFIG.PROXIMITY_RADIUS_KM,
        top_k: int = DACM_CONFIG.TOP_K_NEIGHBORS,
        max_distance_km: float = DACM_CONFIG.MAX_CANDIDATE_DISTANCE_KM,
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.proximity_radius_km = proximity_radius_km
        self.top_k = top_k
        self.max_distance_km = max_distance_km
        self.historical_relationships: Dict[Tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # Historical relationship management
    # ------------------------------------------------------------------

    def set_historical_relationship(self, stn_a: str, stn_b: str, c_ab: float) -> None:
        """Sets empirical historical co-movement relationship C_AB in [0, 1]."""
        self.historical_relationships[(stn_a, stn_b)] = float(np.clip(c_ab, 0.0, 1.0))

    def get_historical_relationship(self, stn_a: str, stn_b: str) -> float:
        """Returns stored historical relationship; default 0.80 for unknown pairs."""
        return self.historical_relationships.get((stn_a, stn_b), 0.80)

    # ------------------------------------------------------------------
    # Stage 1 - Geographic Proximity Gate
    # ------------------------------------------------------------------

    def select_proximity_neighbors(
        self,
        obs_source: AWSObservation,
        all_observations: List[AWSObservation],
        stations_metadata: Dict[str, StationMetadata],
    ) -> List[ProximityNeighbor]:
        """
        STAGE 1: Select geographically close station candidates.

        Rules:
        - Exclude the source station itself.
        - Include only stations within PROXIMITY_RADIUS_KM.
        - Sort by distance (closest first).
        - Return at most TOP_K_NEIGHBORS results.

        Wind direction, alignment, speed - NONE influence Stage 1.
        This is a pure geographic filter per master spec S62.
        """
        candidates: List[Tuple[float, AWSObservation]] = []

        for obs in all_observations:
            if obs.station_id == obs_source.station_id:
                continue
            dist_km = haversine_distance_km(
                obs_source.latitude, obs_source.longitude,
                obs.latitude, obs.longitude
            )
            if dist_km <= self.proximity_radius_km:
                candidates.append((dist_km, obs))

        # Sort nearest-first, cap at top_k
        candidates.sort(key=lambda x: x[0])
        candidates = candidates[: self.top_k]

        neighbors: List[ProximityNeighbor] = []
        for rank, (dist_km, obs) in enumerate(candidates, start=1):
            meta = stations_metadata.get(
                obs.station_id,
                StationMetadata(obs.station_id, obs.station_id, obs.latitude, obs.longitude)
            )
            neighbors.append(ProximityNeighbor(obs=obs, metadata=meta, distance_km=dist_km, rank=rank))

        return neighbors

    # ------------------------------------------------------------------
    # Stage 2 - Wind-Enhanced Coupling (per proximity-selected pair)
    # ------------------------------------------------------------------

    def compute_station_coupling(
        self,
        obs_a: AWSObservation,
        neighbor: ProximityNeighbor,
        recent_wind_directions_a: Optional[List[float]] = None,
        recent_wind_speeds_a: Optional[List[float]] = None,
    ) -> StationCouplingResult:
        """
        STAGE 2: Compute full DACM coupling weight for a proximity-selected pair A -> B.

        Implements master spec formulation pipeline (S33-S46):
          1. Haversine distance + bearing (pre-computed in Stage 1).
          2. Meteorological wind -> transport vector.
          3. Directional alignment a_ij = max(0, dot(v_i, r_ij) / |v_i|).
          4. Downstream wind component v_parallel = s_i * a_ij.
          5. Wind stability S_w = S_direction * S_speed.
          6. Distance coupling W_distance = exp(-d / gamma).
          7. Wind-enhanced coupling W_phys = exp(-d / [gamma*(1+alpha*s*a)]).
          8. Stability adjustment W_stable = W_phys * S_w.
          9. Historical relationship C_ij.
         10. Final: W_DACM = W_stable * C_ij; W_effective = W_DACM * R_j.
        """
        dist_km = neighbor.distance_km
        obs_b = neighbor.obs
        meta_b = neighbor.metadata

        # Bearing and direction unit vector from A to B
        r_ab, bearing_deg = station_direction_vector(
            obs_a.latitude, obs_a.longitude,
            obs_b.latitude, obs_b.longitude
        )

        # Wind vector at source station A (met convention -> transport vector)
        ws_a = obs_a.wind_speed if obs_a.wind_speed is not None else 0.0
        wd_a = obs_a.wind_direction if obs_a.wind_direction is not None else 0.0
        wind_available = (obs_a.wind_speed is not None and obs_a.wind_speed > 0.05
                          and obs_a.wind_direction is not None)
        v_a = meteorological_wind_to_vector(ws_a, wd_a)

        # Raw (pre-clamp) alignment cosine — needed for UPSTREAM detection
        norm_v = float(np.linalg.norm(v_a))
        if norm_v > 1e-3 and ws_a > 1e-3:
            raw_alignment_cos = float(np.dot(v_a, r_ab) / norm_v)
        else:
            raw_alignment_cos = 0.0

        # Directional alignment a_AB = max(0, dot(v_a, r_ab) / |v_a|)
        alignment, is_calm = compute_wind_alignment(v_a, r_ab, ws_a)

        # Wind direction fields
        wind_from_deg   = float(wd_a)
        wind_toward_deg = (wind_from_deg + 180.0) % 360.0
        wind_u_ms       = float(v_a[0])
        wind_v_ms       = float(v_a[1])
        advective_speed_ms = float(ws_a * alignment)  # parallel component toward B

        # Edge status classification (richer than bool)
        edge_status = _classify_edge_status(raw_alignment_cos, ws_a, wind_available)

        # Pure distance coupling: W_distance = exp(-d / gamma)
        w_distance = float(np.exp(-dist_km / max(1.0, self.gamma)))

        # Wind-enhanced physical coupling: W_phys = exp(-d / [gamma*(1 + alpha*s*a)])
        # Wind alignment boosts effective decay range; distance still gates selection.
        wind_boost_denom = self.gamma * (1.0 + self.alpha * ws_a * alignment)
        w_phys = float(np.exp(-dist_km / max(1.0, wind_boost_denom)))

        # Wind stability S_w = S_direction * S_speed
        if recent_wind_directions_a and len(recent_wind_directions_a) >= 2:
            s_direction = circular_wind_stability(recent_wind_directions_a, recent_wind_speeds_a)
        else:
            s_direction = 0.0 if is_calm else 0.85

        if recent_wind_speeds_a and len(recent_wind_speeds_a) >= 2:
            spd_arr = np.array(recent_wind_speeds_a, dtype=np.float32)
            mu_s = float(np.mean(spd_arr))
            sigma_s = float(np.std(spd_arr))
            s_speed = 1.0 / (1.0 + sigma_s / (mu_s + 1e-3))
        else:
            s_speed = 0.85 if not is_calm else 0.0

        s_w = float(np.clip(s_direction * s_speed, 0.0, 1.0))

        # Historical station relationship C_AB
        c_ab = self.get_historical_relationship(obs_a.station_id, obs_b.station_id)

        # Target station reliability R_B
        r_b = float(np.clip(meta_b.reliability_score, 0.0, 1.0))

        # Final DACM weights
        w_stable = w_phys * s_w
        w_dacm = w_stable * c_ab
        w_effective = w_dacm * r_b

        # Downstream flag: wind meaningfully aligned AND not calm
        # Now derived from edge_status for consistency
        is_downstream = edge_status in ("DOWNSTREAM", "WEAK_DOWNSTREAM")

        # Travel time estimate (only valid for downstream edges)
        travel_time_minutes: Optional[float] = None
        if is_downstream and advective_speed_ms > 0.1:
            prop_window = _propagation_estimator.estimate_travel_time(
                distance_km=dist_km,
                wind_speed_ms=ws_a,
                directional_alignment=alignment,
                wind_stability=s_w,
            )
            if prop_window.is_valid:
                travel_time_minutes = round(prop_window.tau_minutes, 1)

        return StationCouplingResult(
            source_station_id=obs_a.station_id,
            target_station_id=obs_b.station_id,
            distance_km=dist_km,
            bearing_deg=bearing_deg,
            directional_alignment=alignment,
            distance_weight=w_distance,
            physical_coupling=w_phys,
            wind_stability=s_w,
            historical_relationship=c_ab,
            target_reliability=r_b,
            raw_dacm_weight=w_dacm,
            effective_weight=w_effective,
            is_downstream=is_downstream,
            within_proximity=True,
            proximity_rank=neighbor.rank,
            # New scientific fields
            raw_alignment_cos=raw_alignment_cos,
            advective_speed_ms=advective_speed_ms,
            wind_speed_ms=float(ws_a),
            wind_u_ms=wind_u_ms,
            wind_v_ms=wind_v_ms,
            wind_from_deg=wind_from_deg,
            wind_toward_deg=wind_toward_deg,
            travel_time_minutes=travel_time_minutes,
            edge_status=edge_status,
            wind_source=obs_a.source_metadata.get("wind_source", "STATION_OBSERVATION"),
            wind_timestamp=obs_a.timestamp.isoformat() if obs_a.timestamp else "",
            wind_match_method=obs_a.source_metadata.get("wind_match_method", "station_observation_exact"),
        )

    # ------------------------------------------------------------------
    # Orchestrator - combines Stage 1 + Stage 2
    # ------------------------------------------------------------------

    def get_all_couplings_for_source(
        self,
        obs_source: AWSObservation,
        all_current_observations: List[AWSObservation],
        stations_metadata: Dict[str, StationMetadata],
        recent_wind_directions: Optional[List[float]] = None,
        recent_wind_speeds: Optional[List[float]] = None,
    ) -> List[StationCouplingResult]:
        """
        Full two-stage DACM coupling pipeline for the source station.

        Stage 1: geographic proximity gate (-> ProximityNeighbor list, no wind)
        Stage 2: wind-enhanced coupling (-> StationCouplingResult list)

        Returns results sorted by effective_weight descending.
        Only proximity-selected stations (within PROXIMITY_RADIUS_KM, top-K)
        appear in output. Distant stations are excluded entirely regardless of wind.
        """
        neighbors = self.select_proximity_neighbors(
            obs_source, all_current_observations, stations_metadata
        )

        if not neighbors:
            return []

        couplings: List[StationCouplingResult] = []
        for neighbor in neighbors:
            coup = self.compute_station_coupling(
                obs_source,
                neighbor,
                recent_wind_directions_a=recent_wind_directions,
                recent_wind_speeds_a=recent_wind_speeds,
            )
            couplings.append(coup)

        couplings.sort(key=lambda x: x.effective_weight, reverse=True)
        return couplings
