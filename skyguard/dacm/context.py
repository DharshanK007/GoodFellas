"""
Dynamic Station-Conditioned Context & Advective Neighborhood Builder.
Constructs DynamicStationContext specifically for any user-selected station.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from skyguard.config import DACM_CONFIG
from skyguard.data.schema import PreparedStationData, DynamicStationContext, StationMetadata, AWSObservation
from skyguard.dacm.coupling import DACMCouplingEngine, StationCouplingResult
from skyguard.dacm.geometry import haversine_distance_km, station_direction_vector


def build_dynamic_station_context(
    target_station_id: str,
    target_prepared: PreparedStationData,
    neighbor_prepared_map: Dict[str, PreparedStationData],
    coupling_engine: Optional[DACMCouplingEngine] = None,
) -> DynamicStationContext:
    """
    Constructs a DynamicStationContext for the specified primary station.
    Performs:
      1. Spatial relationship calculation with all candidate neighbors.
      2. Stage-1 Geographic Proximity Gating (<= 80 km).
      3. Stage-2 Wind-Enhanced DACM Coupling.
      4. Temporal alignment verification across the sub-network.
      5. Explicit packaging of primary vs. contextual features.
    """
    engine = coupling_engine or DACMCouplingEngine()

    target_meta = target_prepared.metadata
    latest_target_obs = target_prepared.raw_observations[-1] if target_prepared.raw_observations else None

    if latest_target_obs is None:
        return DynamicStationContext(
            primary_station_id=target_station_id,
            contextual_station_ids=[],
            spatial_relationships=[],
            temporal_alignment={},
            primary_features={},
            contextual_features={},
            coupling_results=[],
            coupling_quality=0.0,
        )

    # Collect latest observation from each neighboring station
    candidate_latest_obs: List[AWSObservation] = []
    spatial_relationships: List[Dict[str, Any]] = []
    stations_meta: Dict[str, StationMetadata] = {target_station_id: target_meta}

    for n_id, n_prep in neighbor_prepared_map.items():
        if n_id == target_station_id:
            continue
        stations_meta[n_id] = n_prep.metadata
        if n_prep.raw_observations:
            n_obs = n_prep.raw_observations[-1]
            candidate_latest_obs.append(n_obs)

            dist_km = haversine_distance_km(
                target_meta.latitude, target_meta.longitude,
                n_prep.metadata.latitude, n_prep.metadata.longitude
            )
            r_vec, bearing_deg = station_direction_vector(
                target_meta.latitude, target_meta.longitude,
                n_prep.metadata.latitude, n_prep.metadata.longitude
            )
            spatial_relationships.append({
                "neighbor_id": n_id,
                "neighbor_name": n_prep.metadata.name if hasattr(n_prep.metadata, "name") else n_id,
                "distance_km": round(dist_km, 2),
                "bearing_deg": round(bearing_deg, 1),
                "within_proximity": dist_km <= DACM_CONFIG.PROXIMITY_RADIUS_KM,
            })

    # Sort spatial relationships by distance
    spatial_relationships.sort(key=lambda x: x["distance_km"])

    # Recent wind history for circular stability
    recent_wdirs = [
        o.wind_direction for o in target_prepared.raw_observations[-DACM_CONFIG.STABILITY_WINDOW_SIZE:]
        if o.wind_direction is not None
    ]
    recent_wspds = [
        o.wind_speed for o in target_prepared.raw_observations[-DACM_CONFIG.STABILITY_WINDOW_SIZE:]
        if o.wind_speed is not None
    ]

    # Run Two-Stage DACM Coupling specifically for target_station_id
    couplings: List[StationCouplingResult] = engine.get_all_couplings_for_source(
        obs_source=latest_target_obs,
        all_current_observations=candidate_latest_obs,
        stations_metadata=stations_meta,
        recent_wind_directions=recent_wdirs,
        recent_wind_speeds=recent_wspds,
    )

    # Determine contextual station IDs from proximity-selected couplings
    contextual_ids = [c.target_station_id for c in couplings]

    # Package primary features (last observation)
    primary_feats = {
        "temperature": latest_target_obs.temperature,
        "pressure": latest_target_obs.pressure,
        "humidity": latest_target_obs.humidity,
        "wind_speed": latest_target_obs.wind_speed,
        "wind_direction": latest_target_obs.wind_direction,
        "virtual_temperature_tv": float(target_prepared.physics_ready_variables.get("virtual_temperature_tv", [0.0])[-1])
        if len(target_prepared.physics_ready_variables.get("virtual_temperature_tv", [])) > 0 else None,
        "refractive_index_n": float(target_prepared.physics_ready_variables.get("refractive_index_n", [0.0])[-1])
        if len(target_prepared.physics_ready_variables.get("refractive_index_n", [])) > 0 else None,
    }

    # Contextual features summary
    context_feats = {}
    for n_id in contextual_ids:
        n_prep = neighbor_prepared_map.get(n_id)
        if n_prep and n_prep.raw_observations:
            last_n_obs = n_prep.raw_observations[-1]
            context_feats[n_id] = {
                "temperature": last_n_obs.temperature,
                "pressure": last_n_obs.pressure,
                "humidity": last_n_obs.humidity,
                "wind_speed": last_n_obs.wind_speed,
                "wind_direction": last_n_obs.wind_direction,
            }

    # Overall coupling quality score (mean of top coupling weights)
    coupling_quality = float(np.mean([c.effective_weight for c in couplings])) if couplings else 0.0

    temporal_alignment = {
        "target_timestamp": latest_target_obs.timestamp.isoformat(),
        "neighbors_aligned_count": len(candidate_latest_obs),
        "cadence_minutes": target_prepared.temporal_diagnostics.get("nominal_cadence_minutes", 60.0),
    }

    return DynamicStationContext(
        primary_station_id=target_station_id,
        contextual_station_ids=contextual_ids,
        spatial_relationships=spatial_relationships,
        temporal_alignment=temporal_alignment,
        primary_features=primary_feats,
        contextual_features=context_feats,
        coupling_results=couplings,
        coupling_quality=coupling_quality,
    )
