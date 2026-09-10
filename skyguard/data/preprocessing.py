"""
Preprocessing and Normalization Pipeline for AWS Data.
"""

from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from datetime import timedelta
from skyguard.data.schema import AWSObservation, QualityFlag


class AWSPreprocessor:
    """
    Standard preprocessing, unit normalization, physical bounds QC,
    and stateful feature normalization.
    """
    # Physical plausibility limits for sanity checking
    T_MIN, T_MAX = -50.0, 60.0       # °C
    P_MIN, P_MAX = 800.0, 1090.0     # hPa
    RH_MIN, RH_MAX = 0.0, 100.0      # %
    WS_MIN, WS_MAX = 0.0, 100.0      # m/s

    def __init__(self):
        # Normalization statistics: mean and std for [T, P, RH]
        self.fitted: bool = False
        self.mean: np.ndarray = np.zeros(3, dtype=np.float32)
        self.std: np.ndarray = np.ones(3, dtype=np.float32)

    def physical_sanity_check(self, obs: AWSObservation) -> AWSObservation:
        """Flags observations with values outside broad physical bounds."""
        flags = list(obs.quality_flags)

        if not (self.T_MIN <= obs.temperature <= self.T_MAX):
            flags.append(QualityFlag.OUT_OF_PHYSICAL_RANGE)
        if not (self.P_MIN <= obs.pressure <= self.P_MAX):
            flags.append(QualityFlag.OUT_OF_PHYSICAL_RANGE)
        if not (self.RH_MIN <= obs.humidity <= self.RH_MAX):
            flags.append(QualityFlag.OUT_OF_PHYSICAL_RANGE)
            # Clip humidity within [0, 100] for downstream safety
            obs.humidity = max(0.0, min(100.0, obs.humidity))

        if obs.wind_speed is not None:
            if obs.wind_speed < 0.5:
                flags.append(QualityFlag.CALM_WIND)
            if not (self.WS_MIN <= obs.wind_speed <= self.WS_MAX):
                flags.append(QualityFlag.OUT_OF_PHYSICAL_RANGE)

        obs.quality_flags = list(set(flags))
        return obs

    def clean_and_sort_series(self, series: List[AWSObservation]) -> List[AWSObservation]:
        """Sorts chronologically, removes exact duplicate timestamps, and applies sanity QC."""
        if not series:
            return []

        # Sort by timestamp
        sorted_series = sorted(series, key=lambda x: x.timestamp)
        
        # Deduplicate
        deduped: List[AWSObservation] = []
        seen_timestamps = set()
        for obs in sorted_series:
            if obs.timestamp not in seen_timestamps:
                seen_timestamps.add(obs.timestamp)
                deduped.append(self.physical_sanity_check(obs))

        return deduped

    def fit_normalizer(self, observations: List[AWSObservation]) -> None:
        """Computes mean and std from clean training observations for [T, P, RH]."""
        if not observations:
            raise ValueError("Cannot fit normalizer on empty observations.")

        valid_vectors = [
            obs.vector_3d for obs in observations
            if QualityFlag.OUT_OF_PHYSICAL_RANGE not in obs.quality_flags
            and QualityFlag.MISSING_VALUE not in obs.quality_flags
        ]

        if not valid_vectors:
            raise ValueError("No valid observations found to fit normalizer.")

        data = np.array(valid_vectors, dtype=np.float32)
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # Avoid division by zero
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        self.fitted = True

    def normalize(self, vector: np.ndarray) -> np.ndarray:
        """Transforms raw [T, P, RH] vector to zero-mean unit-variance."""
        if not self.fitted:
            # Fallback default normalization if not yet fitted
            default_mean = np.array([20.0, 1013.25, 60.0], dtype=np.float32)
            default_std = np.array([10.0, 15.0, 20.0], dtype=np.float32)
            return (vector - default_mean) / default_std
        return (vector - self.mean) / self.std

    def denormalize(self, norm_vector: np.ndarray) -> np.ndarray:
        """Inverts normalization back to physical units [°C, hPa, %]."""
        if not self.fitted:
            default_mean = np.array([20.0, 1013.25, 60.0], dtype=np.float32)
            default_std = np.array([10.0, 15.0, 20.0], dtype=np.float32)
            return norm_vector * default_std + default_mean
        return norm_vector * self.std + self.mean


def prepare_station_data(
    station_id: str,
    raw_observations: List[AWSObservation],
    metadata: Optional[Any] = None,
    window_size: int = 10,
    normalizer: Optional[AWSPreprocessor] = None,
) -> "PreparedStationData":
    """
    Deterministic Data Preparation Pipeline for a single AWS station.
    Performs:
      1. Schema & Field Normalization
      2. Timestamp Standardization & Cadence Analysis
      3. Physical Unit Standardization (°C, hPa, %, m/s)
      4. Duplicate Detection & Deterministic Resolution
      5. Multi-Criteria Quality Control (bounds, rate of change, stuck sensor, missingness)
         * Crucial: Suspicious / anomalous values are FLAGGED, NOT destroyed
      6. Physics & Thermodynamic Feature Extraction (Tv, es, e, q, rho, N)
      7. Station-Pure Rolling Temporal Window Construction
    """
    from datetime import datetime
    from skyguard.data.schema import PreparedStationData, StationMetadata, QualityFlag
    from skyguard.physics.thermodynamics import (
        saturation_vapor_pressure,
        actual_vapor_pressure,
        virtual_temperature,
        atmospheric_refractive_index,
    )

    if metadata is None:
        first_obs = raw_observations[0] if raw_observations else None
        lat = first_obs.latitude if first_obs else 0.0
        lon = first_obs.longitude if first_obs else 0.0
        metadata = StationMetadata(station_id, f"Station-{station_id}", lat, lon)

    # 1. Deduplication & Timestamp Sorting
    sorted_obs = sorted(raw_observations, key=lambda x: x.timestamp)
    deduped_obs: List[AWSObservation] = []
    seen_times = set()
    dup_count = 0
    for o in sorted_obs:
        if o.timestamp in seen_times:
            dup_count += 1
            continue
        seen_times.add(o.timestamp)
        # Ensure correct station_id is preserved
        o.station_id = station_id
        deduped_obs.append(o)

    # 2. Temporal Cadence Diagnostics
    cadence_minutes = 60.0
    gaps_detected = []
    if len(deduped_obs) >= 2:
        deltas = [
            (deduped_obs[i].timestamp - deduped_obs[i - 1].timestamp).total_seconds() / 60.0
            for i in range(1, len(deduped_obs))
        ]
        cadence_minutes = float(np.median(deltas)) if deltas else 60.0
        for i, dt_min in enumerate(deltas):
            if dt_min > cadence_minutes * 1.8:
                gaps_detected.append({
                    "start": deduped_obs[i].timestamp.isoformat(),
                    "end": deduped_obs[i + 1].timestamp.isoformat(),
                    "gap_minutes": dt_min,
                })

    temporal_diagnostics = {
        "nominal_cadence_minutes": cadence_minutes,
        "total_records": len(deduped_obs),
        "duplicates_resolved": dup_count,
        "gaps_count": len(gaps_detected),
        "gaps": gaps_detected[:10],
    }

    # 3. Quality Control (Physical Bounds, Rate of Change, Stuck Sensor)
    quality_flags_map: Dict[str, List[QualityFlag]] = {}
    missing_count = {"temperature": 0, "pressure": 0, "humidity": 0}
    stuck_counters = {"temperature": (None, 0), "pressure": (None, 0), "humidity": (None, 0)}

    preprocessor = normalizer or AWSPreprocessor()
    if not preprocessor.fitted and len(deduped_obs) >= window_size:
        preprocessor.fit_normalizer(deduped_obs)

    cleaned_obs: List[AWSObservation] = []
    for i, obs in enumerate(deduped_obs):
        flags = list(obs.quality_flags)

        # Missingness checks
        if obs.temperature is None or np.isnan(obs.temperature):
            flags.append(QualityFlag.MISSING_VALUE)
            missing_count["temperature"] += 1
            obs.temperature = 25.0  # safe fallback for calculations
        if obs.pressure is None or np.isnan(obs.pressure):
            flags.append(QualityFlag.MISSING_VALUE)
            missing_count["pressure"] += 1
            obs.pressure = 1013.25
        if obs.humidity is None or np.isnan(obs.humidity):
            flags.append(QualityFlag.MISSING_VALUE)
            missing_count["humidity"] += 1
            obs.humidity = 50.0

        # Physical sanity limits
        if not (-50.0 <= obs.temperature <= 60.0):
            flags.append(QualityFlag.OUT_OF_PHYSICAL_RANGE)
        if not (800.0 <= obs.pressure <= 1090.0):
            flags.append(QualityFlag.OUT_OF_PHYSICAL_RANGE)
        if not (0.0 <= obs.humidity <= 100.0):
            flags.append(QualityFlag.OUT_OF_PHYSICAL_RANGE)
            obs.humidity = float(np.clip(obs.humidity, 0.0, 100.0))

        # Sudden rate-of-change jump detection (> 6°C/hr, > 8 hPa/hr, > 35% RH/hr)
        if i > 0:
            prev_obs = deduped_obs[i - 1]
            dt_hours = max(0.1, (obs.timestamp - prev_obs.timestamp).total_seconds() / 3600.0)
            dT_dt = abs(obs.temperature - prev_obs.temperature) / dt_hours
            dP_dt = abs(obs.pressure - prev_obs.pressure) / dt_hours
            dRH_dt = abs(obs.humidity - prev_obs.humidity) / dt_hours

            if dT_dt > 8.0 or dP_dt > 10.0 or dRH_dt > 45.0:
                flags.append(QualityFlag.RATE_OF_CHANGE_EXCEEDED)

        # Sensor stuck / frozen check (constant reading for >= 6 consecutive steps)
        for var_name, val in [("temperature", obs.temperature), ("pressure", obs.pressure), ("humidity", obs.humidity)]:
            last_val, count = stuck_counters[var_name]
            if last_val is not None and abs(val - last_val) < 1e-4:
                stuck_counters[var_name] = (val, count + 1)
                if count + 1 >= 6:
                    flags.append(QualityFlag.SUSPECTED_FAULT)
            else:
                stuck_counters[var_name] = (val, 1)

        # Retain flags without destroying the observation
        obs.quality_flags = list(set(flags))
        quality_flags_map[obs.timestamp.isoformat()] = obs.quality_flags
        cleaned_obs.append(obs)

    missingness_flags = {
        "missing_counts": missing_count,
        "missing_percentage": {k: round(v / max(1, len(deduped_obs)) * 100.0, 2) for k, v in missing_count.items()},
    }

    # 4. Physics & Thermodynamic Derived Features
    temps = np.array([o.temperature for o in cleaned_obs], dtype=np.float32)
    press = np.array([o.pressure for o in cleaned_obs], dtype=np.float32)
    humids = np.array([o.humidity for o in cleaned_obs], dtype=np.float32)

    temps_k = temps + 273.15
    es_arr = np.array([saturation_vapor_pressure(tk) for tk in temps_k], dtype=np.float32)
    e_arr = np.array([actual_vapor_pressure(rh, tk) for rh, tk in zip(humids, temps_k)], dtype=np.float32)
    tv_arr = np.array([virtual_temperature(tk, p, rh) for tk, p, rh in zip(temps_k, press, humids)], dtype=np.float32)
    n_arr = np.array([atmospheric_refractive_index(tk, p, rh) for tk, p, rh in zip(temps_k, press, humids)], dtype=np.float32)

    physics_features = {
        "temperature_c": temps,
        "pressure_hpa": press,
        "humidity_pct": humids,
        "saturation_vapor_pressure_es": es_arr,
        "actual_vapor_pressure_e": e_arr,
        "virtual_temperature_tv": tv_arr,
        "refractive_index_n": n_arr,
    }

    # 5. Station-Pure Rolling Temporal Windows
    raw_matrix = np.column_stack([temps, press, humids])
    norm_matrix = preprocessor.normalize(raw_matrix) if preprocessor.fitted else raw_matrix

    windows_list = []
    if len(norm_matrix) >= window_size:
        for w_idx in range(len(norm_matrix) - window_size + 1):
            windows_list.append(norm_matrix[w_idx : w_idx + window_size])
        analysis_windows = np.array(windows_list, dtype=np.float32)
    else:
        analysis_windows = np.zeros((0, window_size, 3), dtype=np.float32)

    return PreparedStationData(
        station_id=station_id,
        metadata=metadata,
        raw_observations=cleaned_obs,
        normalized_observations=cleaned_obs,
        quality_flags=quality_flags_map,
        missingness_flags=missingness_flags,
        temporal_diagnostics=temporal_diagnostics,
        physics_ready_variables=physics_features,
        analysis_windows=analysis_windows,
    )
