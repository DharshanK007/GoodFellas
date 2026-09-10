"""
Controlled Anomaly Injection & Benchmark Scenario Generator.

Architecture:
  CLEAN BASELINE (steps 0-179)  -> returned separately for model training
  HELD-OUT PERIOD (steps 180-335) -> faults injected on target only -> model inference
  GROUND TRUTH                  -> stored separately, never fed to model

Thermodynamic consistency enforced throughout:
  T  -> e_s(T) = 6.112 * exp[(17.67 * T)/(T + 243.5)]   [Magnus-Tetens]
  e  = (RH/100) * e_s
  T_v = T_K / [1 - (1 - eps) * (e/p)]                    [Virtual temperature]
  n  = 1 + 7.76e-5 * (p/T_K) + 3.73e-1 * (e/T_K^2)     [Refractive index]
"""

import copy
import math
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from skyguard.data.schema import AWSObservation, CanonicalDataset, QualityFlag, StationMetadata


# -- Physical Constants --------------------------------------------------------
_EPSILON  = 0.622         # Rd/Rv
_K_OFFSET = 273.15        # 0°C in Kelvin
_TETENS_A = 6.112         # hPa
_TETENS_B = 17.67
_TETENS_C = 243.5         # °C


def _e_s(T_c: float) -> float:
    """Saturation vapour pressure [hPa] via Magnus-Tetens."""
    return _TETENS_A * math.exp((_TETENS_B * T_c) / (T_c + _TETENS_C))


def _e_actual(T_c: float, rh: float) -> float:
    """Actual vapour pressure [hPa]."""
    return (rh / 100.0) * _e_s(T_c)


def _virtual_temp(T_c: float, p_hpa: float, rh: float) -> float:
    """Virtual temperature [K]."""
    T_k = T_c + _K_OFFSET
    e = _e_actual(T_c, rh)
    return T_k / (1.0 - (1.0 - _EPSILON) * (e / max(p_hpa, 1.0)))


def _rh_from_consistency(T_c: float, T_v_target: float, p_hpa: float) -> float:
    """Back-compute RH that gives a target virtual temperature."""
    T_k = T_c + _K_OFFSET
    es = _e_s(T_c)
    if abs(T_v_target) < 1e-6:
        return 60.0
    e_target = p_hpa * (1.0 - T_k / T_v_target) / (1.0 - _EPSILON)
    rh = max(5.0, min(99.5, 100.0 * e_target / max(es, 1e-6)))
    return rh


def _make_thermodynamically_consistent(T_c: float, p_hpa: float, rh_approx: float) -> Tuple[float, float]:
    """
    Given T and P, compute the RH that is thermodynamically self-consistent
    with a target virtual temperature derived from the approximate RH.
    Returns (consistent_RH, virtual_temp_K).
    """
    T_v = _virtual_temp(T_c, p_hpa, rh_approx)
    rh_consistent = _rh_from_consistency(T_c, T_v, p_hpa)
    return rh_consistent, T_v


# -- Anomaly Type Catalogue ----------------------------------------------------

class AnomalyType(str, Enum):
    SUDDEN_SPIKE           = "SUDDEN_SPIKE"
    FROZEN_SENSOR          = "FROZEN_SENSOR"
    GRADUAL_DRIFT          = "GRADUAL_DRIFT"
    CONSTANT_OFFSET        = "CONSTANT_OFFSET"
    HIGH_NOISE             = "HIGH_NOISE"
    COMMUNICATION_DROPOUT  = "COMMUNICATION_DROPOUT"
    MULTIVARIATE_INCONSISTENCY = "MULTIVARIATE_INCONSISTENCY"
    PROPAGATING_FRONT      = "PROPAGATING_FRONT"
    HARD_NEGATIVE          = "HARD_NEGATIVE"


# -- Point-injection helpers ---------------------------------------------------

class AnomalyInjector:
    """
    Injects synthetic anomalies and genuine propagating weather events
    into clean meteorological datasets for controlled evaluation.
    """

    @staticmethod
    def inject_spike(
        series: List[AWSObservation],
        index: int,
        variable: str = "temperature",
        magnitude: float = 12.0,
    ) -> List[AWSObservation]:
        """Injects an isolated sudden spike at a specific index."""
        corrupted = copy.deepcopy(series)
        if 0 <= index < len(corrupted):
            obs = corrupted[index]
            current_val = getattr(obs, variable)
            setattr(obs, variable, current_val + magnitude)
            obs.quality_flags.append(QualityFlag.INJECTED_ANOMALY)
            obs.source_metadata["injected_anomaly"] = {
                "type": AnomalyType.SUDDEN_SPIKE.value,
                "variable": variable,
                "magnitude": magnitude,
            }
        return corrupted

    @staticmethod
    def inject_freeze(
        series: List[AWSObservation],
        start_idx: int,
        duration: int = 15,
        variable: str = "temperature",
    ) -> List[AWSObservation]:
        """Freezes a sensor value across a duration while atmospheric state evolves."""
        corrupted = copy.deepcopy(series)
        if 0 <= start_idx < len(corrupted):
            frozen_val = getattr(corrupted[start_idx], variable)
            end_idx = min(len(corrupted), start_idx + duration)
            for i in range(start_idx, end_idx):
                setattr(corrupted[i], variable, frozen_val)
                corrupted[i].quality_flags.append(QualityFlag.INJECTED_ANOMALY)
                corrupted[i].source_metadata["injected_anomaly"] = {
                    "type": AnomalyType.FROZEN_SENSOR.value,
                    "variable": variable,
                    "frozen_value": frozen_val,
                }
        return corrupted

    @staticmethod
    def inject_drift(
        series: List[AWSObservation],
        start_idx: int,
        duration: int = 30,
        variable: str = "temperature",
        rate_per_step: float = 0.2,
    ) -> List[AWSObservation]:
        """Simulates gradual sensor calibration drift."""
        corrupted = copy.deepcopy(series)
        end_idx = min(len(corrupted), start_idx + duration)
        for step, i in enumerate(range(start_idx, end_idx)):
            obs = corrupted[i]
            drift_val = getattr(obs, variable) + (step * rate_per_step)
            setattr(obs, variable, drift_val)
            obs.quality_flags.append(QualityFlag.INJECTED_ANOMALY)
            obs.source_metadata["injected_anomaly"] = {
                "type": AnomalyType.GRADUAL_DRIFT.value,
                "variable": variable,
                "cumulative_drift": step * rate_per_step,
            }
        return corrupted

    @staticmethod
    def inject_multivariate_inconsistency(
        series: List[AWSObservation],
        index: int,
    ) -> List[AWSObservation]:
        """Creates a severe unphysical thermodynamic mismatch."""
        corrupted = copy.deepcopy(series)
        if 0 <= index < len(corrupted):
            obs = corrupted[index]
            obs.temperature = 48.0
            obs.humidity = 98.0
            obs.pressure = 890.0
            obs.quality_flags.append(QualityFlag.INJECTED_ANOMALY)
            obs.source_metadata["injected_anomaly"] = {
                "type": AnomalyType.MULTIVARIATE_INCONSISTENCY.value
            }
        return corrupted

    @staticmethod
    def create_clean_training_dataset(
        num_stations: int = 5,
        num_timesteps: int = 288,
        timestep_minutes: int = 5,
    ) -> CanonicalDataset:
        """
        Generates 24-48 hours of thermodynamically consistent normal diurnal baseline
        for training Channel 1 & Channel 2 models without any anomaly corruption.
        """
        dataset = CanonicalDataset()
        base_time = datetime(2026, 8, 29, 0, 0, 0)

        stations_info = [
            ("AWS-01", "Station-West (A)",   12.95, 77.50),
            ("AWS-02", "Station-Center (B)", 12.95, 77.65),
            ("AWS-03", "Station-North (C)",  13.08, 77.58),
            ("AWS-04", "Station-East (D)",   12.95, 77.85),
            ("AWS-05", "Station-South (E)",  12.82, 77.58),
        ]

        for stn_id, name, lat, lon in stations_info:
            dataset.stations[stn_id] = StationMetadata(
                station_id=stn_id, name=name,
                latitude=lat, longitude=lon,
                elevation_m=920.0, reliability_score=1.0,
            )

        for step in range(num_timesteps):
            t_curr = base_time + timedelta(minutes=step * timestep_minutes)
            hour = t_curr.hour + t_curr.minute / 60.0
            base_temp  = 24.0 + 6.0 * math.sin((hour - 9) * math.pi / 12.0)
            base_press = 1012.0 - 2.5 * math.sin((hour - 3) * math.pi / 12.0)
            base_rh_approx = 65.0 - 15.0 * math.sin((hour - 9) * math.pi / 12.0)
            base_ws = 6.0 + 2.0 * math.sin((hour - 12) * math.pi / 12.0)
            base_wd = 270.0

            for stn_id, _, lat, lon in stations_info:
                stn_seed = hash(stn_id) % 100
                stn_t = base_temp  + (stn_seed - 50) * 0.02 + np.random.normal(0, 0.15)
                stn_p = base_press + (stn_seed - 50) * 0.05 + np.random.normal(0, 0.08)
                rh_approx = max(10.0, min(99.0, base_rh_approx + (stn_seed - 50) * 0.1 + np.random.normal(0, 0.3)))
                stn_rh, _ = _make_thermodynamically_consistent(stn_t, stn_p, rh_approx)
                stn_ws = max(0.5, base_ws + np.random.normal(0, 0.2))
                stn_wd = (base_wd + np.random.normal(0, 3.0)) % 360.0

                obs = AWSObservation(
                    station_id=stn_id, timestamp=t_curr,
                    latitude=lat, longitude=lon,
                    temperature=float(stn_t), pressure=float(stn_p),
                    humidity=float(stn_rh), wind_speed=float(stn_ws),
                    wind_direction=float(stn_wd),
                    quality_flags=[QualityFlag.VALID],
                )
                dataset.add_observation(obs)

        return dataset

    @staticmethod
    def create_synthetic_network_benchmark(
        num_stations: int = 5,
        num_timesteps: int = 120,
        timestep_minutes: int = 5,
    ) -> Tuple[CanonicalDataset, Dict[str, Any]]:
        """
        Generates a realistic multi-station synthetic network with
        a genuine propagating cold front and isolated sensor faults.
        Uses thermodynamically consistent baseline throughout.
        """
        dataset = CanonicalDataset()
        base_time = datetime(2026, 8, 30, 8, 0, 0)

        stations_info = [
            ("AWS-01", "Station-West (A)",   12.95, 77.50),
            ("AWS-02", "Station-Center (B)", 12.95, 77.65),
            ("AWS-03", "Station-North (C)",  13.08, 77.58),
            ("AWS-04", "Station-East (D)",   12.95, 77.85),
            ("AWS-05", "Station-South (E)",  12.82, 77.58),
        ]

        for stn_id, name, lat, lon in stations_info:
            dataset.stations[stn_id] = StationMetadata(
                station_id=stn_id, name=name,
                latitude=lat, longitude=lon,
                elevation_m=920.0, reliability_score=1.0,
            )

        for step in range(num_timesteps):
            t_curr = base_time + timedelta(minutes=step * timestep_minutes)
            hour = t_curr.hour + t_curr.minute / 60.0

            base_temp  = 24.0 + 6.0 * math.sin((hour - 9) * math.pi / 12.0)
            base_press = 1012.0 - 2.5 * math.sin((hour - 3) * math.pi / 12.0)
            base_rh_raw = 65.0 - 15.0 * math.sin((hour - 9) * math.pi / 12.0)
            base_ws = 8.0
            base_wd = 270.0

            for stn_id, _, lat, lon in stations_info:
                stn_seed = hash(stn_id) % 100
                stn_t = base_temp  + (stn_seed - 50) * 0.02 + np.random.normal(0, 0.2)
                stn_p = base_press + (stn_seed - 50) * 0.05 + np.random.normal(0, 0.1)
                rh_raw = max(10.0, min(99.0, base_rh_raw + (stn_seed - 50) * 0.1 + np.random.normal(0, 0.5)))
                stn_rh, _ = _make_thermodynamically_consistent(stn_t, stn_p, rh_raw)
                stn_ws = max(0.5, base_ws + np.random.normal(0, 0.3))
                stn_wd = (base_wd + np.random.normal(0, 4.0)) % 360.0

                flags, meta = [], {}

                # Propagating Cold Front: A@40, B@47, D@56
                front_arrivals = {"AWS-01": 40, "AWS-02": 47, "AWS-04": 56}
                if stn_id in front_arrivals:
                    arr = front_arrivals[stn_id]
                    if step >= arr:
                        prog = min(1.0, (step - arr + 1) / 4.0)
                        stn_t  -= 6.0 * prog
                        stn_p  += 4.0 * prog
                        stn_rh  = min(99.0, stn_rh + 25.0 * prog)
                        stn_rh, _ = _make_thermodynamically_consistent(stn_t, stn_p, stn_rh)
                        stn_ws += 4.0 * prog
                        if arr <= step < arr + 4:
                            flags.append(QualityFlag.CONFIRMED_MET_EVENT)
                            meta["event"] = "GENUINE_COLD_FRONT"

                # Isolated temperature spike: AWS-03 at step 30
                if stn_id == "AWS-03" and step == 30:
                    stn_t += 14.0
                    flags.append(QualityFlag.INJECTED_ANOMALY)
                    meta["anomaly"] = "ISOLATED_TEMP_SPIKE"

                # Frozen humidity: AWS-05 at steps 60-85
                if stn_id == "AWS-05" and 60 <= step <= 85:
                    stn_rh = 85.0
                    flags.append(QualityFlag.INJECTED_ANOMALY)
                    meta["anomaly"] = "FROZEN_HUMIDITY"

                obs = AWSObservation(
                    station_id=stn_id, timestamp=t_curr,
                    latitude=lat, longitude=lon,
                    temperature=float(round(stn_t, 2)),
                    pressure=float(round(stn_p, 2)),
                    humidity=float(round(stn_rh, 1)),
                    wind_speed=float(round(stn_ws, 2)),
                    wind_direction=float(round(stn_wd, 1)),
                    quality_flags=flags or [QualityFlag.VALID],
                    source_metadata=meta,
                )
                dataset.add_observation(obs)

        return dataset, {
            "num_stations": len(stations_info),
            "num_timesteps": num_timesteps,
            "front_events": {"AWS-01": 40, "AWS-02": 47, "AWS-04": 56},
            "spike_events": {"AWS-03": 30},
            "freeze_events": {"AWS-05": (60, 85)},
        }

    # -- Primary 6-Scenario Benchmark Generator --------------------------------

    @classmethod
    def generate_station_synthetic_benchmark(
        cls,
        target_station_id: str,
        num_timesteps: int = 336,
        timestep_minutes: int = 60,
        top_k: int = 5,
    ) -> Tuple[CanonicalDataset, Dict[str, Any]]:
        """
        Generates a physics-consistent 6-scenario benchmark dataset for any Indian AWS station.

        ARCHITECTURE:
          Steps   0-179  -> CLEAN BASELINE (all stations clean) -> for model training
          Steps 180-335  -> TEST PERIOD (faults on target only) -> for inference

        THERMODYNAMIC CONSISTENCY:
          All T, P, RH values are generated such that:
            e_s(T) -> e = RH/100 * e_s -> T_v = T_K/[1-(1-eps)(e/p)]
          are physically self-consistent.

        6 SCENARIOS (all in held-out region 180-335):
          Sc1 (t=196): Temperature spike   +13.5°C   -> SENSOR FAULT
          Sc2 (t=215): Frozen temperature  12h stuck -> SENSOR FAULT
          Sc3 (t=252): Pressure drift      -0.32/h   -> SENSOR FAULT (GRADUAL)
          Sc4 (t=280): Humidity saturation 99% stuck -> SENSOR FAULT
          Sc5 (t=200): Propagating front   advects   -> GENUINE EVENT
          Sc6 (t=310): Hard negative       isolated  -> SENSOR FAULT (not genuine)

        GROUND TRUTH stored separately -- never passed to the model.
        """
        from skyguard.data.india_stations import (
            INDIA_AWS_CATALOG,
            get_station_metadata,
            find_nearest_neighbors,
        )
        from skyguard.dacm.geometry import haversine_distance_km, station_direction_vector
        from skyguard.dacm.wind_vector import meteorological_wind_to_vector
        from skyguard.dacm.alignment import compute_wind_alignment

        target_meta = get_station_metadata(target_station_id)
        if not target_meta:
            raise ValueError(f"Station '{target_station_id}' not found in All-India AWS Catalog.")

        # Resolve actual nearest neighbors within 80 km (widen to 250 km if isolated)
        neighbor_tuples = find_nearest_neighbors(target_station_id, max_distance_km=80.0, top_k=top_k)
        if not neighbor_tuples:
            neighbor_tuples = find_nearest_neighbors(target_station_id, max_distance_km=250.0, top_k=top_k)

        all_stations: List[StationMetadata] = [target_meta] + [n[0] for n in neighbor_tuples]
        dataset = CanonicalDataset()
        for stn in all_stations:
            dataset.stations[stn.station_id] = stn

        # -- Atmospheric parameters derived from station metadata --------------
        base_time   = datetime(2024, 6, 1, 0, 0, 0)
        elev_corr   = target_meta.elevation_m * 0.0065
        lat_corr    = abs(target_meta.latitude - 20.0) * 0.25
        base_t_mean = 32.0 - elev_corr - lat_corr
        base_p_mean = 1013.25 * math.pow(
            max(0.0, 1.0 - 2.25577e-5 * target_meta.elevation_m), 5.25588
        )

        # -- Baseline wind field: South-Westerly (240°) at 5.5 m/s ------------
        syn_wind_speed = 5.5
        syn_wind_dir   = 240.0
        wind_vec = meteorological_wind_to_vector(syn_wind_speed, syn_wind_dir)

        # -- Compute downstream advective delays tau (hours) for each neighbor --
        downstream_delays: Dict[str, int] = {}
        for stn in all_stations:
            if stn.station_id == target_station_id:
                downstream_delays[stn.station_id] = 0
                continue
            dist_km = haversine_distance_km(
                target_meta.latitude, target_meta.longitude,
                stn.latitude, stn.longitude,
            )
            r_vec, _ = station_direction_vector(
                target_meta.latitude, target_meta.longitude,
                stn.latitude, stn.longitude,
            )
            align, is_calm = compute_wind_alignment(wind_vec, r_vec, syn_wind_speed)
            if align >= 0.25 and not is_calm:
                v_par_kmh = max(0.5, syn_wind_speed * align * 3.6)
                tau_h = max(1, int(round(dist_km / v_par_kmh)))
                downstream_delays[stn.station_id] = tau_h
            else:
                downstream_delays[stn.station_id] = -1  # crosswind / not downstream

        # -- Scenario anchor timesteps (all in held-out region 180-335) --------
        CLEAN_END   = 180   # Exclusive: steps 0..179 are CLEAN training data
        SC1_SPIKE   = 196   # Temperature spike (+13.5°C) -> 1 timestep
        SC5_FRONT   = 200   # Propagating cold front starts -> lasts until ~210
        SC2_FREEZE  = 215   # Frozen temperature (12 steps stuck)
        SC3_DRIFT   = 252   # Pressure drift (-0.32 hPa/step from here)
        SC4_HUM_SAT = 280   # Humidity stuck at 99%
        SC6_HARD    = 310   # Hard negative: same magnitude as front, but isolated

        # Frozen temperature: anchor value sampled at t=215
        _freeze_anchor_T: Optional[float] = None

        # -- Generate time series ---------------------------------------------
        all_obs_by_time: Dict[datetime, List[AWSObservation]] = {}
        clean_train_observations: List[AWSObservation] = []

        for step in range(num_timesteps):
            t_curr = base_time + timedelta(hours=step)
            hour_of_day = t_curr.hour

            # Realistic diurnal solar heating + semi-diurnal pressure tide
            diurnal_rad    = math.sin((hour_of_day - 8.0) * math.pi / 12.0)
            semi_diurnal_p = math.cos((hour_of_day - 4.0) * math.pi / 12.0)

            t_cycle  = base_t_mean + (diurnal_rad * 6.0) + (math.sin(step * 0.08) * 0.7)
            p_cycle  = base_p_mean + (semi_diurnal_p * 1.8) + (math.cos(step * 0.05) * 0.9)
            rh_raw_cycle = max(18.0, min(92.0, 58.0 - (diurnal_rad * 28.0) + (math.sin(step * 0.12) * 3.5)))
            ws_cycle = max(1.0, syn_wind_speed + (diurnal_rad * 1.5) + (math.sin(step * 0.15) * 0.8))
            wd_cycle = (syn_wind_dir + (math.sin(step * 0.04) * 20.0)) % 360.0

            for stn in all_stations:
                stn_id   = stn.station_id
                stn_seed = hash(stn_id) % 100
                elev_diff = (stn.elevation_m - target_meta.elevation_m) * 0.0065

                # Station microclimate offset
                stn_t    = t_cycle - elev_diff + ((stn_seed - 50) * 0.015) + np.random.normal(0, 0.12)
                stn_p    = p_cycle - (stn.elevation_m - target_meta.elevation_m) * 0.12 \
                           + ((stn_seed - 50) * 0.03) + np.random.normal(0, 0.08)
                rh_raw   = max(15.0, min(95.0, rh_raw_cycle + ((stn_seed - 50) * 0.08) + np.random.normal(0, 0.25)))

                # Enforce thermodynamic consistency: T, P -> consistent RH
                stn_rh, stn_tv = _make_thermodynamically_consistent(stn_t, stn_p, rh_raw)
                stn_ws   = max(0.5, ws_cycle + np.random.normal(0, 0.15))
                stn_wd   = (wd_cycle + np.random.normal(0, 2.5)) % 360.0

                flags:    List[QualityFlag] = [QualityFlag.VALID]
                meta_ann: Dict[str, Any]   = {
                    "station_name": stn.name,
                    "elevation_m": stn.elevation_m,
                    "is_synthetic_benchmark": True,
                    "phase": "CLEAN_BASELINE" if step < CLEAN_END else "TEST_PERIOD",
                    "virtual_temp_k": round(stn_tv, 3),
                }
                is_target = (stn_id == target_station_id)

                # -------------------------------------------------------------
                # HELD-OUT REGION: inject faults (steps 180-335, target only)
                # CLEAN REGION  : no injection (steps 0-179, all stations)
                # -------------------------------------------------------------

                if step >= CLEAN_END:

                    # -- Sc5: Propagating Cold Front (target + downstream) --
                    arr_delay = downstream_delays.get(stn_id, -1)
                    if arr_delay >= 0:
                        stn_arrival = SC5_FRONT + arr_delay
                        if step >= stn_arrival:
                            prog = min(1.0, (step - stn_arrival + 1) / 4.0)
                            stn_t  -= 7.0 * prog
                            stn_p  += 4.5 * prog
                            rh_post = min(98.0, stn_rh + 24.0 * prog)
                            stn_rh, stn_tv = _make_thermodynamically_consistent(stn_t, stn_p, rh_post)
                            stn_ws += 4.5 * prog
                            if stn_arrival <= step < stn_arrival + 5:
                                flags.append(QualityFlag.CONFIRMED_MET_EVENT)
                                meta_ann.update({
                                    "benchmark_scenario": "SC5_PROPAGATING_COLD_FRONT",
                                    "expected_classification": "GENUINE_METEOROLOGICAL_EVENT",
                                    "arrival_step": stn_arrival,
                                    "propagation_delay_h": arr_delay,
                                    "wind_direction_deg": syn_wind_dir,
                                })

                    # -- The remaining scenarios apply ONLY to the target station --
                    if is_target:

                        # -- Sc1: Temperature spike (+13.5°C, 1 timestep) --
                        if step == SC1_SPIKE:
                            clean_t = stn_t
                            stn_t  += 13.5
                            flags.append(QualityFlag.INJECTED_ANOMALY)
                            meta_ann.update({
                                "benchmark_scenario": "SC1_TEMPERATURE_SPIKE",
                                "expected_classification": "STATION_SENSOR_FAULT",
                                "fault_type": "SUDDEN_SPIKE",
                                "affected_parameter": "temperature",
                                "injection_magnitude": 13.5,
                                "clean_value": round(clean_t, 2),
                                "injected_value": round(stn_t, 2),
                            })

                        # -- Sc2: Frozen Temperature Sensor (12 h stuck) --
                        if step == SC2_FREEZE:
                            _freeze_anchor_T = stn_t
                        if SC2_FREEZE <= step < SC2_FREEZE + 12 and _freeze_anchor_T is not None:
                            stn_t = _freeze_anchor_T
                            flags.append(QualityFlag.INJECTED_ANOMALY)
                            meta_ann.update({
                                "benchmark_scenario": "SC2_FROZEN_TEMPERATURE_SENSOR",
                                "expected_classification": "STATION_SENSOR_FAULT",
                                "fault_type": "FROZEN_SENSOR",
                                "affected_parameter": "temperature",
                                "frozen_value": round(_freeze_anchor_T, 2),
                                "duration_h": 12,
                            })

                        # -- Sc3: Pressure Sensor Drift (-0.32 hPa/step) --
                        if SC3_DRIFT <= step < SC3_DRIFT + 28:
                            drift_steps = step - SC3_DRIFT
                            drift_amount = drift_steps * 0.32
                            stn_p -= drift_amount
                            flags.append(QualityFlag.INJECTED_ANOMALY)
                            meta_ann.update({
                                "benchmark_scenario": "SC3_PRESSURE_SENSOR_DRIFT",
                                "expected_classification": "STATION_SENSOR_FAULT",
                                "fault_type": "GRADUAL_DRIFT",
                                "affected_parameter": "pressure",
                                "cumulative_drift_hpa": round(drift_amount, 2),
                                "rate_hpa_per_step": 0.32,
                            })

                        # -- Sc4: Humidity Sensor Stuck at 99% --
                        if SC4_HUM_SAT <= step < SC4_HUM_SAT + 20:
                            stn_rh = 99.0
                            flags.append(QualityFlag.INJECTED_ANOMALY)
                            meta_ann.update({
                                "benchmark_scenario": "SC4_HUMIDITY_SATURATION_FAULT",
                                "expected_classification": "STATION_SENSOR_FAULT",
                                "fault_type": "FROZEN_SENSOR",
                                "affected_parameter": "humidity",
                                "stuck_value": 99.0,
                                "duration_h": 20,
                            })

                        # -- Sc6: Hard Negative (Isolated Disturbance) --
                        if SC6_HARD <= step < SC6_HARD + 4:
                            prog6 = min(1.0, (step - SC6_HARD + 1) / 4.0)
                            stn_t  -= 7.0 * prog6
                            stn_p  += 4.5 * prog6
                            rh_h   = min(98.0, stn_rh + 24.0 * prog6)
                            stn_rh, stn_tv = _make_thermodynamically_consistent(stn_t, stn_p, rh_h)
                            stn_ws += 4.5 * prog6
                            flags.append(QualityFlag.INJECTED_ANOMALY)
                            meta_ann.update({
                                "benchmark_scenario": "SC6_HARD_NEGATIVE_ISOLATED_DISTURBANCE",
                                "expected_classification": "STATION_SENSOR_FAULT",
                                "fault_type": "MULTIVARIATE_INCONSISTENCY",
                                "note": "Same magnitude as Sc5 front, but no downstream propagation. "
                                        "DACM should flag as sensor fault, not genuine event.",
                            })

                obs = AWSObservation(
                    station_id=stn_id,
                    timestamp=t_curr,
                    latitude=stn.latitude,
                    longitude=stn.longitude,
                    temperature=float(round(stn_t, 2)),
                    pressure=float(round(stn_p, 2)),
                    humidity=float(round(max(5.0, min(100.0, stn_rh)), 1)),
                    wind_speed=float(round(stn_ws, 2)),
                    wind_direction=float(round(stn_wd, 1)),
                    quality_flags=flags,
                    source_metadata=meta_ann,
                )

                if t_curr not in all_obs_by_time:
                    all_obs_by_time[t_curr] = []
                all_obs_by_time[t_curr].append(obs)

                # Collect clean training observations (STRICT: only steps 0-179)
                if step < CLEAN_END:
                    clean_train_observations.append(obs)

        # Ingest into dataset chronologically
        for dt_key in sorted(all_obs_by_time.keys()):
            for o in all_obs_by_time[dt_key]:
                dataset.add_observation(o)

        # -- Scenario catalogue ------------------------------------------------
        scenarios = [
            {
                "id": "sc1_spike",
                "title": "Sc 1 — Isolated Temperature Spike",
                "type": "STATION_SENSOR_FAULT",
                "fault_type": "SUDDEN_SPIKE",
                "affected_parameter": "temperature",
                "timestep_index": SC1_SPIKE,
                "timestamp": (base_time + timedelta(hours=SC1_SPIKE)).isoformat(),
                "affected_station": target_station_id,
                "magnitude": 13.5,
                "description": (
                    f"Abrupt +13.5 °C temperature spike on {target_station_id} at t={SC1_SPIKE}. "
                    "Neighbor stations remain nominal. Thermodynamic inconsistency (T/P/RH) is strong. "
                    "No DACM propagation. Expected: STATION_SENSOR_FAULT."
                ),
                "expected_evidence": {
                    "temporal": "HIGH — abrupt step discontinuity",
                    "physics": "HIGH — T/P/RH thermodynamic residual elevated",
                    "dacm": "LOW — no downstream response",
                },
            },
            {
                "id": "sc2_freeze",
                "title": "Sc 2 — Frozen Temperature Sensor (12 h)",
                "type": "STATION_SENSOR_FAULT",
                "fault_type": "FROZEN_SENSOR",
                "affected_parameter": "temperature",
                "timestep_index": SC2_FREEZE + 6,
                "timestamp": (base_time + timedelta(hours=SC2_FREEZE + 6)).isoformat(),
                "affected_station": target_station_id,
                "description": (
                    f"Temperature sensor frozen at constant value from t={SC2_FREEZE} for 12 h. "
                    "Neighbours continue normal diurnal evolution. Near-zero temporal variance. "
                    "Expected: STATION_SENSOR_FAULT."
                ),
                "expected_evidence": {
                    "temporal": "HIGH — near-zero temporal variance in T",
                    "physics": "HIGH — T no longer tracking evolving P/RH",
                    "dacm": "LOW — no downstream temperature freeze",
                },
            },
            {
                "id": "sc3_drift",
                "title": "Sc 3 — Gradual Pressure Sensor Drift",
                "type": "STATION_SENSOR_FAULT",
                "fault_type": "GRADUAL_DRIFT",
                "affected_parameter": "pressure",
                "timestep_index": SC3_DRIFT + 14,
                "timestamp": (base_time + timedelta(hours=SC3_DRIFT + 14)).isoformat(),
                "affected_station": target_station_id,
                "description": (
                    f"Pressure sensor drifting at -0.32 hPa/step from t={SC3_DRIFT}. "
                    "Deviation grows slowly and becomes suspicious over hours. "
                    "Expected: STATION_SENSOR_FAULT (GRADUAL)."
                ),
                "expected_evidence": {
                    "temporal": "MODERATE->HIGH — slow growing deviation",
                    "physics": "MODERATE->HIGH — growing thermodynamic inconsistency",
                    "dacm": "LOW — neighbours do not drift",
                },
            },
            {
                "id": "sc4_hum_sat",
                "title": "Sc 4 — Humidity Saturation Fault (Stuck 99%)",
                "type": "STATION_SENSOR_FAULT",
                "fault_type": "FROZEN_SENSOR",
                "affected_parameter": "humidity",
                "timestep_index": SC4_HUM_SAT + 6,
                "timestamp": (base_time + timedelta(hours=SC4_HUM_SAT + 6)).isoformat(),
                "affected_station": target_station_id,
                "description": (
                    f"Humidity sensor stuck at 99 % saturation from t={SC4_HUM_SAT} for 20 h. "
                    "Atmosphere is hot and dry -> thermodynamic inconsistency detectable by Ch2. "
                    "Expected: STATION_SENSOR_FAULT."
                ),
                "expected_evidence": {
                    "temporal": "HIGH — flatlined RH during drying conditions",
                    "physics": "HIGH — vapour pressure inconsistency",
                    "dacm": "LOW — no regional moisture event",
                },
            },
            {
                "id": "sc5_front",
                "title": "Sc 5 — Genuine Propagating Cold Front",
                "type": "GENUINE_METEOROLOGICAL_EVENT",
                "fault_type": None,
                "affected_parameter": "all",
                "timestep_index": SC5_FRONT + 1,
                "timestamp": (base_time + timedelta(hours=SC5_FRONT + 1)).isoformat(),
                "affected_station": target_station_id,
                "downstream_delays": downstream_delays,
                "wind_direction_deg": syn_wind_dir,
                "description": (
                    f"Coherent cold front arrives at {target_station_id} (t={SC5_FRONT}) "
                    "then propagates to downstream neighbours with delay tau proportional to distance/wind. "
                    "Thermodynamically consistent. Wind-aligned. Expected: GENUINE_METEOROLOGICAL_EVENT."
                ),
                "expected_evidence": {
                    "temporal": "HIGH — coherent multi-variable transition",
                    "physics": "LOW — state change is internally consistent",
                    "dacm": "HIGH — downstream propagation confirmed with delay tau",
                },
            },
            {
                "id": "sc6_hard_neg",
                "title": "Sc 6 — Hard Negative (Isolated Disturbance)",
                "type": "STATION_SENSOR_FAULT",
                "fault_type": "MULTIVARIATE_INCONSISTENCY",
                "affected_parameter": "all",
                "timestep_index": SC6_HARD + 1,
                "timestamp": (base_time + timedelta(hours=SC6_HARD + 1)).isoformat(),
                "affected_station": target_station_id,
                "description": (
                    f"Same magnitude T/P/RH change as Sc5 front at t={SC6_HARD}, "
                    "but applied ONLY to {target_station_id}. No downstream propagation. "
                    "DACM evidence: absent. Expected: STATION_SENSOR_FAULT (not genuine event)."
                ),
                "expected_evidence": {
                    "temporal": "HIGH — same signal as Sc5",
                    "physics": "MODERATE — consistent internally, inconsistent vs neighbours",
                    "dacm": "LOW — no downstream response -> fault diagnosis",
                },
            },
        ]

        benchmark_metadata = {
            "target_station_id":   target_station_id,
            "target_station_name": target_meta.name,
            "num_stations":        len(all_stations),
            "num_timesteps":       num_timesteps,
            "clean_end_step":      CLEAN_END,
            "num_clean_train_obs": len(clean_train_observations),
            "downstream_delays":   downstream_delays,
            "syn_wind_speed_mps":  syn_wind_speed,
            "syn_wind_dir_deg":    syn_wind_dir,
            "base_t_mean_c":       round(base_t_mean, 2),
            "base_p_mean_hpa":     round(base_p_mean, 2),
            "thermodynamic_consistency": True,
            "scenarios":           scenarios,
            "ground_truth": {
                sc["id"]: {
                    "expected_classification": sc["type"],
                    "timestep_index": sc["timestep_index"],
                    "affected_station": target_station_id,
                }
                for sc in scenarios
            },
            "_clean_train_obs_ref": clean_train_observations,
        }

        return dataset, benchmark_metadata
