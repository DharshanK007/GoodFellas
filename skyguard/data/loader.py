"""
Modular AWS Data Loader with Schema & Column Auto-Detection.
"""

import json
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from skyguard.data.schema import AWSObservation, CanonicalDataset, StationMetadata, QualityFlag


class AWSDataLoader:
    """
    Ingests meteorological observation files (CSV, JSON, DataFrame, etc.)
    with automatic column synonym mapping and unit normalization.
    """

    COLUMN_SYNONYMS = {
        "station_id": ["station_id", "station", "stn_id", "id", "stnid", "site_id", "wban", "usaf"],
        "timestamp": ["timestamp", "time", "date", "datetime", "dt", "valid_time", "ob_time"],
        "latitude": ["latitude", "lat", "stn_lat", "y"],
        "longitude": ["longitude", "lon", "long", "stn_lon", "x"],
        "temperature": ["temperature", "temp", "t", "air_temp", "air_temperature", "t_c", "temp_c", "dry_bulb_temp"],
        "pressure": ["pressure", "p", "press", "baro", "mslp", "slp", "p_hpa", "atm_press", "stn_press"],
        "humidity": ["humidity", "rh", "relative_humidity", "rel_hum", "hum", "rel_humidity"],
        "wind_speed": ["wind_speed", "ws", "wspd", "speed", "wind_spd", "wind_velocity", "ff"],
        "wind_direction": ["wind_direction", "wd", "wdir", "direction", "wind_dir", "w_dir", "dd"],
    }

    @classmethod
    def resolve_column(cls, available_cols: List[str], target_field: str) -> Optional[str]:
        """Finds matching column name from synonyms (case-insensitive)."""
        lower_to_orig = {col.lower().strip(): col for col in available_cols}
        for synonym in cls.COLUMN_SYNONYMS.get(target_field, []):
            if synonym.lower() in lower_to_orig:
                return lower_to_orig[synonym.lower()]
        return None

    @classmethod
    def parse_timestamp(cls, val: Any) -> datetime:
        """Parses diverse timestamp formats to UTC datetime."""
        if isinstance(val, (datetime, pd.Timestamp)):
            if val.tzinfo is not None:
                return val.astimezone(timezone.utc).replace(tzinfo=None)
            return val
        if isinstance(val, (int, float)):
            # Check epoch seconds vs milliseconds
            if val > 1e11:
                return datetime.utcfromtimestamp(val / 1000.0)
            return datetime.utcfromtimestamp(val)
        
        str_val = str(val).strip()
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y%m%d%H%M",
            "%Y/%m/%d %H:%M:%S",
            "%d/%m/%Y %H:%M",
        ):
            try:
                dt = datetime.strptime(str_val, fmt)
                return dt
            except ValueError:
                continue
        # Fallback to dateutil/pandas parser
        return pd.to_datetime(str_val).to_pydatetime()

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, default_station: str = "AWS-DEFAULT") -> CanonicalDataset:
        """Constructs CanonicalDataset from pandas DataFrame."""
        cols = list(df.columns)
        mapping = {field: cls.resolve_column(cols, field) for field in cls.COLUMN_SYNONYMS}

        dataset = CanonicalDataset()
        for _, row in df.iterrows():
            stn_id = str(row[mapping["station_id"]]) if mapping["station_id"] else default_station
            ts_raw = row[mapping["timestamp"]] if mapping["timestamp"] else datetime.utcnow()
            ts = cls.parse_timestamp(ts_raw)

            lat = float(row[mapping["latitude"]]) if mapping["latitude"] and pd.notnull(row[mapping["latitude"]]) else 0.0
            lon = float(row[mapping["longitude"]]) if mapping["longitude"] and pd.notnull(row[mapping["longitude"]]) else 0.0

            # Mandatory variables
            t_raw = float(row[mapping["temperature"]]) if mapping["temperature"] and pd.notnull(row[mapping["temperature"]]) else 20.0
            # If temperature in Kelvin (>150), convert to °C
            if t_raw > 150.0:
                t_raw -= 273.15

            p_raw = float(row[mapping["pressure"]]) if mapping["pressure"] and pd.notnull(row[mapping["pressure"]]) else 1013.25
            # If pressure in Pa (>2000), convert to hPa
            if p_raw > 2000.0:
                p_raw /= 100.0

            rh_raw = float(row[mapping["humidity"]]) if mapping["humidity"] and pd.notnull(row[mapping["humidity"]]) else 50.0
            # If humidity in [0, 1] fraction, convert to %
            if 0.0 <= rh_raw <= 1.0 and rh_raw != 0.0 and rh_raw != 1.0:
                rh_raw *= 100.0

            ws = float(row[mapping["wind_speed"]]) if mapping["wind_speed"] and pd.notnull(row[mapping["wind_speed"]]) else None
            wd = float(row[mapping["wind_direction"]]) if mapping["wind_direction"] and pd.notnull(row[mapping["wind_direction"]]) else None

            obs = AWSObservation(
                station_id=stn_id,
                timestamp=ts,
                latitude=lat,
                longitude=lon,
                temperature=t_raw,
                pressure=p_raw,
                humidity=rh_raw,
                wind_speed=ws,
                wind_direction=wd,
                quality_flags=[]
            )
            dataset.add_observation(obs)

        return dataset

    @classmethod
    def from_csv(cls, file_path: Union[str, Path]) -> CanonicalDataset:
        """Loads canonical dataset from CSV."""
        df = pd.read_csv(file_path)
        return cls.from_dataframe(df)

    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> CanonicalDataset:
        """Loads canonical dataset from JSON."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            df = pd.DataFrame(data)
            return cls.from_dataframe(df)
        elif isinstance(data, dict) and "observations" in data:
            df = pd.DataFrame(data["observations"])
            return cls.from_dataframe(df)
        raise ValueError("Unsupported JSON structure for AWS observations.")
