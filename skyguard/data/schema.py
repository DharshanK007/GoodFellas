"""
Canonical Data Schema and Structures for SkyGuard AI.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np


class QualityFlag(str, Enum):
    VALID = "VALID"
    MISSING_VALUE = "MISSING_VALUE"
    OUT_OF_PHYSICAL_RANGE = "OUT_OF_PHYSICAL_RANGE"
    RATE_OF_CHANGE_EXCEEDED = "RATE_OF_CHANGE_EXCEEDED"
    CALM_WIND = "CALM_WIND"
    INTERPOLATED_SHORT_GAP = "INTERPOLATED_SHORT_GAP"
    INJECTED_ANOMALY = "INJECTED_ANOMALY"
    SUSPECTED_FAULT = "SUSPECTED_FAULT"
    CONFIRMED_MET_EVENT = "CONFIRMED_MET_EVENT"


@dataclass
class AWSObservation:
    """Canonical representation of a single AWS observation."""
    station_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    temperature: float          # Air Temperature in °C
    pressure: float             # Atmospheric Pressure in hPa / mbar
    humidity: float             # Relative Humidity in % (0 - 100)
    wind_speed: Optional[float] = None       # Wind Speed in m/s
    wind_direction: Optional[float] = None   # Wind Direction in degrees (0 - 360)
    quality_flags: List[QualityFlag] = field(default_factory=list)
    source_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def temperature_kelvin(self) -> float:
        """Thermodynamic temperature in Kelvin."""
        return self.temperature + 273.15

    @property
    def pressure_pa(self) -> float:
        """Atmospheric pressure in Pascals (Pa) for thermodynamic equations (1 hPa = 100 Pa)."""
        return self.pressure * 100.0

    @property
    def relative_humidity_fraction(self) -> float:
        """Relative humidity as fractional ratio in [0.0, 1.0]."""
        return float(np.clip(self.humidity / 100.0, 0.0, 1.0))

    @property
    def vector_3d(self) -> np.ndarray:
        """Core measurement vector X = [T, P, RH]."""
        return np.array([self.temperature, self.pressure, self.humidity], dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "station_id": self.station_id,
            "timestamp": self.timestamp.isoformat(),
            "latitude": self.latitude,
            "longitude": self.longitude,
            "temperature": float(self.temperature),
            "pressure": float(self.pressure),
            "humidity": float(self.humidity),
            "wind_speed": float(self.wind_speed) if self.wind_speed is not None else None,
            "wind_direction": float(self.wind_direction) if self.wind_direction is not None else None,
            "quality_flags": [q.value for q in self.quality_flags],
            "source_metadata": self.source_metadata,
        }


@dataclass
class PreparedStationData:
    """
    Deterministic Intermediate Representation of Analysis-Ready Station Data.
    Constructed strictly per single station_id without cross-contamination.
    """
    station_id: str
    metadata: "StationMetadata"
    raw_observations: List[AWSObservation]
    normalized_observations: List[AWSObservation]
    quality_flags: Dict[str, List[QualityFlag]]  # iso_timestamp -> quality flags
    missingness_flags: Dict[str, Any]
    temporal_diagnostics: Dict[str, Any]
    physics_ready_variables: Dict[str, np.ndarray]
    analysis_windows: np.ndarray  # Shape: (N_windows, Window_Size, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "station_id": self.station_id,
            "metadata": self.metadata.to_dict() if hasattr(self.metadata, "to_dict") else str(self.metadata),
            "total_observations": len(self.raw_observations),
            "total_windows": int(self.analysis_windows.shape[0]) if self.analysis_windows is not None else 0,
            "temporal_diagnostics": self.temporal_diagnostics,
            "missingness_summary": self.missingness_flags,
            "start_time": self.raw_observations[0].timestamp.isoformat() if self.raw_observations else None,
            "end_time": self.raw_observations[-1].timestamp.isoformat() if self.raw_observations else None,
        }


@dataclass
class DynamicStationContext:
    """
    Station-Conditioned Dynamic Spatial & Advective Context.
    Explicitly pairs primary station with its dynamically determined geographic neighbors.
    """
    primary_station_id: str
    contextual_station_ids: List[str]
    spatial_relationships: List[Dict[str, Any]]
    temporal_alignment: Dict[str, Any]
    primary_features: Dict[str, Any]
    contextual_features: Dict[str, Any]
    coupling_results: List[Any]
    coupling_quality: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_station_id": self.primary_station_id,
            "contextual_station_ids": self.contextual_station_ids,
            "spatial_relationships": self.spatial_relationships,
            "temporal_alignment": self.temporal_alignment,
            "coupling_quality": round(self.coupling_quality, 4),
            "active_couplings_count": len(self.coupling_results),
        }


@dataclass
class CorrectedObservation:
    """
    Self-Healing Data Container.
    The raw observation is NEVER modified. Imputed values are stored alongside
    confidence, attribution reason, and processing timestamp.
    """
    raw_observation: AWSObservation
    imputed_temperature: Optional[float] = None
    imputed_pressure: Optional[float] = None
    imputed_humidity: Optional[float] = None
    imputation_reason: str = ""
    imputation_confidence: float = 0.0
    processed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw": self.raw_observation.to_dict(),
            "imputed_temperature": self.imputed_temperature,
            "imputed_pressure": self.imputed_pressure,
            "imputed_humidity": self.imputed_humidity,
            "imputation_reason": self.imputation_reason,
            "imputation_confidence": self.imputation_confidence,
            "processed_at": self.processed_at.isoformat(),
        }


@dataclass
class StationMetadata:
    """Metadata for an Automatic Weather Station."""
    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation_m: float = 0.0
    reliability_score: float = 1.0   # Dynamic score R_i in [0, 1]
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "station_id": self.station_id,
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation_m": self.elevation_m,
            "reliability_score": self.reliability_score,
            "active": self.active,
        }


class CanonicalDataset:
    """Container for multi-station AWS observations and station registry."""
    def __init__(self, observations: Optional[List[AWSObservation]] = None, stations: Optional[Dict[str, StationMetadata]] = None):
        self.observations: List[AWSObservation] = observations or []
        self.stations: Dict[str, StationMetadata] = stations or {}

    def add_observation(self, obs: AWSObservation) -> None:
        self.observations.append(obs)
        if obs.station_id not in self.stations:
            self.stations[obs.station_id] = StationMetadata(
                station_id=obs.station_id,
                name=f"Station-{obs.station_id}",
                latitude=obs.latitude,
                longitude=obs.longitude
            )

    def get_station_series(self, station_id: str) -> List[AWSObservation]:
        """Returns chronological series for a given station."""
        series = [obs for obs in self.observations if obs.station_id == station_id]
        series.sort(key=lambda x: x.timestamp)
        return series

    def get_time_slice(self, timestamp: datetime) -> List[AWSObservation]:
        """Returns all station observations at a specific timestamp."""
        return [obs for obs in self.observations if obs.timestamp == timestamp]

    def all_station_ids(self) -> List[str]:
        return list(self.stations.keys())

    def __len__(self) -> int:
        return len(self.observations)
