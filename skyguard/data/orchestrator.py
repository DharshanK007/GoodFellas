"""
SkyGuard AI — Data Orchestration Contracts.

Defines explicit lifecycle states, run identity, and station execution context
so that real/historical and synthetic benchmark pipelines can never silently
share or overwrite each other's data.

NO MODEL / SCIENTIFIC LOGIC IS DEFINED HERE.
This file is a pure data-contract layer.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class DataLifecycleState(str, Enum):
    IDLE                = "IDLE"
    STATION_SELECTED    = "STATION_SELECTED"
    FETCHING            = "FETCHING"
    FETCHED             = "FETCHED"
    VALIDATING          = "VALIDATING"
    ALIGNED             = "ALIGNED"
    READY_FOR_TRAINING  = "READY_FOR_TRAINING"
    TRAINING            = "TRAINING"
    MODEL_READY         = "MODEL_READY"
    INFERENCE_READY     = "INFERENCE_READY"
    ERROR               = "ERROR"


class BenchmarkLifecycleState(str, Enum):
    NOT_READY           = "NOT_READY"
    READY_FOR_GENERATION= "READY_FOR_GENERATION"
    GENERATING          = "GENERATING"
    GENERATED           = "GENERATED"
    RUNNING             = "RUNNING"
    COMPLETE            = "COMPLETE"
    ERROR               = "ERROR"


class DataSourceType(str, Enum):
    REAL_HISTORICAL     = "REAL_HISTORICAL"
    SYNTHETIC_BENCHMARK = "SYNTHETIC_BENCHMARK"
    UNKNOWN             = "UNKNOWN"


@dataclass
class NeighbourInfo:
    station_id: str
    name: str
    latitude: float
    longitude: float
    distance_km: float
    bearing_deg: float
    elevation_m: float = 0.0
    dacm_weight: float = 0.0


@dataclass
class StationExecutionContext:
    station_id: str
    station_name: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    latitude: float = 0.0
    longitude: float = 0.0
    elevation_m: float = 0.0
    state: str = ""
    district: str = ""
    region: str = ""
    timezone: str = "UTC"
    neighbour_ids: List[str] = field(default_factory=list)
    neighbour_info: List[NeighbourInfo] = field(default_factory=list)
    requested_start: Optional[datetime] = None
    requested_end: Optional[datetime] = None
    expected_frequency_minutes: int = 60
    source_type: DataSourceType = DataSourceType.REAL_HISTORICAL
    real_lifecycle: DataLifecycleState = DataLifecycleState.IDLE
    benchmark_lifecycle: BenchmarkLifecycleState = BenchmarkLifecycleState.NOT_READY
    created_at: datetime = field(default_factory=datetime.utcnow)
    dataset_coverage: Dict[str, Any] = field(default_factory=dict)
    benchmark_results: Optional[Dict[str, Any]] = None

    def is_model_ready(self) -> bool:
        return self.real_lifecycle == DataLifecycleState.MODEL_READY

    def can_run_benchmark(self) -> bool:
        return self.is_model_ready()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "station_id": self.station_id,
            "station_name": self.station_name,
            "run_id": self.run_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation_m": self.elevation_m,
            "state": self.state,
            "district": self.district,
            "source_type": self.source_type.value,
            "real_lifecycle": self.real_lifecycle.value,
            "benchmark_lifecycle": self.benchmark_lifecycle.value,
            "neighbour_ids": self.neighbour_ids,
            "requested_start": self.requested_start.isoformat() if self.requested_start else None,
            "requested_end": self.requested_end.isoformat() if self.requested_end else None,
            "created_at": self.created_at.isoformat(),
            "dataset_coverage": self.dataset_coverage,
        }


@dataclass
class ScenarioBenchmarkResult:
    scenario_id: str
    scenario_title: str
    station_id: str
    run_id: str
    expected_class: str
    predicted_class: str
    confidence: float
    local_anomaly_score: float
    temporal_score: float
    physics_score: float
    dacm_propagation_evidence: float
    passed: bool
    model_reasoning: str
    source_type: DataSourceType = DataSourceType.SYNTHETIC_BENCHMARK

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_title": self.scenario_title,
            "station_id": self.station_id,
            "run_id": self.run_id,
            "expected_class": self.expected_class,
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 4),
            "local_anomaly_score": round(self.local_anomaly_score, 4),
            "temporal_score": round(self.temporal_score, 4),
            "physics_score": round(self.physics_score, 4),
            "dacm_propagation_evidence": round(self.dacm_propagation_evidence, 4),
            "passed": self.passed,
            "model_reasoning": self.model_reasoning,
            "source_type": self.source_type.value,
        }
